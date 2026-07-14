"""Reader-free tests for the registered M6 input factory (§2.8.10.4).

Always runnable: they exercise each factory step -- the pe-us version gate,
the claiming sha256 tamper gate, the NAWI/wage-base replacement rule, the NCHS
2010 band collapse, and the ``<= T*`` mortality-exposure adapter -- on
synthetic frames and the committed references. They deliberately do **not**
run ``build_inputs()`` end-to-end (which reads staged PSID and is reserved for
the registered gate), execute no scored run, compute no gate cell, and never
touch ``gates.yaml``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.engine.forward_earnings import fit_projected_wage_index
from populace_dynamics.engine.refit import (
    prepare_mortality_refit_inputs,
    validate_external_vintage,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mortality_floors as mf  # noqa: E402
import registered_m6_inputs as factory  # noqa: E402

_MORTALITY_COLUMNS = [
    "event_year",
    "required_interview_year",
    "age_band",
    "sex",
    "start_weight",
    "exposure",
    "death",
]


# --------------------------------------------------------------------------
# Step 1 -- the policyengine-us version gate
# --------------------------------------------------------------------------
def test__version_gate__passes_on_the_pinned_version():
    assert factory.assert_pe_us_version("1.752.2") == "1.752.2"


def test__version_gate__fires_on_a_wrong_version():
    with pytest.raises(RuntimeError, match="1.752.2"):
        factory.assert_pe_us_version("9.9.9")


def test__version_gate__default_reads_importlib_metadata(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.752.2")
    assert factory.assert_pe_us_version() == "1.752.2"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.700.0")
    with pytest.raises(RuntimeError):
        factory.assert_pe_us_version()


def test__version_gate__pins_the_certified_frame_constant():
    from populace_dynamics.data.deployment_frame import CERTIFIED_PIN

    assert CERTIFIED_PIN["model_version"] == factory.PINNED_PE_US_VERSION
    assert factory.PINNED_PE_US_VERSION == "1.752.2"


# --------------------------------------------------------------------------
# Step 2 -- the leakage-safe NAWI / wage-base surface
# --------------------------------------------------------------------------
def _synthetic_nawi() -> dict[int, float]:
    # Realized trailing decade [2005, 2014] plus post-boundary realized values
    # that differ from the projection, so a correct replacement is visible.
    nawi = {
        year: 1000.0 * (1.03 ** (year - 2005)) for year in range(2005, 2015)
    }
    nawi.update({2015: 90000.0, 2016: 91000.0, 2020: 95000.0})
    return nawi


def test__nawi_rule__replaces_post_boundary_with_iproj_exactly():
    nawi = _synthetic_nawi()
    wage_base = {2013: 113700.0, 2014: 117000.0, 2015: 118500.0}
    new_nawi, _ = factory.replace_nawi_wage_base(nawi, wage_base)
    projection = fit_projected_wage_index(nawi, boundary_year=2014)
    for year in (2015, 2016, 2020):
        assert new_nawi[year] == projection.projected(year)
        # the realized post-boundary value is gone (replaced, not kept).
        assert new_nawi[year] != nawi[year]


def test__nawi_rule__keeps_realized_pre_boundary_and_the_key_range():
    nawi = _synthetic_nawi()
    new_nawi, _ = factory.replace_nawi_wage_base(nawi, {2014: 117000.0})
    assert set(new_nawi) == set(nawi)  # max(nawi) key unchanged
    for year in range(2005, 2015):
        assert new_nawi[year] == nawi[year]


def test__wage_base_rule__truncates_change_years_after_the_boundary():
    nawi = _synthetic_nawi()
    wage_base = {
        2013: 113700.0,
        2014: 117000.0,
        2015: 118500.0,
        2017: 127200.0,
    }
    _, new_wage_base = factory.replace_nawi_wage_base(nawi, wage_base)
    assert sorted(new_wage_base) == [2013, 2014]
    assert new_wage_base[2014] == 117000.0


# --------------------------------------------------------------------------
# Step 3 -- the claiming reference sha256 tamper gate
# --------------------------------------------------------------------------
def test__claiming_json__sha256_equals_the_pinned_factory_constant():
    digest = hashlib.sha256(
        factory.CLAIMING_REFERENCE_PATH.read_bytes()
    ).hexdigest()
    assert digest == factory.CLAIMING_REFERENCE_SHA256


def test__sha_gate__loads_the_untampered_reference():
    reference = factory.load_claiming_reference()
    assert reference.supplement_year == 2014
    assert reference.schema_version == "ssa_claim_ages.v1"


def test__sha_gate__fires_on_a_tampered_reference(tmp_path):
    tampered = tmp_path / "claim.json"
    doc = json.loads(factory.CLAIMING_REFERENCE_PATH.read_text())
    doc["data"]["male"]["2013"]["number_thousands"] += 1
    tampered.write_text(json.dumps(doc, indent=2) + "\n")
    with pytest.raises(ValueError, match="sha256"):
        factory.load_claiming_reference(tampered)


# --------------------------------------------------------------------------
# Step 4a -- NCHS 2010 external central rates
# --------------------------------------------------------------------------
def test__nchs_rates__have_the_pinned_columns_and_fourteen_rows():
    rates = factory.nchs_2010_external_rates()
    assert list(rates.columns) == [
        "lower_age",
        "upper_age",
        "age_band",
        "sex",
        "central_rate",
    ]
    # seven MORTALITY_BANDS x {female, male}.
    assert len(rates) == 14
    assert set(rates["sex"]) == {"female", "male"}
    assert rates.set_index(["age_band", "sex"]).index.is_unique


def test__nchs_rates__reshape_the_canonical_band_collapse():
    rates = factory.nchs_2010_external_rates()
    canonical = mf.nchs_band_rates(
        json.loads(factory.NCHS_2010_PATH.read_text())
    )
    for row in rates.itertuples(index=False):
        assert row.central_rate == canonical[f"{row.age_band}|{row.sex}"]
        # the endpoints are the band bounds; the open top band is 85+.
        assert row.age_band == mf.band_label(row.lower_age, row.upper_age)
    top = rates[rates["age_band"] == "85+"].iloc[0]
    assert (int(top.lower_age), int(top.upper_age)) == (85, 120)


def test__nchs_rates__reject_a_wrong_vintage(tmp_path):
    doc = json.loads(factory.NCHS_2010_PATH.read_text())
    doc["vintage_year"] = 2023
    path = tmp_path / "nchs.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="vintage"):
        factory.nchs_2010_external_rates(path)


# --------------------------------------------------------------------------
# Step 4b -- the <= T* mortality-exposure adapter
# --------------------------------------------------------------------------
def _synthetic_panels():
    # Biennial waves 2011, 2013, 2015. Person 1 survives; person 2 dies in
    # 2012 (inside the 2011->2013 interval).
    demo = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2],
            "period": [2011, 2013, 2015, 2011, 2013],
            "age": [40, 42, 44, 60, 62],
            "weight": [1.0, 1.0, 1.0, 2.0, 2.0],
            "interview": [1, 1, 1, 1, 1],
        }
    )
    death_records = pd.DataFrame(
        {
            "person_id": [1, 2],
            "sex": ["female", "male"],
            "death_year": pd.array([pd.NA, 2012], dtype="Int64"),
        }
    )
    return demo, death_records


def test__adapter__emits_exactly_the_seven_pinned_columns():
    demo, deaths = _synthetic_panels()
    adapted = factory.mortality_exposure_adapter(demo, deaths)
    assert list(adapted.columns) == _MORTALITY_COLUMNS


def test__adapter__dates_every_slice_by_its_interval_closing_wave():
    demo, deaths = _synthetic_panels()
    adapted = factory.mortality_exposure_adapter(demo, deaths)
    grid = sorted(demo["period"].unique())
    closing = {grid[i]: grid[i + 1] for i in range(len(grid) - 1)}
    # required_interview_year is the interval's closing wave on every slice.
    for row in adapted.itertuples(index=False):
        # each slice's start wave is event_year (slice 0) or event_year-1.
        assert row.required_interview_year in closing.values()
    # event_year advances 0/1 within a biennial interval.
    assert set(adapted["event_year"]) == {2011, 2012, 2013, 2014}


def test__adapter__marks_the_death_year_slice():
    demo, deaths = _synthetic_panels()
    adapted = factory.mortality_exposure_adapter(demo, deaths)
    death_slice = adapted[
        (adapted["sex"] == "male") & (adapted["event_year"] == 2012)
    ]
    assert len(death_slice) == 1
    assert death_slice.iloc[0]["exposure"] == 0.5
    assert death_slice.iloc[0]["death"] == 1.0
    # survivor slices carry full exposure and no death.
    survivors = adapted[adapted["death"] == 0.0]
    assert (survivors["exposure"] == 1.0).all()


def test__adapter__drops_the_2013_to_2015_interval_under_truncation():
    demo, deaths = _synthetic_panels()
    adapted = factory.mortality_exposure_adapter(demo, deaths)
    # the boundary-straddling interval is emitted (closing wave 2015)...
    assert (adapted["required_interview_year"] == 2015).any()
    prepared = prepare_mortality_refit_inputs(
        adapted,
        factory.nchs_2010_external_rates(),
        external_vintage_year=2010,
        boundary_year=2014,
    )
    kept = prepared.exposure
    # ...and dropped wholesale by the closing-wave truncation.
    assert (kept["required_interview_year"] == 2015).sum() == 0
    assert int(kept["event_year"].max()) == 2012  # window ends 2011->2013


def test__adapter__age_band_uses_the_shared_band_label():
    demo, deaths = _synthetic_panels()
    adapted = factory.mortality_exposure_adapter(demo, deaths)
    assert set(adapted["age_band"]) <= set(mf.BAND_LABELS)


def test__adapter__dates_event_year_by_true_start_wave_age():
    # A person aged 24 at the start wave: build_exposure_slices drops the
    # age-24 slice (out of band) and keeps the age-25 second slice. event_year
    # must be start_wave + 1 (the age-25 slice's calendar year), read from the
    # panel's true start-wave age, not inferred from the surviving slice.
    demo = pd.DataFrame(
        {
            "person_id": [7, 7],
            "period": [2011, 2013],
            "age": [24, 26],
            "weight": [1.0, 1.0],
            "interview": [1, 1],
        }
    )
    death_records = pd.DataFrame(
        {
            "person_id": [7],
            "sex": ["female"],
            "death_year": pd.array([pd.NA], dtype="Int64"),
        }
    )
    adapted = factory.mortality_exposure_adapter(demo, death_records)
    band_25 = adapted[adapted["age_band"] == "25-34"]
    assert set(band_25["event_year"]) == {2012}
    assert set(band_25["required_interview_year"]) == {2013}


# --------------------------------------------------------------------------
# Declared vintages (leakage-safety of the hardcoded bindings)
# --------------------------------------------------------------------------
def test__declared_vintages__are_all_pre_boundary_and_admitted():
    assert factory.SSA_PARAMS_VINTAGE == 2014
    assert factory.MORTALITY_EXTERNAL_VINTAGE == 2010
    for name, vintage in (
        ("SSA earnings-index parameters", factory.SSA_PARAMS_VINTAGE),
        ("mortality reference", factory.MORTALITY_EXTERNAL_VINTAGE),
    ):
        validate_external_vintage(name, vintage, boundary_year=2014)


def test__factory__has_no_gates_yaml_dependency():
    # The build/test lane never scores or edits the pre-registration
    # contract; the factory has no gates.yaml read or write.
    source = (SCRIPTS / "registered_m6_inputs.py").read_text()
    assert "gates.yaml" not in source
