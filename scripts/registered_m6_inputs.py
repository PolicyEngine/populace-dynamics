"""The registered M6 input factory (design amendment 3d, §2.8.10.4).

``run_gate_m6_candidate1.py`` is not self-starting: it needs an
``--input-factory module:callable`` returning an ``M6HarnessInputs``. This
module is that factory. Invoked as::

    python scripts/run_gate_m6_candidate1.py \\
        --input-factory registered_m6_inputs:build_inputs

``importlib.import_module("registered_m6_inputs")`` resolves because
``scripts/`` is ``sys.path[0]`` when the runner is launched from the repo
root (``run_gate_m6_candidate1.py:42``).

Every binding is **hardcoded** to a single ``<= T*`` source and vintage; there
is no argument and no environment-derived vintage selection beyond
``POPULACE_DYNAMICS_PE_US_DIR`` -> the pinned policyengine-us 1.752.2 install
(``ss/params.py`` resolves it). The five deterministic steps
(§2.8.10.4), in order:

1. Assert ``importlib.metadata.version("policyengine-us") ==
   CERTIFIED_PIN["model_version"] == "1.752.2"``, assert the directory
   ``load_ssa_parameters`` will read resolves to that same distribution's
   on-disk install (``assert_pe_us_param_dir``, §2.8.10.5 F2), and load
   ``params_full = load_ssa_parameters()`` (its load-time cross-check runs on
   the realized series and passes).
2. ``params = dataclasses.replace(params_full, nawi=…, wage_base=…)``:
   realized NAWI for years ``<= T*`` kept, every year ``> T*`` **replaced**
   (not truncated) with the §2.7.6.3 projection ``I_proj`` byte-identical to
   ``engine.forward_earnings.fit_projected_wage_index`` over the same year-key
   range; wage-base change-years truncated to ``<= T*``. ``replace`` follows
   the load so the frozen dataclass's absent ``__post_init__`` bypasses the
   bend-point cross-check (which is intended for the realized series).
3. ``claiming_reference = claiming.load_claim_age_reference(DATA /
   "ssa_claim_ages_2014supplement.json")`` and **assert the JSON file's
   sha256 == a hardcoded constant** -- the artifact actually consumed. (The
   raw-source HTML's hash is build-time provenance, not this tamper gate.)
4. ``mortality_external_rates`` from the committed NCHS 2010 life table (band
   collapse) and ``mortality_exposure`` from the ``<= T*`` PSID person-interval
   slices via the §2.8.10.3 adapter, then the inert ``(0, 24)``
   projection-coverage pad appended to **both** (§2.8.10.5 F1) so the fitted
   model spans 0->120 with a ``<25`` hazard of exactly ``0``.
5. ``return load_m6_inputs(ssa_params=params, ssa_params_vintage=2014,
   claiming_reference=…, mortality_exposure=…, mortality_external_rates=…,
   mortality_external_vintage=2010)`` (boundary_year=2014, earnings_seed=5200).
   ``load_m6_inputs`` re-validates every vintage at assembly, so the harness
   re-checks the factory's bindings before any refit, score, or artifact write.

The factory is exercised on the committed references + staged PSID **only at
the registered run**; the build/test lane never runs it against real data
(runner docstring, §2.8.8). Its helper steps are individually unit-tested on
synthetic/committed inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from populace_dynamics import claiming
from populace_dynamics.data import deaths, panels
from populace_dynamics.data.deployment_frame import CERTIFIED_PIN
from populace_dynamics.engine.forward_earnings import fit_projected_wage_index
from populace_dynamics.engine.refit import BOUNDARY_YEAR
from populace_dynamics.harness.m6_cells import MORTALITY_BANDS
from populace_dynamics.harness.m6_inputs import (
    DEFAULT_EARNINGS_SEED,
    M6HarnessInputs,
    load_m6_inputs,
)
from populace_dynamics.ss.params import (
    _SSA,
    _resolve_pe_us,
    load_ssa_parameters,
)

# ``scripts/`` is sys.path[0] under the runner, but make the sibling
# ``build_mortality_floors`` importable from anywhere (tests, ad-hoc use) so
# the mortality helpers resolve regardless of the caller's CWD.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

#: Repo-root-anchored external-data directory (mirrors ``claiming._ROOT``);
#: never CWD-relative.
DATA = Path(__file__).resolve().parents[1] / "data" / "external"

#: The pinned policyengine-us vintage (§2.8.10.2), shared with every
#: ``gate_w1`` artifact via ``CERTIFIED_PIN``.
PINNED_PE_US_VERSION = "1.752.2"

#: Declared SSA-parameter and mortality-reference vintages (both ``<= T*``).
SSA_PARAMS_VINTAGE = 2014
MORTALITY_EXTERNAL_VINTAGE = 2010

#: The committed claiming artifact the factory consumes, and its pinned
#: sha256 -- the **load-time tamper gate** (§2.8.10.4). Produced byte-
#: reproducibly by ``scripts/extract_ssa_claim_ages_2014.py`` from the
#: committed raw HTML source. Any drift in the consumed JSON fails the build.
CLAIMING_REFERENCE_PATH = DATA / "ssa_claim_ages_2014supplement.json"
CLAIMING_REFERENCE_SHA256 = (
    "b88e45a08909f0f88a0ff37074c757891759ec988fd7e9fe14362eebd0abe462"
)

#: The committed NCHS United States Life Tables, 2010 (NVSR 63-7); the
#: ``<= T*`` mortality vintage (§2.8.10.3).
NCHS_2010_PATH = DATA / "nchs_life_tables_2010.json"


# --------------------------------------------------------------------------
# Step 1 -- the policyengine-us 1.752.2 version gate
# --------------------------------------------------------------------------
def assert_pe_us_version(version: str | None = None) -> str:
    """Assert the installed policyengine-us equals the pinned vintage.

    ``importlib.metadata.version("policyengine-us") ==
    CERTIFIED_PIN["model_version"] == "1.752.2"`` (§2.8.10.2). ``version`` may
    be supplied to exercise the guard without an install; it defaults to the
    live ``importlib.metadata`` reading in the env the parameters load from.
    """
    pinned = CERTIFIED_PIN["model_version"]
    if pinned != PINNED_PE_US_VERSION:
        raise RuntimeError(
            f"CERTIFIED_PIN model_version {pinned!r} != pinned "
            f"{PINNED_PE_US_VERSION!r}; the deployment frame drifted."
        )
    resolved = (
        version
        if version is not None
        else importlib.metadata.version("policyengine-us")
    )
    if resolved != PINNED_PE_US_VERSION:
        raise RuntimeError(
            f"policyengine-us {resolved!r} != pinned {PINNED_PE_US_VERSION!r}; "
            "the M6 factory requires the certified parameter vintage. Install "
            "policyengine-us==1.752.2 and point POPULACE_DYNAMICS_PE_US_DIR at "
            "it."
        )
    return resolved


def assert_pe_us_param_dir(pe_us_dir: Path | None = None) -> Path:
    """Assert the SSA parameter dir is the metadata-versioned install.

    ``assert_pe_us_version`` reads ``importlib.metadata``, but
    ``load_ssa_parameters`` reads YAML from ``_resolve_pe_us(None)`` --
    ``POPULACE_DYNAMICS_PE_US_DIR`` or the default checkout -- **decoupled**
    from the metadata, so a mismatched directory would pass the version gate
    and silently load another parameter vintage (§2.8.10.5, F2). This binds
    the directory the loader **will read** to the on-disk location of the
    **same** distribution the version gate reads. ``pe_us_dir`` may be
    supplied to exercise the guard against an injected root; it defaults to
    the live resolution ``load_ssa_parameters`` itself uses.
    """
    resolved = (_resolve_pe_us(pe_us_dir) / _SSA).resolve()
    versioned = (
        Path(
            importlib.metadata.distribution("policyengine-us").locate_file(
                "policyengine_us"
            )
        ).resolve()
        / "parameters"
        / "gov"
        / "ssa"
    )
    if resolved != versioned:
        raise RuntimeError(
            "POPULACE_DYNAMICS_PE_US_DIR resolves SSA parameters to a "
            "directory that is not the metadata-versioned policyengine-us "
            "install; point it at the 1.752.2 install."
        )
    return resolved


# --------------------------------------------------------------------------
# Step 2 -- the leakage-safe NAWI / wage-base surface
# --------------------------------------------------------------------------
def replace_nawi_wage_base(
    nawi: Mapping[int, float],
    wage_base: Mapping[int, float],
    *,
    boundary_year: int = BOUNDARY_YEAR,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return the ``<= T*``-admissible NAWI and wage-base mappings.

    NAWI: realized values for years ``<= boundary_year`` kept verbatim; every
    year ``> boundary_year`` **replaced** with ``I_proj`` from
    ``fit_projected_wage_index`` (the §2.7.6.3 log-linear fit over the trailing
    decade ``[boundary_year-9, boundary_year]``) -- **byte-identical to the
    forward law's own re-derivation**, over the **same year-key range** (so the
    admitted universe, a function of the key range not the values, is
    unchanged). Wage base: change-years ``> boundary_year`` dropped.

    Pure and reader-free, so it is unit-tested directly; the factory applies it
    through ``dataclasses.replace`` on the frozen ``SSAParameters``.
    """
    projection = fit_projected_wage_index(nawi, boundary_year=boundary_year)
    new_nawi = {
        int(year): (
            float(value)
            if int(year) <= boundary_year
            else projection.projected(int(year))
        )
        for year, value in nawi.items()
    }
    new_wage_base = {
        int(year): float(value)
        for year, value in wage_base.items()
        if int(year) <= boundary_year
    }
    return new_nawi, new_wage_base


def boundary_ssa_parameters():
    """Load policyengine-us 1.752.2 parameters and apply the NAWI rule.

    Order is pinned: **load (cross-check on realized) ->
    ``dataclasses.replace``**. ``SSAParameters`` is a frozen dataclass with no
    ``__post_init__``, so ``replace`` bypasses the bend-point cross-check --
    which is intended for the realized series and would spuriously trip on the
    ``I_proj``-derived post-2014 bend points.
    """
    params_full = load_ssa_parameters()
    new_nawi, new_wage_base = replace_nawi_wage_base(
        params_full.nawi, params_full.wage_base
    )
    return dataclasses.replace(
        params_full, nawi=new_nawi, wage_base=new_wage_base
    )


# --------------------------------------------------------------------------
# Step 3 -- the claiming reference (with the sha256 tamper gate)
# --------------------------------------------------------------------------
def load_claiming_reference(
    path: Path = CLAIMING_REFERENCE_PATH,
    *,
    expected_sha256: str = CLAIMING_REFERENCE_SHA256,
) -> claiming.ClaimAgeReference:
    """Load the 2014-Supplement claiming reference, asserting its sha256.

    The sha256 is of the JSON file **actually consumed** -- the load-time
    tamper gate (§2.8.10.4). The raw-source HTML's hash lives separately in the
    JSON's ``provenance.source_sha256`` as build-time provenance.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"claiming reference {path} sha256 {digest} != pinned "
            f"{expected_sha256}; the consumed artifact has changed (tamper "
            "gate). Rebuild via scripts/extract_ssa_claim_ages_2014.py and "
            "re-pin only after verifying provenance."
        )
    return claiming.load_claim_age_reference(path)


# --------------------------------------------------------------------------
# Step 4a -- NCHS 2010 external central rates (band collapse)
# --------------------------------------------------------------------------
def nchs_2010_external_rates(
    path: Path = NCHS_2010_PATH,
) -> pd.DataFrame:
    """Band x sex NCHS 2010 central death rates in the pinned shape.

    Mirrors ``build_mortality_floors.nchs_band_rates`` (the single source of
    the ``(l_a - l_{b+1}) / (T_a - T_{b+1})`` formula, open top band =
    ``l_85 / T_85``) and reshapes to the ``mortality_external_rates`` columns
    ``{lower_age, upper_age, age_band, sex, central_rate}`` over the seven
    ``MORTALITY_BANDS``.
    """
    import build_mortality_floors as mf

    if tuple(MORTALITY_BANDS) != tuple(mf.BANDS):
        raise RuntimeError(
            "MORTALITY_BANDS and build_mortality_floors.BANDS diverged; the "
            "external-rate band collapse must use one shared band set."
        )
    nchs = json.loads(Path(path).read_text())
    vintage = int(nchs.get("vintage_year", -1))
    if vintage != MORTALITY_EXTERNAL_VINTAGE:
        raise ValueError(
            f"{path} vintage_year {vintage} != pinned "
            f"{MORTALITY_EXTERNAL_VINTAGE}"
        )
    rates = mf.nchs_band_rates(nchs)
    rows = [
        {
            "lower_age": lower,
            "upper_age": upper,
            "age_band": mf.band_label(lower, upper),
            "sex": sex,
            "central_rate": rates[f"{mf.band_label(lower, upper)}|{sex}"],
        }
        for lower, upper in MORTALITY_BANDS
        for sex in mf.SEXES
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "lower_age",
            "upper_age",
            "age_band",
            "sex",
            "central_rate",
        ],
    )


# --------------------------------------------------------------------------
# Step 4b -- the <= T* PSID mortality-exposure adapter (§2.8.10.3)
# --------------------------------------------------------------------------
def mortality_exposure_adapter(
    demographic_panel: pd.DataFrame,
    death_records: pd.DataFrame,
    *,
    boundary_year: int = BOUNDARY_YEAR,
) -> pd.DataFrame:
    """Re-derive the ``<= T*`` person-interval slices in the pinned shape.

    ``build_mortality_floors.build_exposure_slices`` is the single source of
    the slice math -- one row per observed single-year person-interval slice,
    ``exposure = 0.5`` in the death-year slice else ``1.0``, ``death = 1.0``
    there else ``0.0``, ``start_weight`` the F6 realized start-wave anchor
    weight, ``sex`` and ``age_band`` from the slice. This adapter attaches the
    two dating fields the harness needs and emits the seven pinned columns
    ``{event_year, required_interview_year, age_band, sex, start_weight,
    exposure, death}``:

    * ``event_year`` -- the slice's own calendar year (``start_wave`` for the
      slice at the interval start, ``start_wave + 1`` for the second slice of a
      biennial interval; the death year for the death slice).
    * ``required_interview_year`` -- the interval's **closing wave**
      (``next_wave`` from the demographic panel's wave grid) on **every** slice,
      death and survivor alike -- the symmetric conservative dating rule.

    Only intervals anchored at a wave ``<= boundary_year`` are emitted, so the
    boundary-straddling 2013->2015 interval (start wave 2013, closing wave
    2015) is present for ``prepare_mortality_refit_inputs`` to drop wholesale:
    that truncation keeps only rows with **both** ``event_year <= T*`` and
    ``required_interview_year <= T*``, so the ``<= T*`` window ends with the
    2011->2013 interval.
    """
    import build_mortality_floors as mf

    slices = mf.build_exposure_slices(demographic_panel, death_records)

    grid = sorted(int(wave) for wave in demographic_panel["period"].unique())
    next_wave = {wave: grid[i + 1] for i, wave in enumerate(grid[:-1])}

    frame = slices[slices["start_wave"] <= int(boundary_year)].copy()
    # Closing-wave dating: every slice carries its interval's next grid wave.
    frame["required_interview_year"] = frame["start_wave"].map(next_wave)
    if frame["required_interview_year"].isna().any():
        raise ValueError(
            "an emitted interval has no closing wave; every non-terminal "
            "wave must have a grid successor"
        )
    frame["required_interview_year"] = frame["required_interview_year"].astype(
        "int64"
    )
    # event_year = start_wave + intra-interval age progression (0 for the
    # start-wave slice, 1 for the second slice of a biennial interval). The
    # progression is (slice age - the person's age at the start wave), read
    # from the demographic panel directly -- not inferred from the surviving
    # slices, because build_exposure_slices drops out-of-band slices (e.g. an
    # age-24 start-wave slice when the age-25 second slice is retained), which
    # would bias a per-interval age minimum.
    start_age = (
        demographic_panel[["person_id", "period", "age"]]
        .drop_duplicates(["person_id", "period"])
        .rename(columns={"period": "start_wave", "age": "start_wave_age"})
    )
    frame = frame.merge(start_age, on=["person_id", "start_wave"], how="left")
    if frame["start_wave_age"].isna().any():
        raise ValueError(
            "a slice's start wave is absent from the demographic panel; the "
            "exposure slices and the panel must share the wave grid"
        )
    frame["event_year"] = (
        frame["start_wave"] + (frame["age"] - frame["start_wave_age"])
    ).astype("int64")

    return frame.rename(
        columns={"band": "age_band", "weight": "start_weight"}
    )[
        [
            "event_year",
            "required_interview_year",
            "age_band",
            "sex",
            "start_weight",
            "exposure",
            "death",
        ]
    ].reset_index(
        drop=True
    )


# --------------------------------------------------------------------------
# Step 4c -- the inert (0, 24) projection-coverage pad (§2.8.10.5)
# --------------------------------------------------------------------------
#: The pad band: ``AgeSexMortalityModel`` requires bands starting at age 0
#: and contiguous through 120, while every estimation, truth, and scored
#: mortality surface is 25+ only.
PAD_BAND = (0, 24)


def _pad_below_25_projection_coverage(
    external_rates: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    boundary_year: int = BOUNDARY_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append the inert ``(0, 24)`` pad to both mortality fit inputs.

    The bridge between the 25+ estimation surface and the projection model's
    total-population band invariant (§2.8.10.5, F1) -- one appended
    ``(0, 24)`` band per sex on **both** fit inputs, the seven-band
    construction untouched:

    * **external pad**: the NCHS 2010 ``(l_0 - l_25) / (T_0 - T_25)`` central
      rate, for provenance honesty only -- it is outcome-inert, because
      ``aligned_rate = central_rate x (psid_rate / central_rate) =
      psid_rate`` cancels it, so any positive value yields byte-identical
      fitted probabilities;
    * **exposure pad**: one slice per sex dated ``event_year =
      required_interview_year = boundary_year`` (= ``T*``, so it survives the
      ``<= T*`` flow truncation) with ``start_weight = exposure = 1.0`` and
      ``death = 0.0``, so ``psid_rate = 0`` and the fitted ``<25`` hazard is
      ``-expm1(0) = 0`` exactly: **no modeled mortality below the age-25 PSID
      exposure floor.**

    The 25+ fitted cells are byte-identical to the unpadded arithmetic and
    invariant to the pad value and weight; the ``0-24`` band feeds only
    ``apply_mortality`` (a zero hazard for materialized births and any
    ``<25`` members) and appears in no scored or disclosed mortality cell.
    """
    import build_mortality_floors as mf

    lo, hi = PAD_BAND
    label = mf.band_label(lo, hi)
    nchs = json.loads(NCHS_2010_PATH.read_text())
    external_pad = []
    exposure_pad = []
    for sex in mf.SEXES:
        rows = {r["age"]: r for r in nchs["tables"][sex]}
        deaths_band = rows[lo]["lx"] - rows[hi + 1]["lx"]
        person_years = rows[lo]["Tx"] - rows[hi + 1]["Tx"]
        external_pad.append(
            {
                "lower_age": lo,
                "upper_age": hi,
                "age_band": label,
                "sex": sex,
                "central_rate": deaths_band / person_years,
            }
        )
        exposure_pad.append(
            {
                "event_year": int(boundary_year),
                "required_interview_year": int(boundary_year),
                "age_band": label,
                "sex": sex,
                "start_weight": 1.0,
                "exposure": 1.0,
                "death": 0.0,
            }
        )
    padded_external = pd.concat(
        [external_rates, pd.DataFrame(external_pad)], ignore_index=True
    )
    padded_exposure = pd.concat(
        [exposure, pd.DataFrame(exposure_pad)], ignore_index=True
    )
    return padded_external, padded_exposure


# --------------------------------------------------------------------------
# The factory contract -- zero-argument build_inputs()
# --------------------------------------------------------------------------
def build_inputs() -> M6HarnessInputs:
    """Assemble the complete M6 harness input bundle (§2.8.10.4).

    Zero-argument; every binding hardcoded to a single ``<= T*`` source and
    vintage. Reads staged PSID (mortality exposure here, and household /
    earnings / disability inside ``load_m6_inputs``); intended to run only at
    the registered gate, never in the build/test lane.
    """
    # (1) policyengine-us 1.752.2 gate (version + param-dir binding) +
    # realized-series load.
    assert_pe_us_version()
    assert_pe_us_param_dir()
    # (2) leakage-safe NAWI / wage-base surface.
    params = boundary_ssa_parameters()
    # (3) claiming reference + sha256 tamper gate.
    claiming_reference = load_claiming_reference()
    # (4) mortality external rates + <= T* exposure slices + the inert
    # (0, 24) projection-coverage pad on both (§2.8.10.5).
    mortality_external_rates, mortality_exposure = (
        _pad_below_25_projection_coverage(
            nchs_2010_external_rates(),
            mortality_exposure_adapter(
                panels.demographic_panel(),
                deaths.read_death_records(),
            ),
            boundary_year=BOUNDARY_YEAR,
        )
    )
    # (5) assemble; load_m6_inputs re-validates every vintage at assembly.
    return load_m6_inputs(
        ssa_params=params,
        ssa_params_vintage=SSA_PARAMS_VINTAGE,
        claiming_reference=claiming_reference,
        mortality_exposure=mortality_exposure,
        mortality_external_rates=mortality_external_rates,
        mortality_external_vintage=MORTALITY_EXTERNAL_VINTAGE,
        boundary_year=BOUNDARY_YEAR,
        earnings_seed=DEFAULT_EARNINGS_SEED,
    )
