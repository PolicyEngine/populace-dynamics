"""Tests for the PSID death-records reader.

Two tiers, mirroring the other data-reader tests:

* Always-runnable: the pure ``decode_death_code`` logic across every
  codebook case, and a synthetic ``ind2023er`` fixture exercising the
  label-verified read, the ``person_id`` join convention, and the
  decode -- no real PSID needed.
* Skipped off-machine (needs the staged individual file): the real read
  matches the codebook tallies, the death-year domain is sane, and the
  ``person_id`` convention joins cleanly onto the earnings panel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from populace_dynamics.data import deaths, family

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_real_ind = pytest.mark.skipif(
    not (REAL_DATA / "ind2023er").is_dir(),
    reason="PSID ind2023er not staged",
)
needs_real_family = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files not staged",
)


# --------------------------------------------------------------------------
# decode_death_code -- pure, always runnable
# --------------------------------------------------------------------------
def test_decode_not_deceased_and_na():
    assert deaths.decode_death_code(0) == ("not_deceased", None, None, None)
    assert deaths.decode_death_code(9999) == ("na_dk", None, None, None)


@pytest.mark.parametrize("year", [1967, 1990, 2019, 2023])
def test_decode_exact_year(year):
    assert deaths.decode_death_code(year) == ("exact", year, year, year)


@pytest.mark.parametrize(
    "code, lo, hi",
    [
        (709, 2007, 2009),  # 0709
        (103, 2001, 2003),  # 0103
        (2123, 2021, 2023),
        (6768, 1967, 1968),
        (1016, 2010, 2016),
        (9799, 1997, 1999),  # cross into late-1990s bound
    ],
)
def test_decode_range_codes(code, lo, hi):
    status, year, got_lo, got_hi = deaths.decode_death_code(code)
    assert status == "range"
    assert year is None
    assert (got_lo, got_hi) == (lo, hi)


def test_decode_exact_beats_range_ambiguity():
    """A real year in [1967, 2023] is exact, never a 'YY-YY' range."""
    # 2019 could be misread as 20->19 (a lo>hi range) if the exact-year
    # window were not checked first.
    assert deaths.decode_death_code(2019)[0] == "exact"


def test_decode_unknown_when_range_invalid():
    # 2530 -> 25-30 -> 2025 > 1930, lo>hi, not a valid range.
    assert deaths.decode_death_code(2530) == ("unknown", None, None, None)


def test_sex_codes_constant():
    assert deaths.SEX_CODES == {1: "male", 2: "female", 9: "na"}


# --------------------------------------------------------------------------
# Synthetic ind2023er fixture -- always runnable
# --------------------------------------------------------------------------
def _write_ind_fixture(tmp_path: Path, fields) -> Path:
    """Write a tiny ind2023er product from a field table (positions fixed).

    ``fields`` is a list of ``(name, width, label, values)``; the setup
    file's DATA LIST positions derive from the widths so they cannot
    drift from the fixed-width text.
    """
    product = tmp_path / "ind2023er"
    product.mkdir()
    specs = []
    position = 1
    for name, width, _, _ in fields:
        end = position + width - 1
        specs.append(f"      {name:<15} {position} - {end}")
        position += width
    label_lines = [f'   {name:<12} "{label}"' for name, _, label, _ in fields]
    sps = (
        "DATA LIST FILE = PSID FIXED /\n"
        + "\n".join(specs)
        + "\n.\n\nVARIABLE LABELS\n"
        + "\n".join(label_lines)
        + "\n.\n"
    )
    (product / "IND2023ER.sps").write_text(sps)
    n = len(fields[0][3])
    lines = [
        "".join(f"{values[i]:>{width}}" for _, width, _, values in fields)
        for i in range(n)
    ]
    (product / "IND2023ER.txt").write_text("\n".join(lines) + "\n")
    return tmp_path


_FIXTURE_FIELDS = [
    ("ER30001", 2, "1968 INTERVIEW NUMBER", [1, 1, 2, 2, 3]),
    ("ER30002", 3, "PERSON NUMBER   68", [1, 2, 1, 3, 1]),
    ("ER32000", 1, "SEX OF INDIVIDUAL", [1, 2, 1, 2, 9]),
    ("ER32050", 4, "YEAR OF DEATH", [0, 2019, 709, 9999, 1967]),
]


def test_read_fixture_join_sex_and_decode(tmp_path):
    data_dir = _write_ind_fixture(tmp_path, _FIXTURE_FIELDS)
    df = deaths.read_death_records(data_dir=data_dir).set_index("person_id")

    # person_id = ER30001 * 1000 + ER30002 (the panels.py convention).
    assert list(df.index) == [1001, 1002, 2001, 2003, 3001]
    # Sex mapping (including the NA code).
    assert df.loc[1001, "sex"] == "male"
    assert df.loc[1002, "sex"] == "female"
    assert df.loc[3001, "sex"] == "na"
    # Death decoding.
    assert df.loc[1001, "death_status"] == "not_deceased"
    assert df.loc[1002, "death_status"] == "exact"
    assert df.loc[1002, "death_year"] == 2019
    assert df.loc[2001, "death_status"] == "range"
    assert (df.loc[2001, "death_year_lo"], df.loc[2001, "death_year_hi"]) == (
        2007,
        2009,
    )
    assert df.loc[2003, "death_status"] == "na_dk"
    assert df.loc[3001, "death_year"] == 1967


def test_read_fixture_label_verification_fails_on_changed_label(tmp_path):
    bad = list(_FIXTURE_FIELDS)
    bad[3] = ("ER32050", 4, "YR OF DEATH RENAMED", [0, 2019, 709, 9999, 1967])
    data_dir = _write_ind_fixture(tmp_path, bad)
    with pytest.raises(ValueError, match="does not match the adjudicated"):
        deaths.read_death_records(data_dir=data_dir)


# --------------------------------------------------------------------------
# Real ind2023er -- skipped off-machine
# --------------------------------------------------------------------------
@needs_real_ind
def test_real_counts_match_codebook():
    df = deaths.read_death_records()
    assert len(df) == 85536
    assert not df.person_id.duplicated().any()

    sex = df.sex.value_counts(dropna=False).to_dict()
    assert sex["male"] == 42384
    assert sex["female"] == 43151
    assert sex["na"] == 1

    status = df.death_status.value_counts().to_dict()
    assert status["not_deceased"] == 77415
    assert status["exact"] == 7816
    assert status["range"] == 293
    assert status["na_dk"] == 12


@needs_real_ind
def test_real_death_year_domain_is_sane():
    df = deaths.read_death_records()
    exact = df[df.death_status == "exact"]
    # PSID death file covers 1967-2023; every exact year sits inside it.
    assert int(exact.death_year.min()) >= 1967
    assert int(exact.death_year.max()) <= 2023
    # The overwhelming majority are 1968 or later (the study's span).
    assert (exact.death_year >= 1968).mean() > 0.99


@needs_real_family
def test_join_coverage_vs_earnings_panel():
    """The person_id convention joins the earnings panel onto deaths 1:1.

    deaths is read from the same individual file the panels key off, so
    every earnings-panel person must resolve to a death record. Pin that
    coverage share (1.0) -- it is the cross-module person_id check -- and
    the deceased share among earnings-panel persons for context.
    """
    dr = deaths.read_death_records()
    panel = family.family_earnings_panel()
    panel_ids = set(panel.person_id.unique())
    death_ids = set(dr.person_id.unique())

    coverage = len(panel_ids & death_ids) / len(panel_ids)
    assert coverage == 1.0

    deceased = dr[dr.death_status == "exact"]
    deceased_share = len(set(deceased.person_id) & panel_ids) / len(panel_ids)
    # A minority of the (prime-age-observed) earnings panel has since
    # died with an exact recorded year; sanity-bound it well inside (0, 1).
    assert 0.05 < deceased_share < 0.6
