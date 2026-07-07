"""Tests for the Family Relationship Matrix reader (MX23REL).

Always-runnable tier: a miniature synthetic MX product spanning both
code eras, plus the relationship/wave/self filters and the chunked read
path. Integration tier: the real staged 3.5M-row file, pinned on shape,
the self-diagonal and spouse-slice counts, code-frame domains, and
earnings-panel join coverage; skipped when the PSID files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from populace_dynamics.data import family, relmap

from .psid_fixtures import write_product

REAL_DATA = Path("~/PolicyEngine/psid-data").expanduser()
needs_relmap = pytest.mark.skipif(
    not (REAL_DATA / "MX23REL" / "MX23REL.txt").is_file(),
    reason="PSID relationship matrix (MX23REL) not staged",
)
needs_earnings = pytest.mark.skipif(
    not (REAL_DATA / "family" / "2023").is_dir(),
    reason="PSID family files (earnings panel) not staged",
)

# Five synthetic MX pairs: a 1970 (pre-1983 frame) couple with its self
# diagonal, and a 1990 (1983+ frame) parent/child pair. Full 12-variable
# layout so label verification sees everything.
_MX_ROWS: list[tuple[str, int, str, list[int]]] = [
    ("MX1", 1, "RELEASE NUMBER", [1, 1, 1, 1, 1]),
    ("MX2", 4, "INTERVIEW YEAR", [1970, 1970, 1970, 1990, 1990]),
    ("MX3", 5, "INTERVIEW NUMBER", [100, 100, 100, 200, 200]),
    ("MX4", 2, "EGO SEQUENCE NUMBER", [1, 2, 1, 3, 1]),
    ("MX5", 4, "EGO 1968 INTERVIEW NUMBER", [5, 5, 5, 6, 6]),
    ("MX6", 3, "EGO PERSON NUMBER", [1, 2, 1, 3, 1]),
    (
        "MX7",
        2,
        "EGO RELATION TO REFERENCE PERSON",
        [1, 2, 1, 30, 10],
    ),
    ("MX8", 2, "EGO RELATION TO ALTER", [20, 20, 10, 30, 50]),
    ("MX9", 2, "ALTER SEQUENCE NUMBER", [2, 1, 1, 1, 3]),
    ("MX10", 4, "ALTER 1968 INTERVIEW NUMBER", [5, 5, 5, 6, 6]),
    ("MX11", 3, "ALTER PERSON NUMBER", [2, 1, 1, 1, 3]),
    (
        "MX12",
        2,
        "ALTER RELATION TO REFERENCE PERSON",
        [2, 1, 1, 10, 30],
    ),
]


@pytest.fixture
def mini_relmap_dir(tmp_path: Path) -> Path:
    write_product(tmp_path / "MX23REL", "MX23REL.sps", "MX23REL.txt", _MX_ROWS)
    return tmp_path


# --- always-runnable --------------------------------------------------


def test_build_constructs_pair_ids(mini_relmap_dir: Path):
    rm = relmap.relationship_map(data_dir=mini_relmap_dir)
    assert list(rm.columns) == [
        "interview_year",
        "interview_number",
        "ego_person_id",
        "ego_sequence",
        "ego_rel_to_rp",
        "ego_rel_to_alter",
        "alter_person_id",
        "alter_sequence",
        "alter_rel_to_rp",
    ]
    assert list(rm["ego_person_id"]) == [5001, 5002, 5001, 6003, 6001]
    assert list(rm["alter_person_id"]) == [5002, 5001, 5001, 6001, 6003]
    assert list(rm["ego_rel_to_alter"]) == [20, 20, 10, 30, 50]
    assert rm["ego_person_id"].dtype == "int64"


def test_filters_relationship_wave_and_self(mini_relmap_dir: Path):
    spouses = relmap.relationship_map(
        data_dir=mini_relmap_dir, ego_rel_to_alter=relmap.SPOUSE
    )
    assert len(spouses) == 2
    assert set(spouses["ego_rel_to_alter"]) == {relmap.SPOUSE}

    no_self = relmap.relationship_map(data_dir=mini_relmap_dir, drop_self=True)
    assert len(no_self) == 4
    assert relmap.SELF not in set(no_self["ego_rel_to_alter"])

    wave_1990 = relmap.relationship_map(data_dir=mini_relmap_dir, waves=[1990])
    assert set(wave_1990["interview_year"]) == {1990}
    assert len(wave_1990) == 2


def test_chunked_read_matches_unchunked(mini_relmap_dir: Path):
    full = relmap.relationship_map(
        data_dir=mini_relmap_dir, ego_rel_to_alter=relmap.SPOUSE
    )
    chunked = relmap.relationship_map(
        data_dir=mini_relmap_dir,
        ego_rel_to_alter=relmap.SPOUSE,
        chunksize=2,
    )
    assert chunked.equals(full)


def test_rel_to_reference_person_era_split():
    assert (
        relmap.rel_to_reference_person(1970)
        is relmap.REL_TO_REFERENCE_PERSON_PRE1983
    )
    assert (
        relmap.rel_to_reference_person(1990)
        is relmap.REL_TO_REFERENCE_PERSON_1983_PLUS
    )
    # 1968-1982 codes and 1983+ codes are disjoint frames.
    assert relmap.REL_TO_REFERENCE_PERSON_PRE1983[1] == "reference_person"
    assert relmap.REL_TO_REFERENCE_PERSON_1983_PLUS[10] == "reference_person"
    assert relmap.EGO_REL_TO_ALTER[relmap.SPOUSE] == "legal_spouse"


def test_label_mismatch_raises(mini_relmap_dir: Path):
    sps = mini_relmap_dir / "MX23REL" / "MX23REL.sps"
    sps.write_text(
        sps.read_text().replace(
            "EGO RELATION TO ALTER", "EGO RELATION TO OTHER"
        )
    )
    with pytest.raises(ValueError, match="does not match"):
        relmap.relationship_map(data_dir=mini_relmap_dir)


# --- integration: real staged MX23REL ---------------------------------

# Pinned on the staged release (March 2026), verified 2026-07-07.
_N_ROWS = 3_485_034
_N_EGO_PERSONS = 84_983
_N_SELF = 910_446
_N_SPOUSE_PAIRS = 327_970


@pytest.fixture(scope="module")
def real_relmap() -> object:
    return relmap.relationship_map()


@needs_relmap
def test_real_shape_and_pins(real_relmap):
    rm = real_relmap
    assert len(rm) == _N_ROWS
    assert rm["ego_person_id"].nunique() == _N_EGO_PERSONS
    assert int((rm["ego_rel_to_alter"] == relmap.SELF).sum()) == _N_SELF
    assert int(rm["interview_year"].min()) == 1968
    assert int(rm["interview_year"].max()) == 2023


@needs_relmap
def test_real_code_frames_are_documented(real_relmap):
    rm = real_relmap
    assert set(rm["ego_rel_to_alter"].unique()) <= set(relmap.EGO_REL_TO_ALTER)
    rp_union = set(relmap.REL_TO_REFERENCE_PERSON_PRE1983) | set(
        relmap.REL_TO_REFERENCE_PERSON_1983_PLUS
    )
    assert set(rm["ego_rel_to_rp"].unique()) <= rp_union
    assert set(rm["alter_rel_to_rp"].unique()) <= rp_union
    # The era split holds: pre-1983 uses the abbreviated frame, 1983+
    # the detailed one.
    pre = rm[rm["interview_year"] <= 1982]
    post = rm[rm["interview_year"] >= 1983]
    assert set(pre["ego_rel_to_rp"].unique()) <= set(
        relmap.REL_TO_REFERENCE_PERSON_PRE1983
    )
    assert set(post["ego_rel_to_rp"].unique()) <= set(
        relmap.REL_TO_REFERENCE_PERSON_1983_PLUS
    )


@needs_relmap
def test_real_spouse_slice_chunked_matches_full(real_relmap):
    """The streamed spouse slice equals the full-then-filter slice."""
    sliced = relmap.relationship_map(
        ego_rel_to_alter=relmap.SPOUSE, chunksize=1_000_000
    )
    assert len(sliced) == _N_SPOUSE_PAIRS
    assert bool((sliced["ego_rel_to_alter"] == relmap.SPOUSE).all())
    full = real_relmap
    assert len(sliced) == int(
        (full["ego_rel_to_alter"] == relmap.SPOUSE).sum()
    )


@needs_relmap
@needs_earnings
def test_real_join_coverage_vs_earnings_panel(real_relmap):
    """Every earnings-panel person appears as an ego in the matrix."""
    earn_persons = set(family.family_earnings_panel().person_id.unique())
    ego_persons = set(real_relmap["ego_person_id"].unique())
    share = len(earn_persons & ego_persons) / len(earn_persons)
    assert share == pytest.approx(1.0, abs=0.001)
