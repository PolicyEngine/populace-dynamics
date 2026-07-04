"""Tests for the PSID fixed-width data loader.

Unit tests run against a tiny synthetic fixture at
``tests/data/fixtures/mini_product/`` (3 variables, 5 rows) so they
never touch the real, multi-hundred-MB staged files. The fixture is
passed in through the ``data_dir`` argument of :func:`read_psid`
under a test-only product key registered directly on
``populace_dynamics.data.psid.PRODUCTS`` for the duration of the
test module -- it is never added to the real registry.

Integration smoke tests run against the real PSID data staged at
``~/PolicyEngine/psid-data`` (or the
``POPULACE_DYNAMICS_PSID_DIR`` override) and are skipped when that
directory is absent.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from populace_dynamics.data.psid import (
    PRODUCTS,
    find_variables,
    parse_sps_labels,
    parse_sps_layout,
    read_psid,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_DIR = FIXTURES_DIR / "mini_product"
MINI_SPS = MINI_DIR / "MINI.sps"
MINI_TXT = MINI_DIR / "MINI.txt"

# Test-only product key/registration, so read_psid() can resolve
# "mini" through the same code path real products use, without ever
# touching the real PRODUCTS registry that ships to users.
_MINI_PRODUCT_KEY = "mini"
_MINI_REGISTRATION = ("mini_product", "MINI.txt", "MINI.sps")

_REAL_DATA_DIR = Path(
    os.environ.get("POPULACE_DYNAMICS_PSID_DIR", "~/PolicyEngine/psid-data")
).expanduser()


def _real_data_staged() -> bool:
    return all(
        (_REAL_DATA_DIR / subdir / txt_name).is_file()
        for subdir, txt_name, _ in PRODUCTS.values()
    )


@pytest.fixture
def mini_registered():
    """Temporarily register the mini fixture under PRODUCTS."""
    assert (
        _MINI_PRODUCT_KEY not in PRODUCTS
    ), "test-only product key leaked into the real registry"
    PRODUCTS[_MINI_PRODUCT_KEY] = _MINI_REGISTRATION
    try:
        yield _MINI_PRODUCT_KEY
    finally:
        del PRODUCTS[_MINI_PRODUCT_KEY]


# --- parse_sps_layout -------------------------------------------------


def test_parse_sps_layout_returns_expected_columns():
    layout = parse_sps_layout(MINI_SPS)
    assert list(layout.columns) == ["name", "start", "end", "width"]


def test_parse_sps_layout_finds_all_three_variables():
    layout = parse_sps_layout(MINI_SPS)
    assert list(layout["name"]) == ["MN1", "MN2", "MN3"]


def test_parse_sps_layout_positions_are_1_indexed_inclusive():
    layout = parse_sps_layout(MINI_SPS).set_index("name")
    # MN1 is a single character at position 1.
    assert layout.loc["MN1", "start"] == 1
    assert layout.loc["MN1", "end"] == 1
    assert layout.loc["MN1", "width"] == 1
    # MN2 spans positions 2-5 inclusive: 4 characters wide.
    assert layout.loc["MN2", "start"] == 2
    assert layout.loc["MN2", "end"] == 5
    assert layout.loc["MN2", "width"] == 4
    # MN3 spans positions 6-8 inclusive: 3 characters wide.
    assert layout.loc["MN3", "start"] == 6
    assert layout.loc["MN3", "end"] == 8
    assert layout.loc["MN3", "width"] == 3


def test_parse_sps_layout_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_sps_layout(MINI_DIR / "does_not_exist.sps")


def test_parse_sps_layout_non_sps_file_raises_value_error(tmp_path):
    bogus = tmp_path / "not_a_layout.sps"
    bogus.write_text("this file has no DATA LIST block at all\n")
    with pytest.raises(ValueError):
        parse_sps_layout(bogus)


# --- parse_sps_labels --------------------------------------------------


def test_parse_sps_labels_returns_dict_for_all_variables():
    labels = parse_sps_labels(MINI_SPS)
    assert set(labels) == {"MN1", "MN2", "MN3"}


def test_parse_sps_labels_strips_whitespace_and_keeps_year_suffix():
    labels = parse_sps_labels(MINI_SPS)
    assert labels["MN1"] == "RELEASE NUMBER"
    # The trailing "68" is the PSID convention for encoding the
    # survey wave/year directly in the label text, since variable
    # names themselves are opaque codes.
    assert labels["MN2"] == "AGE OF INDIVIDUAL                     68"
    assert labels["MN2"].split()[-1] == "68"
    assert labels["MN3"].split()[-1] == "68"


def test_parse_sps_labels_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_sps_labels(MINI_DIR / "does_not_exist.sps")


def test_parse_sps_labels_non_sps_file_raises_value_error(tmp_path):
    bogus = tmp_path / "not_a_labels_file.sps"
    bogus.write_text("this file has no VARIABLE LABELS block\n")
    with pytest.raises(ValueError):
        parse_sps_labels(bogus)


# --- find_variables -----------------------------------------------------


def test_find_variables_matches_case_insensitively():
    labels = parse_sps_labels(MINI_SPS)
    matches = find_variables(labels, "age of individual")
    assert matches == {"MN2": labels["MN2"]}


def test_find_variables_regex_pattern_matches_multiple():
    labels = parse_sps_labels(MINI_SPS)
    matches = find_variables(labels, r"INCOME|AGE")
    assert set(matches) == {"MN2", "MN3"}


def test_find_variables_no_match_returns_empty_dict():
    labels = parse_sps_labels(MINI_SPS)
    assert find_variables(labels, "no such label text") == {}


# --- read_psid: column-subset correctness ------------------------------


def test_read_psid_all_columns_shape_and_values(mini_registered):
    df = read_psid(mini_registered, columns=None, data_dir=FIXTURES_DIR)
    assert df.shape == (5, 3)
    assert list(df.columns) == ["MN1", "MN2", "MN3"]
    assert list(df["MN1"]) == [1, 2, 3, 1, 2]
    assert list(df["MN2"]) == [25, 34, 42, 19, 61]
    assert list(df["MN3"]) == [100, 250, 75, 999, 0]


def test_read_psid_single_column_subset(mini_registered):
    df = read_psid(mini_registered, columns=["MN2"], data_dir=FIXTURES_DIR)
    assert list(df.columns) == ["MN2"]
    assert df.shape == (5, 1)
    assert list(df["MN2"]) == [25, 34, 42, 19, 61]


def test_read_psid_two_column_subset_preserves_order(mini_registered):
    df = read_psid(
        mini_registered,
        columns=["MN3", "MN1"],
        data_dir=FIXTURES_DIR,
    )
    assert list(df.columns) == ["MN3", "MN1"]
    assert list(df["MN3"]) == [100, 250, 75, 999, 0]
    assert list(df["MN1"]) == [1, 2, 3, 1, 2]


def test_read_psid_nrows_limits_rows_read(mini_registered):
    df = read_psid(
        mini_registered,
        columns=["MN1"],
        data_dir=FIXTURES_DIR,
        nrows=2,
    )
    assert df.shape == (2, 1)
    assert list(df["MN1"]) == [1, 2]


def test_read_psid_unknown_column_raises_key_error(mini_registered):
    with pytest.raises(KeyError):
        read_psid(
            mini_registered,
            columns=["NOT_A_REAL_COLUMN"],
            data_dir=FIXTURES_DIR,
        )


def test_read_psid_unknown_product_raises_key_error():
    with pytest.raises(KeyError):
        read_psid("not_a_real_product", data_dir=FIXTURES_DIR)


def test_read_psid_missing_data_dir_raises_file_not_found(
    mini_registered, tmp_path
):
    with pytest.raises(FileNotFoundError):
        read_psid(mini_registered, data_dir=tmp_path / "nowhere")


def test_read_psid_resolves_data_dir_from_env_var(
    mini_registered, monkeypatch
):
    monkeypatch.setenv("POPULACE_DYNAMICS_PSID_DIR", str(FIXTURES_DIR))
    df = read_psid(mini_registered, columns=["MN1"])
    assert list(df["MN1"]) == [1, 2, 3, 1, 2]


def test_read_psid_all_columns_none_product_not_flagged_warns(
    mini_registered,
):
    # "mini" is not in the large-product warning set, so reading
    # every column should not emit the size warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_psid(mini_registered, columns=None, data_dir=FIXTURES_DIR)


# --- integration smoke tests against the real staged PSID data --------


@pytest.mark.skipif(
    not _real_data_staged(),
    reason="PSID data not staged",
)
def test_real_ind2023er_layout_has_thousands_of_variables():
    subdir, _, sps_name = PRODUCTS["ind2023er"]
    sps_path = _REAL_DATA_DIR / subdir / sps_name
    layout = parse_sps_layout(sps_path)
    assert len(layout) > 5000 or len(layout) > 2000
    # The real IND2023ER.sps documents 2771 columns; assert
    # generously in case of a future refresh, but require it to be
    # a serious, multi-thousand-variable layout either way.
    assert len(layout) >= 2000


@pytest.mark.skipif(
    not _real_data_staged(),
    reason="PSID data not staged",
)
def test_real_ind2023er_labels_exist_for_layout_variables():
    subdir, _, sps_name = PRODUCTS["ind2023er"]
    sps_path = _REAL_DATA_DIR / subdir / sps_name
    layout = parse_sps_layout(sps_path)
    labels = parse_sps_labels(sps_path)
    assert len(labels) > 0
    # Most (allowing for a handful of unlabeled layout columns)
    # layout variables should have a label.
    labeled_fraction = sum(
        1 for name in layout["name"] if name in labels
    ) / len(layout)
    assert labeled_fraction > 0.9


@pytest.mark.skipif(
    not _real_data_staged(),
    reason="PSID data not staged",
)
def test_real_ind2023er_read_psid_column_subset():
    subdir, _, sps_name = PRODUCTS["ind2023er"]
    sps_path = _REAL_DATA_DIR / subdir / sps_name
    layout = parse_sps_layout(sps_path)
    first_three = list(layout["name"][:3])

    df = read_psid(
        "ind2023er",
        columns=first_three,
        data_dir=_REAL_DATA_DIR,
        nrows=100,
    )
    assert df.shape == (100, 3)
    assert list(df.columns) == first_three


@pytest.mark.skipif(
    not _real_data_staged(),
    reason="PSID data not staged",
)
def test_real_mh85_23_read_psid_smoke():
    subdir, _, sps_name = PRODUCTS["mh85_23"]
    sps_path = _REAL_DATA_DIR / subdir / sps_name
    layout = parse_sps_layout(sps_path)
    first_two = list(layout["name"][:2])

    df = read_psid(
        "mh85_23",
        columns=first_two,
        data_dir=_REAL_DATA_DIR,
        nrows=50,
    )
    assert df.shape == (50, 2)
