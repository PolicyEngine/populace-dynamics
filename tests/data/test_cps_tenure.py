"""Tests for the CPS January tenure-supplement reader and E3
tabulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import cps_tenure

REAL_DATA = Path("~/PolicyEngine/cps-tenure-data").expanduser()
needs_real_tenure = pytest.mark.skipif(
    not REAL_DATA.is_dir(),
    reason="CPS January supplement files not staged",
)


def _write_person_file(
    directory: Path,
    year: int,
    rows: list[dict],
) -> Path:
    """Write a fixture Census-style January supplement person CSV.

    Defaults describe an employed 40-year-old private-sector worker
    with 5.00 years of tenure and weight 1000.0 (PWTENWGT carries
    four implied decimals), so tests only state what they exercise.
    """
    directory.mkdir(parents=True, exist_ok=True)
    defaults = {
        "HRHHID": "110011234567890",
        "HRHHID2": "11011",
        "HRINTSTA": 1,
        "PULINENO": 1,
        "PTST1TN": 500,
        "PWTENWGT": 10_000_000,
        "PRTAGE": 40,
        "PESEX": 1,
        "GESTFIPS": 36,
        "PEMLR": 1,
        "PEIO1COW": 4,
    }
    frame = pd.DataFrame(
        [{**defaults, "PULINENO": i + 1, **row} for i, row in enumerate(rows)]
    )
    path = directory / f"jan{year % 100:02d}pub.csv"
    frame.to_csv(path, index=False)
    return path


class TestReadCpsTenure:
    def test_implied_decimals_and_ids(self, tmp_path):
        rows = [{"PTST1TN": 1600, "PRTAGE": 66, "PWTENWGT": 17_721_799}]
        path = _write_person_file(tmp_path, 2024, rows)
        out = cps_tenure.read_cps_tenure(2024, path=path)
        assert list(out["tenure_years"]) == [16.0]
        assert list(out["weight"]) == [1772.1799]
        assert list(out["person_id"]) == ["110011234567890-11011-1"]
        assert (out["year"] == 2024).all()

    def test_topcode_flag_is_per_year(self, tmp_path):
        # The topcode moves: 3300 (2020) / 3200 (2022) / 3100 (2024).
        for year, top in ((2020, 3300), (2022, 3200), (2024, 3100)):
            rows = [
                {"PTST1TN": top, "PRTAGE": 70},
                {"PTST1TN": top - 1, "PRTAGE": 70},
            ]
            path = _write_person_file(tmp_path / str(year), year, rows)
            out = cps_tenure.read_cps_tenure(year, path=path)
            assert list(out["tenure_topcoded"]) == [True, False]

    def test_above_topcode_refuses(self, tmp_path):
        path = _write_person_file(
            tmp_path, 2024, [{"PTST1TN": 3200, "PRTAGE": 70}]
        )
        with pytest.raises(ValueError, match="out-of-dictionary"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_unsupported_year_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2018, [{}])
        with pytest.raises(ValueError, match="dictionary-verified"):
            cps_tenure.read_cps_tenure(2018, path=path)
        with pytest.raises(ValueError, match="dictionary-verified"):
            cps_tenure.read_cps_tenure(2023, path=path)

    def test_year_and_filename_must_agree(self, tmp_path):
        path = _write_person_file(tmp_path, 2022, [{}])
        with pytest.raises(ValueError, match="January 2022"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_resolves_staged_directory_and_env(self, tmp_path, monkeypatch):
        _write_person_file(tmp_path, 2020, [{}])
        out = cps_tenure.read_cps_tenure(2020, data_dir=tmp_path)
        assert len(out) == 1
        monkeypatch.setenv("POPULACE_DYNAMICS_CPS_TENURE_DIR", str(tmp_path))
        assert len(cps_tenure.read_cps_tenure(2020)) == 1

    def test_missing_staged_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="jan20pub"):
            cps_tenure.read_cps_tenure(2020, data_dir=tmp_path)

    def test_missing_column_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{}])
        frame = pd.read_csv(path).drop(columns="PWTENWGT")
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match=r"\['PWTENWGT'\]"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_out_of_dictionary_tenure_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"PTST1TN": 3200}])
        with pytest.raises(ValueError, match="out-of-dictionary"):
            cps_tenure.read_cps_tenure(2024, path=path)
        path = _write_person_file(tmp_path, 2024, [{"PTST1TN": -5}])
        with pytest.raises(ValueError, match="out-of-dictionary"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_negative_weight_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"PWTENWGT": -1}])
        with pytest.raises(ValueError, match="PWTENWGT"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_usable_outside_employed_universe_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"PEMLR": 5}])
        with pytest.raises(ValueError, match="employed universe"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_age_minus_tenure_rule_raises(self, tmp_path):
        # Age 30 with 20.00 years of tenure: 10 < 14.
        path = _write_person_file(
            tmp_path, 2024, [{"PRTAGE": 30, "PTST1TN": 2000}]
        )
        with pytest.raises(ValueError, match="age-minus-tenure"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_nonresponse_excluded_but_countable(self, tmp_path):
        rows = [
            {},
            {"PTST1TN": -1, "PEMLR": 5},
            {"PTST1TN": -2},
            {"PTST1TN": -9},
        ]
        path = _write_person_file(tmp_path, 2024, rows)
        out = cps_tenure.read_cps_tenure(2024, path=path)
        assert len(out) == 1
        full = cps_tenure.read_cps_tenure(
            2024, path=path, include_nonresponse=True
        )
        assert len(full) == 4
        assert full["tenure_years"].isna().sum() == 3
        assert list(full["response"]) == [
            "usable",
            "niu",
            "dont_know",
            "no_response",
        ]

    def test_blank_id_component_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"HRHHID2": ""}])
        with pytest.raises(ValueError, match="blank HRHHID2"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_duplicate_person_id_raises(self, tmp_path):
        rows = [{"PULINENO": 1}, {"PULINENO": 1}]
        path = _write_person_file(tmp_path, 2024, rows)
        frame = pd.read_csv(path)
        frame["PULINENO"] = 1
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="duplicated person id"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_empty_file_gets_friendly_error(self, tmp_path):
        path = tmp_path / "jan24pub.csv"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_noninterview_placeholders_excluded(self, tmp_path):
        # Non-interviewed households (HRINTSTA 2-4) carry
        # PULINENO = -1 / weight 0 / all-NIU placeholder rows; they
        # are excluded before validation rather than tripping the
        # person-record checks.
        rows = [
            {},
            {
                "HRINTSTA": 2,
                "PULINENO": -1,
                "PTST1TN": -1,
                "PWTENWGT": 0,
                "PRTAGE": -1,
                "PESEX": -1,
                "PEMLR": -1,
                "PEIO1COW": -1,
            },
        ]
        path = _write_person_file(tmp_path, 2024, rows)
        out = cps_tenure.read_cps_tenure(2024, path=path)
        assert len(out) == 1

    def test_bad_pulineno_on_interviewed_row_still_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"PULINENO": -1}])
        with pytest.raises(ValueError, match="PULINENO"):
            cps_tenure.read_cps_tenure(2024, path=path)

    def test_class_of_worker_labels(self, tmp_path):
        rows = [{"PEIO1COW": 2}, {"PEIO1COW": 7}]
        path = _write_person_file(tmp_path, 2024, rows)
        out = cps_tenure.read_cps_tenure(2024, path=path)
        assert list(out["class_of_worker"]) == [
            "state",
            "self_employed_unincorporated",
        ]

    def test_gcfip_alias_for_2020_layout(self, tmp_path):
        # The published jan20pub.csv names the dictionary's GESTFIPS
        # column GCFIP; the loader accepts exactly one of the two.
        path = _write_person_file(tmp_path, 2020, [{}])
        frame = pd.read_csv(path).rename(columns={"GESTFIPS": "GCFIP"})
        frame.to_csv(path, index=False)
        out = cps_tenure.read_cps_tenure(2020, path=path)
        assert list(out["state_fips"]) == [36]

    def test_both_fips_names_refused(self, tmp_path):
        path = _write_person_file(tmp_path, 2020, [{}])
        frame = pd.read_csv(path)
        frame["GCFIP"] = frame["GESTFIPS"]
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="ambiguous"):
            cps_tenure.read_cps_tenure(2020, path=path)


class TestTenureTabulation:
    def test_weighted_quantiles_by_age_band(self, tmp_path):
        # Three 25-34 workers with equal weights: quantile midpoints
        # of tenure 2.00 / 5.00 / 8.00 give p50 = 5.0 exactly.
        rows = [
            {"PRTAGE": 30, "PTST1TN": 200},
            {"PRTAGE": 30, "PTST1TN": 500},
            {"PRTAGE": 30, "PTST1TN": 800},
            {"PRTAGE": 50, "PTST1TN": 3100},
        ]
        path = _write_person_file(tmp_path, 2024, rows)
        records = cps_tenure.read_cps_tenure(2024, path=path)
        out = cps_tenure.tenure_tabulation(records)
        young = out[out["age_band"] == "25_34"].iloc[0]
        assert young["p50"] == 5.0
        assert young["unweighted_n"] == 3
        assert young["weighted_persons"] == 3000.0
        assert young["topcoded_share"] == 0.0
        older = out[out["age_band"] == "45_54"].iloc[0]
        assert older["p50"] == 31.0
        assert older["topcoded_share"] == 1.0

    def test_extra_grouping_columns(self, tmp_path):
        rows = [
            {"PRTAGE": 30, "PESEX": 1, "PTST1TN": 200},
            {"PRTAGE": 30, "PESEX": 2, "PTST1TN": 800},
        ]
        path = _write_person_file(tmp_path, 2024, rows)
        records = cps_tenure.read_cps_tenure(2024, path=path)
        out = cps_tenure.tenure_tabulation(records, by=("sex",))
        assert len(out) == 2
        assert set(out["sex"]) == {1, 2}

    def test_nonresponse_rows_ignored(self, tmp_path):
        rows = [{}, {"PTST1TN": -2}]
        path = _write_person_file(tmp_path, 2024, rows)
        records = cps_tenure.read_cps_tenure(
            2024, path=path, include_nonresponse=True
        )
        out = cps_tenure.tenure_tabulation(records)
        assert out["unweighted_n"].sum() == 1

    def test_zero_weight_rows_excluded_from_quantiles(self, tmp_path):
        # A zero-weight knot at tenure 10 would drag the weighted
        # median from 6.0 to 10.0 if included.
        rows = [
            {"PRTAGE": 30, "PTST1TN": 100, "PWTENWGT": 10_000_000},
            {"PRTAGE": 30, "PTST1TN": 1000, "PWTENWGT": 0},
            {"PRTAGE": 30, "PTST1TN": 1100, "PWTENWGT": 10_000_000},
        ]
        path = _write_person_file(tmp_path, 2024, rows)
        records = cps_tenure.read_cps_tenure(2024, path=path)
        out = cps_tenure.tenure_tabulation(records)
        band = out[out["age_band"] == "25_34"].iloc[0]
        assert band["p50"] == 6.0
        assert band["unweighted_n"] == 2
        assert band["weighted_persons"] == 2000.0

    def test_custom_bands_with_gap_exclude_gap_ages(self, tmp_path):
        rows = [
            {"PRTAGE": 18, "PTST1TN": 100},
            {"PRTAGE": 22, "PTST1TN": 100},  # in the gap
            {"PRTAGE": 30, "PTST1TN": 100},
        ]
        path = _write_person_file(tmp_path, 2024, rows)
        records = cps_tenure.read_cps_tenure(2024, path=path)
        out = cps_tenure.tenure_tabulation(
            records, age_bands=((16, 19), (25, 34))
        )
        assert set(out["age_band"]) == {"16_19", "25_34"}
        assert out["unweighted_n"].sum() == 2

    def test_overlapping_or_reversed_bands_raise(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"PRTAGE": 30}])
        records = cps_tenure.read_cps_tenure(2024, path=path)
        with pytest.raises(ValueError, match="overlap"):
            cps_tenure.tenure_tabulation(
                records, age_bands=((16, 30), (25, 44))
            )
        with pytest.raises(ValueError, match="reversed"):
            cps_tenure.tenure_tabulation(records, age_bands=((30, 16),))

    def test_touching_boundary_bands_raise(self, tmp_path):
        # (20,24) and (24,30) share age 24 under closed="both" —
        # the natural adjacent-bands mistake must refuse, not
        # double-assign.
        path = _write_person_file(tmp_path, 2024, [{"PRTAGE": 24}])
        records = cps_tenure.read_cps_tenure(2024, path=path)
        with pytest.raises(ValueError, match="overlap"):
            cps_tenure.tenure_tabulation(
                records, age_bands=((20, 24), (24, 30))
            )

    def test_unsorted_bands_raise(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"PRTAGE": 30}])
        records = cps_tenure.read_cps_tenure(2024, path=path)
        with pytest.raises(ValueError, match="ascending"):
            cps_tenure.tenure_tabulation(
                records, age_bands=((25, 34), (16, 19))
            )

    def test_non_midpoint_weighted_quantile(self, tmp_path):
        # Unevenly spaced values with uneven weights, hand-computed
        # under the cumulative-midpoint convention. Tenures 1.00 /
        # 4.00 / 10.00 with weights 1000 / 2000 / 1000 (raw x10^4)
        # put knots at 0.125 / 0.500 / 0.875 of total weight, so
        # p25 interpolates one third of the way from 1.0 to 4.0
        # (= 2.0), p50 lands on the middle knot (= 4.0), and p75
        # two thirds of the way from 4.0 to 10.0 (= 8.0).
        rows = [
            {"PRTAGE": 30, "PTST1TN": 100, "PWTENWGT": 10_000_000},
            {"PRTAGE": 30, "PTST1TN": 400, "PWTENWGT": 20_000_000},
            {"PRTAGE": 30, "PTST1TN": 1000, "PWTENWGT": 10_000_000},
        ]
        path = _write_person_file(tmp_path, 2024, rows)
        records = cps_tenure.read_cps_tenure(2024, path=path)
        out = cps_tenure.tenure_tabulation(records)
        band = out[out["age_band"] == "25_34"].iloc[0]
        assert band["p25"] == pytest.approx(2.0)
        assert band["p50"] == pytest.approx(4.0)
        assert band["p75"] == pytest.approx(8.0)

    def test_leading_zero_ids_survive(self, tmp_path):
        rows = [{"HRHHID": "000011234567890", "HRHHID2": "01011"}]
        path = _write_person_file(tmp_path, 2024, rows)
        out = cps_tenure.read_cps_tenure(2024, path=path)
        assert list(out["person_id"]) == ["000011234567890-01011-1"]

    def test_empty_input_returns_empty_schema(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"PTST1TN": -2}])
        records = cps_tenure.read_cps_tenure(
            2024, path=path, include_nonresponse=True
        )
        out = cps_tenure.tenure_tabulation(records)
        assert len(out) == 0
        assert "age_band" in out.columns
        assert "p50" in out.columns

    def test_wrong_frame_raises(self):
        with pytest.raises(ValueError, match="read_cps_tenure"):
            cps_tenure.tenure_tabulation(pd.DataFrame({"x": [1]}))


@needs_real_tenure
class TestRealData:
    def test_reads_any_staged_year(self):
        staged = sorted(REAL_DATA.glob("jan*pub.csv*"))
        if not staged:
            pytest.skip("no jan{yy}pub files staged")
        year = 2000 + int(staged[-1].name[3:5])
        out = cps_tenure.read_cps_tenure(year, path=staged[-1])
        assert len(out) > 10_000
        assert out["person_id"].is_unique
        assert out["tenure_years"].between(0, 31).all()
        # Median tenure for prime-age workers is a few years; a
        # gross unit error (months vs years) would blow this band.
        tab = cps_tenure.tenure_tabulation(out)
        prime = tab[tab["age_band"] == "35_44"].iloc[0]
        assert 2.0 < prime["p50"] < 15.0
