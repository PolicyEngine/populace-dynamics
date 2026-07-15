"""Tests for the ASEC firm-size (NOEMP) reader and tabulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import asec_firm_size

REAL_DATA = Path("~/PolicyEngine/asec-data").expanduser()
needs_real_asec = pytest.mark.skipif(
    not REAL_DATA.is_dir(),
    reason="ASEC person files not staged",
)


def _write_person_file(
    directory: Path,
    year: int,
    rows: list[dict],
    filename: str | None = None,
) -> Path:
    """Write a fixture Census-style ASEC person CSV.

    Each row dict may override any raw column; defaults describe a
    private-sector worker with an unallocated NOEMP of 2 who worked
    all year at raw weight 100000 (MARSUPWT carries two implied
    decimals, so that is 1000.00 persons). WORKYN defaults to match
    the row's WKSWORK unless overridden.
    """
    directory.mkdir(parents=True, exist_ok=True)
    defaults = {
        "PERIDNUM": 0,
        "NOEMP": 2,
        "I_NOEMP": 0,
        "LJCW": 1,
        "INDUSTRY": 770,
        "WEIND": 4,
        "WKSWORK": 52,
        "MARSUPWT": 100_000,
    }
    records = []
    for i, row in enumerate(rows):
        record = {**defaults, "PERIDNUM": 10_000 + i, **row}
        record.setdefault(
            "WORKYN", 1 if record["WKSWORK"] not in (0, "0") else 0
        )
        records.append(record)
    frame = pd.DataFrame(records)
    path = directory / (filename or f"pppub{year % 100:02d}.csv")
    frame.to_csv(path, index=False)
    return path


class TestBandMaps:
    def test_2016_maps_code_2_to_10_49(self):
        assert asec_firm_size.noemp_band_map(2016)[2] == "10_49"
        assert asec_firm_size.noemp_band_map(2016)[3] == "50_99"

    def test_2024_maps_code_2_to_10_24(self):
        assert asec_firm_size.noemp_band_map(2024)[2] == "10_24"
        assert asec_firm_size.noemp_band_map(2024)[3] == "25_99"

    def test_regime_boundaries(self):
        assert asec_firm_size.band_regime(2011) == "2011_2018"
        assert asec_firm_size.band_regime(2018) == "2011_2018"
        assert asec_firm_size.band_regime(2019) == "2019_plus"
        assert asec_firm_size.band_regime(2025) == "2019_plus"

    def test_unverified_year_raises(self):
        with pytest.raises(ValueError, match="dictionary-verified"):
            asec_firm_size.noemp_band_map(2010)
        with pytest.raises(ValueError, match="dictionary-verified"):
            asec_firm_size.noemp_band_map(2026)


class TestReadAsecFirmSize:
    def test_bands_by_regime(self, tmp_path):
        rows = [{"NOEMP": 2}, {"NOEMP": 3}]
        for year, expected in (
            (2016, ["10_49", "50_99"]),
            (2024, ["10_24", "25_99"]),
        ):
            path = _write_person_file(tmp_path / str(year), year, rows)
            out = asec_firm_size.read_asec_firm_size(year, path=path)
            assert list(out["firm_size_band"]) == expected
            assert (
                out["band_regime"] == asec_firm_size.band_regime(year)
            ).all()
            assert (out["income_year"] == year - 1).all()

    def test_year_and_filename_must_agree(self, tmp_path):
        path = _write_person_file(tmp_path, 2016, [{}])
        with pytest.raises(ValueError, match="mis-band"):
            asec_firm_size.read_asec_firm_size(2024, path=path)

    def test_resolves_staged_directory(self, tmp_path):
        _write_person_file(tmp_path, 2021, [{"NOEMP": 3}])
        out = asec_firm_size.read_asec_firm_size(2021, data_dir=tmp_path)
        assert list(out["firm_size_band"]) == ["25_99"]

    def test_missing_staged_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="pppub21"):
            asec_firm_size.read_asec_firm_size(2021, data_dir=tmp_path)

    def test_missing_column_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{}])
        frame = pd.read_csv(path).drop(columns="LJCW")
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match=r"\['LJCW'\]"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_out_of_dictionary_noemp_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"NOEMP": 7}])
        with pytest.raises(ValueError, match="out-of-dictionary"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_noemp_outside_universe_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"NOEMP": 2, "WKSWORK": 0}])
        with pytest.raises(ValueError, match="universe"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_zero_noemp_inside_universe_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"NOEMP": 0}])
        with pytest.raises(ValueError, match="NOEMP universe"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_zero_ljcw_inside_universe_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"LJCW": 0}])
        with pytest.raises(ValueError, match="LJCW universe"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_mixed_type_domain_violation_stays_a_value_error(self, tmp_path):
        rows = [{"NOEMP": "A"}, {"NOEMP": 7}]
        path = _write_person_file(tmp_path, 2021, rows)
        with pytest.raises(ValueError, match="out-of-dictionary"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_negative_weight_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"MARSUPWT": -1.0}])
        with pytest.raises(ValueError, match="MARSUPWT"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_blank_person_id_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"PERIDNUM": ""}])
        with pytest.raises(ValueError, match="blank PERIDNUM"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_duplicate_person_id_raises(self, tmp_path):
        rows = [{"PERIDNUM": 77}, {"PERIDNUM": 77}]
        path = _write_person_file(tmp_path, 2021, rows)
        with pytest.raises(ValueError, match="not unique"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_fractional_code_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"NOEMP": 2.9}])
        with pytest.raises(ValueError, match="NOEMP"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_industry_garbage_gets_friendly_error(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"INDUSTRY": "XX"}])
        with pytest.raises(ValueError, match="INDUSTRY"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_infinite_weight_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2021, [{"MARSUPWT": "inf"}])
        with pytest.raises(ValueError, match="MARSUPWT"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_empty_file_gets_friendly_error(self, tmp_path):
        path = tmp_path / "pppub21.csv"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            asec_firm_size.read_asec_firm_size(2021, path=path)

    def test_wide_person_ids_survive_exactly(self, tmp_path):
        # 22-digit PERIDNUMs differing only in the last digit round
        # together under any numeric read (int64 tops out at 19
        # digits; float64 keeps ~15-16), so exact string survival
        # is the regression test for the dtype pin.
        ids = [
            "8812345678901234567891",
            "8812345678901234567892",
        ]
        path = _write_person_file(
            tmp_path, 2021, [{"PERIDNUM": i} for i in ids]
        )
        out = asec_firm_size.read_asec_firm_size(2021, path=path)
        assert list(out["person_id"]) == ids

    def test_universe_and_flags(self, tmp_path):
        rows = [
            {"NOEMP": 1, "I_NOEMP": 1, "LJCW": 6},
            {"NOEMP": 0, "WKSWORK": 0, "LJCW": 0},  # NIU, dropped
            {"NOEMP": 4, "LJCW": 3},
        ]
        path = _write_person_file(tmp_path, 2021, rows)
        out = asec_firm_size.read_asec_firm_size(2021, path=path)
        assert len(out) == 2
        assert list(out["noemp_allocated"]) == [True, False]
        assert list(out["class_of_worker"]) == [
            "self_employed_unincorporated",
            "state",
        ]

    def test_env_var_staging(self, tmp_path, monkeypatch):
        _write_person_file(tmp_path, 2021, [{}])
        monkeypatch.setenv("POPULACE_DYNAMICS_ASEC_DIR", str(tmp_path))
        out = asec_firm_size.read_asec_firm_size(2021)
        assert len(out) == 1


class TestFirmSizeTabulation:
    def test_weighted_counts_and_allocated_share(self, tmp_path):
        rows = [
            {"NOEMP": 2, "MARSUPWT": 100_000, "I_NOEMP": 1},
            {"NOEMP": 2, "MARSUPWT": 300_000, "I_NOEMP": 0},
            {"NOEMP": 5, "MARSUPWT": 50_000, "I_NOEMP": 0, "LJCW": 2},
        ]
        path = _write_person_file(tmp_path, 2024, rows)
        records = asec_firm_size.read_asec_firm_size(2024, path=path)
        out = asec_firm_size.firm_size_tabulation(records)
        ten_24 = out[out["firm_size_band"] == "10_24"].iloc[0]
        assert ten_24["weighted_persons"] == 4000.0
        assert ten_24["unweighted_n"] == 2
        assert ten_24["allocated_share"] == 0.25
        federal = out[out["class_of_worker"] == "federal"].iloc[0]
        assert federal["weighted_persons"] == 500.0
        assert federal["firm_size_band"] == "500_999"

    def test_regime_break_is_visible_across_years(self, tmp_path):
        code_2 = [{"NOEMP": 2}]
        frames = [
            asec_firm_size.read_asec_firm_size(
                year,
                path=_write_person_file(tmp_path / str(year), year, code_2),
            )
            for year in (2018, 2019)
        ]
        out = asec_firm_size.firm_size_tabulation(pd.concat(frames))
        assert set(out["firm_size_band"]) == {"10_49", "10_24"}

    def test_zero_weight_group_share_is_nan(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"MARSUPWT": 0.0}])
        records = asec_firm_size.read_asec_firm_size(2024, path=path)
        out = asec_firm_size.firm_size_tabulation(records)
        assert out.iloc[0]["weighted_persons"] == 0.0
        assert pd.isna(out.iloc[0]["allocated_share"])

    def test_nan_group_keys_are_kept(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{}, {}])
        records = asec_firm_size.read_asec_firm_size(2024, path=path)
        records.loc[0, "class_of_worker"] = None
        out = asec_firm_size.firm_size_tabulation(records)
        assert out["weighted_persons"].sum() == records["weight"].sum()

    def test_wrong_frame_raises(self):
        with pytest.raises(ValueError, match="read_asec_firm_size"):
            asec_firm_size.firm_size_tabulation(pd.DataFrame({"x": [1]}))


class TestWeightScaling:
    def test_raw_weight_has_two_implied_decimals(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"MARSUPWT": "158007"}])
        out = asec_firm_size.read_asec_firm_size(2024, path=path)
        assert list(out["weight"]) == [1580.07]

    def test_stray_allocation_flag_code_raises(self, tmp_path):
        path = _write_person_file(tmp_path, 2024, [{"I_NOEMP": 5}])
        with pytest.raises(ValueError, match="I_NOEMP"):
            asec_firm_size.read_asec_firm_size(2024, path=path)

    def test_workyn_wkswork_mismatch_raises_pre_2019(self, tmp_path):
        # WORKYN = 1 is the stated universe in the 2011-2018
        # dictionaries, so the coincidence with WKSWORK is enforced
        # there.
        path = _write_person_file(tmp_path, 2016, [{"WORKYN": 2}])
        with pytest.raises(ValueError, match="WORKYN"):
            asec_firm_size.read_asec_firm_size(2016, path=path)

    def test_workyn_mismatch_tolerated_2019_plus(self, tmp_path):
        # 2019+ dictionaries state the universe as WKSWORK > 0
        # directly; real 2024 data carries ~0.4% WORKYN = 2 rows
        # beside a fully edited work block, so no coincidence
        # requirement there.
        path = _write_person_file(tmp_path, 2024, [{"WORKYN": 2}])
        out = asec_firm_size.read_asec_firm_size(2024, path=path)
        assert len(out) == 1


@needs_real_asec
class TestRealData:
    def test_reads_any_staged_year(self):
        staged = sorted(REAL_DATA.glob("pppub*.csv*"))
        if not staged:
            pytest.skip("no pppub files staged")
        year = 2000 + int(staged[-1].name[5:7])
        out = asec_firm_size.read_asec_firm_size(year, path=staged[-1])
        assert len(out) > 10_000
        assert out["person_id"].is_unique
        assert set(out["firm_size_band"]) <= set(
            asec_firm_size.noemp_band_map(year).values()
        )
        # Magnitude tripwire for the two-implied-decimals weight
        # convention: the worked-last-year universe is ~170 million
        # persons; an unscaled read lands near 17 billion.
        assert 1.0e8 < out["weight"].sum() < 3.0e8
