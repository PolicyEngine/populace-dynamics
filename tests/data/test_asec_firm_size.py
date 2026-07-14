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
    all year, so tests only state what they exercise.
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
        "MARSUPWT": 1000.0,
    }
    frame = pd.DataFrame(
        [
            {**defaults, "PERIDNUM": 10_000 + i, **row}
            for i, row in enumerate(rows)
        ]
    )
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
            {"NOEMP": 2, "MARSUPWT": 1000.0, "I_NOEMP": 1},
            {"NOEMP": 2, "MARSUPWT": 3000.0, "I_NOEMP": 0},
            {"NOEMP": 5, "MARSUPWT": 500.0, "I_NOEMP": 0, "LJCW": 2},
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

    def test_wrong_frame_raises(self):
        with pytest.raises(ValueError, match="read_asec_firm_size"):
            asec_firm_size.firm_size_tabulation(pd.DataFrame({"x": [1]}))


@needs_real_asec
class TestRealData:
    def test_reads_any_staged_year(self):
        staged = sorted(REAL_DATA.glob("pppub*.csv*"))
        if not staged:
            pytest.skip("no pppub files staged")
        year = 2000 + int(staged[-1].name[5:7])
        out = asec_firm_size.read_asec_firm_size(year, path=staged[-1])
        assert len(out) > 10_000
        assert set(out["firm_size_band"]) <= set(
            asec_firm_size.noemp_band_map(year).values()
        )
