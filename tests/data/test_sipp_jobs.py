"""Tests for the SIPP job-level monthly reader and spell collapse."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace_dynamics.data import sipp_jobs

REAL_DATA = Path("~/PolicyEngine/sipp-data").expanduser()
needs_real_sipp = pytest.mark.skipif(
    not REAL_DATA.is_dir(),
    reason="SIPP pu files not staged",
)

_PERSON_DEFAULTS = {
    "SSUID": "000114552888",
    "PNUM": 101,
    "SWAVE": 1,
    "WPFINWGT": 5000.5,
    "TAGE": 40,
    "ESEX": 1,
}

_SLOT_DEFAULTS = {
    "JOBID": 101,
    "BMONTH": 1,
    "EMONTH": 12,
    "CLWRK": 5,
    "JBORSE": 1,
    "EMPSIZE": 3,
    "IND": "7380",
    "MSUM": 3000,
}


def _write_pu_file(
    directory: Path,
    year: int,
    months: list[dict],
    slots: int = 2,
) -> Path:
    """Write a fixture pipe-delimited SIPP pu file.

    ``months`` holds one dict per person-month row; job-slot values
    are given as ``{"job1": {...}, "job2": {...}}`` overrides (an
    absent job key leaves that slot structurally empty). Defaults
    describe a private-sector job with monthly earnings 3000.
    """
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in months:
        row: dict = {**_PERSON_DEFAULTS}
        row["MONTHCODE"] = spec.get("month", 1)
        for key, value in spec.items():
            if key in ("month",) or key.startswith("job"):
                continue
            row[key] = value
        for n in range(1, slots + 1):
            job = spec.get(f"job{n}")
            for template in sipp_jobs._SLOT_TEMPLATES:
                column = template.format(n=n)
                suffix = column.split("_", 1)[1]
                if job is None:
                    row[column] = ""
                else:
                    row[column] = job.get(suffix, _SLOT_DEFAULTS[suffix])
        records.append(row)
    frame = pd.DataFrame(records)
    path = directory / f"pu{year}.csv"
    frame.to_csv(path, index=False, sep="|")
    return path


class TestReadSippJobMonths:
    def test_basic_read(self, tmp_path):
        months = [{"month": m, "job1": {}} for m in (1, 2, 3)]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert len(out) == 3
        assert (out["person_id"] == "000114552888-101").all()
        assert (out["ref_year"] == 2022).all()
        assert (out["earnings_share"] == 1.0).all()
        assert out["top_earner"].all()
        assert (out["class_of_worker"] == "private_for_profit").all()
        assert (out["industry"] == "7380").all()

    def test_concurrent_jobs_share_and_top(self, tmp_path):
        months = [
            {
                "month": 1,
                "job1": {"MSUM": 3000},
                "job2": {"JOBID": 202, "MSUM": 1000, "CLWRK": 3},
            }
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert len(out) == 2
        assert list(out["earnings_share"]) == [0.75, 0.25]
        assert list(out["top_earner"]) == [True, False]
        assert list(out["class_of_worker"]) == [
            "private_for_profit",
            "state",
        ]

    def test_missing_earnings_code(self, tmp_path):
        months = [{"month": 1, "job1": {"MSUM": -999}}]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert out["earnings"].isna().all()
        assert out["earnings_share"].isna().all()
        assert not out["top_earner"].any()

    def test_inactive_slot_dropped(self, tmp_path):
        months = [{"month": 1, "job1": {"JOBID": -999}}]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert len(out) == 0

    def test_unsupported_year_raises(self, tmp_path):
        path = _write_pu_file(tmp_path, 2021, [{"month": 1, "job1": {}}])
        with pytest.raises(ValueError, match="API-verified"):
            sipp_jobs.read_sipp_job_months(2021, path=path)

    def test_year_and_filename_must_agree(self, tmp_path):
        path = _write_pu_file(tmp_path, 2022, [{"month": 1, "job1": {}}])
        with pytest.raises(ValueError, match="SIPP 2022"):
            sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_missing_person_column_raises(self, tmp_path):
        path = _write_pu_file(tmp_path, 2023, [{"month": 1, "job1": {}}])
        frame = pd.read_csv(path, sep="|").drop(columns="WPFINWGT")
        frame.to_csv(path, index=False, sep="|")
        with pytest.raises(ValueError, match=r"\['WPFINWGT'\]"):
            sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_missing_slot_column_raises(self, tmp_path):
        path = _write_pu_file(tmp_path, 2023, [{"month": 1, "job1": {}}])
        frame = pd.read_csv(path, sep="|").drop(columns="EJB1_CLWRK")
        frame.to_csv(path, index=False, sep="|")
        with pytest.raises(ValueError, match="EJB1_CLWRK"):
            sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_domain_violations_raise(self, tmp_path):
        for spec, match in (
            ({"month": 13, "job1": {}}, "MONTHCODE"),
            ({"month": 1, "job1": {"CLWRK": 9}}, "EJB1_CLWRK"),
            ({"month": 1, "job1": {"JBORSE": 4}}, "EJB1_JBORSE"),
            ({"month": 1, "job1": {"EMPSIZE": 0}}, "EJB1_EMPSIZE"),
            ({"month": 1, "job1": {"BMONTH": 13}}, "EJB1_BMONTH"),
            ({"month": 1, "job1": {"JOBID": 0}}, "EJB1_JOBID"),
            ({"month": 1, "job1": {"MSUM": -5}}, "TJB1_MSUM"),
            ({"month": 1, "WPFINWGT": -1, "job1": {}}, "WPFINWGT"),
        ):
            path = _write_pu_file(tmp_path / match, 2023, [spec])
            with pytest.raises(ValueError, match=match):
                sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_missing_codes_tolerated_on_active_slots(self, tmp_path):
        months = [
            {
                "month": 1,
                "job1": {"CLWRK": -9, "EMPSIZE": -9, "BMONTH": -9},
            }
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert list(out["class_of_worker"]) == ["missing"]
        assert list(out["empsize_code"]) == [-9]

    def test_blank_empsize_on_self_employment_tolerated(self, tmp_path):
        # EMPSIZE's universe is employer-location jobs: on the real
        # 2023 file every self-employment job (JBORSE 2) carries a
        # structurally blank EMPSIZE.
        months = [
            {
                "month": 1,
                "job1": {"JBORSE": 2, "CLWRK": 8, "EMPSIZE": ""},
            }
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert len(out) == 1
        assert out["empsize_code"].isna().all()
        assert list(out["work_arrangement"]) == ["self_employed"]

    def test_blank_empsize_on_employer_job_refuses(self, tmp_path):
        months = [{"month": 1, "job1": {"JBORSE": 1, "EMPSIZE": ""}}]
        path = _write_pu_file(tmp_path, 2023, months)
        with pytest.raises(ValueError, match="EJB1_EMPSIZE"):
            sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_corrupt_empsize_on_self_employment_refuses(self, tmp_path):
        months = [
            {"month": 1, "job1": {"JBORSE": 2, "CLWRK": 8, "EMPSIZE": 42}}
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        with pytest.raises(ValueError, match="EJB1_EMPSIZE"):
            sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_garbage_string_empsize_on_se_slot_refuses(self, tmp_path):
        # A non-numeric string coerces to NaN like a structural
        # blank; the raw-cell guard keeps it a loud refusal on
        # no-establishment slots too (post-merge review note).
        months = [
            {
                "month": 1,
                "job1": {"JBORSE": 2, "CLWRK": 8, "EMPSIZE": "GARBAGE"},
            }
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        with pytest.raises(ValueError, match="EJB1_EMPSIZE"):
            sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_corrupt_code_on_active_slot_refuses(self, tmp_path):
        # Only the -9/-999 sentinels excuse a non-domain value on an
        # active slot; an unparseable string or a blank must refuse
        # rather than silently map to "missing".
        for value in ("GARBAGE", ""):
            months = [{"month": 1, "job1": {"CLWRK": value}}]
            path = _write_pu_file(
                tmp_path / f"c{value or 'blank'}", 2023, months
            )
            with pytest.raises(ValueError, match="EJB1_CLWRK"):
                sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_duplicate_job_id_across_slots_raises(self, tmp_path):
        months = [{"month": 1, "job1": {"JOBID": 101}, "job2": {"JOBID": 101}}]
        path = _write_pu_file(tmp_path, 2023, months)
        with pytest.raises(ValueError, match="more than one slot"):
            sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_person_attribute_sentinels_tolerated(self, tmp_path):
        months = [{"month": 1, "TAGE": -9, "ESEX": -9, "job1": {}}]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert len(out) == 1
        assert out["age"].isna().all()
        assert out["sex"].isna().all()

    def test_structural_sentinel_refused(self, tmp_path):
        months = [{"month": 1, "PNUM": -999, "job1": {}}]
        path = _write_pu_file(tmp_path, 2023, months)
        with pytest.raises(ValueError, match="PNUM"):
            sipp_jobs.read_sipp_job_months(2023, path=path)

    def test_top_earner_tie_breaks_to_lowest_slot(self, tmp_path):
        months = [
            {
                "month": 1,
                "job1": {"MSUM": 2000},
                "job2": {"JOBID": 202, "MSUM": 2000},
            }
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert list(out["top_earner"]) == [True, False]
        assert out["top_earner"].sum() == 1

    def test_all_seven_job_slots_read(self, tmp_path):
        spec = {"month": 1}
        for n in range(1, 8):
            spec[f"job{n}"] = {"JOBID": 100 + n, "MSUM": 100 * n}
        path = _write_pu_file(tmp_path, 2023, [spec], slots=7)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert len(out) == 7
        assert list(out["job_slot"]) == list(range(1, 8))
        assert out["earnings_share"].sum() == pytest.approx(1.0)
        assert list(out[out["top_earner"]]["job_slot"]) == [7]

    def test_industry_sentinel_passes_through_raw(self, tmp_path):
        # TJB{n}_IND is documented as unvalidated pass-through.
        months = [{"month": 1, "job1": {"IND": "-9"}}]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert list(out["industry"]) == ["-9"]

    def test_known_zero_earnings_semantics(self, tmp_path):
        months = [{"month": 1, "job1": {"MSUM": 0}}]
        path = _write_pu_file(tmp_path, 2023, months)
        out = sipp_jobs.read_sipp_job_months(2023, path=path)
        assert out["earnings_share"].isna().all()
        assert out["top_earner"].all()

    def test_staging_and_env(self, tmp_path, monkeypatch):
        _write_pu_file(tmp_path, 2022, [{"month": 1, "job1": {}}])
        out = sipp_jobs.read_sipp_job_months(2022, data_dir=tmp_path)
        assert len(out) == 1
        monkeypatch.setenv("POPULACE_DYNAMICS_SIPP_DIR", str(tmp_path))
        assert len(sipp_jobs.read_sipp_job_months(2022)) == 1

    def test_missing_staged_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="pu2023"):
            sipp_jobs.read_sipp_job_months(2023, data_dir=tmp_path)


class TestJobSpells:
    def test_gap_splits_spell(self, tmp_path):
        months = [
            {"month": 1, "job1": {}},
            {"month": 2, "job1": {}},
            {"month": 5, "job1": {}},
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        spells = sipp_jobs.job_spells(
            sipp_jobs.read_sipp_job_months(2023, path=path)
        )
        assert len(spells) == 2
        first, second = spells.iloc[0], spells.iloc[1]
        assert (first["start_month"], first["end_month"]) == (1, 2)
        assert first["n_months"] == 2
        assert (second["start_month"], second["end_month"]) == (5, 5)
        assert list(spells["spell_id"]) == [1, 2]

    def test_attribute_change_is_surfaced(self, tmp_path):
        months = [
            {"month": 1, "job1": {"EMPSIZE": 3}},
            {"month": 2, "job1": {"EMPSIZE": 4}},
            {"month": 3, "job1": {"EMPSIZE": 4}},
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        spells = sipp_jobs.job_spells(
            sipp_jobs.read_sipp_job_months(2023, path=path)
        )
        assert len(spells) == 1
        assert not spells.iloc[0]["attributes_constant"]
        assert spells.iloc[0]["empsize_code"] == 4  # modal

    def test_modal_attribute_even_tie_takes_smallest(self, tmp_path):
        # A 50/50 within-spell split: mode() sorts, so the smallest
        # value wins, deterministically, and the change is surfaced.
        months = [
            {"month": 1, "job1": {"EMPSIZE": 5}},
            {"month": 2, "job1": {"EMPSIZE": 3}},
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        spells = sipp_jobs.job_spells(
            sipp_jobs.read_sipp_job_months(2023, path=path)
        )
        assert len(spells) == 1
        assert spells.iloc[0]["empsize_code"] == 3
        assert not spells.iloc[0]["attributes_constant"]

    def test_earnings_share_and_primary(self, tmp_path):
        months = [
            {
                "month": m,
                "job1": {"MSUM": 3000},
                "job2": {"JOBID": 202, "MSUM": 1000},
            }
            for m in (1, 2)
        ]
        path = _write_pu_file(tmp_path, 2023, months)
        spells = sipp_jobs.job_spells(
            sipp_jobs.read_sipp_job_months(2023, path=path)
        )
        assert len(spells) == 2
        main = spells[spells["job_id"] == 101].iloc[0]
        side = spells[spells["job_id"] == 202].iloc[0]
        assert main["earnings_share"] == 0.75
        assert side["earnings_share"] == 0.25
        assert bool(main["primary_job"]) is True
        assert bool(side["primary_job"]) is False
        assert main["total_earnings"] == 6000.0

    def test_empty_input_returns_empty_schema(self, tmp_path):
        months = [{"month": 1, "job1": {"JOBID": -999}}]
        path = _write_pu_file(tmp_path, 2023, months)
        empty = sipp_jobs.read_sipp_job_months(2023, path=path)
        spells = sipp_jobs.job_spells(empty)
        assert len(spells) == 0
        assert "spell_id" in spells.columns
        assert "person_id" in spells.columns

    def test_wrong_frame_raises(self):
        with pytest.raises(ValueError, match="read_sipp_job_months"):
            sipp_jobs.job_spells(pd.DataFrame({"x": [1]}))

    def test_multi_year_input_refuses(self, tmp_path):
        # Two files concatenated span two ref_years (2021 and 2022);
        # the month-keyed break/run logic would mis-collapse them, so
        # job_spells must refuse rather than silently link across years.
        p22 = _write_pu_file(
            tmp_path / "y22", 2022, [{"month": m, "job1": {}} for m in (1, 2)]
        )
        p23 = _write_pu_file(
            tmp_path / "y23", 2023, [{"month": m, "job1": {}} for m in (1, 2)]
        )
        both = pd.concat(
            [
                sipp_jobs.read_sipp_job_months(2022, path=p22),
                sipp_jobs.read_sipp_job_months(2023, path=p23),
            ],
            ignore_index=True,
        )
        assert sorted(both["ref_year"].unique()) == [2021, 2022]
        with pytest.raises(ValueError, match="multiple ref_years"):
            sipp_jobs.job_spells(both)


@needs_real_sipp
class TestRealData:
    def test_reads_any_staged_year(self):
        staged = [
            p
            for year in sipp_jobs.SIPP_JOB_YEARS
            for p in REAL_DATA.glob(f"pu{year}.csv*")
        ]
        if not staged:
            pytest.skip("no pu files staged")
        path = staged[-1]
        year = int(path.name[2:6])
        out = sipp_jobs.read_sipp_job_months(year, path=path)
        assert len(out) > 100_000
        # min_count=1 keeps all-NaN person-months (fully missing
        # earnings) as NaN instead of a spurious 0.0 sum.
        shares = out.groupby(["person_id", "month"])["earnings_share"].sum(
            min_count=1
        )
        assert ((shares.dropna() - 1.0).abs() < 1e-9).all()
        spells = sipp_jobs.job_spells(out)
        assert (spells["n_months"].between(1, 12)).all()
