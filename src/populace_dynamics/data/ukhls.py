"""Understanding Society (UKHLS) reader and transition estimator.

Salvaged and adapted from the archived
`policyengine-uk-data#346 <https://github.com/PolicyEngine/policyengine-uk-data/pull/346>`_
per the UK-extension decision thread
`populace#148 <https://github.com/PolicyEngine/populace/issues/148>`_.
This module is the UK analogue of the PSID readers: it stages
UKDS-licensed microdata locally, and only aggregated,
disclosure-controlled tables ever leave the loader.

Scope note: this is **estimation-side data machinery only**. Nothing
here applies transitions to a population — transition-application
models are gate candidates and go through registration (issue #42)
and the pre-registered evaluation protocol, exactly like the US side.

Data staging
------------
Raw UKHLS main-survey ``*_indresp.dta`` files (UKDS End User Licence,
SN 6614) are read from the directory named by the
``POPULACE_DYNAMICS_UKHLS_DIR`` environment variable, defaulting to
``~/PolicyEngine/ukhls-data``. Raw microdata is never committed;
tests that need it skip when the directory is absent.

Disclosure control
------------------
Any ``(age_band, sex, state_from, state_to)`` (or decile) cell
observed fewer than :data:`MIN_CELL_COUNT` times is suppressed before
a table is saved, following the ONS/UKDS safeguarded-microdata
convention, and probabilities are re-normalised over the surviving
rows. The committed tables in ``data/external/`` were produced this
way from Waves 1-15 (601,795 person-wave observations).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_CELL_COUNT",
    "WAVE_LETTERS",
    "WAVE_NUMBERS",
    "WAVE_YEAR_START",
    "ukhls_dir",
    "load_wave",
    "load_all_waves",
    "four_state_label",
    "annotate_four_state",
    "estimate_employment_transitions",
    "estimate_income_decile_transitions",
    "save_transition_tables",
    "load_employment_transitions",
    "load_income_decile_transitions",
]

_DATA_DIR_ENV = "POPULACE_DYNAMICS_UKHLS_DIR"
_DEFAULT_DATA_DIR = Path("~/PolicyEngine/ukhls-data").expanduser()

#: Committed aggregate tables live with the other external references.
EXTERNAL_DIR = Path(__file__).resolve().parents[3] / "data" / "external"
EMPLOYMENT_TABLE = EXTERNAL_DIR / "ukhls_employment_state_transitions.csv"
DECILE_TABLE = EXTERNAL_DIR / "ukhls_income_decile_transitions.csv"

#: Disclosure control: suppress any cell observed fewer than this many
#: times before saving (ONS/UKDS safeguarded-microdata convention).
MIN_CELL_COUNT = 10

# Wave-letter -> wave number mapping. UKHLS uses single-letter
# prefixes: a=Wave 1 (2009-10) ... o=Wave 15 (2023-24).
_LETTERS = "abcdefghijklmno"
WAVE_LETTERS: dict[str, int] = {ch: i + 1 for i, ch in enumerate(_LETTERS)}
WAVE_NUMBERS: dict[int, str] = {v: k for k, v in WAVE_LETTERS.items()}

# Each wave spans two calendar years; we adopt the year of first
# interview (the DWP Income Dynamics publication's convention).
WAVE_YEAR_START: dict[int, int] = {
    i + 1: 2009 + i for i in range(len(_LETTERS))
}

# Minimal column set for income / employment transition analysis. The
# full indresp file has ~1,400 columns, so loading is filtered.
BASE_COLUMNS_UNPREFIXED = [
    "age_dv",
    "sex",
    "jbstat",
    "fimnlabgrs_dv",  # total monthly labour income (gross)
    "fimngrs_dv",  # total monthly personal income (gross)
    "fimnsben_dv",  # social benefit income (monthly)
    "gor_dv",  # Government Office Region
    "hidp",  # within-wave household identifier
]

# JBSTAT is an ordinal enum — keep the raw codes but expose a compact
# harmonised label that transitions are estimated on.
JBSTAT_LABELS: dict[int, str] = {
    1: "SELF_EMPLOYED",
    2: "EMPLOYED",
    3: "UNEMPLOYED",
    4: "RETIRED",
    5: "OTHER_INACTIVE",  # maternity leave
    6: "OTHER_INACTIVE",  # family care
    7: "STUDENT",
    8: "OTHER_INACTIVE",  # LT sick / disabled
    9: "OTHER_INACTIVE",  # govt training
    10: "SELF_EMPLOYED",  # unpaid family business
    11: "OTHER_INACTIVE",
    12: "OTHER_INACTIVE",
    13: "OTHER_INACTIVE",
    97: "OTHER_INACTIVE",
}

# Collapsed four-state labour-market state for transition matrices.
FOUR_STATE_MAP: dict[str, str] = {
    "SELF_EMPLOYED": "IN_WORK",
    "EMPLOYED": "IN_WORK",
    "UNEMPLOYED": "UNEMPLOYED",
    "RETIRED": "RETIRED",
    "STUDENT": "INACTIVE",
    "OTHER_INACTIVE": "INACTIVE",
}

# 5-year age bands keep cell counts comfortably above MIN_CELL_COUNT
# even for sparse states (e.g. retired 30-year-olds).
AGE_BAND_EDGES = [16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 121]


def ukhls_dir() -> Path:
    """Resolve the UKHLS staging directory.

    ``POPULACE_DYNAMICS_UKHLS_DIR`` wins over the default
    ``~/PolicyEngine/ukhls-data``.
    """
    env_value = os.environ.get(_DATA_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return _DEFAULT_DATA_DIR


def _wave_path(wave: int | str) -> Path:
    letter = wave if isinstance(wave, str) else WAVE_NUMBERS[int(wave)]
    path = ukhls_dir() / f"{letter}_indresp.dta"
    if not path.exists():
        raise FileNotFoundError(
            f"UKHLS indresp for wave {wave!r} not found at {path}. "
            f"Stage the UKDS download at {ukhls_dir()} or set "
            f"{_DATA_DIR_ENV}."
        )
    return path


def load_wave(
    wave: int | str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one wave's individual-response file, keyed on ``pidp``.

    Returned column names are stripped of their wave-letter prefix
    (``a_age_dv`` -> ``age_dv``) and ``wave`` / ``year`` columns are
    attached, so callers can iterate over waves uniformly.
    """
    letter = wave if isinstance(wave, str) else WAVE_NUMBERS[int(wave)]
    wave_num = WAVE_LETTERS[letter]
    path = _wave_path(letter)
    if columns is None:
        columns = BASE_COLUMNS_UNPREFIXED

    prefixed = [f"{letter}_{col}" for col in columns]
    df = pd.read_stata(
        path, convert_categoricals=False, columns=["pidp"] + prefixed
    )
    df = df.rename(columns=dict(zip(prefixed, columns, strict=True)))
    df["wave"] = wave_num
    df["year"] = WAVE_YEAR_START[wave_num]
    return df


def load_all_waves(
    columns: list[str] | None = None,
    waves: list[int] | None = None,
) -> pd.DataFrame:
    """Stack waves into one long frame keyed on ``(pidp, wave)``."""
    waves = waves or list(WAVE_LETTERS.values())
    frames = []
    for w in waves:
        try:
            frames.append(load_wave(w, columns=columns))
        except FileNotFoundError as exc:
            logger.warning("Skipping wave %s: %s", w, exc)
    if not frames:
        raise FileNotFoundError(
            "No UKHLS waves could be loaded. Check that "
            f"{ukhls_dir()} contains *_indresp.dta files."
        )
    return pd.concat(frames, ignore_index=True)


def four_state_label(jbstat_code: float) -> str:
    """Map a jbstat code to the compact four-state label.

    Missing / refused / inapplicable codes (negative values) return
    ``"MISSING"`` so callers can filter them explicitly.
    """
    try:
        code = int(jbstat_code)
    except (TypeError, ValueError):
        return "MISSING"
    if code < 0:
        return "MISSING"
    detail = JBSTAT_LABELS.get(code, "OTHER_INACTIVE")
    return FOUR_STATE_MAP.get(detail, "INACTIVE")


def annotate_four_state(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``state`` column derived from ``jbstat``."""
    if "jbstat" not in df.columns:
        raise KeyError("jbstat column required to compute four-state label")
    df = df.copy()
    df["state"] = df["jbstat"].map(four_state_label)
    return df


def _age_band(age: float) -> str | None:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return None
    if a < AGE_BAND_EDGES[0] or a >= AGE_BAND_EDGES[-1]:
        return None
    for lo, hi in zip(AGE_BAND_EDGES, AGE_BAND_EDGES[1:], strict=False):
        if lo <= a < hi:
            return f"{lo}-{hi - 1}"
    return None


def _panel_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Pair consecutive waves on ``pidp``.

    Each output row is a (pidp, wave_t -> wave_{t+1}) transition with
    ``_t`` / ``_t1`` suffixes on every non-key column.
    """
    df = df.sort_values(["pidp", "wave"])
    left = df.rename(columns={c: f"{c}_t" for c in df.columns if c != "pidp"})
    right = df.rename(
        columns={c: f"{c}_t1" for c in df.columns if c != "pidp"}
    )
    right["_join_wave"] = right["wave_t1"] - 1
    merged = left.merge(
        right,
        left_on=["pidp", "wave_t"],
        right_on=["pidp", "_join_wave"],
        how="inner",
    )
    return merged.drop(columns=["_join_wave"])


def _suppress_and_normalise(
    grouped: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    grouped = grouped[grouped["count"] >= MIN_CELL_COUNT].reset_index(
        drop=True
    )
    totals = grouped.groupby(group_cols)["count"].transform("sum")
    grouped["probability"] = grouped["count"] / totals
    return grouped


def estimate_employment_transitions(
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate ``P(state_{t+1} | state_t, age_band, sex)``.

    Returns a long-format frame with columns ``age_band, sex,
    state_from, state_to, count, probability``, cell-suppressed and
    re-normalised per :data:`MIN_CELL_COUNT`.
    """
    if df is None:
        df = load_all_waves()
    if "state" not in df.columns:
        df = annotate_four_state(df)

    pairs = _panel_pairs(df)
    pairs["age_band"] = pairs["age_dv_t"].map(_age_band)
    pairs = pairs.dropna(subset=["age_band"])
    pairs = pairs[pairs["state_t"] != "MISSING"]
    pairs = pairs[pairs["state_t1"] != "MISSING"]
    pairs = pairs[pairs["sex_t"].isin([1, 2])]

    grouped = (
        pairs.groupby(
            ["age_band", "sex_t", "state_t", "state_t1"], observed=True
        )
        .size()
        .rename("count")
        .reset_index()
        .rename(
            columns={
                "sex_t": "sex",
                "state_t": "state_from",
                "state_t1": "state_to",
            }
        )
    )
    grouped["sex"] = grouped["sex"].map({1: "MALE", 2: "FEMALE"})
    return _suppress_and_normalise(grouped, ["age_band", "sex", "state_from"])


def estimate_income_decile_transitions(
    df: pd.DataFrame | None = None,
    *,
    income_col: str = "fimngrs_dv",
) -> pd.DataFrame:
    """Estimate ``P(decile_{t+1} | decile_t, age_band, sex)``.

    Income is ranked into within-wave deciles so the estimator is
    scale-invariant across years.
    """
    if df is None:
        df = load_all_waves()
    df = df.copy()
    df[income_col] = pd.to_numeric(df[income_col], errors="coerce")
    # Negative or missing -> drop (imputation flags / refusals).
    df = df[df[income_col].notna()]
    df = df[df[income_col] >= 0]

    df["decile"] = df.groupby("wave")[income_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=10, labels=False) + 1
    )

    pairs = _panel_pairs(df[["pidp", "wave", "age_dv", "sex", "decile"]])
    pairs["age_band"] = pairs["age_dv_t"].map(_age_band)
    pairs = pairs.dropna(subset=["age_band", "decile_t", "decile_t1"])
    pairs = pairs[pairs["sex_t"].isin([1, 2])]

    grouped = (
        pairs.groupby(
            ["age_band", "sex_t", "decile_t", "decile_t1"], observed=True
        )
        .size()
        .rename("count")
        .reset_index()
        .rename(
            columns={
                "sex_t": "sex",
                "decile_t": "decile_from",
                "decile_t1": "decile_to",
            }
        )
    )
    grouped["sex"] = grouped["sex"].map({1: "MALE", 2: "FEMALE"})
    grouped[["decile_from", "decile_to"]] = grouped[
        ["decile_from", "decile_to"]
    ].astype(int)
    return _suppress_and_normalise(grouped, ["age_band", "sex", "decile_from"])


def save_transition_tables(
    output_dir: Path | str = EXTERNAL_DIR,
    df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Compute both transition tables and write them as CSVs.

    Only the aggregated, suppressed tables are written — small enough
    to commit alongside the other ``data/external`` references.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_all_waves()

    emp = estimate_employment_transitions(df)
    dec = estimate_income_decile_transitions(df)

    emp_path = output_dir / EMPLOYMENT_TABLE.name
    dec_path = output_dir / DECILE_TABLE.name
    emp.to_csv(emp_path, index=False)
    dec.to_csv(dec_path, index=False)
    return {
        "employment_state_transitions": emp_path,
        "income_decile_transitions": dec_path,
    }


def load_employment_transitions(
    path: Path | str | None = None,
) -> dict[tuple[str, str, str], dict[str, float]]:
    """Read the committed employment table into a nested dict.

    Returns ``{(age_band, sex, state_from): {state_to: p, ...}}``.
    """
    path = Path(path) if path else EMPLOYMENT_TABLE
    if not path.exists():
        raise FileNotFoundError(
            f"Transition table not found at {path}. Run "
            "save_transition_tables() after staging UKHLS data."
        )
    df = pd.read_csv(path)
    nested: dict[tuple[str, str, str], dict[str, float]] = {}
    for (age_band, sex, state_from), group in df.groupby(
        ["age_band", "sex", "state_from"], observed=True
    ):
        nested[(str(age_band), str(sex), str(state_from))] = dict(
            zip(
                group["state_to"].astype(str),
                group["probability"].astype(float),
                strict=True,
            )
        )
    return nested


def load_income_decile_transitions(
    path: Path | str | None = None,
) -> dict[tuple[str, str, int], dict[int, float]]:
    """Read the committed decile table into a nested dict."""
    path = Path(path) if path else DECILE_TABLE
    if not path.exists():
        raise FileNotFoundError(
            f"Transition table not found at {path}. Run "
            "save_transition_tables() after staging UKHLS data."
        )
    df = pd.read_csv(path)
    nested: dict[tuple[str, str, int], dict[int, float]] = {}
    for (age_band, sex, decile_from), group in df.groupby(
        ["age_band", "sex", "decile_from"], observed=True
    ):
        nested[(str(age_band), str(sex), int(decile_from))] = dict(
            zip(
                group["decile_to"].astype(int),
                group["probability"].astype(float),
                strict=True,
            )
        )
    return nested
