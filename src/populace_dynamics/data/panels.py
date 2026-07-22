"""Person-period panels from the staged PSID products.

Bridges the raw fixed-width readers (:mod:`populace_dynamics.data.psid`)
and the longitudinal views (:mod:`populace_dynamics.harness.panel`):
label-driven wave maps melt the wide cross-year individual file into
tidy person-period frames.

Conventions, verified against the real IND2023ER labels (2026-07-04):

* **Person identifier.** PSID's unique person key is the 1968 family
  interview number (``ER30001``) with the person number (``ER30002``);
  ``person_id = ER30001 * 1000 + ER30002`` (person numbers are below
  1000, so the pair packs losslessly).
* **Wave years in labels.** Wave-specific labels end in a two-digit
  year (``"AGE OF INDIVIDUAL  68"``, ``"...  23"``); interview-number
  labels lead with a four-digit year (``"1999 INTERVIEW NUMBER"``).
  Two-digit years 30-99 map to 19xx and 00-29 to 20xx.
* **Presence.** A person is present in a responding family unit in a
  wave when that wave's sequence number is 1-20 (the PSID codebook's
  "individuals in the family at the time of the interview" range);
  sequence numbers exist from 1969 on.
* **Weights.** No single weight series spans every wave: the
  cross-section individual weight covers 36 waves and the
  longitudinal weight 1993+, so concepts accept ordered fallback
  patterns tried per wave.

The cross-year individual file carries **no uniform individual labor
income series** (only other-family-unit members from the mid-1990s,
plus a discontinued 1999-2003 total-earnings series), so the earnings
panel requires the per-wave family files; see the staged-data README
for their enumerated download IDs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from populace_dynamics.data import psid

__all__ = [
    "label_year",
    "wave_variables",
    "ind_person_period",
    "demographic_panel",
    "individual_earnings_panel",
    "DEMOGRAPHIC_CONCEPTS",
]

_ID_1968_INTERVIEW = "ER30001"
_ID_PERSON_NUMBER = "ER30002"

#: Concept -> ordered label-regex fallbacks, verified against the real
#: IND2023ER label space (collision-free per wave after these regexes;
#: ``relationship`` excludes 1995's "RELATIONSHIP TO RESPONDENT").
DEMOGRAPHIC_CONCEPTS: dict[str, tuple[str, ...]] = {
    "age": (r"^AGE OF INDIVIDUAL",),
    "sequence": (r"^SEQUENCE NUMBER",),
    "relationship": (r"^RELATION(SHIP)? TO (HEAD|REFERENCE PERSON)",),
    # 1990-92 carry core/Latino/combined weights; the core series
    # continues the individual-weight sequence on either side (the
    # Latino oversample integrates only 1990-95), so it is the
    # fallback there.
    "weight": (
        r"^INDIVIDUAL WEIGHT|CROSS-SECTION WT",
        r"^CORE IND WEIGHT",
        r"LONGITUDINAL W",
    ),
    "interview": (r"INTERVIEW NUMBER",),
}

_IN_FAMILY_SEQUENCE = (1, 20)

#: The individual file's one uniform earnings series is a discontinued
#: "total annual earnings" recode collected in three biennial waves.
#: Each variable is *labeled* by its collection wave but measures the
#: prior odd year's earnings, so keying on the label wave would
#: mislabel every observation by two years. The mapping is
#: earnings reference year -> (variable code, collection wave); both
#: are verified against the file's own labels at read time.
_INDIVIDUAL_EARNINGS_SERIES: dict[int, tuple[str, int]] = {
    1997: ("ER33537N", 1999),
    1999: ("ER33628N", 2001),
    2001: ("ER33728N", 2003),
}

#: PSID recodes reserve a large sentinel for missing earnings; the
#: total-annual-earnings recodes top out well below it.
_EARNINGS_MISSING = 9_999_999


def label_year(label: str) -> int | None:
    """Extract the wave year from a PSID variable label.

    Trailing two-digit years (``"AGE OF INDIVIDUAL  68"``) map to
    1968; leading four-digit years (``"1999 INTERVIEW NUMBER"``) are
    taken as-is. Returns ``None`` when the label carries no year
    (e.g. ``"1968 INTERVIEW NUMBER"`` still returns 1968 via the
    leading form).
    """
    match = re.search(r"(\d{2})\s*$", label)
    if match:
        two = int(match.group(1))
        return 1900 + two if two >= 30 else 2000 + two
    match = re.match(r"^(19|20)\d{2}\b", label)
    if match:
        return int(match.group(0))
    return None


def wave_variables(labels: dict[str, str], pattern: str) -> dict[int, str]:
    """Map wave year -> variable name for labels matching ``pattern``.

    Args:
        labels: Name -> label, as returned by
            :func:`populace_dynamics.data.psid.parse_sps_labels`.
        pattern: Case-insensitive regex over label text.

    Returns:
        ``{year: variable_name}`` for every matching label with a
        recognizable year.

    Raises:
        ValueError: If two matching variables carry the same year —
            the pattern is ambiguous and must be tightened (silent
            arbitrary choice is exactly the failure this guards).
    """
    hits = psid.find_variables(labels, pattern)
    by_year: dict[int, str] = {}
    collisions: list[str] = []
    for name, label in hits.items():
        year = label_year(label)
        if year is None:
            continue
        if year in by_year:
            collisions.append(
                f"{year}: {by_year[year]} and {name} ({label!r})"
            )
        else:
            by_year[year] = name
    if collisions:
        raise ValueError(
            f"Pattern {pattern!r} is ambiguous within wave(s): "
            + "; ".join(collisions)
        )
    return by_year


def _resolve_concepts(
    labels: dict[str, str],
    concepts: dict[str, tuple[str, ...] | str],
) -> tuple[dict[str, dict[int, str]], list[int]]:
    """Resolve fallback patterns per concept and the common waves."""
    resolved: dict[str, dict[int, str]] = {}
    for concept, patterns in concepts.items():
        if isinstance(patterns, str):
            patterns = (patterns,)
        merged: dict[int, str] = {}
        for pattern in patterns:
            for year, name in wave_variables(labels, pattern).items():
                merged.setdefault(year, name)
        if not merged:
            raise ValueError(
                f"Concept {concept!r} matched no wave variables; "
                f"patterns tried: {patterns}"
            )
        resolved[concept] = merged
    waves = sorted(set.intersection(*(set(m) for m in resolved.values())))
    if not waves:
        raise ValueError(
            "No wave carries every requested concept; per-concept "
            "coverage: "
            + ", ".join(
                f"{c}={min(m)}-{max(m)} (n={len(m)})"
                for c, m in resolved.items()
            )
        )
    return resolved, waves


def ind_person_period(
    concepts: dict[str, tuple[str, ...] | str],
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    waves: list[int] | None = None,
    max_period: int | None = None,
) -> pd.DataFrame:
    """Melt the cross-year individual file into a person-period frame.

    Args:
        concepts: Concept name -> label regex (or ordered fallback
            regexes tried per wave). Each concept becomes a column.
        data_dir: Staged-data directory (see
            :func:`populace_dynamics.data.psid.read_psid`).
        nrows: Row cap on the underlying fixed-width read, for
            smoke tests.
        waves: Restrict to these years; default is every wave that
            carries all concepts.
        max_period: Read no wave later than this year. This field cap is
            applied before selecting columns from the wide source product.

    Returns:
        Tidy frame with columns ``person_id``, ``period``, and one
        column per concept, sorted by person and period. Every
        observed person-period row is returned; presence filtering
        (sequence-number semantics) is the caller's decision.
    """
    sps_path = (
        psid._resolve_data_dir(data_dir)
        / psid.PRODUCTS["ind2023er"][0]
        / psid.PRODUCTS["ind2023er"][2]
    )
    labels = psid.parse_sps_labels(sps_path)
    resolved, common_waves = _resolve_concepts(labels, concepts)
    use_waves = waves if waves is not None else common_waves
    if max_period is not None:
        boundary = int(max_period)
        use_waves = [wave for wave in use_waves if wave <= boundary]
        if not use_waves:
            raise ValueError(
                f"No requested wave is at or before max_period={boundary}."
            )
    missing_waves = [
        w for w in use_waves if any(w not in m for m in resolved.values())
    ]
    if missing_waves:
        raise ValueError(
            f"Wave(s) {missing_waves} lack at least one requested " "concept."
        )

    columns = [_ID_1968_INTERVIEW, _ID_PERSON_NUMBER]
    for wave in use_waves:
        columns.extend(resolved[c][wave] for c in concepts)
    wide = psid.read_psid(
        "ind2023er", columns=columns, data_dir=data_dir, nrows=nrows
    )

    person_id = wide[_ID_1968_INTERVIEW].astype("int64") * 1000 + wide[
        _ID_PERSON_NUMBER
    ].astype("int64")
    frames = []
    for wave in use_waves:
        frame = pd.DataFrame({"person_id": person_id, "period": wave})
        for concept in concepts:
            frame[concept] = wide[resolved[concept][wave]]
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    return long.sort_values(["person_id", "period"]).reset_index(drop=True)


def demographic_panel(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    in_family_only: bool = True,
    max_period: int | None = None,
) -> pd.DataFrame:
    """The verified demographic person-period panel, 1969-2023.

    Columns: ``person_id``, ``period``, ``age``, ``sequence``,
    ``relationship``, ``weight``, ``interview``. With
    ``in_family_only`` (default) rows are restricted to sequence
    numbers 1-20 — persons present in a responding family unit at
    the interview — which is the presence notion trajectory windows
    should treat as observation.

    1968 drops out because sequence numbers begin in 1969.
    """
    panel = ind_person_period(
        DEMOGRAPHIC_CONCEPTS,
        data_dir=data_dir,
        nrows=nrows,
        max_period=max_period,
    )
    if in_family_only:
        low, high = _IN_FAMILY_SEQUENCE
        present = (panel["sequence"] >= low) & (panel["sequence"] <= high)
        panel = panel.loc[present].reset_index(drop=True)
    return panel


def individual_earnings_panel(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    positive_only: bool = False,
) -> pd.DataFrame:
    """Real biennial individual earnings, reference years 1997-2001.

    Built from the cross-year individual file's one uniform earnings
    series (:data:`_INDIVIDUAL_EARNINGS_SERIES`). Each variable is
    labeled by its collection wave but measures the prior odd year's
    earnings, so ``period`` is set to the earnings reference year, and
    ``age`` and ``weight`` are read from the same collection wave.
    Every variable code is verified against the file's own label
    before use, so a different release fails loudly rather than
    silently mislabeling.

    Columns: ``person_id``, ``period`` (1997/1999/2001),
    ``earnings`` (missing sentinel dropped), ``age``, ``weight``.
    Rows with in-universe zero earnings are kept unless
    ``positive_only`` is set — zeros are the nonemployment margin the
    scoring protocol's zero-inflation gate targets.

    This is the first *real earnings* input to the noise floor. It is
    a thin three-wave series by design; the full head/wife earnings
    history across every wave lives in the per-wave family files (see
    the staged-data README).
    """
    sps_path = (
        psid._resolve_data_dir(data_dir)
        / psid.PRODUCTS["ind2023er"][0]
        / psid.PRODUCTS["ind2023er"][2]
    )
    labels = psid.parse_sps_labels(sps_path)
    age_by_wave = wave_variables(labels, r"^AGE OF INDIVIDUAL")
    weight_by_wave: dict[int, str] = {}
    for pattern in DEMOGRAPHIC_CONCEPTS["weight"]:
        for year, name in wave_variables(labels, pattern).items():
            weight_by_wave.setdefault(year, name)

    plan: list[tuple[int, str, str, str]] = []
    for ref_year, (earn_var, wave) in _INDIVIDUAL_EARNINGS_SERIES.items():
        actual = labels.get(earn_var, "")
        if f"TOTAL ANNUAL EARNINGS IN {ref_year}" not in actual:
            raise ValueError(
                f"Earnings variable {earn_var} for reference year "
                f"{ref_year} does not match the expected label; found "
                f"{actual!r}. The release layout may have changed."
            )
        if wave not in age_by_wave or wave not in weight_by_wave:
            raise ValueError(
                f"Collection wave {wave} lacks an age or weight "
                f"variable for the {ref_year} earnings series."
            )
        plan.append(
            (ref_year, earn_var, age_by_wave[wave], weight_by_wave[wave])
        )

    columns = [_ID_1968_INTERVIEW, _ID_PERSON_NUMBER]
    for _, earn_var, age_var, weight_var in plan:
        columns += [earn_var, age_var, weight_var]
    wide = psid.read_psid(
        "ind2023er", columns=columns, data_dir=data_dir, nrows=nrows
    )
    person_id = wide[_ID_1968_INTERVIEW].astype("int64") * 1000 + wide[
        _ID_PERSON_NUMBER
    ].astype("int64")

    frames = []
    for ref_year, earn_var, age_var, weight_var in plan:
        frame = pd.DataFrame(
            {
                "person_id": person_id,
                "period": ref_year,
                "earnings": wide[earn_var].astype("float64"),
                "age": wide[age_var].astype("int64"),
                "weight": wide[weight_var].astype("float64"),
            }
        )
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)

    # Drop the missing sentinel and out-of-universe zero-weight rows;
    # keep in-universe zero earnings as the nonemployment margin.
    keep = (long["earnings"] < _EARNINGS_MISSING) & (long["weight"] > 0)
    if positive_only:
        keep &= long["earnings"] > 0
    long = long.loc[keep]
    return long.sort_values(["person_id", "period"]).reset_index(drop=True)
