"""PSID work-limitation (self-reported disability) status and the DI
incidence / recovery / conversion hazards it supports (roadmap #113, M4).

The micro basis
---------------
The cross-year individual file (``ind2023er``) carries a per-wave
``EMPLOYMENT STATUS`` recode whose code ``5`` is
``"Permanently disabled"`` -- the person's own self-reported labor-force
status. This is PSID's cleanest all-adult, every-wave work-limitation
signal, and the micro basis the M4 row names ("PSID work-limitation as
micro basis"). It is resolved label-driven through
:func:`populace_dynamics.data.panels.ind_person_period` (so a re-layout
of the release fails loudly), and the value-code map (``5`` = disabled)
is separately verified against the release's own SAS ``VALUE`` blocks by
:func:`verify_employment_status_codes` -- the same "verify every code
mapping against the formats file" discipline the demographic readers
apply to variable labels, extended here to value labels.

**Wave coverage (verified 2026-07-10 against IND2023ER.sps).** The
individual-file ``EMPLOYMENT STATUS`` recode is present in twenty waves::

    1982, 1983, then every wave 1993-1997 (annual) and 1999-2023
    (biennial).

1968-1981 and 1984-1992 carry no individual-file employment-status
recode under this label (head/wife status lived only in the per-wave
family files then), so those years contribute no person-years here. The
question wording is stable across the covered waves -- code ``5`` reads
``"Permanently disabled"`` verbatim in 1982, 1993, 2009 and 2023 -- but
the 1993+ block is the dense, continuous panel and 1982-1983 are two
isolated early snapshots (no adjacent wave within two years, so they
contribute person-year *occupancy* but almost no grid-adjacent
transition *pairs*; see :data:`MAX_INTERVAL`).

The concept delta (named here, never conflated downstream)
----------------------------------------------------------
PSID's ``EMPLOYMENT STATUS == 5`` is a **self-reported labor-force
status**, NOT an SSA Disability Insurance (DI) award. The two differ in
definition (self-report vs a medical-vocational adjudication),
population (all adults vs disability-insured workers below the full
retirement age), severity threshold, and timing. Empirically the PSID
self-report is far more transient than a DI award -- interval recovery
(exit from self-reported disability) runs 25-50% in these data, orders
of magnitude above SSA DI recovery -- because respondents cycle in and
out of "permanently disabled" and relabel toward "retired" near the
full retirement age. This module therefore reports its hazards as a
work-limitation series and **never equates them to DI award, recovery,
or termination rates**; the ratio to any SSA administrative series is
reported honestly (as the mortality foundation reports the PSID
undercount), never calibrated away. The named deltas are carried in
full in ``runs/m4_disability_v1.json`` (``concept_deltas``).

DI in policyengine-us
---------------------
policyengine-us models SSDI dollars as a single uprated survey **input**
(``social_security_disability``, added into ``social_security`` beside
retirement/survivors/dependents) with no incidence, recovery, or
full-retirement-age conversion formula -- the same input-only shape the
402(b)/(c)/(e)/(f) auxiliary benefits have (see
:mod:`populace_dynamics.ss.params`). So the DI *dynamics* and the
DI->retirement conversion are not sourced from pe-us; the empirical
hazards live here on the PSID basis and the statutory conversion factor
is carried, statute-cited, in
:mod:`populace_dynamics.disability_conversion`.

Construction
------------
* **Status person-years.** One row per person-wave from
  :func:`read_disability_status`, restricted to a valid self-reported
  status (codes 1-8; code 0 "Inap." and code 9 "NA/DK" are not a state)
  and, by default, to persons present in a responding family unit
  (sequence 1-20). ``disabled`` is ``status_code == 5``.
* **Sex.** Attached person-constant from a
  :func:`populace_dynamics.data.deaths.read_death_records`-shaped frame
  (``ER32000``), exactly as the mortality foundation attaches it.
* **Transition pairs.** For each person, consecutive observed waves
  ``(w, w')`` with ``w' - w <= MAX_INTERVAL`` (the annual/biennial grid
  step; longer attrition-and-return gaps are dropped, not interpolated)
  form one transition record carrying the age and weight at ``w`` and
  the from/to disability state. Incidence is the not-disabled ->
  disabled transition over not-disabled exposure; recovery is the
  disabled -> not-disabled transition over disabled exposure; both are
  banded by age-at-``w`` x sex. A death or attrition between waves leaves
  no ``w'`` observation, so the pair simply does not form -- the person
  is right-censored, never miscounted as a recovery.

Rates are **per observation interval** (numerator transitions over
at-risk intervals). Because the grid mixes 1- and 2-year steps, a cell's
exposure-weighted mean interval length is carried alongside every rate;
annualization is a documented future-gate decision, not silently
applied here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from populace_dynamics.data import panels, psid

__all__ = [
    "EMPLOYMENT_STATUS_CODES",
    "DISABLED_CODE",
    "RETIRED_CODE",
    "VALID_STATUS_CODES",
    "SEXES",
    "DI_AGE_BANDS",
    "MAX_INTERVAL",
    "CONVERSION_WINDOW",
    "MAX_AGE",
    "DisabilityPanel",
    "band_label",
    "parse_sas_value_labels",
    "parse_sas_format_assignments",
    "employment_status_formats_path",
    "verify_employment_status_codes",
    "read_disability_status",
    "attach_sex",
    "build_transition_pairs",
    "build_disability_panel",
    "incidence_cells",
    "recovery_cells",
    "conversion_cells",
    "prevalence_cells",
    "reference_moments",
]

#: ``EMPLOYMENT STATUS`` value codes, verified 2026-07-10 against the
#: IND2023ER SAS ``VALUE`` blocks (ER35116F for 2023; the coding is
#: stable across every covered wave). Code 5 is the disability signal;
#: 0 ("Inap.") and 9 ("NA/DK") are not a self-reported status.
EMPLOYMENT_STATUS_CODES: dict[int, str] = {
    0: "inap",
    1: "working",
    2: "temporarily_laid_off",
    3: "unemployed",
    4: "retired",
    5: "disabled",
    6: "keeping_house",
    7: "student",
    8: "other",
    9: "na_dk",
}

#: The self-reported disability code (``"Permanently disabled"``).
DISABLED_CODE = 5
#: The retired code, used only by the conversion analog (a DI ->
#: retirement conversion shows up as disabled -> retired near FRA).
RETIRED_CODE = 4
#: Codes that count as an ascertained self-reported status (a person-year
#: state). 0 ("Inap.") and 9 ("NA/DK") are excluded from every numerator
#: and denominator.
VALID_STATUS_CODES: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7, 8})

SEXES: tuple[str, ...] = ("female", "male")

#: Working-age-through-FRA incidence/recovery bands (closed single-year
#: bounds; the top band ends at 66, just below the modern FRA, because a
#: disabled worker auto-converts to a retirement benefit at FRA -- past
#: that age the DI hazard is not defined, it is the conversion).
DI_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (20, 29),
    (30, 39),
    (40, 49),
    (50, 59),
    (60, 66),
)

#: Longest wave gap (years) that counts as a grid-adjacent transition
#: interval. The PSID individual grid is annual 1993-1997 and biennial
#: 1999-2023, so 2 keeps every clean step and drops attrition-and-return
#: gaps (and the isolated 1983->1993 decade jump) rather than
#: interpolating a hazard across them.
MAX_INTERVAL = 2

#: Age window over which a disabled -> retired transition is read as the
#: PSID analog of the 6.B5.1 disability-conversion column (the
#: FRA-crossing years).
CONVERSION_WINDOW: tuple[int, int] = (60, 67)

MAX_AGE = 120

_ID_1968_INTERVIEW = "ER30001"
_ID_PERSON_NUMBER = "ER30002"

#: Concept -> label-regex fallbacks for the status melt. Age, sequence
#: and weight reuse the verified demographic patterns so the resolved
#: variables are identical to :func:`panels.demographic_panel`'s; the
#: employment-status pattern is anchored so the (non-existent) 2nd/3rd
#: mention columns could never collide into a wave.
_STATUS_CONCEPTS: dict[str, tuple[str, ...]] = {
    "status": (r"^EMPLOYMENT STATUS",),
    "age": panels.DEMOGRAPHIC_CONCEPTS["age"],
    "sequence": panels.DEMOGRAPHIC_CONCEPTS["sequence"],
    "weight": panels.DEMOGRAPHIC_CONCEPTS["weight"],
}

_IN_FAMILY_SEQUENCE = (1, 20)


# --------------------------------------------------------------------------
# Value-label verification against the release's SAS format blocks
# --------------------------------------------------------------------------
#: A ``VALUE <fmt>`` block header in a PSID ``*_formats.sas`` file.
_VALUE_HEADER_RE = re.compile(r"^\s*VALUE\s+(\S+)\s*$", re.MULTILINE)
#: A single-line integer ``N = 'label'`` value-label pair (the short
#: labels; multi-line "Inap." continuations are irrelevant to the codes
#: this module verifies and are skipped).
_VALUE_LINE_RE = re.compile(
    r"^\s*(-?\d+)\s*=\s*'((?:[^']|'')*)'", re.MULTILINE
)
#: A ``VAR  VARFMT.`` format-assignment line (variable starts with a
#: letter; distinguishes it from the ``N = 'label'`` value lines).
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z]\w*)\s+(\w+)\.\s*$", re.MULTILINE)


def employment_status_formats_path(data_dir: Path | None = None) -> Path:
    """Resolve the ``IND2023ER_formats.sas`` path for the staged file."""
    subdir = psid.PRODUCTS["ind2023er"][0]
    return psid._resolve_data_dir(data_dir) / subdir / "IND2023ER_formats.sas"


def parse_sas_value_labels(path: str | Path) -> dict[str, dict[int, str]]:
    """Parse ``VALUE <fmt> ... ;`` blocks from a PSID SAS formats file.

    Returns ``{format_name: {code: label}}`` for the integer-coded,
    single-line value labels in each block (the short labels; multi-line
    "Inap." continuation labels are not integer-decoded here and are
    skipped -- this reader verifies the ascertained-status codes only).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no ``VALUE`` block can be found.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PSID SAS formats file not found: {path} "
            "(see ~/PolicyEngine/psid-data/README.md)"
        )
    text = path.read_text(errors="replace")
    headers = list(_VALUE_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError(
            f"No VALUE block found in {path}; not a PSID SAS formats file?"
        )
    out: dict[str, dict[int, str]] = {}
    for i, header in enumerate(headers):
        fmt = header.group(1)
        block_start = header.end()
        block_end = (
            headers[i + 1].start() if i + 1 < len(headers) else len(text)
        )
        semicolon = text.find(";", block_start)
        if semicolon != -1:
            block_end = min(block_end, semicolon)
        block = text[block_start:block_end]
        codes: dict[int, str] = {}
        for code, label in _VALUE_LINE_RE.findall(block):
            codes[int(code)] = label.replace("''", "'").strip()
        out[fmt] = codes
    return out


def parse_sas_format_assignments(path: str | Path) -> dict[str, str]:
    """Parse ``VAR  VARFMT.`` format assignments from a SAS formats file.

    Returns ``{variable_name: format_name}``. Value-label lines
    (``N = 'label'``) do not match the assignment pattern, so the two
    sections do not cross-contaminate.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PSID SAS formats file not found: {path} "
            "(see ~/PolicyEngine/psid-data/README.md)"
        )
    text = path.read_text(errors="replace")
    return {var: fmt for var, fmt in _ASSIGN_RE.findall(text)}


def _status_variables(
    data_dir: Path | None, waves: list[int] | None
) -> dict[int, str]:
    """``{wave: employment-status variable}`` for the requested waves."""
    sps_path = psid.product_sps_path("ind2023er", data_dir)
    labels = psid.parse_sps_labels(sps_path)
    by_wave = panels.wave_variables(labels, _STATUS_CONCEPTS["status"][0])
    if waves is not None:
        missing = [w for w in waves if w not in by_wave]
        if missing:
            raise ValueError(
                f"Wave(s) {missing} carry no EMPLOYMENT STATUS variable."
            )
        by_wave = {w: by_wave[w] for w in waves}
    return by_wave


def verify_employment_status_codes(
    *,
    data_dir: Path | None = None,
    waves: list[int] | None = None,
) -> dict[int, dict[str, str]]:
    """Verify code 5 = disabled and code 1 = working in the formats file.

    For every covered wave's ``EMPLOYMENT STATUS`` variable, resolve the
    SAS format it is assigned (falling back to the ``<var>F`` naming
    convention when the assignment block omits it) and assert that value
    code :data:`DISABLED_CODE` maps to a label containing ``"disabl"``
    and code 1 to one containing ``"work"`` (case-insensitive). This is
    the value-label analog of the demographic readers' variable-label
    verification: a release that renumbered the employment-status recode
    fails loudly here instead of silently mislabelling every disability
    person-year.

    Returns ``{wave: {"variable", "format", "code5_label",
    "code1_label"}}`` on success.

    Raises:
        ValueError: On the first wave whose format is missing or whose
            code 5 / code 1 label does not match.
    """
    by_wave = _status_variables(data_dir, waves)
    fmt_path = employment_status_formats_path(data_dir)
    value_labels = parse_sas_value_labels(fmt_path)
    assignments = parse_sas_format_assignments(fmt_path)

    out: dict[int, dict[str, str]] = {}
    for wave in sorted(by_wave):
        var = by_wave[wave]
        fmt = assignments.get(var, f"{var}F")
        codes = value_labels.get(fmt)
        if not codes:
            raise ValueError(
                f"Wave {wave}: no VALUE block for format {fmt!r} "
                f"(variable {var}) in {fmt_path.name}; the release "
                "format layout may have changed."
            )
        code5 = codes.get(DISABLED_CODE, "")
        code1 = codes.get(1, "")
        if "disabl" not in code5.lower():
            raise ValueError(
                f"Wave {wave}: EMPLOYMENT STATUS code {DISABLED_CODE} in "
                f"format {fmt!r} is {code5!r}, expected a 'disabled' "
                "label. The employment-status recode may have been "
                "renumbered; re-verify disability.EMPLOYMENT_STATUS_CODES."
            )
        if "work" not in code1.lower():
            raise ValueError(
                f"Wave {wave}: EMPLOYMENT STATUS code 1 in format "
                f"{fmt!r} is {code1!r}, expected a 'working' label."
            )
        out[wave] = {
            "variable": var,
            "format": fmt,
            "code5_label": code5,
            "code1_label": code1,
        }
    return out


# --------------------------------------------------------------------------
# The status person-period reader
# --------------------------------------------------------------------------
def read_disability_status(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
    in_family_only: bool = True,
) -> pd.DataFrame:
    """Read the per-wave self-reported disability status person-period.

    Melts the twenty ``EMPLOYMENT STATUS`` waves of the cross-year
    individual file (label-resolved via
    :func:`populace_dynamics.data.panels.ind_person_period`, so a
    re-layout fails loudly) and keeps only ascertained states
    (:data:`VALID_STATUS_CODES`; code 0 "Inap." and 9 "NA/DK" dropped).

    Columns: ``person_id``, ``period`` (wave year), ``status_code`` (raw
    ``EMPLOYMENT STATUS``), ``status`` (its label), ``disabled``
    (``status_code == 5``), ``retired`` (``== 4``), ``age``, ``weight``,
    ``sequence``. With ``in_family_only`` (default) rows are restricted
    to sequence 1-20 -- persons present in a responding family unit at
    the interview, the presence notion the demographic panel uses.

    Only the variable *labels* are verified at read time (through
    ``ind_person_period``); the value-code map (5 = disabled) is verified
    separately by :func:`verify_employment_status_codes` against the SAS
    format blocks.
    """
    panel = panels.ind_person_period(
        _STATUS_CONCEPTS, data_dir=data_dir, nrows=nrows
    )
    panel["status_code"] = panel["status"].astype("int64")
    if in_family_only:
        low, high = _IN_FAMILY_SEQUENCE
        present = (panel["sequence"] >= low) & (panel["sequence"] <= high)
        panel = panel.loc[present]
    valid = panel["status_code"].isin(VALID_STATUS_CODES)
    age_ok = (panel["age"] >= 0) & (panel["age"] <= MAX_AGE)
    weight_ok = panel["weight"] > 0
    panel = panel.loc[valid & age_ok & weight_ok].copy()

    panel["status"] = (
        panel["status_code"].map(EMPLOYMENT_STATUS_CODES).astype("string")
    )
    panel["disabled"] = panel["status_code"] == DISABLED_CODE
    panel["retired"] = panel["status_code"] == RETIRED_CODE
    panel["age"] = panel["age"].astype("int64")
    panel["weight"] = panel["weight"].astype("float64")
    cols = [
        "person_id",
        "period",
        "status_code",
        "status",
        "disabled",
        "retired",
        "age",
        "weight",
        "sequence",
    ]
    return (
        panel[cols].sort_values(["person_id", "period"]).reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Sex attach + grid-adjacent transition pairs
# --------------------------------------------------------------------------
def attach_sex(
    status: pd.DataFrame, death_records: pd.DataFrame
) -> pd.DataFrame:
    """Attach person-constant sex from a death-records-shaped frame.

    ``death_records`` need only carry ``person_id`` and ``sex`` (the
    :func:`populace_dynamics.data.deaths.read_death_records` shape, which
    reads sex from the same individual file). Rows whose sex is not
    ``male``/``female`` are dropped, matching the mortality foundation.
    """
    sex_by_person = death_records.set_index("person_id")["sex"]
    out = status.copy()
    out["sex"] = out["person_id"].map(sex_by_person).astype("string")
    return out[out["sex"].isin(SEXES)].reset_index(drop=True)


def build_transition_pairs(
    status_with_sex: pd.DataFrame, *, max_interval: int = MAX_INTERVAL
) -> pd.DataFrame:
    """Grid-adjacent (``<= max_interval`` yr) wave-pair transitions.

    One row per person per consecutive observed-wave pair ``(w, w')``
    with ``w' - w <= max_interval``: ``person_id``, ``sex``,
    ``age`` (at ``w``), ``weight`` (at ``w``), ``interval`` (``w' - w``),
    ``from_disabled`` / ``to_disabled`` / ``to_retired`` booleans. Pairs
    across longer gaps (attrition-and-return, or the isolated
    1983->1993 jump) are dropped, not interpolated.
    """
    s = status_with_sex.sort_values(["person_id", "period"])
    grp = s.groupby("person_id", sort=False)
    next_period = grp["period"].shift(-1)
    next_disabled = grp["disabled"].shift(-1)
    next_retired = grp["retired"].shift(-1)
    interval = next_period - s["period"]
    keep = next_period.notna() & (interval >= 1) & (interval <= max_interval)

    return pd.DataFrame(
        {
            "person_id": s["person_id"][keep].to_numpy(),
            "sex": s["sex"][keep].to_numpy(),
            "age": s["age"][keep].astype("int64").to_numpy(),
            "weight": s["weight"][keep].astype("float64").to_numpy(),
            "interval": interval[keep].astype("int64").to_numpy(),
            "from_disabled": s["disabled"][keep].to_numpy(dtype=bool),
            "to_disabled": next_disabled[keep].to_numpy(dtype=bool),
            "to_retired": next_retired[keep].to_numpy(dtype=bool),
        }
    ).reset_index(drop=True)


@dataclass(frozen=True)
class DisabilityPanel:
    """Self-reported disability person-years and grid-adjacent pairs.

    Attributes:
        person_years: One row per valid person-wave with ``person_id``,
            ``period``, ``age``, ``sex``, ``weight``, ``status_code``,
            ``disabled``, ``retired`` -- the DI occupancy (a maximal run
            of ``disabled`` person-years is one work-limitation episode).
        pairs: One row per grid-adjacent wave transition
            (:func:`build_transition_pairs`) -- the incidence / recovery
            / conversion hazard basis.
    """

    person_years: pd.DataFrame
    pairs: pd.DataFrame


def build_disability_panel(
    status: pd.DataFrame,
    death_records: pd.DataFrame,
    *,
    max_interval: int = MAX_INTERVAL,
) -> DisabilityPanel:
    """Assemble the disability person-year panel and its transitions.

    Pure transform over a :func:`read_disability_status`-shaped frame and
    a :func:`populace_dynamics.data.deaths.read_death_records`-shaped
    frame (sex source) -- unit-testable on synthetic fixtures.
    """
    with_sex = attach_sex(status, death_records)
    cols = [
        "person_id",
        "period",
        "age",
        "sex",
        "weight",
        "status_code",
        "disabled",
        "retired",
    ]
    person_years = with_sex[cols].reset_index(drop=True)
    pairs = build_transition_pairs(with_sex, max_interval=max_interval)
    return DisabilityPanel(person_years=person_years, pairs=pairs)


# --------------------------------------------------------------------------
# Hazard / prevalence cells
# --------------------------------------------------------------------------
def band_label(lo: int, hi: int) -> str:
    """Flat band label, e.g. ``"50-59"`` or open-topped ``"60+"``."""
    return f"{lo}-{hi}" if hi < MAX_AGE else f"{lo}+"


def _band_of(age: object, bands: tuple[tuple[int, int], ...]) -> str | None:
    if pd.isna(age):
        return None
    a = int(age)
    for lo, hi in bands:
        if lo <= a <= hi:
            return band_label(lo, hi)
    return None


def _rate_cell(
    num_weight: float,
    den_weight: float,
    n_events: int,
    n_at_risk: int,
    mean_interval: float,
) -> dict[str, float]:
    return {
        "rate": float(num_weight / den_weight) if den_weight > 0 else 0.0,
        "num_wt": float(num_weight),
        "den_wt": float(den_weight),
        "n_events": int(n_events),
        "n_at_risk": int(n_at_risk),
        "mean_interval_yr": float(mean_interval),
    }


def _transition_cells(
    at_risk: pd.DataFrame,
    event_col: str,
    *,
    prefix: str,
    weighted: bool,
) -> dict[str, dict[str, float]]:
    """Weighted per-interval transition rate by :data:`DI_AGE_BANDS` x sex.

    ``at_risk`` is the subset of pairs in the origin state; ``event_col``
    the boolean transition indicator. Every band x sex cell is emitted
    (zero-exposure cells carry a zero rate) so the reference-moment key
    set is fixed regardless of the person subset.
    """
    df = at_risk.copy()
    df["band"] = df["age"].map(lambda a: _band_of(a, DI_AGE_BANDS))
    df = df[df["band"].notna()]
    w = df["weight"] if weighted else pd.Series(1.0, index=df.index)
    df = df.assign(_w=w)

    out: dict[str, dict[str, float]] = {}
    for lo, hi in DI_AGE_BANDS:
        band = band_label(lo, hi)
        for sex in SEXES:
            cell = df[(df["band"] == band) & (df["sex"] == sex)]
            den_wt = float(cell["_w"].sum())
            events = cell[cell[event_col]]
            num_wt = float(events["_w"].sum())
            mean_iv = (
                float((cell["_w"] * cell["interval"]).sum() / den_wt)
                if den_wt > 0
                else 0.0
            )
            out[f"{prefix}.{band}|{sex}"] = _rate_cell(
                num_wt, den_wt, len(events), len(cell), mean_iv
            )
    return out


def incidence_cells(
    panel: DisabilityPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """DI-onset hazard: not-disabled -> disabled per interval, band x sex."""
    pairs = panel.pairs
    if person_ids is not None:
        pairs = pairs[pairs["person_id"].isin(person_ids)]
    at_risk = pairs[~pairs["from_disabled"]]
    return _transition_cells(
        at_risk, "to_disabled", prefix="incidence", weighted=weighted
    )


def recovery_cells(
    panel: DisabilityPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """DI-recovery hazard: disabled -> not-disabled per interval, band x sex.

    NOTE (named delta): this is exit from *self-reported* disability, an
    order of magnitude more transient than an SSA DI termination; it is
    never equated to a DI recovery rate.
    """
    pairs = panel.pairs
    if person_ids is not None:
        pairs = pairs[pairs["person_id"].isin(person_ids)]
    at_risk = pairs[pairs["from_disabled"]].copy()
    at_risk["recovered"] = ~at_risk["to_disabled"]
    return _transition_cells(
        at_risk, "recovered", prefix="recovery", weighted=weighted
    )


def prevalence_cells(
    panel: DisabilityPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """Weighted disabled prevalence among person-years, band x sex.

    The occupancy stock (share of valid person-years self-reporting
    ``disabled``), the integrative quantity a payable-baseline
    composition rides on. ``n_events`` is the disabled person-year count.
    """
    py = panel.person_years
    if person_ids is not None:
        py = py[py["person_id"].isin(person_ids)]
    py = py.copy()
    py["band"] = py["age"].map(lambda a: _band_of(a, DI_AGE_BANDS))
    py = py[py["band"].notna()]
    w = py["weight"] if weighted else pd.Series(1.0, index=py.index)
    py = py.assign(_w=w)

    out: dict[str, dict[str, float]] = {}
    for lo, hi in DI_AGE_BANDS:
        band = band_label(lo, hi)
        for sex in SEXES:
            cell = py[(py["band"] == band) & (py["sex"] == sex)]
            den_wt = float(cell["_w"].sum())
            dis = cell[cell["disabled"]]
            num_wt = float(dis["_w"].sum())
            out[f"prevalence.{band}|{sex}"] = _rate_cell(
                num_wt, den_wt, len(dis), len(cell), 0.0
            )
    return out


def conversion_cells(
    panel: DisabilityPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """PSID analog of the 6.B5.1 disability-conversion column, by sex.

    Among grid-adjacent pairs that transition INTO ``retired`` at the
    FRA-crossing ages (:data:`CONVERSION_WINDOW`), the weighted share
    whose origin state was ``disabled`` -- a disabled -> retired
    transition read as a DI -> retirement conversion. This is
    structurally parallel to 6.B5.1 (conversions over retired-worker
    entries) but on a self-reported-labor-force basis; the artifact
    reports the ratio to the administrative share with the concept delta
    named, never a calibrated match.
    """
    pairs = panel.pairs
    if person_ids is not None:
        pairs = pairs[pairs["person_id"].isin(person_ids)]
    lo, hi = CONVERSION_WINDOW
    entries = pairs[
        pairs["to_retired"] & (pairs["age"] >= lo) & (pairs["age"] <= hi)
    ].copy()
    w = entries["weight"] if weighted else pd.Series(1.0, index=entries.index)
    entries = entries.assign(_w=w)

    out: dict[str, dict[str, float]] = {}
    for sex in SEXES:
        cell = entries[entries["sex"] == sex]
        den_wt = float(cell["_w"].sum())
        conv = cell[cell["from_disabled"]]
        num_wt = float(conv["_w"].sum())
        mean_iv = (
            float((cell["_w"] * cell["interval"]).sum() / den_wt)
            if den_wt > 0
            else 0.0
        )
        out[f"conversion.retired_from_disabled|{sex}"] = _rate_cell(
            num_wt, den_wt, len(conv), len(cell), mean_iv
        )
    return out


def reference_moments(
    panel: DisabilityPanel,
    person_ids: set[int] | None = None,
    *,
    weighted: bool = True,
) -> dict[str, dict[str, float]]:
    """Every M4 reference-moment cell for a person subset.

    The union of incidence, recovery, prevalence and conversion cells.
    Calling with ``person_ids=None`` gives the committed reference
    moments; calling on each half of a person-disjoint split gives the
    noise-floor inputs (as in the mortality foundation).
    """
    cells: dict[str, dict[str, float]] = {}
    cells.update(incidence_cells(panel, person_ids, weighted=weighted))
    cells.update(recovery_cells(panel, person_ids, weighted=weighted))
    cells.update(prevalence_cells(panel, person_ids, weighted=weighted))
    cells.update(conversion_cells(panel, person_ids, weighted=weighted))
    return cells
