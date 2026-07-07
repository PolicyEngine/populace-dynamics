"""Per-child records from the PSID Childbirth & Adoption History File.

The Childbirth and Adoption History File (PSID packaged product 1109,
collection waves 1985-2023, ``cah85_23``) carries one record per
childbirth or adoption event reported by an eligible parent, plus one
placeholder "denial" record per parent who reported no biological
children and one per parent who reported no adoptions. Its retrospective
fertility timing drives the childbirth-transition moments and the
caregiver anchors; the parent and child keys join it to the earnings
panel (:mod:`populace_dynamics.data.family`) and the demographic panel
(:mod:`populace_dynamics.data.panels`).

Person identifiers
------------------
Same convention as the rest of the stack: the 1968 family interview
number with the person number, ``id = interview * 1000 + person``. The
parent is ``CAH3``/``CAH4`` and the child ``CAH10``/``CAH11``, so both
``parent_person_id`` and ``child_person_id`` are directly joinable to
the individual file's ``ER30001 * 1000 + ER30002``.

Record structure and the denial placeholder
--------------------------------------------
The file's own documentation: "If an individual has never had any
children, one record indicates that report. Similarly, if the individual
never adopted any children, one record contains the denial." Those
placeholder records carry the child fields at the inapplicable code
(``CAH12 = 9`` "no child", ``CAH15 = 9999``, no child id). They are
kept, flagged ``is_event = False``, so the row count is stable and a
parent's zero-fertility report is visible; :func:`birth_events`
restricts to actual events.

Documented judgment calls (verified against the staged Release 2 file,
2026-07-07)
-----------------------------------------------------------------------
* **Child identifier joinability.** ``child_person_id`` is
  ``CAH10 * 1000 + CAH11`` only for an enumerable PSID sample-member
  child (``CAH10`` in 1-9308 and ``CAH11`` in 1-399); otherwise NA. The
  excluded cases, confirmed in the data: ``CAH11`` in 800-995 = a child
  who has never been in any PSID sample family (a real child, but not a
  PSID person -- common for adoptees and for children who left or died
  before enumeration), ``== 0`` = inapplicable (denial placeholder),
  ``== 999`` / ``CAH10 == 9999`` = NA. Of 148,739 records, 90,065 are
  actual events (``is_event``) and 67,819 carry a joinable child id
  (66,776 births, 1,043 adoptions) -- most adopted children and many
  departed/deceased children are non-sample and so are not joinable.
* **Year missing sentinels.** Year fields use ``9998`` = NA/DK and
  ``9999`` = inapplicable, both mapped to NA. ``CAH7`` (parent birth)
  carries only ``9998``; ``CAH15`` (child birth) and ``CAH26`` (moved
  out or died) carry both; ``CAH114`` (year the child count was last
  reported) carries neither.
* **Month sentinels and season codes.** Month fields use ``98`` = NA/DK
  and ``99`` = inapplicable (both -> NA); child-birth-month codes
  ``21``-``24`` are *seasons* and are kept as coarse timing, not NA.
* **Birth order.** ``CAH9`` is the actual birth order 1-18; ``98`` =
  NA/DK and ``99`` = inapplicable (adoption record or denial) map to NA.
  Twins/multiples were assigned consecutive orders.
* **Record type and event flag.** ``record_type`` decodes ``CAH2``
  (1 = birth, 2 = adoption). ``is_event`` is ``CAH12 != 9`` -- True for
  an actual birth/adoption (the child's sex is known or NA-but-present),
  False for a denial placeholder. Adoption detail lives in the raw
  ``adoptive_relationship_code`` (``CAH117``); it is inapplicable (99)
  on childbirth records, mapped to NA there.
* **Coded-field domains.** ``parent_sex``/``child_sex`` (male/female),
  ``mother_marital_status_at_birth`` (``CAH8``), and
  ``where_child_last_reported`` (``CAH24``, whose code 6 = deceased is
  the survivorship signal) are decoded from their exact documented code
  sets, with the inapplicable ("no child") code mapped to NA and any
  undocumented code raising rather than passing through.
* **Single harmonized coding.** Like the marriage file, this is a single
  retrospective product with one coding across collection waves, so no
  era-varmap is needed; the label check still guards a re-layout.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from populace_dynamics.data import psid

__all__ = [
    "PRODUCT",
    "birth_history",
    "birth_events",
    "RECORD_TYPE",
    "MOTHER_MARITAL_STATUS",
    "WHERE_CHILD_LAST_REPORTED",
    "ADOPTIVE_RELATIONSHIP",
]

#: PSID product key for the Childbirth & Adoption History File.
PRODUCT = "cah85_23"

#: Variables read and their exact documented labels; the whole set is
#: label-verified at read so a re-layout fails loudly.
_VARS: dict[str, str] = {
    "CAH1": "RELEASE NUMBER",
    "CAH2": "RECORD TYPE",
    "CAH3": "1968 INTERVIEW NUMBER OF PARENT",
    "CAH4": "PERSON NUMBER OF PARENT",
    "CAH5": "SEX OF PARENT",
    "CAH6": "MONTH PARENT BORN",
    "CAH7": "YEAR PARENT BORN",
    "CAH8": "MARITAL STATUS OF MOTHER WHEN IND BORN",
    "CAH9": "BIRTH ORDER",
    "CAH10": "1968 INTERVIEW NUMBER OF CHILD",
    "CAH11": "PERSON NUMBER OF CHILD",
    "CAH12": "OS1. SEX OF CHILD",
    "CAH13": "OS2. MONTH CHILD BORN",
    "CAH15": "OS2. YEAR CHILD BORN",
    "CAH24": "OS5. WHERE CHILD WAS WHEN LAST REPORTED",
    "CAH25": "OS6. MONTH CHILD MOVED OUT OR DIED",
    "CAH26": "OS6. YEAR CHILD MOVED OUT OR DIED",
    "CAH114": "YR MOST RECENTLY REPORTED NUMBER OF KIDS",
    "CAH115": "YEAR MOST RECENTLY REPORTED THIS CHILD",
    "CAH116": "NUMBER OF NATURAL OR ADOPTED CHILDREN",
    "CAH117": "RELATIONSHIP TO ADOPTIVE PARENT",
    "CAH118": "NUMBER OF BIRTH OR ADOPTION RECORDS",
}

#: CAH2 record type.
RECORD_TYPE: dict[int, str] = {1: "birth", 2: "adoption"}

#: CAH5 / CAH12 sex codes (child sex adds 8 = NA, 9 = no child as NA).
_SEX: dict[int, str] = {1: "male", 2: "female"}

#: CAH8 marital status of the mother when the child was born.
MOTHER_MARITAL_STATUS: dict[int, str] = {
    1: "married",
    2: "never_married",
    3: "widowed",
    4: "divorced",
    5: "separated",
    7: "other",
    8: "unknown",
}

#: CAH24 where the child was when last reported (code 6 = deceased is the
#: survivorship signal used by the caregiver anchors).
WHERE_CHILD_LAST_REPORTED: dict[int, str] = {
    1: "in_family_unit",
    2: "with_father_outside_fu",
    3: "with_mother_outside_fu",
    4: "with_other_relative",
    5: "own_household",
    6: "deceased",
    7: "elsewhere",
    8: "adopted_out_to_relative",
    9: "adopted_out_to_nonrelative",
    10: "adopted_out_unknown",
    11: "fostered_out",
    98: "unknown",
}

#: CAH117 relationship of the adopted child to the adoptive parent
#: (raw codes retained; documented here for reference). 98 = NA/DK,
#: 99 = inapplicable (childbirth record) and is mapped to NA.
ADOPTIVE_RELATIONSHIP: dict[int, str] = {
    30: "natural_child",
    33: "stepchild",
    35: "child_of_cohabitor",
    38: "foster_child",
    40: "sibling",
    47: "sibling_in_law",
    48: "sibling_of_cohabitor",
    60: "grandchild",
    65: "great_grandchild",
    70: "niece_or_nephew",
    71: "niece_or_nephew_of_spouse",
    74: "cousin",
    75: "cousin_of_spouse",
    83: "child_of_first_year_cohabitor",
    94: "other_nonrelative",
    95: "other_relative",
    96: "other_relative_of_spouse",
    97: "other_relative_of_cohabitor",
    98: "unknown",
}

#: Year fields: 9998 = NA/DK, 9999 = inapplicable.
_YEAR_NA: tuple[int, ...] = (9998, 9999)
#: Month fields: 98 = NA/DK, 99 = inapplicable (season codes 21-24 kept).
_MONTH_NA: tuple[int, ...] = (98, 99)
#: Birth-order field: 98 = NA/DK, 99 = inapplicable.
_ORDER_NA: tuple[int, ...] = (98, 99)
#: Count-of-children field: 98 = NA/DK.
_COUNT_NA: tuple[int, ...] = (98,)
#: CAH117 inapplicable code (childbirth record) -> NA.
_ADOPT_INAP: tuple[int, ...] = (99,)

_INTERVIEW_RANGE = (1, 9308)
_PERSON_RANGE = (1, 399)


def _na(series: pd.Series, sentinels: tuple[int, ...]) -> pd.Series:
    """Cast a raw integer column to nullable Int64, sentinels -> NA."""
    return series.astype("Int64").mask(series.isin(list(sentinels)))


def _decode(
    series: pd.Series,
    mapping: dict[int, str],
    field: str,
    *,
    na_codes: Iterable[int] = (),
) -> pd.Series:
    """Map integer codes to labels; ``na_codes`` -> NA; else raise.

    A code that is neither documented in ``mapping`` nor listed in
    ``na_codes`` raises, so a changed coding fails loudly rather than
    silently dropping to NA.
    """
    na_set = set(na_codes)
    observed = {int(v) for v in pd.unique(series.dropna())}
    unexpected = observed - set(mapping) - na_set
    if unexpected:
        raise ValueError(
            f"{field}: undocumented code(s) {sorted(unexpected)}; the "
            f"documented domain is {sorted(mapping)}. The release coding "
            "may have changed."
        )
    full: dict[int, object] = {code: pd.NA for code in na_set}
    full.update(mapping)
    return series.map(full).astype("string")


def _build(raw: pd.DataFrame) -> pd.DataFrame:
    """Assemble the tidy per-record frame from a raw CAH column frame.

    Pure transform over the read columns, split out so the event logic
    and sentinel/dtype handling are unit-testable on synthetic rows.
    """
    cah10 = raw["CAH10"].astype("int64")
    cah11 = raw["CAH11"].astype("int64")
    joinable = cah11.between(*_PERSON_RANGE) & cah10.between(*_INTERVIEW_RANGE)
    child_id = (cah10 * 1000 + cah11).astype("Int64").mask(~joinable)

    out = pd.DataFrame(
        {
            "parent_person_id": (
                raw["CAH3"].astype("int64") * 1000
                + raw["CAH4"].astype("int64")
            ).astype("int64"),
            "parent_sex": _decode(raw["CAH5"], _SEX, "CAH5 parent sex"),
            "parent_birth_year": _na(raw["CAH7"], _YEAR_NA),
            "parent_birth_month": _na(raw["CAH6"], _MONTH_NA),
            "record_type": _decode(raw["CAH2"], RECORD_TYPE, "CAH2 type"),
            "child_person_id": child_id,
            "child_sex": _decode(
                raw["CAH12"], _SEX, "CAH12 child sex", na_codes=(8, 9)
            ),
            "birth_year": _na(raw["CAH15"], _YEAR_NA),
            "birth_month": _na(raw["CAH13"], _MONTH_NA),
            "birth_order": _na(raw["CAH9"], _ORDER_NA),
            "mother_marital_status_at_birth": _decode(
                raw["CAH8"],
                MOTHER_MARITAL_STATUS,
                "CAH8 mother status",
                na_codes=(9,),
            ),
            "adoptive_relationship_code": _na(raw["CAH117"], _ADOPT_INAP),
            "where_child_last_reported": _decode(
                raw["CAH24"],
                WHERE_CHILD_LAST_REPORTED,
                "CAH24 where last reported",
                na_codes=(99,),
            ),
            "moved_out_or_died_year": _na(raw["CAH26"], _YEAR_NA),
            "moved_out_or_died_month": _na(raw["CAH25"], _MONTH_NA),
            "most_recent_child_report_year": _na(raw["CAH115"], _YEAR_NA),
            "children_count_report_year": raw["CAH114"].astype("Int64"),
            "n_children": _na(raw["CAH116"], _COUNT_NA),
            "n_records": raw["CAH118"].astype("int64"),
            "is_event": (raw["CAH12"] != 9).to_numpy(),
        }
    )
    return out


def birth_history(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read the Childbirth & Adoption History File into a per-record frame.

    Returns one row per file record -- one per birth/adoption event, plus
    per-parent denial placeholders (``is_event = False``). Every column
    is resolved from the file's own labels and verified before the
    fixed-width read; coded fields are decoded from their documented
    domains with an undocumented code raising.

    Columns: ``parent_person_id``, ``parent_sex``, ``parent_birth_year``,
    ``parent_birth_month``, ``record_type`` (birth/adoption),
    ``child_person_id`` (NA unless a joinable PSID person), ``child_sex``,
    ``birth_year``/``birth_month``, ``birth_order``,
    ``mother_marital_status_at_birth``, ``adoptive_relationship_code``,
    ``where_child_last_reported`` (code 6 = deceased),
    ``moved_out_or_died_year``/``_month``, ``most_recent_child_report_year``,
    ``children_count_report_year``, ``n_children``, ``n_records``,
    ``is_event``. Year/month/order/count fields are nullable ``Int64``
    with PSID missing sentinels mapped to NA (see the module docstring).
    """
    sps_path = psid.product_sps_path(PRODUCT, data_dir)
    labels = psid.parse_sps_labels(sps_path)
    psid.verify_labels(labels, _VARS, context=PRODUCT)
    raw = psid.read_psid(
        PRODUCT, columns=list(_VARS), data_dir=data_dir, nrows=nrows
    )
    return _build(raw)


def birth_events(records: pd.DataFrame) -> pd.DataFrame:
    """Actual birth/adoption events, ordered per parent by date.

    Pure transform over a :func:`birth_history`-shaped frame (so it is
    unit-testable on synthetic rows): drops denial placeholders
    (``is_event = False``) and orders each parent's events by birth year
    then birth order. The result is the fertility-event timeline the
    childbirth-transition moments build on.
    """
    ev = records[records["is_event"]].copy()
    ev = ev.sort_values(
        ["parent_person_id", "birth_year", "birth_order"]
    ).reset_index(drop=True)
    return ev
