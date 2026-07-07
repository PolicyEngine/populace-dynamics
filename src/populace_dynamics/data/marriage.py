"""Per-marriage records from the PSID Marriage History File (mh85_23).

The Marriage History File (PSID packaged product 1121, collection waves
1985-2023) carries one record per marriage per individual, plus one
placeholder record per never-married individual. Its retrospective
marriage timing and dissolution codes drive the gate-2
marriage/divorce/survivorship moments; the person keys join it to the
earnings panel (:mod:`populace_dynamics.data.family`) and the
demographic panel (:mod:`populace_dynamics.data.panels`).

Person identifier
-----------------
Same convention as the rest of the stack (see
:mod:`populace_dynamics.data.panels`): the 1968 family interview number
(``MH2``) with the person number (``MH3``),
``person_id = MH2 * 1000 + MH3``. Person numbers are below 1000, so the
pair packs losslessly and the key is identical to the individual file's
``ER30001 * 1000 + ER30002``, so marriage records join the earnings and
demographic panels directly. The spouse carries the same kind of key
(``MH7``/``MH8``); ``spouse_person_id`` is populated only when the
spouse is a joinable PSID sample-member person (see judgment calls).

Record structure and the never-married placeholder
--------------------------------------------------
``MH18`` (number of marriages) is 0 for never-married individuals, who
get a single placeholder record with the marriage fields set to the
"never married" inapplicable code (``MH9 = MH12 = 99``/``9``). These
rows are kept -- they carry ``last_known_status = "never_married"`` and
are the file's own denominator for ever-married rates -- but flagged
``is_marriage = False`` and left with NA marriage fields.
:func:`marriage_history` therefore returns every file record (a stable
row count), while :func:`marital_trajectories` restricts to actual
marriages.

Documented judgment calls (verified against the staged Release 2 file,
2026-07-07)
-----------------------------------------------------------------------
* **Year missing sentinels.** PSID year fields use ``9998`` = NA/DK and
  ``9999`` = inapplicable (never married, or the marriage has not ended
  for an end/separation year). Both map to NA. Observed: ``MH6`` (birth)
  carries only ``9998``; ``MH11``/``MH14``/``MH16`` carry both. Real
  years span 1884-2008 (birth) and 1905-2023 (marriage events).
* **Month sentinels and season codes.** Month fields use ``98`` = NA/DK
  and ``99`` = inapplicable, both mapped to NA. Codes ``21``-``24`` are
  *seasons* (winter/spring/summer/fall) the PSID accepts when a
  respondent cannot give a month; they are kept as-is (real, if coarse,
  timing), NOT mapped to NA. Callers wanting month-precision only should
  drop months >= 21.
* **Marriage-order sentinels.** ``MH9`` = ``98`` (NA/DK order) and
  ``99`` (inapplicable: never married) map to NA ``marriage_order``;
  actual orders are 1-13. 613 actual-marriage records carry an unknown
  (98) order.
* **Spouse identifier joinability.** ``spouse_person_id`` is
  ``MH7 * 1000 + MH8`` only when the spouse is an enumerable PSID
  sample-member person: ``MH7`` in 1-9308 and ``MH8`` in 1-399.
  Otherwise it is NA. The excluded cases are documented in the codebook
  and confirmed in the data: ``MH8`` in 800-995 = "spouse who has never
  been in any sample family" (a real spouse, but not a PSID person, so
  not joinable); ``MH7``/``MH8`` == 0 = inapplicable (never married);
  ``MH7`` == 9999 / ``MH8`` == 999 = NA whether ever married. Of 65,226
  records, 32,522 carry a joinable spouse id.
* **Sex, status, and last-known-status domains.** Decoded from their
  exact documented code sets and re-checked at read; an unexpected code
  raises rather than passing through. ``how_ended`` (``MH12``) maps
  1->intact, 3->widowhood, 4->divorce, 5->separated, 7->other (a more
  recent marriage was also reported with no evidence of divorce or
  widowhood -- a probable bigamist), 8->unknown (NA/DK), 9->
  never_married (the placeholder). ``MH12`` has no codes 2 or 6.
* **Single harmonized coding.** Unlike the family files, this product is
  a single retrospective file with one coding across all collection
  waves, so no era-varmap is needed; the label check still guards
  against a re-layout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from populace_dynamics.data import psid

__all__ = [
    "PRODUCT",
    "marriage_history",
    "marital_trajectories",
    "marriage_episodes",
    "HOW_ENDED",
    "LAST_KNOWN_STATUS",
]

#: PSID product key for the Marriage History File (see
#: :data:`populace_dynamics.data.psid.PRODUCTS`).
PRODUCT = "mh85_23"

#: Every variable on the file, mapped to its exact documented label.
#: The whole layout is label-verified at read (:func:`psid.verify_labels`)
#: so a shifted or renamed column fails loudly instead of silently
#: mismapping a marriage field.
_VARS: dict[str, str] = {
    "MH1": "RELEASE NUMBER",
    "MH2": "1968 INTERVIEW NUMBER OF INDIVIDUAL",
    "MH3": "PERSON NUMBER OF INDIVIDUAL",
    "MH4": "SEX OF INDIVIDUAL",
    "MH5": "MONTH INDIVIDUAL BORN",
    "MH6": "YEAR INDIVIDUAL BORN",
    "MH7": "1968 INTERVIEW NUMBER OF SPOUSE",
    "MH8": "PERSON NUMBER OF SPOUSE",
    "MH9": "ORDER OF THIS MARRIAGE",
    "MH10": "MONTH MARRIED",
    "MH11": "YEAR MARRIED",
    "MH12": "STATUS OF THIS MARRIAGE",
    "MH13": "MONTH WIDOWED OR DIVORCED",
    "MH14": "YEAR WIDOWED OR DIVORCED",
    "MH15": "MONTH SEPARATED",
    "MH16": "YEAR SEPARATED",
    "MH17": "YEAR MOST RECENTLY REPORTED MARRIAGE",
    "MH18": "NUMBER OF MARRIAGES OF THIS INDIVIDUAL",
    "MH19": "LAST KNOWN MARITAL STATUS",
    "MH20": "NUMBER OF MARRIAGE RECORDS",
}

#: MH4 sex of individual.
_SEX: dict[int, str] = {1: "male", 2: "female"}

#: MH12 status of this marriage -> how it ended.
HOW_ENDED: dict[int, str] = {
    1: "intact",
    3: "widowhood",
    4: "divorce",
    5: "separated",
    7: "other",
    8: "unknown",
    9: "never_married",
}

#: MH19 last known marital status.
LAST_KNOWN_STATUS: dict[int, str] = {
    1: "married",
    2: "never_married",
    3: "widowed",
    4: "divorced",
    5: "separated",
    8: "unknown",
}

#: Year fields: 9998 = NA/DK, 9999 = inapplicable.
_YEAR_NA: tuple[int, ...] = (9998, 9999)
#: Month fields: 98 = NA/DK, 99 = inapplicable (season codes 21-24 kept).
_MONTH_NA: tuple[int, ...] = (98, 99)
#: Marriage-order field: 98 = NA/DK, 99 = inapplicable (never married).
_ORDER_NA: tuple[int, ...] = (98, 99)
#: Count-of-marriages field: 98 = NA/DK.
_COUNT_NA: tuple[int, ...] = (98,)

#: A joinable spouse is a PSID sample-member person: MH7 in this range
#: (a 1968 family interview number) and MH8 a real person number.
_INTERVIEW_RANGE = (1, 9308)
_PERSON_RANGE = (1, 399)


def _na(series: pd.Series, sentinels: tuple[int, ...]) -> pd.Series:
    """Cast a raw integer column to nullable Int64, sentinels -> NA."""
    return series.astype("Int64").mask(series.isin(list(sentinels)))


def _decode(
    series: pd.Series, mapping: dict[int, str], field: str
) -> pd.Series:
    """Map integer codes to labels, raising on any undocumented code."""
    observed = {int(v) for v in pd.unique(series.dropna())}
    unexpected = observed - set(mapping)
    if unexpected:
        raise ValueError(
            f"{field}: undocumented code(s) {sorted(unexpected)}; the "
            f"documented domain is {sorted(mapping)}. The release coding "
            "may have changed."
        )
    return series.map(mapping).astype("string")


def _build(raw: pd.DataFrame) -> pd.DataFrame:
    """Assemble the tidy per-record frame from a raw MH column frame.

    Pure transform over columns ``MH2``..``MH20`` (a
    :func:`psid.read_psid` result), split out so the episode logic and
    dtype/sentinel handling are unit-testable on synthetic rows without
    the staged file.
    """
    mh7 = raw["MH7"].astype("int64")
    mh8 = raw["MH8"].astype("int64")
    joinable = mh8.between(*_PERSON_RANGE) & mh7.between(*_INTERVIEW_RANGE)
    spouse_id = (mh7 * 1000 + mh8).astype("Int64").mask(~joinable)

    out = pd.DataFrame(
        {
            "person_id": (
                raw["MH2"].astype("int64") * 1000 + raw["MH3"].astype("int64")
            ).astype("int64"),
            "sex": _decode(raw["MH4"], _SEX, "MH4 sex"),
            "birth_year": _na(raw["MH6"], _YEAR_NA),
            "birth_month": _na(raw["MH5"], _MONTH_NA),
            "marriage_order": _na(raw["MH9"], _ORDER_NA),
            "spouse_person_id": spouse_id,
            "start_year": _na(raw["MH11"], _YEAR_NA),
            "start_month": _na(raw["MH10"], _MONTH_NA),
            "end_year": _na(raw["MH14"], _YEAR_NA),
            "end_month": _na(raw["MH13"], _MONTH_NA),
            "separation_year": _na(raw["MH16"], _YEAR_NA),
            "separation_month": _na(raw["MH15"], _MONTH_NA),
            "how_ended": _decode(raw["MH12"], HOW_ENDED, "MH12 status"),
            "last_known_status": _decode(
                raw["MH19"], LAST_KNOWN_STATUS, "MH19 last status"
            ),
            "most_recent_report_year": raw["MH17"].astype("Int64"),
            "n_marriages": _na(raw["MH18"], _COUNT_NA),
            "n_records": raw["MH20"].astype("int64"),
            "is_marriage": (raw["MH12"] != 9).to_numpy(),
        }
    )
    return out


def marriage_history(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read the Marriage History File into a tidy per-record frame.

    Returns one row per file record -- one per marriage per individual,
    plus one never-married placeholder per never-married individual
    (``is_marriage = False``). Every column is resolved from the file's
    own labels and verified before the fixed-width read, and every coded
    field (sex, status, last-known status) is decoded from its
    documented domain with an undocumented code raising rather than
    passing through.

    Columns: ``person_id``, ``sex``, ``birth_year``, ``birth_month``,
    ``marriage_order`` (1-13, NA if unknown/inapplicable),
    ``spouse_person_id`` (NA unless a joinable PSID person),
    ``start_year``/``start_month`` (this marriage began),
    ``end_year``/``end_month`` (ended in widowhood or divorce),
    ``separation_year``/``separation_month``, ``how_ended``,
    ``last_known_status``, ``most_recent_report_year``, ``n_marriages``,
    ``n_records``, ``is_marriage``. Year/month/order/count fields are
    nullable ``Int64`` with PSID missing sentinels mapped to NA (see the
    module docstring for the sentinel conventions and season codes).
    """
    sps_path = psid.product_sps_path(PRODUCT, data_dir)
    labels = psid.parse_sps_labels(sps_path)
    psid.verify_labels(labels, _VARS, context=PRODUCT)
    raw = psid.read_psid(
        PRODUCT, columns=list(_VARS), data_dir=data_dir, nrows=nrows
    )
    return _build(raw)


def marriage_episodes(records: pd.DataFrame) -> pd.DataFrame:
    """Actual marriages as per-person ordered episodes with an end year.

    Pure transform over a :func:`marriage_history`-shaped frame (so it
    is unit-testable on synthetic rows): drops never-married
    placeholders, orders each person's marriages, and coalesces a single
    ``episode_end_year`` -- the widowhood/divorce ``end_year`` for those
    dissolutions, the ``separation_year`` for separations, NA for intact
    marriages -- plus ``episode_duration_years`` where both ends are
    known.

    Columns: ``person_id``, ``marriage_order``, ``start_year``,
    ``start_month``, ``episode_end_year``, ``how_ended``,
    ``episode_duration_years``, ``spouse_person_id``,
    ``last_known_status``.
    """
    ep = records[records["is_marriage"]].copy()
    ep = ep.sort_values(["person_id", "marriage_order"]).reset_index(drop=True)
    end_year = ep["end_year"].where(
        ep["how_ended"] != "separated", ep["separation_year"]
    )
    ep["episode_end_year"] = end_year.astype("Int64")
    ep["episode_duration_years"] = (
        ep["episode_end_year"] - ep["start_year"]
    ).astype("Int64")
    return ep[
        [
            "person_id",
            "marriage_order",
            "start_year",
            "start_month",
            "episode_end_year",
            "how_ended",
            "episode_duration_years",
            "spouse_person_id",
            "last_known_status",
        ]
    ].reset_index(drop=True)


def marital_trajectories(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Person-level marital episode timeline (actual marriages only).

    Convenience wrapper: :func:`marriage_episodes` applied to
    :func:`marriage_history`. One row per actual marriage, ordered by
    person and marriage order, with a coalesced episode end year. Never
    married individuals (placeholder records) drop out here; use
    :func:`marriage_history` to see them.
    """
    return marriage_episodes(marriage_history(data_dir=data_dir, nrows=nrows))
