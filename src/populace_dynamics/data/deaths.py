"""Death records and sex from the cross-year individual file.

The PSID individual file (``ind2023er``) carries, alongside the
per-wave demographic series, two person-constant variables this module
recovers for the differential-mortality component:

* ``ER32000`` -- ``"SEX OF INDIVIDUAL"`` (1 = male, 2 = female, 9 = NA);
* ``ER32050`` -- ``"YEAR OF DEATH"``, the "more precise year of death
  from the 1968-2023 PSID Death File".

Both are cross-year (one value per person), so this reader returns one
row per person, keyed by the same ``person_id`` convention the rest of
the data layer uses (``ER30001 * 1000 + ER30002``; see
:mod:`populace_dynamics.data.panels`).

Year-of-death coding (``ER32050``)
----------------------------------
Verified against ``IND2023ER_codebook.pdf`` (2026-07-07); the counts
below are the codebook's own tallies over the 85,536 individual
records:

* **0** -- ``"Inap.: This person is not deceased"`` (77,415; 90.51%).
  This is the alive-or-not-known-dead sentinel: 0 does NOT mean "died
  in year 0", it means no death is recorded for the person.
* **1967-2023** -- an exact single calendar year of death (7,816;
  9.14%). These are the clean, unambiguous death years.
* **four-digit range codes** -- when only a range of possible death
  years was reported, the code packs it as ``AABB`` where ``AA`` is the
  last two digits of the first possible year and ``BB`` the last two of
  the last possible year (codebook: "the first two digits represent the
  first possible year of death, and the last two digits represent the
  last possible year"). Examples: ``709`` -> ``0709`` -> 2007-2009;
  ``6768`` -> 1967-1968; ``103`` -> 2001-2003; ``2123`` -> 2021-2023.
  Every range code falls OUTSIDE [1967, 2023] numerically, so it never
  collides with an exact year (see :func:`decode_death_code`). ~300
  people carry range codes.
* **9999** -- ``"NA year of death"`` (12): a death is known but the
  year is not.

A two-digit year ``yy`` maps to ``1900 + yy`` when ``yy >= 30`` and
``2000 + yy`` otherwise (:data:`_TWO_DIGIT_YEAR_PIVOT`), the same pivot
:func:`populace_dynamics.data.panels.label_year` uses; it separates the
PSID death years cleanly (67-99 -> 1967-1999, 00-23 -> 2000-2023).

There is **no month-of-death variable** in ``ind2023er`` (the death
file records year, or a year range, only) -- a search of the setup
file's label block finds no ``MONTH ... DEATH`` variable. Mortality
timing below the year is therefore not available from this product.

Discipline
----------
Mirrors :mod:`populace_dynamics.data.psid` and
:mod:`populace_dynamics.data.family`: every variable is resolved by its
opaque PSID code and its human-readable label is verified (under
whitespace normalization) against the adjudicated table before the
fixed-width read, so a changed release layout fails loudly rather than
silently reading the wrong column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from populace_dynamics.data import psid

__all__ = [
    "SEX_CODES",
    "DEATH_NOT_DECEASED_CODE",
    "DEATH_NA_YEAR_CODE",
    "EXACT_YEAR_MIN",
    "EXACT_YEAR_MAX",
    "decode_death_code",
    "read_death_records",
]

#: The ID variables (1968 interview number, person number) and the two
#: person-constant variables this reader needs, each paired with its
#: exact expected label for verification against the setup file.
_VARS: dict[str, str] = {
    "ER30001": "1968 INTERVIEW NUMBER",
    "ER30002": "PERSON NUMBER 68",
    "ER32000": "SEX OF INDIVIDUAL",
    "ER32050": "YEAR OF DEATH",
}

_ID_1968_INTERVIEW = "ER30001"
_ID_PERSON_NUMBER = "ER30002"
_SEX_VAR = "ER32000"
_DEATH_YEAR_VAR = "ER32050"

#: ER32000 sex codes (verified against the codebook, 2026-07-07:
#: 1 = Male 49.55%, 2 = Female 50.45%, 9 = NA one record).
SEX_CODES: dict[int, str] = {1: "male", 2: "female", 9: "na"}

#: ER32050 sentinels and the exact-year window (see the module
#: docstring). 0 = not deceased; 9999 = death known, year NA; a real
#: single year of death lies in [1967, 2023].
DEATH_NOT_DECEASED_CODE = 0
DEATH_NA_YEAR_CODE = 9999
EXACT_YEAR_MIN = 1967
EXACT_YEAR_MAX = 2023

#: Two-digit-year pivot (same as panels.label_year): >= 30 -> 19xx,
#: else 20xx. Separates PSID death years 1967-2023 without collision.
_TWO_DIGIT_YEAR_PIVOT = 30


def _two_digit_year(yy: int) -> int:
    """Expand a two-digit year with the PSID pivot (>=30 -> 19xx)."""
    return 1900 + yy if yy >= _TWO_DIGIT_YEAR_PIVOT else 2000 + yy


def decode_death_code(
    code: int,
) -> tuple[str, int | None, int | None, int | None]:
    """Decode a raw ``ER32050`` value into ``(status, year, lo, hi)``.

    Returns a status string and, where defined, the exact death year
    and the [lo, hi] bounds of the possible-death-year window:

    * ``code == 0`` -> ``("not_deceased", None, None, None)``.
    * ``code == 9999`` -> ``("na_dk", None, None, None)`` -- death
      known, year not ascertained.
    * ``1967 <= code <= 2023`` -> ``("exact", year, year, year)``.
    * otherwise, decode as a range ``AABB`` (first two digits = last
      two of the first possible year, last two = last two of the last
      possible year); if the decoded ``lo <= hi`` ->
      ``("range", None, lo, hi)`` (no single year), else
      ``("unknown", None, None, None)``.

    Checking the exact-year window BEFORE the range decode is what keeps
    a real year like 2019 from being misread as a "20-19" range: every
    range code is numerically outside [1967, 2023].
    """
    code = int(code)
    if code == DEATH_NOT_DECEASED_CODE:
        return ("not_deceased", None, None, None)
    if code == DEATH_NA_YEAR_CODE:
        return ("na_dk", None, None, None)
    if EXACT_YEAR_MIN <= code <= EXACT_YEAR_MAX:
        return ("exact", code, code, code)
    digits = f"{code:04d}"
    lo = _two_digit_year(int(digits[:2]))
    hi = _two_digit_year(int(digits[2:]))
    if lo <= hi:
        return ("range", None, lo, hi)
    return ("unknown", None, None, None)


def _verify_labels(labels: dict[str, str]) -> None:
    """Verify each needed variable's label matches the adjudicated one.

    Whitespace-normalized comparison (PSID pads labels with variable
    spacing), mirroring :func:`populace_dynamics.data.family._verified`.
    A mismatch means the release layout changed and the ``_VARS`` table
    must be re-verified against the setup file.
    """
    for var, expected in _VARS.items():
        actual = " ".join(labels.get(var, "").split())
        wanted = " ".join(expected.split())
        if actual != wanted:
            raise ValueError(
                f"Variable {var} label {actual!r} does not match the "
                f"adjudicated label {wanted!r}. The IND2023ER release "
                "layout may have changed; re-verify deaths._VARS."
            )


def read_death_records(
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read per-person sex and year-of-death from ``ind2023er``.

    Every variable is resolved from the file's own labels and verified
    (:func:`_verify_labels`) before the fixed-width read.

    Returns one row per individual record with columns:

    * ``person_id`` -- ``ER30001 * 1000 + ER30002`` (the panels.py key);
    * ``sex_code`` / ``sex`` -- raw ``ER32000`` and its label
      (``"male"``/``"female"``/``"na"``);
    * ``death_code`` -- raw ``ER32050``;
    * ``death_status`` -- one of ``"not_deceased"``, ``"exact"``,
      ``"range"``, ``"na_dk"``, ``"unknown"`` (see
      :func:`decode_death_code`);
    * ``death_year`` -- the exact year of death for ``"exact"`` rows,
      else ``<NA>`` (nullable ``Int64``);
    * ``death_year_lo`` / ``death_year_hi`` -- the possible-death-year
      bounds (equal to ``death_year`` for ``"exact"``, the range bounds
      for ``"range"``, ``<NA>`` otherwise).

    A person with ``death_status == "not_deceased"`` (``death_code``
    0) is alive or not known to have died; that is the only "alive"
    reading -- there is no separate alive flag in this product.
    """
    sps_path = (
        psid._resolve_data_dir(data_dir)
        / psid.PRODUCTS["ind2023er"][0]
        / psid.PRODUCTS["ind2023er"][2]
    )
    labels = psid.parse_sps_labels(sps_path)
    _verify_labels(labels)

    raw = psid.read_psid(
        "ind2023er",
        columns=[
            _ID_1968_INTERVIEW,
            _ID_PERSON_NUMBER,
            _SEX_VAR,
            _DEATH_YEAR_VAR,
        ],
        data_dir=data_dir,
        nrows=nrows,
    )

    person_id = raw[_ID_1968_INTERVIEW].astype("int64") * 1000 + raw[
        _ID_PERSON_NUMBER
    ].astype("int64")
    sex_code = raw[_SEX_VAR].astype("int64")
    death_code = raw[_DEATH_YEAR_VAR].astype("int64")

    decoded = [decode_death_code(c) for c in death_code]
    status = [d[0] for d in decoded]
    year = [d[1] for d in decoded]
    lo = [d[2] for d in decoded]
    hi = [d[3] for d in decoded]

    frame = pd.DataFrame(
        {
            "person_id": person_id,
            "sex_code": sex_code,
            "sex": sex_code.map(SEX_CODES).astype("string"),
            "death_code": death_code,
            "death_status": pd.array(status, dtype="string"),
            "death_year": pd.array(year, dtype="Int64"),
            "death_year_lo": pd.array(lo, dtype="Int64"),
            "death_year_hi": pd.array(hi, dtype="Int64"),
        }
    )
    return frame.sort_values("person_id").reset_index(drop=True)
