"""Head and spouse labor income from the per-wave family files.

The cross-year individual file carries no uniform individual labor
income, so real earnings histories come from the family files: each
wave reports the family's head/reference-person and wife/spouse labor
income for the *prior* calendar year, and the individual file's
per-wave interview number, relationship, and sequence attach those
amounts to persons.

Scope: waves 1994-2023 (income reference years 1993-2022), where the
labels resolve uniquely under one pattern family (verified against
every staged file on 2026-07-05):

* interview: ``"<wave> INTERVIEW #"`` (1994-1997) or
  ``"<wave> FAMILY INTERVIEW (ID) NUMBER"`` (1999+);
* head labor income: ``"LABOR INCOME OF HEAD-<yyyy>"``,
  ``"LABOR INCOME-HEAD"`` (1997/1999), or
  ``"LABOR INCOME OF REF PERSON-<yyyy>"`` (2017+);
* spouse labor income: the WIFE/SPOUSE forms of the same.

Waves 1968-1993 use era-specific abbreviations (``"HDS LABOR
INCOME"``, ``"HEAD LABOR Y"``, ``"WIFE 84 LABOR/WAGE"``) and are a
documented extension, not silently included.

Interviews are annual through 1997 and biennial from 1999. The
usable panel starts at reference year 1968: the merge requires the
individual file's sequence numbers, which begin in 1969, so the 1968
wave's 1967 income is not attachable to persons. Around the
1992/1993 reference-year seam the head series shows the expected
concept dip (about 7 percent in the raw median), from the pre-1994
totals including farm/business labor parts that the ER era carries
separately.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from populace_dynamics.data import panels, psid

__all__ = [
    "FAMILY_WAVES",
    "WAVES_WITHOUT_ACC",
    "read_family_labor",
    "family_earnings_panel",
]

#: Waves whose labels the loader resolves; annual to 1997, biennial after.
FAMILY_WAVES: tuple[int, ...] = (
    *range(1968, 1998),
    *range(1999, 2024, 2),
)

#: Pre-1994 waves use era-specific variable names and labels; this
#: table is the adjudicated map (wave -> interview, head labor, wife
#: labor variables with their exact labels), built by reading every
#: staged file's own label block (2026-07-05). Labels verify at read
#: time under whitespace normalization. wife_concept flags 1976-1978,
#: where no edited wife labor-income total exists and the wife series
#: is her wage item only — a documented series limitation, not a
#: silent substitution. Note the 1992->1993 seam: pre-1994 head
#: totals include the labor part of farm and business income, while
#: the ER-era series carries those parts in separate variables, so
#: windows straddling reference years 1992/1993 mix concepts.
_PRE94: dict[int, dict[str, tuple[str, str] | str]] = {
    1968: {
        "interview": ("V2", "INTERVIEW NUMBER 68"),
        "head": ("V74", "HDS LABOR INCOME"),
        "wife": ("V75", "WIFE LBR INCOME"),
        "wife_concept": "total",
    },
    1969: {
        "interview": ("V442", "1969 INT NUMBER"),
        "head": ("V514", "LABOR INC HEAD"),
        "wife": ("V516", "LABOR INC WIFE"),
        "wife_concept": "total",
    },
    1970: {
        "interview": ("V1102", "1970 INT #"),
        "head": ("V1196", "LABOR INC HEAD"),
        "wife": ("V1198", "LABOR INC WIFE"),
        "wife_concept": "total",
    },
    1971: {
        "interview": ("V1802", "71 ID NO."),
        "head": ("V1897", "LABOR INC HEAD"),
        "wife": ("V1899", "LABOR INC WIFE"),
        "wife_concept": "total",
    },
    1972: {
        "interview": ("V2402", "1972 INT #"),
        "head": ("V2498", "LABOR INC HEAD"),
        "wife": ("V2500", "LABOR INC WIFE"),
        "wife_concept": "total",
    },
    1973: {
        "interview": ("V3002", "1973 INT #"),
        "head": ("V3051", "HDS TOT LABOR Y"),
        "wife": ("V3053", "WFS LABOR INC"),
        "wife_concept": "total",
    },
    1974: {
        "interview": ("V3402", "1974 ID NUMBER"),
        "head": ("V3463", "TOT LABOR INC-HD"),
        "wife": ("V3465", "TOT LABOR INC-WF"),
        "wife_concept": "total",
    },
    1975: {
        "interview": ("V3802", "1975 INT #"),
        "head": ("V3863", "HEAD LABOR Y"),
        "wife": ("V3865", "WIFE LABOR Y"),
        "wife_concept": "total",
    },
    1976: {
        "interview": ("V4302", "1976 ID NUMBER"),
        "head": ("V5031", "HEAD TOTAL LABOR Y"),
        "wife": ("V4379", "WIFES ANNUAL WAGE H25"),
        "wife_concept": "wages_only",
    },
    1977: {
        "interview": ("V5202", "1977 ID"),
        "head": ("V5627", "TOT 1976 LABOR INCM HEAD"),
        "wife": ("V5289", "WIFE 1976 WAGES"),
        "wife_concept": "wages_only",
    },
    1978: {
        "interview": ("V5702", "1978 ID"),
        "head": ("V6174", "TOT 1977 HEAD LABOR Y"),
        "wife": ("V5788", "WIFE 1977 WAGE"),
        "wife_concept": "wages_only",
    },
    1979: {
        "interview": ("V6302", "1979 ID"),
        "head": ("V6767", "TOT 1978 HEAD LABOR Y"),
        "wife": ("V6398", "WIFE 1978 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1980: {
        "interview": ("V6902", "1980 INTERVIEW NUMBER"),
        "head": ("V7413", "TOT HD LABOR $ Y 79"),
        "wife": ("V6988", "WIFE 1979 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1981: {
        "interview": ("V7502", "1981 INTERVIEW NUMBER"),
        "head": ("V8066", "TOT HD LABOR $ $ Y 80"),
        "wife": ("V7580", "WIFE 1980 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1982: {
        "interview": ("V8202", "1982 INTERVIEW NUMBER"),
        "head": ("V8690", "TOT HD LABOR $ $ Y 81"),
        "wife": ("V8273", "WIFE 1981 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1983: {
        "interview": ("V8802", "1983 INTERVIEW NUMBER"),
        "head": ("V9376", "TOTAL HEAD LABOR Y 82"),
        "wife": ("V8881", "WIFE 1982 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1984: {
        "interview": ("V10002", "1984 INTERVIEW NUMBER"),
        "head": ("V11023", "TOTAL HEAD LABOR Y 83"),
        "wife": ("V10263", "WIFE 1983 LABOR/Y"),
        "wife_concept": "total",
    },
    1985: {
        "interview": ("V11102", "1985 INTERVIEW NUMBER"),
        "head": ("V12372", "TOTAL HEAD LABOR Y 84"),
        "wife": ("V11404", "WIFE 84 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1986: {
        "interview": ("V12502", "1986 INTERVIEW NUMBER"),
        "head": ("V13624", "TOTAL HEAD LABOR Y 85"),
        "wife": ("V12803", "WIFE 85 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1987: {
        "interview": ("V13702", "1987 INTERVIEW NUMBER"),
        "head": ("V14671", "TOTAL HEAD LABOR Y 86"),
        "wife": ("V13905", "WIFE 86 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1988: {
        "interview": ("V14802", "1988 INTERVIEW NUMBER"),
        "head": ("V16145", "TOTAL HEAD LABOR Y 87"),
        "wife": ("V14920", "WIFE 87 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1989: {
        "interview": ("V16302", "1989 INTERVIEW NUMBER"),
        "head": ("V17534", "TOTAL HEAD LABOR Y 88"),
        "wife": ("V16420", "WIFE 88 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1990: {
        # "INTERVEW" is the source file's own typo.
        "interview": ("V17702", "1990 INTERVEW NUMBER"),
        "head": ("V18878", "TOTAL HEAD LABOR Y 89"),
        "wife": ("V17836", "WIFE 89 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1991: {
        "interview": ("V19002", "1991 INTERVIEW NUMBER"),
        "head": ("V20178", "TOTAL HEAD LABOR Y 90"),
        "wife": ("V19136", "WIFE 90 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1992: {
        "interview": ("V20302", "1992 INTERVIEW NUMBER"),
        "head": ("V21484", "TOTAL HEAD LABOR Y 91"),
        "wife": ("V20436", "WIFE 91 LABOR/WAGE"),
        "wife_concept": "total",
    },
    1993: {
        "interview": ("V21602", "1993 INTERVIEW NUMBER"),
        "head": ("V23323", "HD 1992 TOTAL LABOR INCOME"),
        "wife": ("V23324", "WF 1992 TOTAL LABOR INCOME"),
        "wife_concept": "total",
    },
}

#: Waves the loader resolves that carry NO labor-income accuracy
#: variable for either role, so :func:`family_earnings_panel` emits
#: ``earnings_acc = 0`` there. Reading a family setup file's whole label
#: block found no ``ACC ...`` variable on head or wife labor income in
#: the two earliest waves (verified 2026-07-05). ``earnings_acc == 0``
#: therefore means "PSID did not flag this observation as
#: assigned/imputed OR this wave carries no labor-income accuracy flag";
#: this constant enumerates the second case (see also the per-role
#: ``None`` gaps in :data:`_ACC_COMPONENTS`, e.g. 1976 head).
WAVES_WITHOUT_ACC: tuple[int, ...] = (1968, 1969)

#: Per-wave, per-role labor-income accuracy variables, adjudicated wave
#: by wave from each family setup file's own label block (the ``_PRE94``
#: discipline, extended to accuracy). Each role maps to a tuple of
#: ``(variable, exact_label)`` components; ``earnings_acc`` for a
#: person in that role/wave is the MAXIMUM of these components' codes
#: (PSID accuracy/assignment codes: 0 = not assigned, 1-9 = assigned or
#: edited, the digit encoding the method). A role absent from a wave's
#: entry has no labor-income accuracy variable that wave and scores 0.
#: Only wage-and-salary and other/misc labor-income components enter;
#: family-level business, farm, and asset accuracy flags are excluded
#: by construction, so ``earnings_acc`` flags exactly the labor income
#: the panel carries. Every label is verified at read under whitespace
#: normalization, so a changed release layout fails loudly.
#:
#: Adjudications and era boundaries (all lined up against the labor
#: income total :func:`read_family_labor` reads for the same role):
#:
#: * **1970-1975** carry a single direct accuracy flag on each role's
#:   labor-income total (``ACC LABOR Y HEAD`` / ``ACC LABOR Y WIFE`` and
#:   their era spellings); used directly.
#: * **1976** carries an accuracy flag only on the wife's wage item
#:   (``ACC WIFES ANN WAG``, matching her ``wages_only`` income series)
#:   and none on the head's total labor income, so head scores 0 that
#:   wave (a documented per-role gap, not a silent zero).
#: * **1977-1978** split the head total into a wage flag
#:   (``ACC HEAD <yyyy> WAGES``) and a non-wage flag
#:   (``ACC HD LABOR Y EXCL WAGE``); ``earnings_acc`` is their max. The
#:   wife carries a wage flag only (``wages_only`` income series).
#: * **1979-1992** keep the head wage + non-wage split (the non-wage
#:   flag is spelled ``EXCL``/``EX``/``EXC WAGES`` across years) and
#:   give the wife a single direct total flag (``ACC WF <yy>
#:   LABOR/WAGE``). Note 1982's wife flag label reads ``ACC WF 80
#:   LABOR/WAGE`` in the source (a PSID label typo carried into the
#:   1982 file; the variable is the 1981 wife labor accuracy).
#: * **1993** itemizes the head labor total into wages, bonus,
#:   overtime, tips, commission, and extra-job wages, with no single
#:   "misc labor" flag; ``earnings_acc`` is the max over those six.
#:   Because the 1993 head total also folds in the labor part of
#:   farm/business income (the 1992/1993 concept seam the module
#:   docstring notes) while these six cover only the wage-type parts,
#:   1993 head accuracy under-covers that farm/business labor slice by
#:   design — the "no business/farm flags" rule. The wife's flag is her
#:   direct total excluding business/farm (``ACC WF 1992 LABOR INCOME
#:   EXCL BUS/FARM``), matching her total-labor income concept.
#: * **1994-1996, 2001, 2003, 2005-2015** carry the head as a
#:   wage-and-salary flag plus a misc-labor flag on the SAME
#:   reference-year total the income variable reads (``ACC WAGES AND
#:   SALARIES OF HEAD-<yyyy>`` + ``ACC MISC LABOR INCOME OF
#:   HEAD-<yyyy>``, or the ``... LAST YEAR`` wording in 2003);
#:   ``earnings_acc`` is their max. These waves also carry a distinct
#:   *current-interview* ``ACCURACY OF WAGES/SALARY-HEAD`` bracket flag
#:   (0/1 only) on a different question; it is deliberately NOT used,
#:   because the reference-year ``ACC ... OF HEAD-<yyyy>`` family is the
#:   assignment flag on the edited prior-year total the panel carries
#:   (verified on 2013: the reference-year family carries the full
#:   0/1/2/3/5 edit-code range, the current bracket flag only 0/1).
#: * **1997, 1999** read the SHORT-form ``LABOR INCOME-HEAD`` /
#:   ``LABOR INCOME-WIFE`` totals, so they pair with the SHORT-form
#:   direct accuracy flags ``ACC LABOR INCOME-HD`` / ``ACC LABOR
#:   INCOME-WF`` (clean 0-3 codes), NOT the ``-<yyyy>`` wage/misc family
#:   those two waves also carry (which belongs to the detailed
#:   prior-year breakdown and shows a distinct code-9 scheme).
#: * **2015-2023** give both roles the wage + misc split
#:   (``... OF SPOUSE-<yyyy>`` for the spouse, ``HEAD``/``RP`` for the
#:   reference person); ``earnings_acc`` is the per-role max. Before
#:   2015 the spouse carries a single direct total flag (``ACC LABOR
#:   INCOME OF WIFE-<yyyy>``), used directly.
_ACC_COMPONENTS: dict[int, dict[str, tuple[tuple[str, str], ...]]] = {
    1970: {
        "head": (("V1197", "ACC LABOR Y HEAD"),),
        "spouse": (("V1199", "ACC LABOR Y WIFE"),),
    },
    1971: {
        "head": (("V1898", "ACC LABOR Y HEAD"),),
        "spouse": (("V1900", "ACC LABOR Y WIFE"),),
    },
    1972: {
        "head": (("V2499", "ACC LABOR Y HEAD"),),
        "spouse": (("V2501", "ACC LABOR Y WIFE"),),
    },
    1973: {
        "head": (("V3052", "ACC HDS LABOR INC"),),
        "spouse": (("V3054", "ACC WFS LABOR INC"),),
    },
    1974: {
        "head": (("V3464", "ACC LABOR INC-HD"),),
        "spouse": (("V3466", "ACC LABOR INC-WF"),),
    },
    1975: {
        "head": (("V3864", "ACC HD LABOR Y"),),
        "spouse": (("V3866", "ACC WIFE LABOR Y"),),
    },
    1976: {
        # Head total carries no accuracy flag this wave; wife wage only.
        "spouse": (("V4380", "ACC WIFES ANN WAG"),),
    },
    1977: {
        "head": (
            ("V5284", "ACC HEAD 1976 WAGES"),
            ("V5288", "ACC HD LABOR Y EXCL WAGE"),
        ),
        "spouse": (("V5290", "ACC WIFE 1976 WAGES"),),
    },
    1978: {
        "head": (
            ("V5783", "ACC HEAD 1977 WAGES"),
            ("V5787", "ACC HD LABOR Y EXCL WAGE"),
        ),
        "spouse": (("V5789", "ACC WIFE 1977 WAGE"),),
    },
    1979: {
        "head": (
            ("V6392", "ACC HEAD 1978 WAGES"),
            ("V6397", "ACC HD LABOR Y EXCL WAGE"),
        ),
        "spouse": (("V6399", "ACC WF 78 LABOR/WAGE"),),
    },
    1980: {
        "head": (
            ("V6982", "ACC HEAD 1979 WAGES"),
            ("V6987", "ACC HD LABOR Y EXCL WAGE"),
        ),
        "spouse": (("V6989", "ACC WF 79 LABOR/WAGE"),),
    },
    1981: {
        "head": (
            ("V7574", "ACC HEAD 1980 WAGES"),
            ("V7579", "ACC HD LABOR Y EX WAGES"),
        ),
        "spouse": (("V7581", "ACC WF 80 LABOR/WAGE"),),
    },
    1982: {
        "head": (
            ("V8266", "ACC HEAD 1981 WAGES"),
            ("V8271", "ACC HD LABOR Y EX WAGES"),
        ),
        # Source label reads "WF 80" (PSID typo); it is the 1981 wife.
        "spouse": (("V8274", "ACC WF 80 LABOR/WAGE"),),
    },
    1983: {
        "head": (
            ("V8874", "ACC HEAD 1982 WAGES"),
            ("V8879", "ACC HD LABOR Y EX WAGES"),
        ),
        "spouse": (("V8882", "ACC WF 82 LABOR/WAGE"),),
    },
    1984: {
        "head": (
            ("V10257", "ACC HEAD 1983 WAGES"),
            ("V10262", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V10264", "ACC WF 1983 LABOR/Y"),),
    },
    1985: {
        "head": (
            ("V11398", "ACC HEAD 84 WAGES"),
            ("V11403", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V11405", "ACC WF 84 LABOR/WAGE"),),
    },
    1986: {
        "head": (
            ("V12797", "ACC HEAD 85 WAGES"),
            ("V12802", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V12804", "ACC WF 85 LABOR/WAGE"),),
    },
    1987: {
        "head": (
            ("V13899", "ACC HEAD 86 WAGES"),
            ("V13904", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V13906", "ACC WF 86 LABOR/WAGE"),),
    },
    1988: {
        "head": (
            ("V14914", "ACC HEAD 87 WAGES"),
            ("V14919", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V14921", "ACC WF 87 LABOR/WAGE"),),
    },
    1989: {
        "head": (
            ("V16414", "ACC HEAD 88 WAGES"),
            ("V16419", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V16421", "ACC WF 88 LABOR/WAGE"),),
    },
    1990: {
        "head": (
            ("V17830", "ACC HEAD 89 WAGES"),
            ("V17835", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V17837", "ACC WF 89 LABOR/WAGE"),),
    },
    1991: {
        "head": (
            ("V19130", "ACC HEAD 90 WAGES"),
            ("V19135", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V19137", "ACC WF 90 LABOR/WAGE"),),
    },
    1992: {
        "head": (
            ("V20430", "ACC HEAD 91 WAGES"),
            ("V20435", "ACC HD LABOR Y EXC WAGES"),
        ),
        "spouse": (("V20437", "ACC WF 91 LABOR/WAGE"),),
    },
    1993: {
        "head": (
            ("V21740", "ACC HD 1992 WAGES"),
            ("V21742", "ACC HD 1992 BONUS INCOME"),
            ("V21744", "ACC HD 1992 OVERTIME INCOME"),
            ("V21746", "ACC HD 1992 TIPS INCOME"),
            ("V21748", "ACC HD 1992 COMMISSION INCOME"),
            ("V21802", "ACC HD 1992 WAGES FROM EXTRA JOB"),
        ),
        "spouse": (("V21808", "ACC WF 1992 LABOR INCOME EXCL BUS/FARM"),),
    },
    1994: {
        "head": (
            ("ER4123", "ACC WAGES AND SALARIES OF HEAD-1993"),
            ("ER4137", "ACC MISC LABOR INCOME OF HEAD-1993"),
        ),
        "spouse": (("ER4145", "ACC LABOR INCOME OF WIFE-1993"),),
    },
    1995: {
        "head": (
            ("ER6963", "ACC WAGES AND SALARIES OF HEAD-1994"),
            ("ER6977", "ACC MISC LABOR INCOME OF HEAD-1994"),
        ),
        "spouse": (("ER6985", "ACC LABOR INCOME OF WIFE-1994"),),
    },
    1996: {
        "head": (
            ("ER9214", "ACC WAGES AND SALARIES OF HEAD-1995"),
            ("ER9228", "ACC MISC LABOR INCOME OF HEAD-1995"),
        ),
        "spouse": (("ER9236", "ACC LABOR INCOME OF WIFE-1995"),),
    },
    1997: {
        # Short-form LABOR INCOME-HEAD/-WIFE totals -> short-form direct
        # accuracy flags (not the -1996 wage/misc breakdown family).
        "head": (("ER12081", "ACC LABOR INCOME-HD"),),
        "spouse": (("ER12083", "ACC LABOR INCOME-WF"),),
    },
    1999: {
        "head": (("ER16464", "ACC LABOR INCOME-HD"),),
        "spouse": (("ER16466", "ACC LABOR INCOME-WF"),),
    },
    2001: {
        "head": (
            ("ER20426", "ACC WAGES AND SALARIES OF HEAD-2000"),
            ("ER20440", "ACC MISC LABOR INCOME OF HEAD-2000"),
        ),
        "spouse": (("ER20448", "ACC LABOR INCOME OF WIFE-2000"),),
    },
    2003: {
        "head": (
            ("ER24118", "ACC WAGES AND SALARIES OF HEAD LAST YEAR"),
            ("ER24134", "ACC MISC LABOR INCOME OF HEAD LAST YEAR"),
        ),
        "spouse": (("ER24136", "ACC LABOR INCOME OF WIFE LAST YEAR"),),
    },
    2005: {
        "head": (
            ("ER27914", "ACC WAGES AND SALARIES OF HEAD-2004"),
            ("ER27930", "ACC MISC LABOR INCOME OF HEAD-2004"),
        ),
        "spouse": (("ER27944", "ACC LABOR INCOME OF WIFE-2004"),),
    },
    2007: {
        "head": (
            ("ER40904", "ACC WAGES AND SALARIES OF HEAD-2006"),
            ("ER40920", "ACC MISC LABOR INCOME OF HEAD-2006"),
        ),
        "spouse": (("ER40934", "ACC LABOR INCOME OF WIFE-2006"),),
    },
    2009: {
        "head": (
            ("ER46812", "ACC WAGES AND SALARIES OF HEAD-2008"),
            ("ER46828", "ACC MISC LABOR INCOME OF HEAD-2008"),
        ),
        "spouse": (("ER46842", "ACC LABOR INCOME OF WIFE-2008"),),
    },
    2011: {
        "head": (
            ("ER52220", "ACC WAGES AND SALARIES OF HEAD-2010"),
            ("ER52236", "ACC MISC LABOR INCOME OF HEAD-2010"),
        ),
        "spouse": (("ER52250", "ACC LABOR INCOME OF WIFE-2010"),),
    },
    2013: {
        "head": (
            ("ER58021", "ACC WAGES AND SALARIES OF HEAD-2012"),
            ("ER58037", "ACC MISC LABOR INCOME OF HEAD-2012"),
        ),
        "spouse": (("ER58051", "ACC LABOR INCOME OF WIFE-2012"),),
    },
    2015: {
        "head": (
            ("ER65201", "ACC WAGES AND SALARIES OF HEAD-2014"),
            ("ER65215", "ACC MISC LABOR INCOME OF HEAD-2014"),
        ),
        "spouse": (
            ("ER65229", "ACC WAGES AND SALARIES OF SPOUSE-2014"),
            ("ER65243", "ACC MISC LABOR INCOME OF SPOUSE-2014"),
        ),
    },
    2017: {
        "head": (
            ("ER71278", "ACC WAGES AND SALARIES OF RP-2016"),
            ("ER71292", "ACC MISC LABOR INCOME OF RP-2016"),
        ),
        "spouse": (
            ("ER71306", "ACC WAGES AND SALARIES OF SPOUSE-2016"),
            ("ER71320", "ACC MISC LABOR INCOME OF SPOUSE-2016"),
        ),
    },
    2019: {
        "head": (
            ("ER77300", "ACC WAGES AND SALARIES OF RP-2018"),
            ("ER77314", "ACC MISC LABOR INCOME OF RP-2018"),
        ),
        "spouse": (
            ("ER77328", "ACC WAGES AND SALARIES OF SPOUSE-2018"),
            ("ER77342", "ACC MISC LABOR INCOME OF SPOUSE-2018"),
        ),
    },
    2021: {
        "head": (
            ("ER81627", "ACC WAGES AND SALARIES OF RP-2020"),
            ("ER81641", "ACC MISC LABOR INCOME OF RP-2020"),
        ),
        "spouse": (
            ("ER81655", "ACC WAGES AND SALARIES OF SPOUSE-2020"),
            ("ER81669", "ACC MISC LABOR INCOME OF SPOUSE-2020"),
        ),
    },
    2023: {
        "head": (
            ("ER85481", "ACC WAGES AND SALARIES OF RP-2022"),
            ("ER85495", "ACC MISC LABOR INCOME OF RP-2022"),
        ),
        "spouse": (
            ("ER85509", "ACC WAGES AND SALARIES OF SPOUSE-2022"),
            ("ER85523", "ACC MISC LABOR INCOME OF SPOUSE-2022"),
        ),
    },
}

#: Accuracy codes are one-digit assignment codes; family labor income
#: sits far below any missing sentinel, so a plain integer max over the
#: role's components is the wave's ``earnings_acc``.
_ACC_DEFAULT = 0

_HEAD_LABOR = r"^LABOR INCOME( OF)?[ -](HEAD|REF PERSON)"
_SPOUSE_LABOR = r"^LABOR INCOME( OF)?[ -](WIFE|SPOUSE)"

#: Individual-file relationship codes attaching family amounts to
#: persons. The coding changed in 1983 (verified empirically on the
#: cross-year file): single digits through 1982 (1 = head,
#: 2 = wife), two digits from 1983 (10 = head/reference person,
#: 20 = legal wife/spouse, 22 = cohabiting partner).
_RELATIONSHIP_ERA_BREAK = 1983
_RELATIONSHIP_HEAD_PRE83 = (1,)
_RELATIONSHIP_SPOUSE_PRE83 = (2,)
_RELATIONSHIP_HEAD = (10,)
_RELATIONSHIP_SPOUSE = (20, 22)


def _relationship_codes(wave: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if wave < _RELATIONSHIP_ERA_BREAK:
        return _RELATIONSHIP_HEAD_PRE83, _RELATIONSHIP_SPOUSE_PRE83
    return _RELATIONSHIP_HEAD, _RELATIONSHIP_SPOUSE


#: Defensive missing sentinel; family labor income is edited and far
#: below it, which the integration tests assert.
_MISSING = 9_999_998


def _family_paths(wave: int, data_dir: Path | None) -> tuple[Path, Path]:
    base = psid._resolve_data_dir(data_dir) / "family" / str(wave)
    if not base.is_dir():
        raise FileNotFoundError(
            f"Family wave directory not found: {base} "
            f"({psid._README_POINTER})"
        )
    sps = sorted(
        p
        for p in base.glob("*.sps")
        if not p.name.lower().endswith("_formats.sps")
    )
    txt = sorted(base.glob("*.txt"))
    if len(sps) != 1 or len(txt) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .sps and one .txt in {base}; "
            f"found {len(sps)} and {len(txt)}."
        )
    return sps[0], txt[0]


def _single(labels: dict[str, str], pattern: str, wave: int, what: str) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    hits = {n: v for n, v in labels.items() if regex.search(v)}
    if len(hits) != 1:
        raise ValueError(
            f"Wave {wave}: pattern for {what} matched {len(hits)} "
            f"variables ({dict(list(hits.items())[:4])}); expected "
            "exactly one."
        )
    return next(iter(hits))


def _verified(
    labels: dict[str, str], var: str, expected: str, wave: int
) -> str:
    """Return ``var`` after checking its label matches the table.

    Comparison normalizes runs of whitespace, because PSID labels pad
    columns with variable spacing. A mismatch means the release
    layout changed and the adjudicated table must be re-verified.
    """
    actual = " ".join(labels.get(var, "").split())
    wanted = " ".join(expected.split())
    if actual != wanted:
        raise ValueError(
            f"Wave {wave}: variable {var} label {actual!r} does not "
            f"match the adjudicated table ({wanted!r}). The release "
            "layout may have changed."
        )
    return var


def _check_reference_year(label: str, wave: int) -> None:
    """Labels that carry a 4-digit year must carry ``wave - 1``."""
    match = re.search(r"(19|20)\d{2}\s*$", label)
    if match and int(match.group(0)) != wave - 1:
        raise ValueError(
            f"Wave {wave}: label {label!r} carries reference year "
            f"{match.group(0)}, expected {wave - 1}. The release "
            "layout may have changed."
        )


def _acc_component_vars(
    labels: dict[str, str], wave: int, role: str
) -> list[str]:
    """Label-verified accuracy-component variables for a wave/role.

    Returns the (possibly empty) list of variable names whose codes are
    maxed into ``earnings_acc`` for ``role`` (``"head"``/``"spouse"``)
    in ``wave``, each verified against :data:`_ACC_COMPONENTS` under the
    same whitespace-normalized label check the income variables use. An
    empty list means the wave carries no labor-income accuracy variable
    for that role (see :data:`WAVES_WITHOUT_ACC` and the per-role gaps
    documented on :data:`_ACC_COMPONENTS`).
    """
    components = _ACC_COMPONENTS.get(wave, {}).get(role, ())
    return [_verified(labels, var, label, wave) for var, label in components]


def read_family_labor(
    wave: int,
    *,
    data_dir: Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read one wave's family-level labor income and accuracy flags.

    Returns a frame with ``interview``, ``head_labor``,
    ``spouse_labor``, and the two per-role accuracy columns
    ``head_acc`` / ``spouse_acc`` for every responding family. Every
    variable is resolved from the file's own labels and reference years
    verified where the label carries one. Each accuracy column is the
    per-row MAXIMUM of that role's label-verified labor-income accuracy
    components (:data:`_ACC_COMPONENTS`); a role with no accuracy
    variable this wave (:data:`WAVES_WITHOUT_ACC`, or the per-role gaps
    like 1976 head) gets ``0``.
    """
    if wave not in FAMILY_WAVES:
        raise ValueError(
            f"Wave {wave} is outside the resolved range "
            f"{FAMILY_WAVES[0]}-{FAMILY_WAVES[-1]}."
        )
    sps_path, txt_path = _family_paths(wave, data_dir)
    labels = psid.parse_sps_labels(sps_path)
    if wave in _PRE94:
        entry = _PRE94[wave]
        interview = _verified(labels, *entry["interview"], wave)
        head = _verified(labels, *entry["head"], wave)
        spouse = _verified(labels, *entry["wife"], wave)
    else:
        interview = _single(
            labels,
            rf"^{wave} (INTERVIEW #|FAMILY INTERVIEW \(ID\) NUMBER)$",
            wave,
            "interview number",
        )
        head = _single(labels, _HEAD_LABOR, wave, "head labor income")
        spouse = _single(labels, _SPOUSE_LABOR, wave, "spouse labor income")
        _check_reference_year(labels[head], wave)
        _check_reference_year(labels[spouse], wave)

    head_acc_vars = _acc_component_vars(labels, wave, "head")
    spouse_acc_vars = _acc_component_vars(labels, wave, "spouse")

    layout = psid.parse_sps_layout(sps_path)
    layout_by_name = layout.set_index("name")
    income_names = [interview, head, spouse]
    acc_names = head_acc_vars + spouse_acc_vars
    colspecs = [
        (
            int(layout_by_name.loc[name, "start"]) - 1,
            int(layout_by_name.loc[name, "end"]),
        )
        for name in (*income_names, *acc_names)
    ]
    raw = pd.read_fwf(
        txt_path,
        colspecs=colspecs,
        names=[*income_names, *acc_names],
        header=None,
        nrows=nrows,
    )
    frame = raw[[interview, head, spouse]].rename(
        columns={
            interview: "interview",
            head: "head_labor",
            spouse: "spouse_labor",
        }
    )
    frame["head_acc"] = _max_acc(raw, head_acc_vars)
    frame["spouse_acc"] = _max_acc(raw, spouse_acc_vars)
    return frame


def _max_acc(raw: pd.DataFrame, acc_vars: list[str]) -> pd.Series:
    """Per-row max of the accuracy-component columns (0 if none)."""
    if not acc_vars:
        return pd.Series(_ACC_DEFAULT, index=raw.index, dtype="int64")
    return raw[acc_vars].max(axis=1).astype("int64")


def family_earnings_panel(
    *,
    waves: tuple[int, ...] | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Person-level labor-income histories from the family files.

    Merges each wave's family head/spouse labor income onto persons
    through the individual file's interview number, keeping persons
    present in a responding family (sequence 1-20) whose relationship
    is head/reference person or wife/spouse/partner. ``period`` is
    the income reference year, ``wave - 1``.

    Columns: ``person_id``, ``period``, ``earnings``, ``role``
    (``"head"``/``"spouse"``), ``age``, ``weight`` (age and weight
    are measured at the collection wave), and ``earnings_acc``.

    ``earnings_acc`` (small int, default 0) is the PSID
    accuracy/assignment code for that person-period's role-specific
    labor income: the MAXIMUM over the role's wave-specific labor-income
    accuracy components (:data:`_ACC_COMPONENTS`). Code 0 means either
    the observation was not flagged as assigned/imputed OR the wave
    carries no labor-income accuracy flag for that role; the two
    flag-carrying cases are enumerated by :data:`WAVES_WITHOUT_ACC`
    (whole waves) and the per-role gaps documented on
    :data:`_ACC_COMPONENTS` (e.g. 1976 head). Positive codes flag an
    assigned/edited value (1-9, the digit encoding the method).
    """
    use_waves = tuple(waves) if waves is not None else FAMILY_WAVES
    demo = panels.demographic_panel(data_dir=data_dir)
    demo = demo[demo.period.isin(use_waves)]

    frames = []
    for wave in use_waves:
        labor = read_family_labor(wave, data_dir=data_dir)
        wave_people = demo[demo.period == wave]
        merged = wave_people.merge(
            labor, left_on="interview", right_on="interview", how="inner"
        )
        head_codes, spouse_codes = _relationship_codes(wave)
        is_head = merged.relationship.isin(head_codes)
        is_spouse = merged.relationship.isin(spouse_codes)
        merged = merged[is_head | is_spouse].copy()
        head_mask = merged.relationship.isin(head_codes)
        merged["earnings"] = merged.head_labor.where(
            head_mask,
            merged.spouse_labor,
        ).astype("float64")
        merged["earnings_acc"] = (
            merged.head_acc.where(head_mask, merged.spouse_acc)
            .fillna(_ACC_DEFAULT)
            .astype("int64")
        )
        merged["role"] = "spouse"
        merged.loc[head_mask, "role"] = "head"
        merged["period"] = wave - 1
        frames.append(
            merged[
                [
                    "person_id",
                    "period",
                    "earnings",
                    "earnings_acc",
                    "role",
                    "age",
                    "weight",
                ]
            ]
        )
    panel = pd.concat(frames, ignore_index=True)
    keep = (panel.earnings < _MISSING) & (panel.weight > 0)
    panel = panel.loc[keep]
    return panel.sort_values(["person_id", "period"]).reset_index(drop=True)
