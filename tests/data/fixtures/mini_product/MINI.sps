
**************************************************************************
   Label           : Mini synthetic fixture for PSID loader tests
   DOI             :
   Rows            : 5
   Columns         : 3
   ASCII File Date : December 2, 2025
*************************************************************************.

FILE HANDLE PSID / NAME = '[PATH]\MINI.TXT' LRECL = 8 .
DATA LIST FILE = PSID FIXED /
      MN1             1 - 1         MN2             2 - 5         MN3             6 - 8
   .
   EXECUTE .
   VARIABLE LABELS
      MN1          "RELEASE NUMBER"
      MN2          "AGE OF INDIVIDUAL                     68"
      MN3          "MONEY INCOME IND                      68"
   .
