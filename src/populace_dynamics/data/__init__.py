"""Data loaders for populace-dynamics.

Currently covers PSID packaged-data products (fixed-width text plus
SPSS setup files) via :mod:`populace_dynamics.data.psid`.
"""

from __future__ import annotations

from populace_dynamics.data.psid import (
    PRODUCTS,
    find_variables,
    parse_sps_labels,
    parse_sps_layout,
    read_psid,
)

__all__ = [
    "PRODUCTS",
    "find_variables",
    "parse_sps_labels",
    "parse_sps_layout",
    "read_psid",
]
