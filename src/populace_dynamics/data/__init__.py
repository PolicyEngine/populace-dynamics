"""Data loaders for populace-dynamics.

Covers the PSID packaged-data products (fixed-width text plus SPSS setup
files) via :mod:`populace_dynamics.data.psid`: the earnings panel
(:mod:`~populace_dynamics.data.family`,
:mod:`~populace_dynamics.data.panels`) and the demographic history
readers -- marriage (:mod:`~populace_dynamics.data.marriage`), childbirth
and adoption (:mod:`~populace_dynamics.data.births`), and the family
relationship matrix (:mod:`~populace_dynamics.data.relmap`).

Also exposes the label-verified CPS ASEC firm-size (NOEMP) reader
(:mod:`~populace_dynamics.data.asec_firm_size`).
"""

from __future__ import annotations

from populace_dynamics.data.asec_firm_size import (
    ASEC_FIRM_SIZE_YEARS,
    NOEMP_BANDS_2011_2018,
    NOEMP_BANDS_2019_PLUS,
    band_regime,
    firm_size_tabulation,
    noemp_band_map,
    read_asec_firm_size,
)
from populace_dynamics.data.births import birth_events, birth_history
from populace_dynamics.data.deaths import (
    decode_death_code,
    read_death_records,
)
from populace_dynamics.data.family import (
    FAMILY_WAVES,
    family_earnings_panel,
    read_family_labor,
)
from populace_dynamics.data.marriage import (
    marital_trajectories,
    marriage_episodes,
    marriage_history,
)
from populace_dynamics.data.panels import (
    DEMOGRAPHIC_CONCEPTS,
    demographic_panel,
    ind_person_period,
    label_year,
    wave_variables,
)
from populace_dynamics.data.psid import (
    PRODUCTS,
    find_variables,
    parse_sps_labels,
    parse_sps_layout,
    product_sps_path,
    read_psid,
    verify_labels,
)
from populace_dynamics.data.relmap import (
    rel_to_reference_person,
    relationship_map,
)

__all__ = [
    "ASEC_FIRM_SIZE_YEARS",
    "NOEMP_BANDS_2011_2018",
    "NOEMP_BANDS_2019_PLUS",
    "band_regime",
    "firm_size_tabulation",
    "noemp_band_map",
    "read_asec_firm_size",
    "decode_death_code",
    "read_death_records",
    "FAMILY_WAVES",
    "family_earnings_panel",
    "read_family_labor",
    "DEMOGRAPHIC_CONCEPTS",
    "demographic_panel",
    "ind_person_period",
    "label_year",
    "wave_variables",
    "PRODUCTS",
    "find_variables",
    "parse_sps_labels",
    "parse_sps_layout",
    "product_sps_path",
    "read_psid",
    "verify_labels",
    "marriage_history",
    "marital_trajectories",
    "marriage_episodes",
    "birth_history",
    "birth_events",
    "relationship_map",
    "rel_to_reference_person",
]
