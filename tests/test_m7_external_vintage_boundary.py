"""Boundary tests shared by the three M7 external-reference artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXTERNAL = ROOT / "data" / "external"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from populace_dynamics.engine.refit import (  # noqa: E402
    validate_external_vintage,
)


@pytest.mark.parametrize(
    "filename",
    (
        "ssa_effective_interest_rates_2014.json",
        "ssa_tr_ultimate_assumptions_2014.json",
        "ssa_trust_fund_opening_reserve_2014.json",
    ),
)
def test__m7_external_vintage__admits_2014_and_rejects_2015(filename):
    doc = json.loads((EXTERNAL / filename).read_text())
    name = doc["schema_version"]
    admitted = validate_external_vintage(
        name, doc["vintage_year"], boundary_year=2014
    )
    assert admitted.vintage_year == 2014

    with pytest.raises(ValueError, match=r"vintage 2015 is post-T\* \(2014\)"):
        validate_external_vintage(name, 2015, boundary_year=2014)
