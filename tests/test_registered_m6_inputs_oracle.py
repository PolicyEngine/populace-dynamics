"""policyengine-us-dependent checks for the M6 factory (§2.8.10.2).

Skipped unless policyengine-us **1.752.2** is importable in the running env
(the certified vintage). When it is -- e.g. the registered-run env, or a
dedicated venv with ``POPULACE_DYNAMICS_PE_US_DIR`` pointed at the install --
these bind the factory's SSA-parameter step to the real parameters: the
version gate reads the pinned metadata, and the NAWI replacement reproduces
the forward law's projection to the digit (I_proj(2015) = 47,252.12) with the
realized post-2014 series absent.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import registered_m6_inputs as factory  # noqa: E402

_HAS_PE_US = importlib.util.find_spec("policyengine_us") is not None
_PE_US_VERSION = (
    importlib.metadata.version("policyengine-us") if _HAS_PE_US else None
)
needs_pinned_pe_us = pytest.mark.skipif(
    _PE_US_VERSION != factory.PINNED_PE_US_VERSION,
    reason=(
        "policyengine-us 1.752.2 not importable in this env; install it and "
        "set POPULACE_DYNAMICS_PE_US_DIR to run the oracle path"
    ),
)


def _installed_pe_us_dir() -> Path:
    """The site-packages root that contains ``policyengine_us/``."""
    import policyengine_us

    return Path(policyengine_us.__file__).resolve().parents[1]


@needs_pinned_pe_us
def test__version_gate__reads_the_pinned_metadata():
    assert factory.assert_pe_us_version() == "1.752.2"


@needs_pinned_pe_us
def test__nawi_replacement__matches_the_forward_law_on_real_parameters(
    monkeypatch,
):
    monkeypatch.setenv(
        "POPULACE_DYNAMICS_PE_US_DIR", str(_installed_pe_us_dir())
    )
    params = factory.boundary_ssa_parameters()

    # Realized pre-boundary anchors, unchanged from the pinned revision.
    assert params.nawi[2005] == pytest.approx(36952.94)
    assert params.nawi[2014] == pytest.approx(46481.52)
    # Post-boundary is the log-linear projection, not the realized series.
    assert params.nawi[2015] == pytest.approx(47252.12, abs=0.01)
    assert abs(params.nawi[2015] - 48098.63) > 1.0  # realized 2015 absent

    from populace_dynamics.engine.forward_earnings import (
        fit_projected_wage_index,
    )
    from populace_dynamics.ss.params import load_ssa_parameters

    full = load_ssa_parameters()
    projection = fit_projected_wage_index(full.nawi, boundary_year=2014)
    for year, value in params.nawi.items():
        if year > 2014:
            assert value == projection.projected(year)
        else:
            assert value == full.nawi[year]


@needs_pinned_pe_us
def test__wage_base__is_truncated_to_the_boundary_on_real_parameters(
    monkeypatch,
):
    monkeypatch.setenv(
        "POPULACE_DYNAMICS_PE_US_DIR", str(_installed_pe_us_dir())
    )
    params = factory.boundary_ssa_parameters()
    assert max(params.wage_base) == 2014
    assert params.wage_base[2014] == pytest.approx(117000.0)
