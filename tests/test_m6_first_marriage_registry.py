"""Guards for the frozen M6 first-marriage registry sibling."""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeWarning

from populace_dynamics.data import transitions
from populace_dynamics.models.family_transitions import registry as ft_registry
from populace_dynamics.models.family_transitions.components import (
    first_marriage_support_aware as support_aware,
)
from populace_dynamics.models.family_transitions.components.first_marriage_support_aware import (
    FirstMarriagePreflightAbort,
    SupportAwareFirstMarriageModel,
    _objective_and_gradient,
    fit_support_aware_first_marriage,
    recompute_support_aware_first_marriage_checksums,
    validate_support_aware_first_marriage_fit,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT / "src/populace_dynamics/models/family_transitions/registry.py"
)
LEGACY_FIRST_MARRIAGE_PATH = (
    ROOT
    / "src/populace_dynamics/models/family_transitions/components"
    / "first_marriage.py"
)
SIBLING_FIRST_MARRIAGE_PATH = (
    ROOT
    / "src/populace_dynamics/models/family_transitions/components"
    / "first_marriage_support_aware.py"
)
PREFREEZE_REGISTRY_SPEC_PATH = (
    ROOT / "docs/analysis/m6_first_marriage_registry_spec_prefreeze_v1.json"
)
SELECTION_FINDINGS_PATH = (
    ROOT / "docs/analysis/m6_first_marriage_c_selection_findings.json"
)
CANDIDATE_16_SPEC_SHA256 = (
    "6d4d2b2beadc87d17404a3deb64a272c2456d7471b3ad6f1cef779d807765aa1"
)
CANDIDATE_16_SOURCE_SHA256 = (
    "230845a869c25f092fb4d921cfc163abbcf20079a2ab05699aa8a6c3ac5b7a70"
)
LEGACY_FIRST_MARRIAGE_SHA256 = (
    "3b90492eb32e779061afb4d771e40d1d0990c0fa4ddfc912bd6a61187efeabb1"
)
INCUMBENT_FITTER_SOURCE_SHA256 = {
    "_fit_initial_states": (
        "a9b53616fe2ee699b3d832bb76b766c6d159ccc1007b070d760fb1858d09d1f3"
    ),
    "_train_frames": (
        "e5c1969f12f0364f7167efb444d0b3e3c90a3cfc96b2c83d9affd7142e937036"
    ),
    "_fit_first_marriage": (
        "d8588aea0b469235230d59678d0ab5cdc47dfd8109c49bc4bad3d495f8fd2bf6"
    ),
    "_fit_divorce": (
        "400ba2b77ca73fec233ffbcbccf0a1725dd8bdd1999ab798e849447774d9c102"
    ),
    "_fit_remarriage": (
        "9ff3f1040c2bf5e109d023cc0160bb99e3e7d48dfa5f6210ee75d8a9c5a6ce95"
    ),
    "_fit_fertility": (
        "1788fc8b2d137b6c6d8e2188c46b8723b26e3841541880cf295923d9e61e01ee"
    ),
    "_fit_gap": (
        "fa11f61ed4c7da63ad4b61dca81c593352e01dc5e45d52859d72668fc6e67404"
    ),
    "_fit_widowhood": (
        "acf8e1540ab1a5d85fae739906ba8a4009e97aaa5aa6b77946d7cc2842ab02c9"
    ),
}


def _balanced_support_frame() -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    patterns = (
        ("female", 1970, (18, 24, 25)),
        ("female", 1980, (18, 24, 25)),
        ("female", 1990, (18, 23)),
        ("male", 1970, (18, 24, 25)),
        ("male", 1980, (18, 24, 25)),
    )
    rows: list[dict[str, object]] = []
    event_years: set[tuple[int, int]] = set()
    person_id = 1
    for sex, cohort, ages in patterns:
        for age in ages:
            for is_event in (False, True):
                year = cohort + age
                rows.append(
                    {
                        "person_id": person_id,
                        "year": year,
                        "age": age,
                        "sex": sex,
                        "weight": 1.0,
                        "birth_decade": cohort,
                    }
                )
                if is_event:
                    event_years.add((person_id, year))
                person_id += 1
    return pd.DataFrame(rows), event_years


def _fit_balanced() -> SupportAwareFirstMarriageModel:
    frame, event_years = _balanced_support_frame()
    return fit_support_aware_first_marriage(frame, event_years, c=0.01)


def _registry_context(
    frame: pd.DataFrame,
    event_years: set[tuple[int, int]],
) -> ft_registry.FitContext:
    person_years = frame.copy()
    person_years["required_interview_year"] = person_years["year"]
    person_years["marital_state"] = "never_married"
    event_mask = [
        (int(person_id), int(year)) in event_years
        for person_id, year in zip(
            person_years["person_id"],
            person_years["year"],
            strict=True,
        )
    ]
    events = person_years.loc[
        event_mask,
        [
            "person_id",
            "year",
            "required_interview_year",
            "age",
            "sex",
            "weight",
        ],
    ].copy()
    events["transition"] = "first_marriage"
    attrs = (
        frame.sort_values(["person_id", "year"])
        .groupby("person_id", as_index=False)
        .first()[["person_id", "birth_decade", "weight"]]
        .rename(columns={"birth_decade": "birth_year"})
    )
    attrs["censor_year"] = 2014
    people = attrs["person_id"].astype(int).tolist()
    panel = transitions.MaritalPanel(
        person_years=person_years,
        events=events,
        attrs=attrs,
    )
    return ft_registry.FitContext(
        panel=panel,
        demographic_panel=pd.DataFrame(
            {"person_id": people, "period": [2014] * len(people)}
        ),
        marriage_records=pd.DataFrame(),
        birth_records=pd.DataFrame(),
        marriage_order_map=pd.DataFrame(),
        train_ids=frozenset(people),
    )


def test_candidate16_source_component_and_spec_remain_byte_unchanged():
    registry_bytes = REGISTRY_PATH.read_bytes()
    marker = b"CANDIDATE_16 = CandidateSpec("
    candidate_16_source = registry_bytes[registry_bytes.index(marker) :]

    assert hashlib.sha256(candidate_16_source).hexdigest() == (
        CANDIDATE_16_SOURCE_SHA256
    )
    assert hashlib.sha256(
        LEGACY_FIRST_MARRIAGE_PATH.read_bytes()
    ).hexdigest() == (LEGACY_FIRST_MARRIAGE_SHA256)
    assert ft_registry.CANDIDATE_16.sha256 == CANDIDATE_16_SPEC_SHA256


def test_prefreeze_registry_spec_artifact_is_exact_historical_snapshot():
    raw = PREFREEZE_REGISTRY_SPEC_PATH.read_text(encoding="utf-8")
    artifact = json.loads(raw)
    assert (
        raw
        == json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    assert set(artifact) == {
        "authority",
        "candidate_16_invariance",
        "implementation_files",
        "registration_disposition",
        "resolved_candidate_spec",
        "schema_version",
        "selection_binding",
        "status",
    }
    resolved = dict(artifact["resolved_candidate_spec"])
    recorded_sha256 = resolved.pop("sha256")
    assert resolved == ft_registry.M6_CANDIDATE_2_PREFREEZE.canonical_dict()
    assert recorded_sha256 == ft_registry.M6_CANDIDATE_2_PREFREEZE.sha256
    canonical = json.dumps(
        resolved,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == recorded_sha256

    # These bind the immutable prefreeze artifact to its historical source.
    # The live registry legitimately adds the separately frozen sibling.
    assert artifact["implementation_files"] == [
        {
            "n_bytes": 22624,
            "path": (
                "src/populace_dynamics/models/family_transitions/registry.py"
            ),
            "sha256": (
                "92491b1282845249ed25027dabcd82b99a47a6ca56842e5bb516b4da8b83d156"
            ),
        },
        {
            "n_bytes": 48264,
            "path": (
                "src/populace_dynamics/models/family_transitions/components/"
                "first_marriage_support_aware.py"
            ),
            "sha256": (
                "d1cac0298b6f1c391fe49d5195e1e1754114d710101933dd717bd0651545958e"
            ),
        },
    ]
    # The live sibling adds only the post-#267 pickle-boundary reducer; the
    # prefreeze artifact above remains its unchanged historical snapshot.
    sibling_payload = SIBLING_FIRST_MARRIAGE_PATH.read_bytes()
    assert len(sibling_payload) == 49673
    assert hashlib.sha256(sibling_payload).hexdigest() == (
        "5bff8cc9fc2d9a57686a2adb99e11f2683017605ba21db9e61d21bd42b137b2a"
    )
    invariance = artifact["candidate_16_invariance"]
    assert invariance == {
        "legacy_first_marriage": {
            "n_bytes": len(LEGACY_FIRST_MARRIAGE_PATH.read_bytes()),
            "path": str(LEGACY_FIRST_MARRIAGE_PATH.relative_to(ROOT)),
            "sha256": LEGACY_FIRST_MARRIAGE_SHA256,
        },
        "literal_candidate_spec_source_sha256": (CANDIDATE_16_SOURCE_SHA256),
        "resolved_candidate_spec_sha256": CANDIDATE_16_SPEC_SHA256,
    }
    assert artifact["selection_binding"] == {
        "final_fit_checksums": None,
        "full_stdout_sha256": None,
        "no_registerable_status": "NO_REGISTERABLE_FIRST_MARRIAGE_FIT",
        "real_data_selection_executed_in_this_lane": False,
        "required_raw_schema": "m6.first_marriage.c_selection.full.v1",
        "selected_c": None,
    }
    assert (
        artifact["registration_disposition"]["registration_may_proceed"]
        is False
    )
    assert artifact["registration_disposition"]["freeze_performed"] is False


def test_incumbent_registry_definitions_dispatch_and_fitters_are_pinned():
    expected_dispatch = {
        "initial_states": (
            ft_registry._fit_initial_states,
            "candidate9:189-232;c12:469-526,1103-1133;c16:410-486",
        ),
        "first_marriage": (
            ft_registry._fit_first_marriage,
            "candidate1:187-315;candidate2:147-175;candidate3:187-238",
        ),
        "divorce": (
            ft_registry._fit_divorce,
            "candidate1:384-414,485-499,884-889",
        ),
        "remarriage": (
            ft_registry._fit_remarriage,
            "candidate10:260-328,395-414;candidate11:163-171",
        ),
        "fertility": (
            ft_registry._fit_fertility,
            "candidate5:312-439",
        ),
        "spousal_age_gap": (
            ft_registry._fit_gap,
            "candidate12:317-460,793-831",
        ),
        "widowhood": (
            ft_registry._fit_widowhood,
            "candidate14:379-430;candidate15:391-423;candidate16:497-634,795-823",
        ),
    }
    incumbent_keys = {
        (component.kind, component.implementation_id)
        for component in ft_registry.CANDIDATE_16.components
    }
    sibling_key = (
        "first_marriage",
        "logit_ncs_age_sex_boundary_flat_cohort_l2.v1",
    )
    assert set(ft_registry.REGISTRY._definitions) == incumbent_keys | {
        sibling_key
    }
    for component in ft_registry.CANDIDATE_16.components:
        definition = ft_registry.REGISTRY.definition(component)
        fitter, source = expected_dispatch[component.kind]
        assert definition.fitter is fitter
        assert definition.source == source
        assert definition.params == component.params

    for name, expected_sha256 in INCUMBENT_FITTER_SOURCE_SHA256.items():
        source = inspect.getsource(getattr(ft_registry, name)).encode()
        assert hashlib.sha256(source).hexdigest() == expected_sha256


def test_candidate16_dispatch_order_and_dependency_state_are_unchanged():
    expected_order = [
        component.kind for component in ft_registry.CANDIDATE_16.components
    ]
    calls: list[str] = []
    context = object()
    definitions = []
    for index, component in enumerate(ft_registry.CANDIDATE_16.components):

        def fitter(
            observed_context: object,
            fitted: dict[str, object],
            *,
            kind: str = component.kind,
            expected_predecessors: tuple[str, ...] = tuple(
                expected_order[:index]
            ),
        ) -> str:
            assert observed_context is context
            assert tuple(fitted) == expected_predecessors
            calls.append(kind)
            return kind

        definitions.append(
            ft_registry.ComponentDefinition(
                kind=component.kind,
                implementation_id=component.implementation_id,
                fitter=fitter,  # type: ignore[arg-type]
                source="candidate16_dispatch_guard",
                params=component.params,
            )
        )
    isolated = ft_registry.ComponentRegistry(tuple(definitions))
    fitted = isolated.fit(
        ft_registry.CANDIDATE_16,
        context,  # type: ignore[arg-type]
    )
    assert calls == expected_order
    assert fitted.implementation_ids == {
        component.kind: component.implementation_id
        for component in ft_registry.CANDIDATE_16.components
    }


def test_prefreeze_and_frozen_specs_are_first_marriage_only_siblings(
    monkeypatch: pytest.MonkeyPatch,
):
    incumbent = {
        component.kind: component
        for component in ft_registry.CANDIDATE_16.components
    }
    prefreeze = {
        component.kind: component
        for component in ft_registry.M6_CANDIDATE_2_PREFREEZE.components
    }
    sibling = {
        component.kind: component
        for component in ft_registry.M6_CANDIDATE_2.components
    }

    assert tuple(prefreeze) == tuple(incumbent)
    assert tuple(sibling) == tuple(incumbent)
    for kind in set(incumbent) - {"first_marriage"}:
        assert prefreeze[kind] == incumbent[kind]
        assert sibling[kind] == incumbent[kind]
    assert prefreeze["first_marriage"].params["selected_c"] is None
    assert sibling["first_marriage"].implementation_id == (
        "logit_ncs_age_sex_boundary_flat_cohort_l2.v1"
    )
    selection = json.loads(SELECTION_FINDINGS_PATH.read_text(encoding="utf-8"))
    params = sibling["first_marriage"].params
    assert params["selected_c"] == selection["selection"]["selected_c"]
    assert params["selected_c"] == 0.001
    assert (
        params["selection_ledger_sha256"]
        == hashlib.sha256(SELECTION_FINDINGS_PATH.read_bytes()).hexdigest()
    )
    assert (
        dict(params["final_fit_checksums"])
        == selection["final_fit"]["fit_audit"]["checksums"]
    )
    assert sibling["first_marriage"].params["substream_codes"] == ()
    assert sibling["first_marriage"].params["solver_tol"] == 1e-8
    assert sibling["first_marriage"].params["solver_gtol"] == 1e-8
    assert (
        sibling["first_marriage"].params["solver_ftol"]
        == np.finfo(np.float64).eps
    )
    incumbent_definition = ft_registry.REGISTRY.definition(
        incumbent["first_marriage"]
    )
    sibling_definition = ft_registry.REGISTRY.definition(
        sibling["first_marriage"]
    )
    assert incumbent_definition.fitter is ft_registry._fit_first_marriage
    assert sibling_definition.fitter is ft_registry._fit_m6_first_marriage
    assert ft_registry.M6_CANDIDATE_2.sha256 == (
        "734a5b04f347c5d4904bbc6d5ab9a1c2876272d35284eedd2f450518acf1cec5"
    )

    frame, event_years = _balanced_support_frame()
    context = _registry_context(frame, event_years)
    fitted = object()

    def fit_sentinel(*_args: object, **kwargs: object) -> object:
        assert kwargs["c"] == 0.001
        assert kwargs["max_iter"] == 10000
        assert kwargs["tol"] == 1e-8
        return fitted

    monkeypatch.setattr(
        ft_registry,
        "fit_support_aware_first_marriage",
        fit_sentinel,
    )
    assert sibling_definition.fitter(context, {}) is fitted


def test_candidate3_reuses_exact_candidate2_frozen_family_spec():
    candidate2 = ft_registry.M6_CANDIDATE_2
    candidate3 = ft_registry.M6_CANDIDATE_3

    assert candidate3 is candidate2
    assert candidate3.canonical_dict() == candidate2.canonical_dict()
    assert candidate3.components is candidate2.components
    assert candidate3.candidate_id == "m6_candidate2_registry_v1"
    assert candidate3.sha256 == candidate2.sha256
    assert candidate3.sha256 == (
        "734a5b04f347c5d4904bbc6d5ab9a1c2876272d35284eedd2f450518acf1cec5"
    )


def test_prefreeze_reference_is_not_a_live_component_definition():
    reference = next(
        component
        for component in ft_registry.M6_CANDIDATE_2_PREFREEZE.components
        if component.kind == "first_marriage"
    )
    with pytest.raises(ValueError, match="parameters"):
        ft_registry.REGISTRY.definition(reference)


def test_prefreeze_candidate_aborts_before_context_or_any_component_fitter():
    class NoContextReads:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"prefreeze read context.{name}")

    with pytest.raises(FirstMarriagePreflightAbort, match="any component"):
        ft_registry.REGISTRY.fit(
            ft_registry.M6_CANDIDATE_2_PREFREEZE,
            NoContextReads(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("frame_name", "field"),
    (
        ("person_years", "year"),
        ("events", "required_interview_year"),
    ),
)
def test_selected_registry_adapter_rejects_every_2015_date_before_fit(
    monkeypatch: pytest.MonkeyPatch,
    frame_name: str,
    field: str,
):
    frame, event_years = _balanced_support_frame()
    context = _registry_context(frame, event_years)
    poisoned = getattr(context.panel, frame_name).copy()
    poisoned.loc[poisoned.index[0], field] = 2015
    panel = transitions.MaritalPanel(
        person_years=(
            poisoned
            if frame_name == "person_years"
            else context.panel.person_years
        ),
        events=poisoned if frame_name == "events" else context.panel.events,
        attrs=context.panel.attrs,
    )
    context = replace(context, panel=panel)
    monkeypatch.setattr(
        ft_registry,
        "_M6_FIRST_MARRIAGE_PARAMS",
        {"selected_c": 0.01},
    )
    fit_calls = 0

    def fit_sentinel(*args: object, **kwargs: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("fit must remain unreachable")

    monkeypatch.setattr(
        ft_registry,
        "fit_support_aware_first_marriage",
        fit_sentinel,
    )
    with pytest.raises(
        FirstMarriagePreflightAbort,
        match=rf"{field} reaches 2015",
    ):
        ft_registry._fit_m6_first_marriage(context, {})
    assert fit_calls == 0


def test_registry_converts_expected_degeneracy_but_selector_keeps_value_error(
    monkeypatch: pytest.MonkeyPatch,
):
    frame, _event_years = _balanced_support_frame()
    context = _registry_context(frame, set())
    monkeypatch.setattr(
        ft_registry,
        "_M6_FIRST_MARRIAGE_PARAMS",
        {"selected_c": 0.01},
    )
    with pytest.raises(
        FirstMarriagePreflightAbort,
        match="registered fit input is degenerate",
    ):
        ft_registry._fit_m6_first_marriage(context, {})
    with pytest.raises(ValueError, match="event and non-event"):
        fit_support_aware_first_marriage(frame, set(), c=0.01)


def test_boundary_flat_design_keeps_global_age_and_clips_cohort_deviation():
    model = _fit_balanced()
    age = np.asarray([23.0, 24.0])
    female = np.asarray([False, False])
    cohort_1990 = np.asarray([1990, 1990])
    raw = model.raw_design(age, female, cohort_1990)

    # The female global spline sees supported age 24 from older cohorts.
    assert not np.array_equal(raw[0, :4], raw[1, :4])
    # The 1990 deviation is held at its pooled support endpoint, age 23.
    # With levels 1970/1980/1990, its four interaction columns are 12/14/16/18.
    assert np.array_equal(raw[0, [12, 14, 16, 18]], raw[1, [12, 14, 16, 18]])

    unseen_new = model.raw_design(
        np.asarray([24.0]), np.asarray([False]), np.asarray([2000])
    )
    seen_newest = model.raw_design(
        np.asarray([24.0]), np.asarray([False]), np.asarray([1990])
    )
    assert np.array_equal(unseen_new, seen_newest)

    # A nearest-cohort tie resolves deterministically to the older decade.
    tie = model.raw_design(
        np.asarray([24.0]), np.asarray([False]), np.asarray([1985])
    )
    older = model.raw_design(
        np.asarray([24.0]), np.asarray([False]), np.asarray([1980])
    )
    assert np.array_equal(tie, older)

    diagnostics = model.transport_diagnostics(age, female, cohort_1990)
    assert diagnostics.canonical_records() == [
        {
            "sex": "female",
            "target_birth_decade": 1990,
            "mapped_birth_decade": 1990,
            "age": 23.0,
            "global_age_evaluated": 23.0,
            "cohort_age_evaluated": 23.0,
            "global_boundary_evaluated": False,
            "cohort_boundary_evaluated": False,
        },
        {
            "sex": "female",
            "target_birth_decade": 1990,
            "mapped_birth_decade": 1990,
            "age": 24.0,
            "global_age_evaluated": 24.0,
            "cohort_age_evaluated": 23.0,
            "global_boundary_evaluated": False,
            "cohort_boundary_evaluated": True,
        },
    ]
    assert not diagnostics.cohort_age_evaluated.flags.writeable


def test_fit_audit_is_canonical_complete_and_records_empty_cartesian_cell():
    frame, event_years = _balanced_support_frame()
    model = fit_support_aware_first_marriage(frame, event_years, c=0.01)
    shuffled = fit_support_aware_first_marriage(
        frame.sample(frac=1.0, random_state=42),
        event_years,
        c=0.01,
    )

    validate_support_aware_first_marriage_fit(model)
    assert model.fit_audit is not None
    assert shuffled.fit_audit is not None
    assert model.fit_audit.eligible
    assert model.fit_audit.eligibility_failures == ()
    assert model.fit_audit.checksums == shuffled.fit_audit.checksums
    empty = model.fit_audit.support["sex_by_cohort"]["male|1990"]
    assert empty == {
        "n_rows": 0,
        "row_weight": 0.0,
        "n_events": 0,
        "event_weight": 0.0,
        "age_min": None,
        "age_max": None,
    }
    assert set(model.audit_dict()) == {
        "c",
        "n_input_rows",
        "n_train_rows",
        "n_train_events",
        "n_features",
        "solver_success",
        "solver_status",
        "solver_message",
        "n_iter",
        "max_iter",
        "max_iter_reached",
        "solver_gtol",
        "solver_ftol",
        "warning_count",
        "warning_messages",
        "convergence_warning_count",
        "convergence_warning_messages",
        "objective_value",
        "gradient_inf_norm",
        "intercept",
        "coefficients",
        "design_finite",
        "coefficients_finite",
        "linear_predictor_finite",
        "linear_predictor_min",
        "linear_predictor_max",
        "probabilities_finite",
        "probabilities_strict_unit_interval",
        "probability_min",
        "probability_max",
        "eligible",
        "eligibility_failures",
        "checksums",
        "support",
    }
    assert model.fit_audit.solver_gtol == 1e-8
    assert model.fit_audit.solver_ftol == np.finfo(np.float64).eps


def test_fit_audit_and_model_pickle_round_trip():
    model = _fit_balanced()
    audit = model.fit_audit
    assert audit is not None

    assert "__reduce__" in type(audit).__dict__
    restored_audit = pickle.loads(pickle.dumps(audit, protocol=5))
    assert restored_audit.canonical_dict() == audit.canonical_dict()
    with pytest.raises(TypeError):
        restored_audit.checksums["coefficient_sha256"] = "0" * 64
    with pytest.raises(TypeError):
        restored_audit.support["sex"]["female"]["age_max"] = 123.0

    restored_model = pickle.loads(pickle.dumps(model, protocol=5))
    assert restored_model.audit_dict() == model.audit_dict()
    validate_support_aware_first_marriage_fit(restored_model)


def test_explicit_penalized_objective_and_gradient_scaling():
    design = np.asarray(
        [[-1.0, 0.5], [0.0, -0.25], [0.75, 1.0], [1.5, -0.5]],
        dtype=np.float64,
    )
    outcomes = np.asarray([0.0, 1.0, 0.0, 1.0])
    normalized_weight = np.asarray([0.5, 1.5, 0.75, 1.25])
    parameters = np.asarray([0.2, -0.4, 0.3])
    c = 0.3
    objective, gradient = _objective_and_gradient(
        parameters,
        design,
        outcomes,
        normalized_weight,
        c,
    )

    linear = parameters[0] + design.dot(parameters[1:])
    expected_objective = float(
        (
            normalized_weight * (np.logaddexp(0.0, linear) - outcomes * linear)
        ).mean()
        + parameters[1:].dot(parameters[1:]) / (2.0 * c * len(outcomes))
    )
    assert objective == pytest.approx(expected_objective, abs=1e-15)

    epsilon = 1e-6
    numerical = np.empty_like(parameters)
    for index in range(len(parameters)):
        step = np.zeros_like(parameters)
        step[index] = epsilon
        upper = _objective_and_gradient(
            parameters + step,
            design,
            outcomes,
            normalized_weight,
            c,
        )[0]
        lower = _objective_and_gradient(
            parameters - step,
            design,
            outcomes,
            normalized_weight,
            c,
        )[0]
        numerical[index] = (upper - lower) / (2.0 * epsilon)
    assert np.allclose(gradient, numerical, rtol=0.0, atol=1e-9)

    residual = normalized_weight * (1.0 / (1.0 + np.exp(-linear)) - outcomes)
    assert gradient[0] == pytest.approx(residual.mean(), abs=1e-15)
    expected_slopes = design.T.dot(residual) / len(outcomes) + parameters[
        1:
    ] / (c * len(outcomes))
    assert np.allclose(gradient[1:], expected_slopes, rtol=0.0, atol=1e-15)


def test_live_state_is_read_only_and_all_checksums_replay_independently():
    model = _fit_balanced()
    assert model.fit_audit is not None
    assert dict(recompute_support_aware_first_marriage_checksums(model)) == (
        dict(model.fit_audit.checksums)
    )
    validate_support_aware_first_marriage_fit(
        model,
        expected_checksums=dict(model.fit_audit.checksums),
    )
    with pytest.raises(ValueError, match="read-only"):
        model.coefficients[0] = 123.0
    with pytest.raises(ValueError, match="read-only"):
        model.sex_age_max[0] = 123.0
    with pytest.raises(TypeError):
        model.fit_audit.checksums["coefficient_sha256"] = "0" * 64
    with pytest.raises(TypeError):
        model.fit_audit.support["sex"]["female"]["age_max"] = 123.0


def test_forced_state_mutation_and_self_referential_hashes_fail_replay():
    model = _fit_balanced()
    assert model.fit_audit is not None
    original_checksums = dict(
        recompute_support_aware_first_marriage_checksums(model)
    )
    bad_audit_checksums = dict(model.fit_audit.checksums)
    bad_audit_checksums["coefficient_sha256"] = "0" * 64
    object.__setattr__(
        model,
        "fit_audit",
        replace(model.fit_audit, checksums=bad_audit_checksums),
    )
    with pytest.raises(
        FirstMarriagePreflightAbort,
        match="registered coefficient_sha256 does not reproduce",
    ):
        # This expected mapping agrees with the corrupted audit. Validation
        # must compare it with the independently replayed live checksum.
        validate_support_aware_first_marriage_fit(
            model,
            expected_checksums=bad_audit_checksums,
        )

    model = _fit_balanced()
    expanded = model.sex_age_max.copy()
    expanded[0] += 100.0
    expanded.setflags(write=False)
    object.__setattr__(model, "sex_age_max", expanded)
    with pytest.raises(
        FirstMarriagePreflightAbort,
        match="sex support maxima differ from fit rows",
    ):
        validate_support_aware_first_marriage_fit(
            model,
            expected_checksums=original_checksums,
        )


def test_only_convergence_warnings_make_an_otherwise_valid_fit_ineligible(
    monkeypatch: pytest.MonkeyPatch,
):
    frame, event_years = _balanced_support_frame()
    original_minimize = support_aware.minimize

    def harmless_warning(*args: object, **kwargs: object):
        warnings.warn("synthetic provenance note", UserWarning, stacklevel=2)
        return original_minimize(*args, **kwargs)

    monkeypatch.setattr(support_aware, "minimize", harmless_warning)
    harmless = fit_support_aware_first_marriage(
        frame,
        event_years,
        c=0.01,
    )
    assert harmless.fit_audit is not None
    assert harmless.fit_audit.warning_count == 1
    assert harmless.fit_audit.convergence_warning_count == 0
    assert harmless.fit_audit.eligible
    validate_support_aware_first_marriage_fit(harmless)

    def convergence_warning(*args: object, **kwargs: object):
        warnings.warn(
            "synthetic convergence failed",
            OptimizeWarning,
            stacklevel=2,
        )
        return original_minimize(*args, **kwargs)

    monkeypatch.setattr(support_aware, "minimize", convergence_warning)
    convergence = fit_support_aware_first_marriage(
        frame,
        event_years,
        c=0.01,
    )
    assert convergence.fit_audit is not None
    assert convergence.fit_audit.convergence_warning_count == 1
    assert not convergence.fit_audit.eligible
    assert "convergence_warning_emitted" in (
        convergence.fit_audit.eligibility_failures
    )
    with pytest.raises(FirstMarriagePreflightAbort, match="convergence"):
        validate_support_aware_first_marriage_fit(convergence)


def test_ineligible_fit_is_publishable_but_preflight_aborts():
    frame, event_years = _balanced_support_frame()
    event_years.remove(sorted(event_years)[0])
    model = fit_support_aware_first_marriage(
        frame,
        event_years,
        c=0.01,
        max_iter=1,
    )

    assert model.fit_audit is not None
    assert not model.fit_audit.eligible
    assert "max_iter_reached" in model.fit_audit.eligibility_failures
    assert model.audit_dict()["eligible"] is False
    with pytest.raises(
        FirstMarriagePreflightAbort,
        match="NO_REGISTERABLE_FIRST_MARRIAGE_FIT",
    ):
        validate_support_aware_first_marriage_fit(model)
