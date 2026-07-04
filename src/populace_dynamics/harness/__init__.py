"""The population-view evaluation harness.

Adapted from PolicyEngine/imputation-paper (src/imputation_paper/experiments/),
the population-view harness described in that paper; extended here for
longitudinal views.

Holds the weighted metrics (:mod:`~populace_dynamics.harness.metrics`), the
donor/receiver holdout split (:mod:`~populace_dynamics.harness.holdout`), and
the survey-view scoring (:mod:`~populace_dynamics.harness.views`).
"""

from __future__ import annotations

from populace_dynamics.harness.holdout import Split, paired_splits, split_frame
from populace_dynamics.harness.metrics import (
    classifier_two_sample_auc,
    energy_distance,
    prdc,
    reweight_fragility,
    weighted_pinball_loss,
    weighted_wasserstein1,
    zero_share_error,
)
from populace_dynamics.harness.views import (
    SurveyView,
    harness_scorecard,
    project_view,
    score_view,
)

__all__ = [
    "Split",
    "SurveyView",
    "classifier_two_sample_auc",
    "energy_distance",
    "harness_scorecard",
    "paired_splits",
    "prdc",
    "project_view",
    "reweight_fragility",
    "score_view",
    "split_frame",
    "weighted_pinball_loss",
    "weighted_wasserstein1",
    "zero_share_error",
]
