import numpy as np
import pandas as pd
import pytest
from collections import deque

from src.models.evaluate_models import (
    choose_f1_threshold,
    chronological_indices,
    threshold_options,
)
from src.graph.build_graph_features import GraphFeatureBuilder


def test_chronological_indices_use_expected_boundaries():
    train, validation, test = chronological_indices(10_000)
    assert (train.start, train.stop) == (0, 7000)
    assert (validation.start, validation.stop) == (7000, 8500)
    assert (test.start, test.stop) == (8500, 10_000)


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_chronological_indices_reject_too_few_rows(rows):
    with pytest.raises(ValueError):
        chronological_indices(rows)


def test_threshold_selection_returns_validation_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.20, 0.55, 0.75, 0.90])
    threshold, f1 = choose_f1_threshold(y_true, probabilities)
    assert 0.0 <= threshold <= 1.0
    assert f1 == pytest.approx(1.0)


def test_recall_operating_points_are_validation_only_candidates():
    y_true = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.20, 0.70, 0.90])
    options = threshold_options(y_true, probabilities, recall_targets=(0.5, 0.9))
    assert options["0.5"] is not None
    assert options["0.5"]["recall"] >= 0.5
    assert options["0.9"] is not None
    assert options["0.9"]["recall"] >= 0.9


def test_graph_history_keeps_event_exactly_at_window_cutoff():
    cutoff = pd.Timestamp("2026-01-02 00:00:00")
    history = deque([
        (cutoff - pd.Timedelta(seconds=1), "old", 10.0),
        (cutoff, "boundary", 20.0),
    ])
    GraphFeatureBuilder.remove_expired(history, cutoff)
    assert list(history) == [(cutoff, "boundary", 20.0)]
