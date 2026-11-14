"""Statistical checks against independently computed sklearn resamples."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.stats.bootstrap import (
    _draw_shared_test_weights,
    bootstrap_arm,
    bootstrap_contrast,
    safe_ratio,
    weighted_statistics,
)


@pytest.mark.parametrize("labels", [[0, 1, 2], [0, 2]])
def test_weighted_f1_matches_literal_resampling(labels):
    rng = np.random.default_rng(12)
    truth = rng.choice(labels, size=47)
    preds = rng.integers(0, 3, size=(4, 47))
    indices = rng.integers(0, 47, size=(25, 47))
    weights = np.array([np.bincount(index, minlength=47) for index in indices])
    actual = weighted_statistics(truth, preds, labels, weights)
    expected = np.array([[f1_score(truth[index], pred[index], labels=labels,
                                  average="macro", zero_division=0)
                          for pred in preds] for index in indices])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_point_is_mean_of_seed_scores_and_bootstrap_repeats_exactly():
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 3, size=48)
    preds = rng.integers(0, 3, size=(3, 48))
    first = bootstrap_arm(truth, preds, [0, 1, 2], boot=79, boot_seed=42, chunk_size=20)
    same = bootstrap_arm(truth, preds, [0, 1, 2], boot=79, boot_seed=42, chunk_size=20)
    other = bootstrap_arm(truth, preds, [0, 1, 2], boot=79, boot_seed=43, chunk_size=20)
    expected = [f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0) for pred in preds]
    assert first["point"] == pytest.approx(np.mean(expected))
    assert first["std"] == pytest.approx(np.std(expected, ddof=1))
    np.testing.assert_array_equal(first["draws"], same["draws"])
    assert not np.array_equal(first["draws"], other["draws"])
    assert first["n_nan"] == 0
    assert first["se"] > 0


def test_perfect_one_class_classifier_has_degenerate_interval():
    # With multiple true classes an ordinary unstratified resample can omit a
    # class. A one-class example isolates the requested perfect-classifier limit.
    truth = np.zeros(20, dtype=int)
    result = bootstrap_arm(truth, truth[None], [0], boot=20)
    assert result["point"] == result["ci_lo"] == result["ci_hi"] == 1
    assert result["se"] == result["std"] == 0


def test_strict_neutral_predictions_are_false_negatives_and_included_in_js():
    truth = np.array([0, 2, 0, 2])
    preds = np.ones((1, 4), dtype=int)
    assert weighted_statistics(truth, preds, [0, 2], np.ones(4))[0, 0] == 0
    assert weighted_statistics(truth, preds, [0, 2], np.ones(4), "js")[0, 0] == pytest.approx(1)
    np.testing.assert_array_equal(weighted_statistics(truth, truth[None], [0, 2], np.ones(4), "js"), [[0]])
    assert weighted_statistics(truth, truth[None], [0, 2], np.ones(4), "f1_2")[0, 0] == 1


def test_same_test_arms_share_the_exact_weights_and_other_tests_do_not():
    truth = np.arange(20) % 3
    arms = {name: {"y_true": truth, "preds_by_seed": truth[None], "labels": [0, 1, 2]}
            for name in ("zs", "ref", "other")}
    weights = _draw_shared_test_weights(np.random.default_rng(2), arms,
                                       {"zs": "NEU", "ref": "NEU", "other": "VSMEC"}, 11)
    assert weights["zs"] is weights["ref"]
    assert weights["zs"] is not weights["other"]
    assert not np.array_equal(weights["zs"], weights["other"])
    np.testing.assert_array_equal(weights["zs"].sum(axis=1), np.full(11, 20))


def test_model_seed_resamples_are_independent_even_when_test_rows_are_paired():
    truth = np.zeros(8, dtype=int)
    preds = np.array([[0] * 8, [1] * 8])
    arm = {"y_true": truth, "preds_by_seed": preds, "labels": [0], "stat": "accuracy"}
    result = bootstrap_contrast({"zs": arm, "ref": arm}, lambda x: x["zs"] - x["ref"], True, boot=40, boot_seed=9)
    assert result["point"] == 0
    assert np.any(result["draws"] != 0)


def test_retention_near_zero_counts_undefined_draws():
    truth = np.zeros(20, dtype=int)
    good = {"y_true": truth, "preds_by_seed": truth[None], "labels": [0]}
    bad = {"y_true": truth, "preds_by_seed": np.ones((1, 20), dtype=int), "labels": [0]}
    result = bootstrap_contrast({"zs": good, "ref": bad},
                                lambda x: safe_ratio(x["zs"], x["ref"]), True, boot=31)
    assert np.isnan(result["point"])
    assert result["n_nan"] == 31
    assert np.isnan(result["ci_lo"]) and np.isnan(result["ci_hi"])
    assert np.isnan(safe_ratio(1, 1e-9))
    assert safe_ratio(1, 1e-8) == 1e8


def test_sharing_incompatible_ordered_test_labels_is_rejected():
    one = {"y_true": [0, 2], "preds_by_seed": [[0, 2]], "labels": [0, 2]}
    two = {"y_true": [2, 0], "preds_by_seed": [[2, 0]], "labels": [0, 2]}
    with pytest.raises(ValueError, match="identical ordered"):
        bootstrap_contrast({"a": one, "b": two}, lambda x: x["a"] - x["b"], True, boot=10)


@pytest.mark.filterwarnings("ignore:Unknown pytest.mark.slow:pytest.PytestUnknownMarkWarning")
@pytest.mark.slow
def test_accuracy_interval_has_reasonable_known_population_coverage():
    rng = np.random.default_rng(981)
    hits = 0
    experiments = 100
    probability = 0.75
    for experiment in range(experiments):
        truth = np.zeros(200, dtype=int)
        preds = (rng.random(200) > probability).astype(int)[None]
        result = bootstrap_arm(truth, preds, [0, 1], "accuracy", boot=299,
                               boot_seed=experiment + 250, chunk_size=299)
        hits += result["ci_lo"] <= probability <= result["ci_hi"]
    assert 0.90 <= hits / experiments <= 0.99
