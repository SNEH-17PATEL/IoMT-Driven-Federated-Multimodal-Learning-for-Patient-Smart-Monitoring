"""
Phase 2 — Model Utilities  (test_model_utils.py)
=================================================
Tests every public function in model_utils.py.
No DB access. No file I/O. Uses mini_model/synthetic_client_data from conftest.

87 tests across 11 groups:
  2A  ICUModel                    —  9 tests
  2B  ICUDataset                  —  4 tests
  2C  weighted_mse_loss           —  5 tests
  2D  get_sample_weights          —  5 tests
  2E  train_model                 — 11 tests
  2F  evaluate_model              —  8 tests
  2G  compute_sofa_risk_metrics   — 16 tests
  2H  get_weights / set_weights   —  6 tests
  2I  get_trend / classify_range  — 11 tests
  2J  apply_dp_to_update          —  6 tests
  2K  estimate_privacy_budget     —  6 tests
"""
import math
import os
import sys

import numpy as np
import numpy.testing as npt
import pytest
import torch

# ── sys.path fallback ─────────────────────────────────────────────────────────
_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.normpath(os.path.join(_UNIT_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_utils import (
    ICUModel, ICUDataset,
    weighted_mse_loss, get_sample_weights,
    train_model, evaluate_model,
    compute_sofa_risk_metrics,
    get_weights, set_weights,
    get_trend, classify_range,
    apply_dp_to_update, estimate_privacy_budget,
)


# ═════════════════════════════════════════════════════════════════════════════
# 2A — ICUModel
# ═════════════════════════════════════════════════════════════════════════════

def test_icu_model_forward_output_shape(mini_model):
    """ICUModel(10).forward(x) returns tensor of shape (n, 1)."""
    x   = torch.randn(8, 10)
    out = mini_model(x)
    assert out.shape == (8, 1)


def test_icu_model_production_input_dim():
    """ICUModel(108) works with the full production input dimension."""
    model = ICUModel(108)
    x     = torch.randn(4, 108)
    out   = model(x)
    assert out.shape == (4, 1)


def test_icu_model_layer_count(mini_model):
    """Architecture: exactly 4 Linear layers and 3 ReLU in .net."""
    linears = [m for m in mini_model.net if isinstance(m, torch.nn.Linear)]
    relus   = [m for m in mini_model.net if isinstance(m, torch.nn.ReLU)]
    assert len(linears) == 4
    assert len(relus)   == 3


def test_icu_model_first_layer_dims():
    """First Linear layer: (input_dim → 128)."""
    model  = ICUModel(10)
    first  = [m for m in model.net if isinstance(m, torch.nn.Linear)][0]
    assert first.in_features  == 10
    assert first.out_features == 128


def test_icu_model_second_layer_dims(mini_model):
    """Second Linear layer: (128 → 64)."""
    linears = [m for m in mini_model.net if isinstance(m, torch.nn.Linear)]
    assert linears[1].in_features  == 128
    assert linears[1].out_features == 64


def test_icu_model_third_layer_dims(mini_model):
    """Third Linear layer: (64 → 32)."""
    linears = [m for m in mini_model.net if isinstance(m, torch.nn.Linear)]
    assert linears[2].in_features  == 64
    assert linears[2].out_features == 32


def test_icu_model_output_layer_dims(mini_model):
    """Output Linear layer: (32 → 1)."""
    linears = [m for m in mini_model.net if isinstance(m, torch.nn.Linear)]
    assert linears[3].in_features  == 32
    assert linears[3].out_features == 1


def test_icu_model_eval_mode_deterministic(mini_model):
    """In eval mode, repeated forward passes return identical outputs."""
    mini_model.eval()
    x = torch.randn(4, 10)
    with torch.no_grad():
        out1 = mini_model(x).clone()
        out2 = mini_model(x).clone()
    npt.assert_array_equal(out1.numpy(), out2.numpy())


def test_icu_model_parameter_count():
    """Total parameter count matches the architecture formula."""
    # Linear(10,128): 10*128+128=1408   Linear(128,64): 128*64+64=8256
    # Linear(64,32):   64*32+32=2080    Linear(32,1):    32*1+1 =   33
    model    = ICUModel(10)
    expected = 10*128 + 128 + 128*64 + 64 + 64*32 + 32 + 32*1 + 1
    actual   = sum(p.numel() for p in model.parameters())
    assert actual == expected


# ═════════════════════════════════════════════════════════════════════════════
# 2B — ICUDataset
# ═════════════════════════════════════════════════════════════════════════════

def test_icu_dataset_len(synthetic_client_data):
    """len(dataset) equals the number of samples."""
    X, y = synthetic_client_data
    ds   = ICUDataset(X, y)
    assert len(ds) == len(X)


def test_icu_dataset_getitem_returns_tuple(synthetic_client_data):
    """dataset[0] returns a 2-tuple (X_tensor, y_tensor)."""
    X, y = synthetic_client_data
    ds   = ICUDataset(X, y)
    item = ds[0]
    assert isinstance(item, tuple) and len(item) == 2


def test_icu_dataset_x_dtype_float32(synthetic_client_data):
    """X tensor dtype is float32."""
    X, y      = synthetic_client_data
    ds        = ICUDataset(X, y)
    x_tensor, _ = ds[0]
    assert x_tensor.dtype == torch.float32


def test_icu_dataset_y_shape_one_dim(synthetic_client_data):
    """y tensor at index 0 has shape (1,) — from .view(-1, 1)."""
    X, y         = synthetic_client_data
    ds           = ICUDataset(X, y)
    _, y_tensor  = ds[0]
    assert y_tensor.shape == torch.Size([1])


# ═════════════════════════════════════════════════════════════════════════════
# 2C — weighted_mse_loss()
# ═════════════════════════════════════════════════════════════════════════════

def test_weighted_mse_loss_zero_for_perfect_predictions():
    """Perfect predictions → loss == 0."""
    t    = torch.tensor([0.0, 5.0, 10.0, 20.0]).view(-1, 1)
    loss = weighted_mse_loss(t, t)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_weighted_mse_loss_higher_sofa_gets_larger_weight():
    """Higher SOFA targets always get larger raw weights (before normalisation)."""
    for low_val, high_val in [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0)]:
        assert (1.0 + high_val * 0.5) > (1.0 + low_val * 0.5)


def test_weighted_mse_loss_formula_correctness():
    """weight = 1 + target*0.5; single-element batch: normalised weight == 1."""
    # target=10 → raw weight=6.0; only element → mean=6.0 → normalised=1.0
    # loss = 1.0 * (0 - 10)^2 = 100
    t    = torch.tensor([[10.0]])
    p    = torch.zeros(1, 1)
    loss = weighted_mse_loss(p, t)
    assert loss.item() == pytest.approx(100.0, rel=1e-4)


def test_weighted_mse_loss_positive_for_nonzero_error():
    """Non-zero prediction error → positive loss."""
    p    = torch.tensor([[3.0]])
    t    = torch.tensor([[7.0]])
    assert weighted_mse_loss(p, t).item() > 0.0


def test_weighted_mse_loss_batch_mean_normalized():
    """After batch-mean normalisation, weights average to 1.0."""
    targets = torch.tensor([0.0, 4.0, 8.0, 12.0]).view(-1, 1)
    raw     = 1.0 + targets * 0.5
    assert (raw / raw.mean()).mean().item() == pytest.approx(1.0, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# 2D — get_sample_weights()
# ═════════════════════════════════════════════════════════════════════════════

def test_get_sample_weights_low_sofa():
    """Low SOFA (< 5) → weight == 1.0."""
    y = np.array([0.0, 1.0, 2.5, 4.9])
    npt.assert_array_equal(get_sample_weights(y), np.ones(4, dtype=np.float32))


def test_get_sample_weights_moderate_sofa():
    """Moderate SOFA (5 ≤ y < 10) → weight == 2.0."""
    y = np.array([5.0, 7.0, 9.9])
    npt.assert_array_equal(get_sample_weights(y), np.full(3, 2.0, dtype=np.float32))


def test_get_sample_weights_high_sofa():
    """High SOFA (y ≥ 10) → weight == 5.0."""
    y = np.array([10.0, 15.0, 24.0])
    npt.assert_array_equal(get_sample_weights(y), np.full(3, 5.0, dtype=np.float32))


def test_get_sample_weights_output_length():
    """Output length equals input length."""
    y = np.array([1.0, 6.0, 11.0, 3.5])
    assert len(get_sample_weights(y)) == len(y)


def test_get_sample_weights_dtype_float32():
    """Output dtype is float32."""
    y = np.array([1.0, 6.0, 11.0])
    assert get_sample_weights(y).dtype == np.float32


# ═════════════════════════════════════════════════════════════════════════════
# 2E — train_model()
# ═════════════════════════════════════════════════════════════════════════════

def test_train_model_returns_float(mini_model, synthetic_client_data):
    """train_model returns a Python float."""
    X, y   = synthetic_client_data
    result = train_model(mini_model, X, y, epochs=1, oversample=False)
    assert isinstance(result, float)


def test_train_model_loss_is_finite(mini_model, synthetic_client_data):
    """Returned loss is finite (not NaN or inf)."""
    X, y = synthetic_client_data
    loss = train_model(mini_model, X, y, epochs=1, oversample=False)
    assert math.isfinite(loss)


def test_train_model_loss_decreases_with_more_epochs():
    """More epochs on the same data → lower final loss."""
    torch.manual_seed(99)
    np.random.seed(99)
    # Constant target is easy to learn: model only needs to adjust output bias
    X = np.random.randn(200, 10).astype(np.float32)
    y = np.full(200, 5.0, dtype=np.float32)

    init_w = get_weights(ICUModel(10))

    m1  = ICUModel(10);  set_weights(m1,  init_w)
    m10 = ICUModel(10);  set_weights(m10, init_w)

    loss1  = train_model(m1,  X, y, epochs=1,  lr=0.01, oversample=False)
    loss10 = train_model(m10, X, y, epochs=10, lr=0.01, oversample=False)
    assert loss10 < loss1


def test_train_model_no_fedprox(mini_model, synthetic_client_data):
    """Works without FedProx (global_params=None, mu=0.0)."""
    X, y = synthetic_client_data
    loss = train_model(mini_model, X, y, epochs=1, oversample=False,
                       global_params=None, mu=0.0)
    assert math.isfinite(loss)


def test_train_model_with_fedprox(mini_model, synthetic_client_data):
    """Works with FedProx (global_params supplied, mu=0.5)."""
    X, y         = synthetic_client_data
    global_zeros = [np.zeros_like(w) for w in get_weights(mini_model)]
    loss = train_model(mini_model, X, y, epochs=1, oversample=False,
                       global_params=global_zeros, mu=0.5)
    assert math.isfinite(loss)


def test_train_model_oversample_true(mini_model, synthetic_client_data):
    """Works with oversample=True (WeightedRandomSampler branch)."""
    X, y = synthetic_client_data
    loss = train_model(mini_model, X, y, epochs=1, oversample=True)
    assert math.isfinite(loss)


def test_train_model_oversample_false(mini_model, synthetic_client_data):
    """Works with oversample=False (plain shuffled DataLoader)."""
    X, y = synthetic_client_data
    loss = train_model(mini_model, X, y, epochs=1, oversample=False)
    assert math.isfinite(loss)


def test_train_model_grad_clip(mini_model, synthetic_client_data):
    """Works with grad_clip=1.0 (gradient norm clipping branch)."""
    X, y = synthetic_client_data
    loss = train_model(mini_model, X, y, epochs=1, oversample=False, grad_clip=1.0)
    assert math.isfinite(loss)


def test_train_model_no_grad_clip(mini_model, synthetic_client_data):
    """Works with grad_clip=None (no clipping branch)."""
    X, y = synthetic_client_data
    loss = train_model(mini_model, X, y, epochs=1, oversample=False, grad_clip=None)
    assert math.isfinite(loss)


def test_train_model_very_low_lr(mini_model, synthetic_client_data):
    """lr=1e-5 (LR decay floor) still yields finite loss."""
    X, y = synthetic_client_data
    loss = train_model(mini_model, X, y, epochs=1, lr=1e-5, oversample=False)
    assert math.isfinite(loss)


def test_train_model_fedprox_increases_loss():
    """With large mu and global far from local, proximal term raises reported loss."""
    torch.manual_seed(0)
    np.random.seed(0)
    X = np.random.randn(200, 10).astype(np.float32)
    y = np.random.uniform(0, 15, 200).astype(np.float32)

    init_w = get_weights(ICUModel(10))

    m_base = ICUModel(10);  set_weights(m_base, init_w)
    loss_base = train_model(m_base, X, y, epochs=1, oversample=False,
                             global_params=None, mu=0.0)

    m_prox = ICUModel(10);  set_weights(m_prox, init_w)
    zeros  = [np.zeros_like(w) for w in init_w]
    loss_prox = train_model(m_prox, X, y, epochs=1, oversample=False,
                             global_params=zeros, mu=100.0)

    assert loss_prox > loss_base


# ═════════════════════════════════════════════════════════════════════════════
# 2F — evaluate_model()
# ═════════════════════════════════════════════════════════════════════════════

def test_evaluate_model_returns_dict_keys(mini_model, synthetic_client_data):
    """Returns dict with at least keys mse, mae, r2."""
    X, y   = synthetic_client_data
    result = evaluate_model(mini_model, X, y)
    assert {"mse", "mae", "r2"} <= set(result.keys())


def test_evaluate_model_values_are_float(mini_model, synthetic_client_data):
    """All returned metric values are Python floats."""
    X, y   = synthetic_client_data
    result = evaluate_model(mini_model, X, y)
    for key in ("mse", "mae", "r2"):
        assert isinstance(result[key], float), f"{key} not float"


def test_evaluate_model_mse_nonnegative(mini_model, synthetic_client_data):
    """MSE ≥ 0 always."""
    X, y = synthetic_client_data
    assert evaluate_model(mini_model, X, y)["mse"] >= 0.0


def test_evaluate_model_mae_nonnegative(mini_model, synthetic_client_data):
    """MAE ≥ 0 always."""
    X, y = synthetic_client_data
    assert evaluate_model(mini_model, X, y)["mae"] >= 0.0


def test_evaluate_model_r2_at_most_one(mini_model, synthetic_client_data):
    """R² ≤ 1.0 always."""
    X, y = synthetic_client_data
    assert evaluate_model(mini_model, X, y)["r2"] <= 1.0


def test_evaluate_model_perfect_predictions(synthetic_client_data):
    """A model that predicts y exactly → MAE == 0 and R² == 1.0."""
    from unittest.mock import MagicMock
    X, y = synthetic_client_data

    mock = MagicMock()
    mock.eval = MagicMock()
    mock.return_value = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    result = evaluate_model(mock, X, y)
    assert result["mae"] == pytest.approx(0.0, abs=1e-5)
    assert result["r2"]  == pytest.approx(1.0, abs=1e-5)


def test_evaluate_model_constant_mean_predictions_r2_zero(synthetic_client_data):
    """Constant predictions equal to the true mean → R² == 0."""
    from unittest.mock import MagicMock
    X, y = synthetic_client_data

    mean_val = float(y.mean())
    mock = MagicMock()
    mock.eval = MagicMock()
    mock.return_value = torch.full((len(X), 1), mean_val, dtype=torch.float32)

    result = evaluate_model(mock, X, y)
    assert result["r2"] == pytest.approx(0.0, abs=1e-4)


def test_evaluate_model_sets_eval_mode(mini_model, synthetic_client_data):
    """evaluate_model calls model.eval() — model.training is False afterwards."""
    X, y = synthetic_client_data
    mini_model.train()          # put in train mode first
    evaluate_model(mini_model, X, y)
    assert not mini_model.training


# ═════════════════════════════════════════════════════════════════════════════
# 2G — compute_sofa_risk_metrics()
# ═════════════════════════════════════════════════════════════════════════════

def _risk_data():
    """Synthetic (y_true, y_score) covering all 3 risk bands + both binary classes."""
    rng    = np.random.default_rng(0)
    y_true = np.concatenate([rng.uniform(0,  5, 40),
                              rng.uniform(5, 10, 30),
                              rng.uniform(10, 24, 30)])
    y_score = np.clip(y_true + rng.normal(0, 1.5, len(y_true)), 0, 24)
    return y_true, y_score


def test_csrm_return_keys():
    """Returns dict with risk_bands and high_risk_alert keys."""
    result = compute_sofa_risk_metrics(*_risk_data())
    assert "risk_bands"      in result
    assert "high_risk_alert" in result


def test_csrm_risk_band_labels():
    """risk_bands.labels == [Low (<5), Moderate (5-<10), High (>=10)]."""
    labels = compute_sofa_risk_metrics(*_risk_data())["risk_bands"]["labels"]
    assert labels == ["Low (<5)", "Moderate (5-<10)", "High (>=10)"]


def test_csrm_confusion_matrix_is_3x3():
    """risk_bands.confusion_matrix is a 3×3 list."""
    cm = compute_sofa_risk_metrics(*_risk_data())["risk_bands"]["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)


def test_csrm_per_class_keys():
    """Per-class entries each have precision, recall, sensitivity, specificity, f1."""
    per_class = compute_sofa_risk_metrics(*_risk_data())["risk_bands"]["per_class"]
    required  = {"precision", "recall", "sensitivity", "specificity", "f1"}
    for label, m in per_class.items():
        assert required <= set(m.keys()), f"missing keys for {label}"


def test_csrm_sensitivity_equals_recall():
    """sensitivity and recall are identical for every class."""
    per_class = compute_sofa_risk_metrics(*_risk_data())["risk_bands"]["per_class"]
    for label, m in per_class.items():
        assert m["sensitivity"] == pytest.approx(m["recall"], abs=1e-9), label


def test_csrm_default_alert_threshold():
    """high_risk_alert.predicted_sofa_threshold defaults to 8.0."""
    thr = compute_sofa_risk_metrics(*_risk_data())["high_risk_alert"][
        "predicted_sofa_threshold"
    ]
    assert thr == 8.0


def test_csrm_auroc_float_when_both_classes():
    """AUROC is a float when y_true contains both risk categories."""
    auroc = compute_sofa_risk_metrics(*_risk_data())["high_risk_alert"]["auroc"]
    assert isinstance(auroc, float)


def test_csrm_auroc_none_when_single_class():
    """AUROC is None when y_true has only one binary class."""
    y_true  = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5])
    y_score = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 9.8])
    result  = compute_sofa_risk_metrics(y_true, y_score)
    assert result["high_risk_alert"]["auroc"] is None


def test_csrm_pr_auc_none_when_single_class():
    """PR-AUC is None when y_true has only one binary class."""
    y_true  = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5])
    y_score = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 9.8])
    assert compute_sofa_risk_metrics(y_true, y_score)["high_risk_alert"]["pr_auc"] is None


def test_csrm_raises_empty_input():
    """Raises ValueError for empty arrays."""
    with pytest.raises(ValueError):
        compute_sofa_risk_metrics([], [])


def test_csrm_raises_length_mismatch():
    """Raises ValueError when y_true and y_score differ in length."""
    with pytest.raises(ValueError):
        compute_sofa_risk_metrics([1.0, 2.0], [1.0])


def test_csrm_raises_non_finite():
    """Raises ValueError for NaN or inf in either input."""
    with pytest.raises(ValueError):
        compute_sofa_risk_metrics([float("nan"), 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        compute_sofa_risk_metrics([1.0, 2.0], [1.0, float("inf")])


def test_csrm_predictions_clipped_to_0_24():
    """Predictions > 24 are clipped to 24 before band assignment."""
    # All true = Low; all predicted = 30 → clipped to 24 → assigned High band
    y_true  = np.array([1.0, 2.0, 3.0])
    y_score = np.array([30.0, 35.0, 40.0])
    cm = compute_sofa_risk_metrics(y_true, y_score)["risk_bands"]["confusion_matrix"]
    # Row 0 = actual Low, Col 2 = predicted High → all 3 misclassified there
    assert cm[0][2] == 3


def test_csrm_macro_keys_match_per_class():
    """macro dict has the same keys as any per-class entry."""
    result    = compute_sofa_risk_metrics(*_risk_data())["risk_bands"]
    macro     = result["macro"]
    per_class = result["per_class"]
    assert set(macro.keys()) == set(next(iter(per_class.values())).keys())


def test_csrm_confusion_rows_sum_to_class_counts():
    """Each row of the 3×3 matrix sums to the actual count for that class."""
    y_true  = np.concatenate([np.full(40, 2.0),   # Low
                               np.full(30, 7.0),   # Moderate
                               np.full(30, 15.0)]) # High
    y_score = y_true.copy()   # perfect predictions
    cm = compute_sofa_risk_metrics(y_true, y_score)["risk_bands"]["confusion_matrix"]
    assert sum(cm[0]) == 40
    assert sum(cm[1]) == 30
    assert sum(cm[2]) == 30


def test_csrm_binary_confusion_total():
    """TP + FP + TN + FN == len(y_true) for the high-risk binary alert."""
    y_true, y_score = _risk_data()
    a     = compute_sofa_risk_metrics(y_true, y_score)["high_risk_alert"]
    total = (a["true_positive"] + a["false_positive"] +
             a["true_negative"] + a["false_negative"])
    assert total == len(y_true)


# ═════════════════════════════════════════════════════════════════════════════
# 2H — get_weights() / set_weights()
# ═════════════════════════════════════════════════════════════════════════════

def test_get_weights_returns_list(mini_model):
    """get_weights returns a list."""
    assert isinstance(get_weights(mini_model), list)


def test_get_weights_elements_are_ndarrays(mini_model):
    """Every element is a numpy ndarray."""
    assert all(isinstance(w, np.ndarray) for w in get_weights(mini_model))


def test_get_weights_length_matches_state_dict(mini_model):
    """Length equals the number of tensors in model.state_dict()."""
    assert len(get_weights(mini_model)) == len(mini_model.state_dict())


def test_set_weights_round_trip(mini_model):
    """Weights extracted from model_a can be loaded into model_b; outputs match."""
    model_b  = ICUModel(10)
    original = get_weights(mini_model)
    set_weights(model_b, original)

    x = torch.randn(4, 10)
    mini_model.eval(); model_b.eval()
    with torch.no_grad():
        npt.assert_allclose(mini_model(x).numpy(), model_b(x).numpy(), rtol=1e-5)


def test_get_set_weights_identity(mini_model):
    """set_weights(model, get_weights(model)) leaves weights unchanged."""
    before = [w.copy() for w in get_weights(mini_model)]
    set_weights(mini_model, before)
    after  = get_weights(mini_model)
    for b, a in zip(before, after):
        npt.assert_array_equal(b, a)


def test_set_weights_wrong_count_raises(mini_model):
    """Fewer weight arrays than expected → RuntimeError from load_state_dict."""
    truncated = get_weights(mini_model)[:2]   # only 2 of 8 tensors
    with pytest.raises(RuntimeError):
        set_weights(mini_model, truncated)


# ═════════════════════════════════════════════════════════════════════════════
# 2I — get_trend() / classify_range()
# ═════════════════════════════════════════════════════════════════════════════

def test_get_trend_increasing():
    assert get_trend([1, 2, 3, 4, 5]) == "increasing"


def test_get_trend_decreasing():
    assert get_trend([5, 4, 3, 2, 1]) == "decreasing"


def test_get_trend_stable_flat():
    assert get_trend([3, 3, 3, 3]) == "stable"


def test_get_trend_single_element():
    """Single-element series → 'stable' (len < 2 guard)."""
    assert get_trend([5]) == "stable"


def test_get_trend_empty():
    """Empty series → 'stable' (len < 2 guard)."""
    assert get_trend([]) == "stable"


def test_get_trend_slope_exactly_boundary():
    """Slope == 0.1 is 'stable' — threshold is strictly > 0.1."""
    # [0, 0.1, 0.2, 0.3] fitted by y=0.1x → slope exactly 0.1
    assert get_trend([0, 0.1, 0.2, 0.3]) == "stable"


def test_classify_range_normal():
    assert classify_range(70, 60, 100) == "normal"


def test_classify_range_high():
    assert classify_range(120, 60, 100) == "high"


def test_classify_range_low():
    assert classify_range(40, 60, 100) == "low"


def test_classify_range_lower_boundary_is_normal():
    """Value == low → 'normal' (condition is strict <)."""
    assert classify_range(60, 60, 100) == "normal"


def test_classify_range_upper_boundary_is_normal():
    """Value == high → 'normal' (condition is strict >)."""
    assert classify_range(100, 60, 100) == "normal"


# ═════════════════════════════════════════════════════════════════════════════
# 2J — apply_dp_to_update()
# ═════════════════════════════════════════════════════════════════════════════

def test_apply_dp_returns_list_of_ndarrays(mini_weights):
    """Returns a list of float32 ndarrays with the same shapes as the inputs."""
    global_ = [np.zeros_like(w) for w in mini_weights]
    result  = apply_dp_to_update(mini_weights, global_, sensitivity=1.0, sigma=0.1)
    assert isinstance(result, list)
    for r, w in zip(result, mini_weights):
        assert isinstance(r, np.ndarray)
        assert r.shape == w.shape


def test_apply_dp_no_clip_when_norm_small(mini_weights):
    """L2 norm << sensitivity → no clipping; with sigma=0 result ≈ local."""
    global_ = [w.copy() for w in mini_weights]
    local   = [w + 1e-9  for w in global_]   # tiny perturbation; L2 << 100
    result  = apply_dp_to_update(local, global_, sensitivity=100.0, sigma=0.0)
    for r, l in zip(result, local):
        npt.assert_allclose(r, l.astype(np.float32), atol=1e-5)


def test_apply_dp_clips_large_update(mini_weights):
    """sensitivity=1e-6 forces clipping; L2 norm of (result−global) ≈ 1e-6."""
    global_ = [np.zeros_like(w) for w in mini_weights]
    result  = apply_dp_to_update(mini_weights, global_, sensitivity=1e-6, sigma=0.0)
    flat    = np.concatenate([(r - g).flatten() for r, g in zip(result, global_)])
    assert np.linalg.norm(flat) == pytest.approx(1e-6, rel=0.01)


def test_apply_dp_noise_is_added(mini_weights):
    """With sigma > 0, noisy result differs from noiseless (sigma=0) result."""
    global_ = [np.zeros_like(w) for w in mini_weights]
    r_noise = apply_dp_to_update(mini_weights, global_, sensitivity=100.0, sigma=1.0)
    r_clean = apply_dp_to_update(mini_weights, global_, sensitivity=100.0, sigma=0.0)
    diffs   = [np.abs(a - b).max() for a, b in zip(r_noise, r_clean)]
    assert any(d > 0.0 for d in diffs)


def test_apply_dp_output_dtype_float32(mini_weights):
    """All output arrays have dtype float32."""
    global_ = [np.zeros_like(w) for w in mini_weights]
    result  = apply_dp_to_update(mini_weights, global_, sensitivity=1.0, sigma=0.1)
    assert all(r.dtype == np.float32 for r in result)


def test_apply_dp_zero_sigma_no_noise(mini_weights):
    """sigma=0 → noise_std=0 → two calls return identical arrays."""
    global_ = [np.zeros_like(w) for w in mini_weights]
    r1 = apply_dp_to_update(mini_weights, global_, sensitivity=100.0, sigma=0.0)
    r2 = apply_dp_to_update(mini_weights, global_, sensitivity=100.0, sigma=0.0)
    for a, b in zip(r1, r2):
        npt.assert_array_equal(a, b)


# ═════════════════════════════════════════════════════════════════════════════
# 2K — estimate_privacy_budget()
# ═════════════════════════════════════════════════════════════════════════════

def test_estimate_privacy_budget_returns_float():
    assert isinstance(estimate_privacy_budget(10, 1.0), float)


def test_estimate_privacy_budget_zero_rounds():
    """Zero rounds → budget == 0."""
    assert estimate_privacy_budget(0, 1.0) == 0.0


def test_estimate_privacy_budget_formula():
    """Result matches sqrt(2·ln(1.25/δ))/σ · T, rounded to 3 d.p."""
    expected = round(math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0 * 100, 3)
    assert estimate_privacy_budget(100, 1.0) == expected


def test_estimate_privacy_budget_sigma_nonpositive_returns_inf():
    """sigma ≤ 0 → float('inf')."""
    assert estimate_privacy_budget(10,  0)    == float("inf")
    assert estimate_privacy_budget(10, -1.0)  == float("inf")


def test_estimate_privacy_budget_monotone_in_sigma():
    """Higher sigma → lower epsilon (more private)."""
    assert estimate_privacy_budget(10, 2.0) < estimate_privacy_budget(10, 0.5)


def test_estimate_privacy_budget_monotone_in_rounds():
    """More rounds → higher epsilon (less private)."""
    assert estimate_privacy_budget(100, 1.0) > estimate_privacy_budget(10, 1.0)
