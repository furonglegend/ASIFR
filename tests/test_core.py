# tests/test_core.py
# Unit tests for core CausalStream prototype components.
# All comments and identifiers are in English.

import numpy as np
import torch
import pytest

# Import modules from the causalstream package (assumes package exists in PYTHONPATH)
from causalstream.data import SyntheticWFTraceDataset, apply_defense_padding, windowize_trace
from causalstream.models import CausalStream
from causalstream.utils import AdaptiveNormalizer, set_seed


def test_synthetic_dataset_basic_properties():
    """
    Test that the synthetic dataset constructs the expected number of samples
    and that each returned trace has the expected 2-column shape (time, signed_size).
    """
    set_seed(123)
    n_sites = 3
    traces_per_site = 4
    ds = SyntheticWFTraceDataset(n_sites=n_sites, traces_per_site=traces_per_site, seed=123)
    expected_len = n_sites * traces_per_site
    assert len(ds) == expected_len, "Dataset length mismatch"
    trace, label = ds[0]
    # trace must be a 2D numpy array with two columns: time and signed_size
    assert isinstance(trace, np.ndarray), "trace must be numpy.ndarray"
    assert trace.ndim == 2 and trace.shape[1] == 2, "trace must have shape (N_packets, 2)"
    assert isinstance(label, int), "label must be integer"


def test_windowize_and_normalizer_produces_valid_matrix():
    """
    Test windowize_trace and AdaptiveNormalizer produce finite-valued matrices
    and normalized columns without NaNs or infinities.
    """
    set_seed(1)
    ds = SyntheticWFTraceDataset(n_sites=1, traces_per_site=1, seed=1)
    trace, _ = ds[0]
    # apply a sample defense padding augmentation
    padded = apply_defense_padding(trace, pad_prob=0.05)
    # windowize
    D = 6
    seq_windows = 64
    M = windowize_trace(padded, w_ms=44, seq_windows=seq_windows, n_channels=D)
    assert M.shape == (D, seq_windows), "window matrix shape mismatch"
    assert np.isfinite(M).all(), "window matrix contains non-finite values"
    # test AdaptiveNormalizer across columns
    norm = AdaptiveNormalizer(channels=D, alpha_mu=0.99, alpha_s=0.995)
    for t in range(M.shape[1]):
        col = M[:, t]
        out = norm.normalize_column(col)
        assert out.shape == (D,), "normalized column shape incorrect"
        assert np.isfinite(out).all(), "normalized column contains non-finite values"


def test_model_forward_output_shapes_and_types():
    """
    Instantiate a small CausalStream model and verify forward pass shapes/types.
    This test uses a small ssm_hidden and reduced num_classes for speed.
    """
    set_seed(7)
    # create a single synthetic trace and windowize it
    ds = SyntheticWFTraceDataset(n_sites=1, traces_per_site=1, seed=7)
    trace, _ = ds[0]
    D = 6
    T = 48
    M = windowize_trace(trace, w_ms=44, seq_windows=T, n_channels=D).astype(np.float32)
    # stack to create a small batch
    B = 2
    M_batch = np.stack([M, M], axis=0)  # shape (B, D, T)
    # instantiate model with matching input channels
    model = CausalStream(in_channels=D, frontend_hidden=32, encoder_dim=32,
                         ssm_hidden=64, confound_dim=16, num_classes=5)
    model.eval()
    with torch.no_grad():
        M_tensor = torch.tensor(M_batch, dtype=torch.float32)
        logits, pooled, corrected, conf_hat = model(M_tensor)
    # check shapes
    assert logits.shape == (B, 5), f"expected logits shape (B,5), got {logits.shape}"
    assert pooled.shape == (B, 64), f"expected pooled shape (B,64), got {pooled.shape}"
    assert corrected.shape == (B, 64), f"expected corrected shape (B,64), got {corrected.shape}"
    assert conf_hat.shape[0] == B and conf_hat.shape[1] == 16, "confounder embedding shape mismatch"
    # check numeric types
    assert logits.dtype == torch.float32, "logits dtype must be float32"


def test_backward_step_runs_and_creates_gradients():
    """
    Basic smoke test to ensure a training step runs and gradients are produced.
    We compute a simple cross-entropy loss for a dummy label and backpropagate.
    """
    set_seed(11)
    D = 6
    T = 40
    # prepare dummy batch (B=3)
    B = 3
    # random synthetic windows
    M_batch = np.random.randn(B, D, T).astype(np.float32) * 0.1
    labels = np.array([0, 1, 2], dtype=np.int64)
    model = CausalStream(in_channels=D, frontend_hidden=32, encoder_dim=32,
                         ssm_hidden=48, confound_dim=8, num_classes=3)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    M_tensor = torch.tensor(M_batch, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    optimizer.zero_grad()
    logits, pooled, corrected, conf_hat = model(M_tensor)
    loss = torch.nn.functional.cross_entropy(logits, labels_t)
    loss.backward()
    optimizer.step()
    # check that at least one parameter has gradient populated
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found after backward()"
    # gradients should be finite
    all_finite = all(torch.isfinite(g).all().item() for g in grads)
    assert all_finite, "Some gradients are non-finite"