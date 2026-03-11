# utils.py
# Utility helpers for training, normalization, and seeding.

from typing import Sequence
import numpy as np
import torch
import random


def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility across numpy, torch and random.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AdaptiveNormalizer:
    """
    Per-channel exponential moving average normalizer with gating blend.
    This mirrors the description in Methodology (EMA for mean/variance, blend with log1p).
    Works on column vectors of shape (channels,).
    """

    def __init__(self, channels: int, alpha_mu: float = 0.99, alpha_s: float = 0.995, eps: float = 1e-6):
        self.channels = channels
        self.alpha_mu = float(alpha_mu)
        self.alpha_s = float(alpha_s)
        self.eps = float(eps)
        self.mu = np.zeros((channels,), dtype=np.float32)
        self.s = np.ones((channels,), dtype=np.float32)

    def normalize_column(self, col: Sequence[float]) -> np.ndarray:
        """
        Update EMAs and return blended normalized column for a single time window.
        Input: col of shape (channels,)
        Output: numpy array (channels,) float32
        """
        col = np.asarray(col, dtype=np.float32)
        self.mu = self.alpha_mu * self.mu + (1.0 - self.alpha_mu) * col
        self.s = self.alpha_s * self.s + (1.0 - self.alpha_s) * ((col - self.mu) ** 2)
        sigma = np.sqrt(self.s + self.eps)
        z = (col - self.mu) / sigma
        log = np.log1p(np.abs(col))
        # simple gating: prefer z-score when values are stable, otherwise log
        gate = 1.0 / (1.0 + np.exp(- (np.abs(col) - 1.0) * 0.5))
        blended = gate * z + (1.0 - gate) * log
        return blended.astype(np.float32)