# data.py
# Synthetic dataset utilities and windowing helpers used by CausalStream prototype.

from typing import List, Tuple, Optional
import numpy as np
from torch.utils.data import Dataset


class SyntheticWFTraceDataset(Dataset):
    """
    Synthetic dataset that emits packet-level traces and labels.
    Each trace is a numpy array with shape (N_packets, 2): [time_ms, signed_size].
    """

    def __init__(self, n_sites: int = 10, traces_per_site: int = 200, seed: int = 42):
        self.samples: List[np.ndarray] = []
        self.labels: List[int] = []
        np.random.seed(seed)
        for s in range(n_sites):
            for _ in range(traces_per_site):
                n_pkts = np.random.randint(60, 500)
                # inter-arrival times exponential to simulate bursts and silences
                iat = np.random.exponential(scale=40.0, size=n_pkts).astype(np.int64)
                times = np.cumsum(iat)
                # packet sizes chosen from a small set, sign indicates direction
                sizes = np.random.choice([60, 150, 500, 1200], size=n_pkts, p=[0.4, 0.3, 0.2, 0.1])
                dirs = np.random.choice([-1, 1], size=n_pkts, p=[0.45, 0.55])
                signed_sizes = dirs * sizes
                trace = np.stack([times, signed_sizes], axis=1)
                self.samples.append(trace)
                self.labels.append(s)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.samples[index], int(self.labels[index])


def apply_defense_padding(trace: np.ndarray, pad_prob: float = 0.02,
                          max_pad_size: int = 600) -> np.ndarray:
    """
    Insert synthetic padding packets into a packet-level trace.
    Padding packets carry size 0 (or small values) and appear near existing times.
    """

    if trace is None or trace.shape[0] == 0:
        return trace
    out = []
    for t, s in trace:
        out.append((int(t), int(s)))
        if np.random.rand() < pad_prob:
            jitter = int(np.random.randint(1, 20))
            pad_size = int(np.random.randint(0, max_pad_size))
            out.append((int(t + jitter), int(pad_size)))
    arr = np.array(out, dtype=np.int64)
    arr = arr[arr[:, 0].argsort()]
    return arr


def windowize_trace(trace: np.ndarray,
                    w_ms: int = 44,
                    seq_windows: int = 128,
                    n_channels: int = 6) -> np.ndarray:
    """
    Convert packet-level trace to a windowed matrix M with shape (D, N):
      channels (example): [out_count, in_count, out_bytes, in_bytes, mean_isi, pkt_count]
    The implementation uses simple statistics per window; this function returns float32.
    """

    M = np.zeros((n_channels, seq_windows), dtype=np.float32)
    if trace is None or trace.shape[0] == 0:
        return M
    t0 = int(trace[0, 0])
    times = (trace[:, 0] - t0).astype(np.int64)
    idx = (times // w_ms).astype(np.int64)
    idx = np.clip(idx, 0, seq_windows - 1)
    for i, (t, s) in enumerate(trace):
        w = idx[i]
        val = int(s)
        if val > 0:
            M[0, w] += 1
            M[2, w] += val
        elif val < 0:
            M[1, w] += 1
            M[3, w] += abs(val)
        M[5, w] += 1
    # mean inter-packet interval proxy per window (avoid division by zero)
    counts = M[5]
    M[4] = w_ms / (counts + 1.0)
    return M