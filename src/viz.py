# viz.py
# Simple visualization helpers (loss/acc plots, embedding scatter, activation heatmap).
# Designed for demo usage with small sample sizes and no heavy t-SNE defaults.

import time
import os
from typing import Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.rcParams.update({"axes.grid": False})  # disable global gridlines for clean plots


def _timestamped_path(base: str) -> str:
    ts = int(time.time())
    safe = base.replace(" ", "_")
    return f"{safe}_{ts}.png"


def save_figure(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss_accuracy(epochs: Sequence[int], train_loss: Sequence[float], val_loss: Optional[Sequence[float]],
                       train_acc: Sequence[float], val_acc: Optional[Sequence[float]], out_prefix: str = "train"):
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, train_loss, label="train loss", linewidth=1.4)
    if val_loss is not None:
        ax1.plot(epochs, val_loss, label="val loss", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(fontsize='small')
    p1 = _timestamped_path(out_prefix + "_loss")
    save_figure(fig1, p1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(epochs, train_acc, label="train acc", linewidth=1.4)
    if val_acc is not None:
        ax2.plot(epochs, val_acc, label="val acc", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(fontsize='small')
    p2 = _timestamped_path(out_prefix + "_acc")
    save_figure(fig2, p2)
    return p1, p2


def plot_embedding_2d(emb: np.ndarray, labels: Sequence[int], method: str = "tsne", n_iter: int = 250):
    N = emb.shape[0]
    # if N large, downsample for quick demo
    if N > 500:
        idx = np.random.choice(N, 500, replace=False)
        emb = emb[idx]
        labels = np.asarray(labels)[idx]
    if method == "tsne":
        ts = TSNE(n_components=2, init="pca", learning_rate="auto", n_iter=n_iter, random_state=42)
        emb2 = ts.fit_transform(emb)
    else:
        raise ValueError("Unsupported method for demo")
    fig, ax = plt.subplots(figsize=(6, 5))
    labs = np.unique(labels)
    for l in labs:
        mask = np.asarray(labels) == l
        ax.scatter(emb2[mask, 0], emb2[mask, 1], s=14, label=str(l), alpha=0.85)
    ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2"); ax.legend(fontsize='small')
    path = _timestamped_path("embedding2d")
    save_figure(fig, path)
    return path


def plot_activation_heatmap(activations: np.ndarray):
    # activations: (time, channels) or (channels, time)
    arr = np.asarray(activations)
    if arr.ndim != 2:
        raise ValueError("activations must be 2D array")
    if arr.shape[0] < arr.shape[1]:
        mat = arr
    else:
        mat = arr.T
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_xlabel("Time"); ax.set_ylabel("Channel")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Activation")
    path = _timestamped_path("activation_heatmap")
    save_figure(fig, path)
    return path