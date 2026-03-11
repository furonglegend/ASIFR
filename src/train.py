# train.py
# Training loop for the CausalStream prototype. English comments only.

import argparse
import os
from typing import List
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SyntheticWFTraceDataset, apply_defense_padding, windowize_trace
from .models import CausalStream
from .utils import set_seed, AdaptiveNormalizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def info_nce_loss(z_q: torch.Tensor, z_k: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Simple InfoNCE between two sets of vectors.
    z_q, z_k: (B, dim)
    """
    z_q = F.normalize(z_q, dim=-1)
    z_k = F.normalize(z_k, dim=-1)
    logits = z_q @ z_k.t() / temperature  # (B, B)
    labels = torch.arange(z_q.size(0), device=z_q.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def collate_batch(batch):
    # batch is list of (trace, label)
    # return lists to process in training loop
    traces = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    return traces, labels


def train_one_epoch(model: CausalStream, dataloader: DataLoader, optimizer, device: torch.device, cfg: dict):
    model.train()
    loss_acc = {"total": 0.0, "ce": 0.0, "contrast": 0.0, "cf": 0.0}
    count = 0
    for traces, labels in tqdm(dataloader, desc="train"):
        B = len(traces)
        norm = AdaptiveNormalizer(channels=cfg["model"]["in_channels"],
                                  alpha_mu=cfg.get("alpha_mu", 0.99),
                                  alpha_s=cfg.get("alpha_s", 0.995))
        M_batch = []
        for trace in traces:
            # simple augmentation: defense padding simulation
            trace_aug = apply_defense_padding(trace, pad_prob=0.02)
            M = windowize_trace(trace_aug, w_ms=cfg["data"]["w_ms"],
                                seq_windows=cfg["data"]["seq_windows"],
                                n_channels=cfg["model"]["in_channels"])
            # normalize each column causally
            for t in range(M.shape[1]):
                M[:, t] = norm.normalize_column(M[:, t])
            M_batch.append(M)
        M_batch = np.stack(M_batch, axis=0)  # (B, D, T)
        M_tensor = torch.tensor(M_batch, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        optimizer.zero_grad()
        logits, pooled, corrected, conf_hat = model(M_tensor)
        ce = F.cross_entropy(logits, labels_t)
        # contrastive: early vs full pooled
        mid = max(1, M_tensor.shape[2] // 4)
        with torch.no_grad():
            logits_early, pooled_early, _, _ = model(M_tensor[:, :, :mid])
        contr = info_nce_loss(pooled_early.detach(), pooled.detach(), temperature=cfg.get("contrast_tau", 0.1))
        # counterfactual regularizer (simple noisy perturbation proxy for demo)
        noisy = pooled + 0.02 * torch.randn_like(pooled)
        cf = F.mse_loss(pooled, noisy)
        loss = ce + cfg.get("lambda_c", 0.5) * contr + cfg.get("lambda_cf", 0.1) * cf
        loss.backward()
        optimizer.step()
        loss_acc["total"] += loss.item()
        loss_acc["ce"] += ce.item()
        loss_acc["contrast"] += contr.item()
        loss_acc["cf"] += cf.item()
        count += 1
        # keep demo epochs short if dataset is large
        if count >= cfg.get("demo_batch_limit", 20):
            break
    for k in loss_acc:
        loss_acc[k] = loss_acc[k] / max(1, count)
    return loss_acc


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    ds = SyntheticWFTraceDataset(n_sites=cfg["data"]["n_sites"], traces_per_site=cfg["data"]["traces_per_site"],
                                 seed=cfg.get("seed", 42))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_batch)
    model = CausalStream(in_channels=cfg["model"]["in_channels"],
                         frontend_hidden=cfg["model"]["frontend_hidden"],
                         encoder_dim=cfg.get("encoder_dim", 128),
                         ssm_hidden=cfg["model"]["ssm_hidden"],
                         confound_dim=cfg["model"]["confound_dim"],
                         num_classes=cfg["model"]["num_classes"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=5e-2)
    save_dir = cfg["logging"].get("save_dir", "runs/demo")
    os.makedirs(save_dir, exist_ok=True)
    epochs = cfg["train"].get("epochs", 3)
    for ep in range(1, epochs + 1):
        stats = train_one_epoch(model, dl, optimizer, device, cfg)
        print(f"[Epoch {ep}] stats: {stats}")
        # save checkpoint each epoch (small demo filename)
        torch.save(model.state_dict(), os.path.join(save_dir, f"causalstream_epoch{ep}.pt"))
    print("Training finished. Models saved to:", save_dir)


if __name__ == "__main__":
    main()