# eval.py
# Simple evaluation script that loads a saved checkpoint and prints a few predictions.

import argparse
import yaml
import os
import torch
import numpy as np

from .data import SyntheticWFTraceDataset, windowize_trace
from .models import CausalStream
from .utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint (optional)")
    return p.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    ds = SyntheticWFTraceDataset(n_sites=cfg["data"]["n_sites"], traces_per_site=8)
    model = CausalStream(in_channels=cfg["model"]["in_channels"],
                         frontend_hidden=cfg["model"]["frontend_hidden"],
                         encoder_dim=cfg.get("encoder_dim", 128),
                         ssm_hidden=cfg["model"]["ssm_hidden"],
                         confound_dim=cfg["model"]["confound_dim"],
                         num_classes=cfg["model"]["num_classes"])
    # load checkpoint if provided, otherwise try default saved file
    ckpt = args.ckpt or os.path.join(cfg["logging"].get("save_dir", "runs/demo"), "causalstream_epoch1.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("Loaded checkpoint:", ckpt)
    else:
        print("No checkpoint found at", ckpt, "-> running with randomly initialized weights")
    model.to(device).eval()
    # run a small evaluation: print top-3 predicted labels and softmax probabilities
    for i in range(5):
        trace, label = ds[i]
        M = windowize_trace(trace, w_ms=cfg["data"]["w_ms"], seq_windows=cfg["data"]["seq_windows"],
                            n_channels=cfg["model"]["in_channels"])
        with torch.no_grad():
            logits, pooled, corrected, conf_hat = model(torch.tensor(M[None, :, :], dtype=torch.float32, device=device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        topk = list(np.argsort(probs)[-3:][::-1])
        print(f"Sample {i}: true={label}, top3_pred={topk}, probs={probs[topk]}")
    print("Evaluation finished.")


if __name__ == "__main__":
    main()