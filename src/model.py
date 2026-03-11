# models.py
# Reference model components for CausalStream prototype.
# All comments and identifiers are in English.

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvFrontEnd(nn.Module):
    """
    Causal 1D convolutional front-end.
    Each convolution adds left context only by trimming right-side padding
    after applying a dilated conv so that the module is causal.
    Input shape: (batch, channels, time)
    Output shape: (batch, hidden, time)
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 kernel_size: int = 5, dilations: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.convs = nn.ModuleList()
        cur = in_channels
        for d in dilations:
            conv = nn.Conv1d(cur, hidden_channels, kernel_size,
                             padding=(kernel_size - 1) * d, dilation=d)
            self.convs.append(conv)
            cur = hidden_channels
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = x
        for conv in self.convs:
            out = conv(out)
            # remove right-padding introduced to preserve causality
            kernel = conv.kernel_size[0]
            dilation = conv.dilation[0]
            remove = (kernel - 1) * dilation
            if remove > 0:
                out = out[:, :, :-remove]
            out = self.activation(out)
        return out


class SelectiveStateSpace(nn.Module):
    """
    A lightweight state-space-like recurrent module for streaming.
    This is a prototype: a simple parametric recurrent update
    is implemented for clarity and ease of replacement by a
    performance-optimized Mamba-2 implementation later.
    Input: (B, T, dim_in)
    Output: (B, T, hidden_dim) and final state (B, hidden_dim)
    """

    def __init__(self, dim_in: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        # small linear transition and input projection
        self.register_parameter("A", nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01))
        self.input_proj = nn.Linear(dim_in, hidden_dim)
        self.state_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, init_state: Optional[torch.Tensor] = None):
        # x: (B, T, dim_in)
        B, T, _ = x.shape
        if init_state is None:
            h = x.new_zeros((B, self.hidden_dim))
        else:
            h = init_state
        outs = []
        for t in range(T):
            xt = x[:, t, :]  # (B, dim_in)
            h = torch.matmul(h, self.A.t()) + self.input_proj(xt) + self.state_bias
            h = self.act(h)
            outs.append(h.unsqueeze(1))
        out = torch.cat(outs, dim=1)  # (B, T, hidden_dim)
        return self.out_proj(out), h  # (B, T, hidden_dim), (B, hidden_dim)


class MetaNet(nn.Module):
    """
    Small MLP that produces a confounder embedding from a short summary.
    """

    def __init__(self, input_dim: int, confound_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, confound_dim)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class FrontDoorAdjust(nn.Module):
    """
    Maps estimated confounder embedding to the same space as sequence pooled features.
    """

    def __init__(self, confound_dim: int = 64, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(confound_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, c_hat: torch.Tensor) -> torch.Tensor:
        return self.net(c_hat)


class MINE(nn.Module):
    """
    Simple neural estimator of mutual information (Donsker-Varadhan variant).
    Returns an estimate and the two networks' outputs for debugging.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x,y: (B, dim)
        joint = torch.cat([x, y], dim=-1)
        t = self.net(joint)  # (B,1)
        # marginal: shuffle y
        idx = torch.randperm(y.size(0), device=y.device)
        y_shuffled = y[idx]
        marg = self.net(torch.cat([x, y_shuffled], dim=-1))
        # DV estimate: mean(t) - log(mean(exp(marg)))
        est = torch.mean(t) - torch.log(torch.mean(torch.exp(marg)) + 1e-8)
        return est, t, marg


class CausalStream(nn.Module):
    """
    High-level assembly of the CausalStream prototype:
      - causal conv front-end that produces per-window features
      - selective state-space operator
      - MetaNet + front-door adjustment for confounder correction
      - classifier head
    """

    def __init__(self,
                 in_channels: int = 6,
                 frontend_hidden: int = 128,
                 encoder_dim: int = 128,
                 ssm_hidden: int = 256,
                 confound_dim: int = 64,
                 num_classes: int = 10):
        super().__init__()
        self.frontend = CausalConvFrontEnd(in_channels, hidden_channels=frontend_hidden)
        # project conv hidden to encoder dim
        self.encoder_proj = nn.Linear(frontend_hidden, encoder_dim)
        self.ssm = SelectiveStateSpace(dim_in=encoder_dim, hidden_dim=ssm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(ssm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.metanet = MetaNet(input_dim=2 * in_channels, confound_dim=confound_dim)
        self.frontdoor = FrontDoorAdjust(confound_dim=confound_dim, out_dim=ssm_hidden)
        self.mine = MINE(x_dim=ssm_hidden, y_dim=ssm_hidden, hidden=128)

    def forward(self, M: torch.Tensor):
        """
        M: windowed input matrix shaped (B, D, T)
        returns:
          logits (B, num_classes),
          pooled sequence features (B, ssm_hidden),
          corrected pooled features (B, ssm_hidden),
          confound_est (B, confound_dim)
        """
        if not torch.is_tensor(M):
            M = torch.tensor(M, dtype=torch.float32)
        B, D, T = M.shape
        # frontend expects (B, C, T)
        feat = self.frontend(M)                # (B, frontend_hidden, T)
        feat = feat.permute(0, 2, 1).contiguous()  # (B, T, frontend_hidden)
        feat = self.encoder_proj(feat)         # (B, T, encoder_dim)
        ssm_out, state = self.ssm(feat)        # ssm_out: (B, T, ssm_hidden)
        pooled = ssm_out.mean(dim=1)           # (B, ssm_hidden) average pooling
        logits = self.classifier(pooled)       # (B, num_classes)
        # confounder estimate: simple summary from raw windows (mean/std across time)
        s_summary = torch.cat([M.mean(dim=2), M.std(dim=2)], dim=1)  # (B, 2D)
        conf_hat = self.metanet(s_summary)     # (B, confound_dim)
        adj = self.frontdoor(conf_hat)         # (B, ssm_hidden)
        corrected = pooled - adj               # (B, ssm_hidden)
        return logits, pooled, corrected, conf_hat