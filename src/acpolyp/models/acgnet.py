"""ACGNet: backbone + active-contour-guided pooling head (Phase XI steps 43-46, Phase XIII step 53).

Forward pass. Let F = backbone(x) in R^{B x C x h x w}, M in [0,1]^{B x 1 x h x w} the contour region
(area-downsampled), V the valid-FOV mask, and s in R^{d} the standardised contour/hand-crafted features.

    g     = LN( sum_p V F / sum_p V )                         global descriptor
    f_in  = LN( sum_p M F / (sum_p M + eps) )                 inside the active contour
    f_out = LN( sum_p (V - M)+ F / (sum_p (V - M)+ + eps) )   outside, still inside the FOV
    e     = GELU( W_2 GELU( W_1 s ) )                          contour feature embedding
    z     = Linear( Dropout( [g, f_in, f_out, e] ) )          logits

Ablation modes (`mode`):
  none        z = Linear(Dropout(g))
  ac_pool     [g, f_in, f_out]
  ac_feat     [g, e]
  acg         [g, f_in, f_out, e]                  (full model)
Control masks (shuffle / ellipse) are selected at the data level, not here.
Regions with (almost) no mass fall back to g, so the head is defined for every input.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("none", "ac_pool", "ac_feat", "acg", "gfeat")


class ACGNet(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny.fb_in22k_ft_in1k",
        n_classes: int = 8,
        mode: str = "none",
        n_feats: int = 0,
        pretrained: bool = True,
        drop_path: float = 0.1,
        head_dropout: float = 0.2,
        feat_dim: int = 128,
    ):
        super().__init__()
        assert mode in MODES, mode
        self.mode = mode
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="", drop_path_rate=drop_path
        )
        c = self.backbone.num_features
        self.c = c
        self.norm_g, self.norm_in, self.norm_out = nn.LayerNorm(c), nn.LayerNorm(c), nn.LayerNorm(c)
        self.use_pool = mode in ("ac_pool", "acg")
        self.use_feat = mode in ("ac_feat", "acg", "gfeat")
        if self.use_feat:
            assert n_feats > 0, "feature modes need n_feats > 0"
            self.feat_mlp = nn.Sequential(
                nn.Linear(n_feats, feat_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
            )
        d = c * (3 if self.use_pool else 1) + (feat_dim if self.use_feat else 0)
        self.drop = nn.Dropout(head_dropout)
        self.fc = nn.Linear(d, n_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
        self.eps = 1e-3

    @staticmethod
    def _pool(fm: torch.Tensor, w: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        mass = w.sum((2, 3))  # (B,1)
        return (fm * w).sum((2, 3)) / (mass + eps), mass

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[1] == 3, f"expected (B,3,H,W), got {tuple(x.shape)}"
        fm = self.backbone.forward_features(x)  # (B,C,h,w)
        B, C, h, w = fm.shape
        assert C == self.c, f"backbone width {C} != head width {self.c}"
        if mask is not None:
            assert mask.shape == (B, 1, *x.shape[-2:]) and mask.device == x.device, (
                "mask must be (B,1,H,W) on the input device"
            )
        v = (
            F.adaptive_avg_pool2d(valid, (h, w))
            if valid is not None
            else torch.ones(B, 1, h, w, device=x.device)
        )
        g, _ = self._pool(fm, v, self.eps)
        g = self.norm_g(g)
        parts = [g]
        if self.use_pool:
            assert mask is not None
            m = F.adaptive_avg_pool2d(mask, (h, w)) * v
            f_in, mass_in = self._pool(fm, m, self.eps)
            f_out, mass_out = self._pool(fm, (v - m).clamp_min(0), self.eps)
            f_in = torch.where(mass_in > 0.05, self.norm_in(f_in), g)
            f_out = torch.where(mass_out > 0.05, self.norm_out(f_out), g)
            parts += [f_in, f_out]
        if self.use_feat:
            assert feats is not None
            parts.append(self.feat_mlp(feats))
        return self.fc(self.drop(torch.cat(parts, 1)))

    def param_groups(self, lr: float, backbone_mult: float, wd: float) -> list[dict]:
        """AdamW groups: backbone at lr*mult, new layers at lr; no decay on biases and norm parameters."""
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_bb = name.startswith("backbone.")
            no_decay = p.ndim <= 1 or name.endswith(".bias")
            groups.setdefault((is_bb, no_decay), []).append(p)
        return [
            {
                "params": ps,
                "lr": lr * (backbone_mult if is_bb else 1.0),
                "weight_decay": 0.0 if nd else wd,
            }
            for (is_bb, nd), ps in groups.items()
        ]


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
