"""Model construction from a config dict."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .acgnet import ACGNet

# short names used in configs / tables -> pretrained timm checkpoints
BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
}
NCHW = {
    "resnet50",
    "efficientnetv2_s",
    "convnext_tiny",
}  # backbones whose feature map ACGNet can pool


class PlainNet(nn.Module):
    """Standard fine-tuning head for backbones with token/NHWC outputs (Swin, ViT)."""

    def __init__(
        self,
        name: str,
        n_classes: int,
        pretrained: bool,
        drop_path: float,
        head_dropout: float,
        size: int,
    ):
        super().__init__()
        kw = dict(
            pretrained=pretrained,
            num_classes=n_classes,
            drop_path_rate=drop_path,
            drop_rate=head_dropout,
        )
        try:
            self.net = timm.create_model(name, img_size=size, **kw)
        except TypeError:
            self.net = timm.create_model(name, **kw)

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_groups(self, lr: float, backbone_mult: float, wd: float) -> list[dict]:
        head = {id(p) for p in self.net.get_classifier().parameters()}
        groups: dict[tuple[bool, bool], list] = {}
        for n, p in self.named_parameters():
            is_head = id(p) in head
            nd = p.ndim <= 1 or n.endswith(".bias")
            groups.setdefault((is_head, nd), []).append(p)
        return [
            {
                "params": ps,
                "lr": lr * (1.0 if h else backbone_mult),
                "weight_decay": 0.0 if nd else wd,
            }
            for (h, nd), ps in groups.items()
        ]


def build_model(cfg: dict, n_feats: int = 0, n_classes: int = 8) -> nn.Module:
    m = cfg["model"]
    short = m["name"]
    mode = m.get("ac_mode", "none")
    name = BACKBONES.get(short, short)
    if short in NCHW:
        return ACGNet(
            name,
            n_classes,
            mode=mode,
            n_feats=n_feats,
            pretrained=m["pretrained"],
            drop_path=m["drop_path"],
            head_dropout=m["head_dropout"],
        )
    if mode != "none":
        raise ValueError(
            f"{short}: guided heads need an NCHW feature map; use one of {sorted(NCHW)}"
        )
    return PlainNet(
        name,
        n_classes,
        m["pretrained"],
        m["drop_path"],
        m["head_dropout"],
        cfg["data"]["input_size"],
    )
