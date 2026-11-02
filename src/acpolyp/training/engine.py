"""Training / validation / inference loops, schedulers, checkpointing (Phase XIII steps 55-58, XIV, XV)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from sklearn.metrics import f1_score

from ..data.augment import eval_transform, train_transform
from .metrics import softmax

AUX_KEYS = ("mask", "valid", "feats")


def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def prepare(b: dict, train: bool, size: int, gen: torch.Generator | None = None) -> dict:
    """Apply the shared GPU transform; replaces image/mask/valid by their model-ready versions."""
    if train:
        x, m, v = train_transform(b["image"], b["mask"], b["valid"], gen, size)
    else:
        x, m, v = eval_transform(b["image"], b["mask"], b["valid"], size)
    return {**b, "image": x.contiguous(memory_format=torch.channels_last), "mask": m, "valid": v}


def forward_batch(model: nn.Module, b: dict) -> torch.Tensor:
    return model(b["image"], **{k: b[k] for k in AUX_KEYS if k in b})


def cosine_with_warmup(step: int, total: int, warmup: int, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


@torch.inference_mode()
def predict(
    model: nn.Module, loader, device: str, amp: bool = True, size: int = 224
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate logits and labels across the whole loader (global metrics are computed afterwards)."""
    model.eval()
    logits, ys = [], []
    for batch in loader:
        b = prepare(to_device(batch, device), False, size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = forward_batch(model, b)
        assert torch.isfinite(out).all(), "non-finite logits at inference"
        logits.append(out.float().cpu())
        ys.append(b["label"].cpu())
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: dict,
    run_dir: Path,
    device: str = "cuda",
    log=print,
) -> dict:
    t = cfg["train"]
    model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(
        model.param_groups(t["lr"], t["backbone_lr_mult"], t["weight_decay"]), betas=(0.9, 0.999)
    )
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    steps_per_epoch = len(train_loader)
    total = t["epochs"] * steps_per_epoch
    warm = t["warmup_epochs"] * steps_per_epoch
    crit = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    best, best_epoch, bad, step = -1.0, -1, 0, 0
    history = []
    size = cfg["data"]["input_size"]
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed_run"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(t["epochs"]):
        model.train()
        run, n = 0.0, 0
        for batch in train_loader:
            b = prepare(to_device(batch, device), True, size, gen)
            scale = cosine_with_warmup(step, total, warm)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=t["amp"]):
                logits = forward_batch(model, b)
            loss = crit(logits.float(), b["label"])  # loss in fp32: logit-space, numerically stable
            assert torch.isfinite(loss), f"non-finite loss at step {step}"
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            run += loss.item() * len(b["label"])
            n += len(b["label"])
            step += 1
        vl, vy = predict(model, val_loader, device, t["amp"], size)
        vp = softmax(vl)
        vf1 = float(f1_score(vy, vp.argmax(1), average="macro"))
        vacc = float((vp.argmax(1) == vy).mean())
        row = {
            "epoch": epoch,
            "train_loss": run / n,
            "val_macro_f1": vf1,
            "val_acc": vacc,
            "grad_norm": float(gn),
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        log(
            f"  ep{epoch:02d} loss {row['train_loss']:.4f} val_f1 {vf1:.4f} val_acc {vacc:.4f} gn {row['grad_norm']:.2f} [{row['elapsed_s']:.0f}s]"
        )
        if vf1 > best:
            best, best_epoch, bad = vf1, epoch, 0
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                str(run_dir / "best.safetensors"),
            )
        else:
            bad += 1
            if bad >= t["patience"]:
                log(f"  early stop at epoch {epoch}")
                break
    train_time = time.time() - t0
    model.load_state_dict(load_file(str(run_dir / "best.safetensors")))
    return {
        "best_val_macro_f1": best,
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "epochs_run": len(history),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        "history": history,
    }


@torch.inference_mode()
def latency(
    model: nn.Module,
    example: dict,
    device: str = "cuda",
    warmup: int = 10,
    iters: int = 50,
    amp: bool = True,
    size: int = 224,
) -> float:
    """Mean per-batch latency in ms after warm-up (eval transform excluded), with CUDA synchronisation."""
    model.eval()
    b = prepare(to_device(example, device), False, size)
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            forward_batch(model, b)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            forward_batch(model, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000
