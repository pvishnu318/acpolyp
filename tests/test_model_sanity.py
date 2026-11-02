"""Debugging sanity checks (Phase XIV): shapes, gradient flow, NaN guards, overfit, batch independence."""

import pytest
import torch

from acpolyp.models.acgnet import ACGNet, count_params
from acpolyp.training.engine import cosine_with_warmup

B, S = 8, 64


def make(mode, n_feats=0):
    torch.manual_seed(0)
    return ACGNet(
        "resnet18", 8, mode=mode, n_feats=n_feats, pretrained=False, drop_path=0.0, head_dropout=0.0
    )


def batch(n_feats=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 3, S, S, generator=g)
    mask = (torch.rand(B, 1, S, S, generator=g) > 0.5).float()
    valid = torch.ones(B, 1, S, S)
    f = torch.randn(B, n_feats, generator=g) if n_feats else None
    y = torch.arange(B) % 8
    return x, mask, valid, f, y


@pytest.mark.parametrize(
    "mode,nf", [("none", 0), ("ac_pool", 0), ("ac_feat", 10), ("acg", 10), ("gfeat", 10)]
)
def test_forward_shapes_dtypes_and_finite(mode, nf):
    m = make(mode, nf).eval()
    x, mask, valid, f, _ = batch(nf)
    out = m(x, mask=mask, valid=valid, feats=f)
    assert out.shape == (B, 8) and out.dtype == torch.float32 and torch.isfinite(out).all()


@pytest.mark.parametrize("mode,nf", [("none", 0), ("acg", 10)])
def test_every_trainable_param_receives_finite_gradient(mode, nf):
    """timm zero-initialises the last BN gain of each residual block, so residual branches have exactly zero
    gradient at step 0 by design. Take one optimiser step first, then require gradient flow everywhere
    that is used by the selected mode."""
    m = make(mode, nf).train()
    x, mask, valid, f, y = batch(nf)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x, mask=mask, valid=valid, feats=f), y).backward()
        opt.step()
    unused = {"norm_in", "norm_out"} if mode == "none" else set()
    checked = [
        (n, p) for n, p in m.named_parameters() if p.requires_grad and n.split(".")[0] not in unused
    ]
    bad = [n for n, p in checked if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"parameters without finite gradient: {bad[:5]}"
    nonzero = [bool(p.grad.abs().sum() > 0) for _, p in checked]
    assert sum(nonzero) / len(nonzero) > 0.95


def test_single_batch_overfit():
    m = make("acg", 10).train()
    x, mask, valid, f, y = batch(10)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(80):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x, mask=mask, valid=valid, feats=f), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"cannot overfit one batch: loss={loss.item():.3f}"


def test_no_cross_sample_mixing_and_order_invariance_in_eval():
    m = make("acg", 10).eval()
    x, mask, valid, f, _ = batch(10)
    out = m(x, mask=mask, valid=valid, feats=f)
    perm = torch.randperm(B, generator=torch.Generator().manual_seed(1))
    out_p = m(x[perm], mask=mask[perm], valid=valid[perm], feats=f[perm])
    assert torch.allclose(out[perm], out_p, atol=1e-5)
    x2 = x.clone()
    x2[0] += 5.0  # perturbing sample 0 must not change any other sample's logits
    out2 = m(x2, mask=mask, valid=valid, feats=f)
    assert torch.allclose(out[1:], out2[1:], atol=1e-5)


def test_empty_region_falls_back_to_global_and_stays_finite():
    m = make("ac_pool").eval()
    x, _, valid, _, _ = batch()
    out = m(x, mask=torch.zeros(B, 1, S, S), valid=valid)
    assert torch.isfinite(out).all()


def test_param_groups_exclude_norm_and_bias_from_decay():
    m = make("acg", 10)
    groups = m.param_groups(1e-3, 0.25, 0.05)
    assert sum(len(g["params"]) for g in groups) == len(
        [p for p in m.parameters() if p.requires_grad]
    )
    no_decay = [g for g in groups if g["weight_decay"] == 0.0]
    assert no_decay and all(p.ndim <= 1 for g in no_decay for p in g["params"])
    assert {g["lr"] for g in groups} == {1e-3, 1e-3 * 0.25}


def test_param_count_and_schedule():
    total, trainable = count_params(make("none"))
    assert total == trainable > 1_000_000
    s = [cosine_with_warmup(i, 100, 10) for i in range(100)]
    assert s[0] < s[9] <= 1.0 and s[-1] < 0.05 and max(s) <= 1.0
