import pytest
import torch

from acpolyp.data import augment as A

S = 256


def synthetic(batch=6):
    """White square on black; the mask marks exactly the square; everything is valid."""
    img = torch.zeros(batch, 3, S, S, dtype=torch.uint8)
    mask = torch.zeros(batch, 1, S, S, dtype=torch.uint8)
    for b in range(batch):
        y, x = 60 + 10 * b, 90 + 5 * b
        img[b, :, y : y + 70, x : x + 60] = 255
        mask[b, :, y : y + 70, x : x + 60] = 1
    return img, mask, torch.ones(batch, 1, S, S, dtype=torch.uint8)


def test_eval_transform_contract():
    img, mask, valid = synthetic()
    valid[:, :, :, :40] = 0
    x, m, v = A.eval_transform(img, mask, valid, 224)
    assert x.shape == (6, 3, 224, 224) and x.dtype == torch.float32 and m.shape == (6, 1, 224, 224)
    assert set(m.unique().tolist()) <= {0.0, 1.0} and set(v.unique().tolist()) <= {0.0, 1.0}
    assert (
        x[:, :, :, :30] == 0
    ).all()  # exterior of the valid FOV is exactly zero after normalisation


@pytest.mark.parametrize("seed", range(5))
def test_train_warp_keeps_mask_aligned_with_image(seed):
    img, mask, valid = synthetic()
    gen = torch.Generator().manual_seed(seed)
    x, m, v = A.train_transform(img, mask, valid, gen, 224)
    # undo colour jitter + normalisation roughly: the bright square is the pixels well above the mean
    bright = (x.mean(1, keepdim=True) > 0.0).float()
    for b in range(x.shape[0]):
        inter = (bright[b] * m[b]).sum()
        union = ((bright[b] + m[b]) > 0).float().sum()
        if m[b].sum() > 200:  # square still (partly) inside the crop
            assert inter / union > 0.85, (
                f"mask drifted from image content (IoU {float(inter / union):.2f})"
            )


def test_train_transform_is_deterministic_given_generator_and_differs_across_seeds():
    img, mask, valid = synthetic()
    a = A.train_transform(img, mask, valid, torch.Generator().manual_seed(1), 224)
    b = A.train_transform(img, mask, valid, torch.Generator().manual_seed(1), 224)
    c = A.train_transform(img, mask, valid, torch.Generator().manual_seed(2), 224)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert not torch.equal(a[0], c[0])


def test_train_transform_outputs_are_finite_and_masks_binary():
    img, mask, valid = synthetic()
    x, m, v = A.train_transform(img, mask, valid, torch.Generator().manual_seed(0), 224)
    assert torch.isfinite(x).all() and set(m.unique().tolist()) <= {0.0, 1.0}
    assert x.abs().max() < 5  # inside the plausible range of ImageNet-normalised pixels


def test_colour_jitter_stays_in_unit_range():
    x = torch.rand(8, 3, 32, 32)
    y = A.color_jitter(x, torch.Generator().manual_seed(0))
    assert y.min() >= 0 and y.max() <= 1
