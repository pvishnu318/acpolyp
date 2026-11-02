import cv2
import numpy as np
import pandas as pd
import pytest

from acpolyp.ac import features as F
from acpolyp.config import load_config
from acpolyp.data import contracts, curate, preprocess


def synth_frame(w=720, h=576):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 2 - 20, (170, 90, 80), -1)  # bright FOV disc
    cv2.putText(
        img, "TEXT 12/03", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )  # thin text in margin
    return img


def test_fov_mask_finds_disc_and_ignores_margin_text():
    img = synth_frame()
    fov = preprocess.fov_mask(img)
    assert abs(fov.mean() - np.pi * (288 - 20) ** 2 / (720 * 576)) < 0.03
    assert fov[100, 5:60].sum() == 0


def test_teal_box_is_detected_but_blue_dye_is_not():
    img = synth_frame()
    img[400:550, 40:240] = (20, 160, 140)  # flat teal box, lower-left
    assert preprocess.teal_mask(img).sum() > 0
    dye = synth_frame()
    dye[250:330, 300:420] = (30, 90, 200)
    dye[250:330, 300:420] += np.random.default_rng(0).integers(0, 30, (80, 120, 3), dtype=np.uint8)
    assert preprocess.teal_mask(dye).sum() == 0


def test_canonicalise_output_contract_and_format_independent_box_mask():
    c = preprocess.canonicalise(synth_frame(), 256)
    assert (
        c.image.shape == (256, 256, 3) and c.image.dtype == np.uint8 and c.fov.shape == (256, 256)
    )
    assert c.fov.max() == 1 and (c.fov == 0).any()
    c2 = preprocess.canonicalise(synth_frame(1280, 1024), 256)
    assert c2.image.shape == c.image.shape


def test_mask_crop_aligns_with_image_crop():
    img = synth_frame()
    gt = np.zeros(img.shape[:2], np.uint8)
    gt[250:350, 330:430] = 255
    c = preprocess.canonicalise(img, 256)
    m = preprocess.crop_resize_mask(gt, c.bbox, 256)
    ys, xs = np.where(m > 0)
    assert 100 < ys.mean() < 156 and 100 < xs.mean() < 156


def test_shape_features_on_disc_and_thin_rectangle():
    valid = np.ones((128, 128), np.uint8)
    disc = np.zeros((128, 128), np.uint8)
    cv2.circle(disc, (64, 64), 30, 1, -1)
    f = F.shape_features(disc, valid)
    assert f["s_circularity"] > 0.9 and f["s_solidity"] > 0.98 and f["s_ecc"] < 0.1
    assert abs(f["s_area_frac"] - np.pi * 900 / 128**2) < 0.01
    rect = np.zeros((128, 128), np.uint8)
    rect[40:60, 10:110] = 1
    g = F.shape_features(rect, valid)
    assert g["s_circularity"] < f["s_circularity"] and g["s_axis_ratio"] < 0.3


def test_glcm_constant_image_has_zero_contrast_and_unit_energy():
    f = F.glcm_features(np.full((32, 32), 120, np.uint8), np.ones((32, 32), np.uint8), "t")
    assert (
        f["t_glcm_contrast"] == 0
        and abs(f["t_glcm_energy"] - 1) < 1e-9
        and f["t_glcm_entropy"] == 0
    )


def test_glcm_checkerboard_hand_calculation():
    gray = (np.indices((8, 8)).sum(0) % 2 * 255).astype(np.uint8)  # levels 0 and 15 only
    f = F.glcm_features(gray, np.ones((8, 8), np.uint8), "t")
    # H/V neighbours differ by 15 levels, diagonals are equal -> mean (i-j)^2 = 0.5 * 225 = 112.5
    assert 100 < f["t_glcm_contrast"] < 125


def test_all_features_finite_and_families_present():
    rgb = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    m = np.zeros((128, 128), np.uint8)
    m[30:80, 30:80] = 1
    f = F.all_features(rgb, np.ones((128, 128), np.uint8), m)
    assert np.isfinite(list(f.values())).all()
    assert {F.family_of(k) for k in f} == {"global", "shape", "boundary", "region"}


def test_all_features_handles_empty_mask():
    rgb = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    f = F.all_features(rgb, np.ones((128, 128), np.uint8), np.zeros((128, 128), np.uint8))
    assert np.isfinite(list(f.values())).all()


def test_contract_rejects_bad_label_and_duplicate_ids():
    row = dict(
        idx=0,
        image_id="a/b",
        label=0,
        label_name="dyed-lifted-polyps",
        width=10,
        height=10,
        sha256="0" * 64,
        phash="0" * 16,
        decode_ok=True,
        teal_frac=0.0,
        fov_frac=0.5,
        n_exif_tags=0,
        has_pii_exif=False,
        mean_luma=100.0,
        lap_var=1.0,
    )
    df = pd.DataFrame([row])
    contracts.validate_metadata(df)
    bad = df.copy()
    bad.loc[0, "label"] = 3
    with pytest.raises(AssertionError):
        contracts.validate_metadata(bad)
    with pytest.raises(AssertionError):
        contracts.validate_metadata(pd.concat([df, df]))


def test_grouped_split_never_straddles_groups_and_is_deterministic():
    y = np.repeat(np.arange(4), 250)
    groups = np.arange(1000) // 2  # consecutive pairs share a scene group
    s1 = curate.make_splits(y, groups, 0.2, 0.1, seed=0)
    s2 = curate.make_splits(y, groups, 0.2, 0.1, seed=0)
    assert (s1["split"] == s2["split"]).all()
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        assert not (set(groups[s1.split == a]) & set(groups[s1.split == b]))
    assert abs((s1.split == "test").mean() - 0.2) < 0.03


def test_config_override_parsing():
    cfg = load_config(overrides=["train.lr=1e-3", "model.pretrained=false", "train.seeds=[1,2]"])
    assert (
        cfg["train"]["lr"] == 1e-3
        and cfg["model"]["pretrained"] is False
        and cfg["train"]["seeds"] == [1, 2]
    )
