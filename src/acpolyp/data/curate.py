"""Deduplication, label-noise audit, grouped splits and leakage audit (Phases VII + IX).

Policy (all thresholds justified by visual inspection of sampled pairs, see docs/DATA_AUDIT.md):
  * byte-identical or cosine >= dup_cos (frozen DINOv2 embedding)  -> duplicate. Same label: keep
    the first. Different labels: drop *both* (label contradiction).
  * cosine >= group_cos -> same 'scene group'. Groups are never split across train/val/test.
  * Kvasir v2 has no patient/video IDs, so scene groups are the best available leakage guard.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..config import load_config
from ..models.embed import cached
from ..paths import INTERIM, PROCESSED, ensure
from .contracts import CLASSES

EMB = "vit_small_patch14_dinov2.lvd142m"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def find_duplicates(df: pd.DataFrame, emb: np.ndarray, dup_cos: float) -> tuple[np.ndarray, dict]:
    """Return boolean `drop` mask over rows and a report."""
    n = len(df)
    drop = np.zeros(n, bool)
    lab = df["label"].to_numpy()
    sim = _l2(emb) @ _l2(emb).T
    np.fill_diagonal(sim, 0.0)
    byte_same = df["sha256"].to_numpy()[:, None] == df["sha256"].to_numpy()[None, :]
    np.fill_diagonal(byte_same, False)
    dup = (sim >= dup_cos) | byte_same
    ii, jj = np.where(np.triu(dup, 1))
    n_cross = 0
    for i, j in zip(ii, jj):
        if lab[i] != lab[j]:
            drop[i] = drop[j] = True
            n_cross += 1
        elif not drop[i]:
            drop[j] = True
    rep = {
        "duplicate_pairs": int(len(ii)),
        "cross_class_pairs": n_cross,
        "rows_dropped": int(drop.sum()),
    }
    return drop, rep


def scene_groups(emb: np.ndarray, group_cos: float) -> np.ndarray:
    s = _l2(emb) @ _l2(emb).T
    np.fill_diagonal(s, 0.0)
    _, g = connected_components(csr_matrix(s >= group_cos), directed=False)
    return g


def label_issues(
    emb: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """cleanlab confident-learning flags from grouped out-of-fold logistic-regression probabilities."""
    from cleanlab.filter import find_label_issues

    oof = np.zeros((len(y), len(CLASSES)))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(emb, y, groups):
        sc = StandardScaler().fit(emb[tr])
        clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(emb[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(emb[te]))
    issues = find_label_issues(labels=y, pred_probs=oof, filter_by="prune_by_noise_rate")
    return issues, oof


def make_splits(
    y: np.ndarray, groups: np.ndarray, test_frac: float, val_frac: float, seed: int
) -> pd.DataFrame:
    n = len(y)
    idx = np.arange(n)
    k_test = round(1 / test_frac)
    sgkf = StratifiedGroupKFold(k_test, shuffle=True, random_state=seed)
    trv, te = next(iter(sgkf.split(idx, y, groups)))
    k_val = round((1 - test_frac) / val_frac)
    sgkf2 = StratifiedGroupKFold(k_val, shuffle=True, random_state=seed + 1)
    tr_i, va_i = next(iter(sgkf2.split(trv, y[trv], groups[trv])))
    split = np.empty(n, dtype=object)
    split[te] = "test"
    split[trv[va_i]] = "val"
    split[trv[tr_i]] = "train"
    # naive (leaky) comparison split: stratified random, ignoring groups, same sizes
    rng_split = np.empty(n, dtype=object)
    skf = StratifiedKFold(k_test, shuffle=True, random_state=seed)
    trv_r, te_r = next(iter(skf.split(idx, y)))
    skf2 = StratifiedKFold(k_val, shuffle=True, random_state=seed + 1)
    tr_r, va_r = next(iter(skf2.split(trv_r, y[trv_r])))
    rng_split[te_r] = "test"
    rng_split[trv_r[va_r]] = "val"
    rng_split[trv_r[tr_r]] = "train"
    # 5-fold grouped CV over train+val for classical models (step 34)
    fold = -np.ones(n, int)
    for f, (_, hold) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=seed + 2).split(trv, y[trv], groups[trv])
    ):
        fold[trv[hold]] = f
    return pd.DataFrame({"split": split, "split_random": rng_split, "cv_fold": fold})


def leakage_audit(df: pd.DataFrame, emb: np.ndarray, split_col: str) -> dict:
    """Zero group / hash overlap across splits; nearest-train cosine for each test image."""
    e = _l2(emb)
    tr, te = (df[split_col] == "train").to_numpy(), (df[split_col] == "test").to_numpy()
    va = (df[split_col] == "val").to_numpy()
    out = {}
    for name, a, b in (("train_test", tr, te), ("train_val", tr, va), ("val_test", va, te)):
        out[f"group_overlap_{name}"] = int(
            len(set(df.loc[a, "group_id"]) & set(df.loc[b, "group_id"]))
        )
        out[f"sha_overlap_{name}"] = int(len(set(df.loc[a, "sha256"]) & set(df.loc[b, "sha256"])))
    nn = (e[te] @ e[tr].T).max(1)
    out["test_nn_train_cos_mean"] = float(nn.mean())
    out["test_nn_train_cos_p95"] = float(np.percentile(nn, 95))
    out["test_frac_nn_above_0.90"] = float((nn >= 0.90).mean())
    out["test_frac_nn_above_0.92"] = float((nn >= 0.92).mean())
    return out


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    ensure(PROCESSED)
    df = pd.read_parquet(INTERIM / "metadata.parquet")
    emb = cached(EMB, "canon", 224)
    report = {"n_ingested": len(df)}

    # --- sanity / outlier checks (step 26) -------------------------------------------------
    bad = (df["fov_frac"] < 0.20) | (df["mean_luma"] < 8) | (df["mean_luma"] > 245)
    report["range_check_failures"] = int(bad.sum())
    q1, q3 = df["lap_var"].quantile([0.25, 0.75])
    report["blur_outliers_iqr"] = int((df["lap_var"] < q1 - 1.5 * (q3 - q1)).sum())

    # --- duplicates (step 25) -------------------------------------------------------------
    drop, rep = find_duplicates(df, emb, cfg["data"]["dedup"]["emb_cosine_dup"])
    report["dedup"] = rep
    keep = ~drop & ~bad.to_numpy()
    df, emb = df[keep].reset_index(drop=True), emb[keep]
    report["n_after_cleaning"] = len(df)
    report["class_counts_after_cleaning"] = df["label_name"].value_counts().sort_index().to_dict()

    # --- scene groups + label-noise audit -------------------------------------------------
    df["group_id"] = scene_groups(emb, cfg["data"]["dedup"]["emb_cosine_group"])
    gsz = df.groupby("group_id")["group_id"].transform("size")
    report["scene_groups"] = {
        "n_groups": int(df["group_id"].nunique()),
        "images_in_multi_groups": int((gsz > 1).sum()),
        "largest_group": int(gsz.max()),
    }
    issues, oof = label_issues(emb, df["label"].to_numpy(), df["group_id"].to_numpy(), seed)
    df["label_issue"] = issues
    df["oof_conf_true"] = oof[np.arange(len(df)), df["label"].to_numpy()]
    report["cleanlab_flagged"] = int(issues.sum())
    report["cleanlab_flagged_by_class"] = (
        df[df["label_issue"]].groupby("label_name").size().to_dict()
    )

    # --- splits (steps 32, 34, 35) --------------------------------------------------------
    sp = make_splits(
        df["label"].to_numpy(),
        df["group_id"].to_numpy(),
        cfg["data"]["test_frac"],
        cfg["data"]["val_frac"],
        seed,
    )
    df = pd.concat([df, sp], axis=1)
    report["leakage_audit_grouped"] = leakage_audit(df, emb, "split")
    report["leakage_audit_random"] = leakage_audit(df, emb, "split_random")
    report["split_sizes"] = df["split"].value_counts().to_dict()
    report["split_class_balance"] = pd.crosstab(df["split"], df["label_name"]).to_dict("index")

    assert report["leakage_audit_grouped"]["group_overlap_train_test"] == 0
    assert report["leakage_audit_grouped"]["sha_overlap_train_test"] == 0
    assert report["leakage_audit_grouped"]["group_overlap_train_val"] == 0
    assert report["leakage_audit_grouped"]["group_overlap_val_test"] == 0

    # immutable artefacts + hashes (step 35)
    df.to_parquet(PROCESSED / "metadata_clean.parquet", index=False)
    np.save(PROCESSED / "kept_idx.npy", np.where(keep)[0])  # maps cleaned rows -> cache rows
    hashes = {}
    for s in ("train", "val", "test"):
        ids = df.loc[df["split"] == s, "image_id"].sort_values().tolist()
        (PROCESSED / f"{s}_ids.txt").write_text("\n".join(ids), encoding="utf-8")
        hashes[s] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    report["split_sha256"] = hashes
    (PROCESSED / "curation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
