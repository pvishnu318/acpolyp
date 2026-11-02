"""Dataset, joint image+mask augmentation and DataLoader factory (Phase XII step 47, Phase X step 40).

Tensor contract of one sample (augmentation/normalisation happen on the GPU, see data/augment.py):
  image  (3, S, S) uint8              canonical frame
  mask   (1, S, S) uint8 in {0,1}     active-contour region (or control mask)
  valid  (1, S, S) uint8 in {0,1}     valid-FOV pixels
  feats  (F,)      float32            standardised hand-crafted features (fit on the train split only)
  label  ()        int64
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..ac.features import family_of
from ..paths import INTERIM, PROCESSED

MASK_FILES = {"ac": "ac_masks.npy", "ellipse": "ac_masks_ellipse.npy", "shuffle": "ac_masks_shuffle.npy"}
FEAT_FILES = {"ac": "ac_features.parquet", "ellipse": "ac_features_ellipse.parquet", "shuffle": "ac_features_shuffle.parquet"}


def feature_columns(df: pd.DataFrame, families: list[str]) -> list[str]:
    return [c for c in df.columns if family_of(c) in families]


class LazyNpy:
    """Memory-mapped .npy opened on first access inside the worker; pickles as a path only, so
    spawned DataLoader workers (Windows) never serialise the arrays."""

    def __init__(self, path):
        self.path = str(path)
        self._a = None

    def _arr(self):
        if self._a is None:
            self._a = np.load(self.path, mmap_mode="r")
        return self._a

    def __getitem__(self, i):
        return self._arr()[i]

    def __len__(self):
        return len(self._arr())

    def __getstate__(self):
        return {"path": self.path, "_a": None}


def clean_cache(name: str, src: str) -> str:
    """Write the cleaned-order (kept rows only) copy of an interim cache once and return its path."""
    out = PROCESSED / f"{name}_clean.npy"
    if not out.exists():
        kept = np.load(PROCESSED / "kept_idx.npy")
        np.save(out, np.ascontiguousarray(np.load(INTERIM / src, mmap_mode="r")[kept]))
    return str(out)


class KvasirData:
    """Everything split-independent, loaded once and shared by the train/val/test Dataset views."""

    def __init__(self, image_key: str = "canon", mask_source: str = "ac", feat_families: list[str] | None = None,
                 split_col: str = "split"):
        self.meta = pd.read_parquet(PROCESSED / "metadata_clean.parquet")
        self.images = LazyNpy(clean_cache(f"images_{image_key}", f"images_{image_key}.npy"))  # (N,256,256,3) uint8
        self.valid = LazyNpy(clean_cache("fov", "fov.npy"))
        self.masks = LazyNpy(PROCESSED / MASK_FILES[mask_source])  # (N,128,128)
        self.labels = self.meta["label"].to_numpy().astype(np.int64)
        self.feats = None
        if feat_families:
            df = pd.read_parquet(PROCESSED / FEAT_FILES[mask_source])
            gdf = pd.read_parquet(PROCESSED / FEAT_FILES["ac"])  # global features never depend on the mask
            cols = feature_columns(df, [f for f in feat_families if f != "global"])
            f = df[cols].to_numpy(np.float32) if cols else np.zeros((len(df), 0), np.float32)
            if "global" in feat_families:
                gcols = feature_columns(gdf, ["global"])
                f = np.concatenate([f, gdf[gcols].to_numpy(np.float32)], axis=1)
            self.feats = f
            self.feat_names = cols + (gcols if "global" in feat_families else [])
        self.split = self.meta[split_col].to_numpy()  # 'split' = grouped; 'split_random' = naive leaky control

    def idx(self, split: str, frac: float = 1.0) -> np.ndarray:
        """Row indices of a split; frac < 1 draws a fixed class-stratified subset (learning-curve study)."""
        rows = np.where(self.split == split)[0]
        if frac >= 1.0:
            return rows
        rng = np.random.default_rng(1234)  # the subset is identical for every model and seed
        keep = [rng.permutation(rows[self.labels[rows] == k])[: max(1, round(frac * (self.labels[rows] == k).sum()))] for k in np.unique(self.labels[rows])]
        return np.sort(np.concatenate(keep))

    def fit_scaler(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Mean/std from the TRAIN rows only; robust-clipped to limit heavy tails."""
        if self.feats is None:
            return None
        tr = self.feats[self.idx("train")]
        lo, hi = np.percentile(tr, [1, 99], axis=0)
        trc = np.clip(tr, lo, hi)
        return trc.mean(0), trc.std(0) + 1e-6, lo, hi  # type: ignore[return-value]


class KvasirSplit(Dataset):
    """Raw (un-augmented) samples; geometry and photometry are applied per batch on the GPU."""

    def __init__(self, data: KvasirData, split: str, scaler=None, indices=None):
        self.d = data
        self.rows = data.idx(split) if indices is None else indices
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.rows[i])
        d = self.d
        img = torch.from_numpy(np.array(d.images[r])).permute(2, 0, 1)  # uint8 (3,S,S)
        S = img.shape[-1]
        mask = torch.from_numpy(np.array(d.masks[r]))[None, None]
        mask = F.interpolate(mask, size=(S, S), mode="nearest")[0]  # uint8 (1,S,S)
        valid = torch.from_numpy(np.array(d.valid[r]))[None]
        out = {"image": img, "mask": mask, "valid": valid, "label": torch.tensor(d.labels[r])}
        if d.feats is not None:
            mu, sd, lo, hi = self.scaler
            out["feats"] = torch.from_numpy(((np.clip(d.feats[r], lo, hi) - mu) / sd).astype(np.float32))
        return out


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int, pin: bool = True) -> DataLoader:
    """In-process loader (num_workers=0): indexing a memory-mapped uint8 cache is cheap, and worker processes
    cost ~2.4 GB of commit memory each on Windows (observed to exhaust the commit limit)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate,
                      pin_memory=pin, drop_last=shuffle, generator=g)
