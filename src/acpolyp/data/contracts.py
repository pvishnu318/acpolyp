"""Strict schema validation and data contracts (Phase VII, step 27)."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

CLASSES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
POLYP_IDX = CLASS_TO_IDX["polyps"]
# Classes that show a polyp or polyp-related procedure: the clinically hard negatives for 'polyps'.
COLON_CLASSES = [
    "normal-cecum",
    "polyps",
    "ulcerative-colitis",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
]


class ImageRecord(BaseModel):
    image_id: str
    label: int = Field(ge=0, lt=len(CLASSES))
    label_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(min_length=64, max_length=64)
    phash: str = Field(min_length=16, max_length=16)
    decode_ok: bool
    teal_frac: float = Field(ge=0.0, le=1.0)
    fov_frac: float = Field(ge=0.0, le=1.0)
    n_exif_tags: int = Field(ge=0)
    has_pii_exif: bool
    mean_luma: float = Field(ge=0.0, le=255.0)
    lap_var: float = Field(ge=0.0)


REQUIRED_COLUMNS = list(ImageRecord.model_fields)


def validate_metadata(df: pd.DataFrame) -> None:
    """Raise AssertionError on any contract violation."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    assert df["image_id"].is_unique, "image_id must be unique"
    assert not df[REQUIRED_COLUMNS].isna().any().any(), "NaN in required columns"
    assert set(df["label_name"]) <= set(CLASSES), "unknown class name"
    assert (df["label"] == df["label_name"].map(CLASS_TO_IDX)).all(), "label / label_name mismatch"
    assert df["decode_ok"].all(), "undecodable images must be removed before validation"
    for rec in df[REQUIRED_COLUMNS].to_dict("records"):  # type + range checks per row
        ImageRecord(**rec)
