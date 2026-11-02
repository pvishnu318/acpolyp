"""Walking skeleton (Phase XII step 48): real DataLoader -> dummy constant model -> loss -> evaluator -> logger,
verified to run end to end in seconds. Skipped when the processed data is not present."""

import time

import numpy as np
import pytest
import torch

from acpolyp.paths import PROCESSED

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "metadata_clean.parquet").exists(), reason="processed data not built"
)


class ConstantModel(torch.nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(k))

    def forward(self, x, **_):
        return self.logits.expand(x.shape[0], -1)


def test_pipeline_runs_end_to_end_with_constant_model(tmp_path):
    from acpolyp.data.dataset import KvasirData, KvasirSplit, make_loader
    from acpolyp.runlog import RunDir
    from acpolyp.training.engine import forward_batch, predict, prepare, to_device
    from acpolyp.training.metrics import full_metrics, softmax

    t0 = time.time()
    data = KvasirData("canon", "ac", None)
    ds = KvasirSplit(data, "val")
    loader = make_loader(ds, 32, False, 0, pin=False)
    batch = next(iter(loader))
    assert batch["image"].dtype == torch.uint8 and batch["image"].shape[1:] == (3, 256, 256)
    assert batch["mask"].shape[1:] == (1, 256, 256) and batch["label"].dtype == torch.int64
    b = prepare(to_device(batch, "cpu"), False, 224)
    assert b["image"].shape[1:] == (3, 224, 224) and b["image"].dtype == torch.float32
    model = ConstantModel()
    loss = torch.nn.functional.cross_entropy(forward_batch(model, b), b["label"])
    assert abs(loss.item() - np.log(8)) < 0.15  # uniform logits on balanced classes
    logits, y = predict(model, loader, "cpu", amp=False)
    m = full_metrics(softmax(logits) + 1e-9 * np.random.default_rng(0).random(logits.shape), y)
    rd = RunDir("skeleton", root=tmp_path)
    rd.log_row({"loss": float(loss), **{k: m[k] for k in ("acc", "macro_f1")}})
    assert (rd.path / "metrics.csv").exists() and (rd.path / "meta.json").exists()
    assert time.time() - t0 < 60, "walking skeleton must run in seconds"
    assert 0.05 < m["acc"] < 0.25  # constant prediction on 8 balanced classes sits at chance
