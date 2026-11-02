"""Global seeding and determinism control (Phase XIII, step 57)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed python, numpy, torch (CPU+CUDA) and return a torch.Generator for DataLoaders."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few CUDA kernels (e.g. adaptive pooling backward) have no deterministic variant.
        torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """Make each DataLoader worker's numpy/random state a deterministic function of the base seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
