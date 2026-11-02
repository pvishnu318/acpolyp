"""Print classical-baseline results (mean over seeds)."""

import json

import numpy as np

from acpolyp.paths import RUNS


def main() -> None:
    for d in sorted((RUNS / "classical").iterdir()):
        if not d.is_dir():
            continue
        rs = [json.loads(f.read_text())["test"] for f in sorted(d.glob("seed*/summary.json"))]
        if rs:
            print(
                f"{d.name:55s} n={len(rs)} F1={np.mean([r['macro_f1'] for r in rs]):.4f} acc={np.mean([r['acc'] for r in rs]):.4f} polyp_auroc={np.mean([r['polyp_auroc'] for r in rs]):.4f}"
            )


if __name__ == "__main__":
    main()
