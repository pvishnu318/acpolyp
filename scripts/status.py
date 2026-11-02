"""Print a one-line summary for every finished run (monitoring helper)."""

import json

from acpolyp.paths import RUNS


def main() -> None:
    rows = []
    for d in sorted(RUNS.glob("*/seed*")):
        f = d / "summary.json"
        if d.parent.name == "classical":
            continue
        if f.exists():
            r = json.loads(f.read_text())
            t = r["test_calibrated"]
            rows.append(
                f"{d.parent.name:20s} {d.name:6s} F1={t['macro_f1']:.4f} acc={t['acc']:.4f} polyp_auroc={t['polyp_auroc']:.4f} "
                f"ece={t['ece']:.4f} val_f1={r['best_val_macro_f1']:.4f} {r['train_time_s']:.0f}s"
            )
        else:
            rows.append(f"{d.parent.name:20s} {d.name:6s} running")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
