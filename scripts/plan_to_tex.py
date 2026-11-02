"""Convert the compliance matrix in docs/PROJECT_PLAN.md into a LaTeX longtable for the report appendix."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = {"✅": "done", "➖": "n/a", "⚠️": "limitation", "⚠": "limitation"}
ORDER = ["done", "limitation", "n/a"]
UNICODE = {
    "θ": r"\ensuremath{\theta}",
    "→": r"\ensuremath{\rightarrow}",
    "↔": r"\ensuremath{\leftrightarrow}",
    "§": r"\S{}",
    "≥": r"\ensuremath{\ge}",
    "≤": r"\ensuremath{\le}",
    "×": r"\ensuremath{\times}",
    "≈": r"\ensuremath{\approx}",
    "·": r"\ensuremath{\cdot}",
    "∈": r"\ensuremath{\in}",
    "–": "--",
    "—": "---",
    "✓": "",
    "➖": "n/a",
    "±": r"\ensuremath{\pm}",
    "**": "",
    "⚠️": "",
    "⚠": "",
}


def esc(s: str) -> str:
    """Escape LaTeX specials outside backtick spans; backtick spans become \\texttt{...}."""
    out = []
    for i, part in enumerate(re.split(r"`([^`]*)`", s)):
        if i % 2:  # inside backticks
            out.append(r"\texttt{" + re.sub(r"([_#&%$])", r"\\\1", part) + "}")
        else:
            out.append(re.sub(r"([&%$#_])", r"\\\1", part))
    res = "".join(out)
    for k, v in UNICODE.items():
        res = res.replace(k, v)
    return res


def main() -> None:
    lines = (ROOT / "docs" / "PROJECT_PLAN.md").read_text(encoding="utf-8").splitlines()
    rows: list[str] = []
    for ln in lines:
        if ln.startswith("## Phase"):
            rows.append(r"\midrule\multicolumn{4}{l}{\textbf{" + esc(ln[3:]) + r"}}\\\midrule")
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 4 or not re.match(r"^\d+([–-]\d+)?$", cells[0]):
            continue
        num, step, status, where = cells[0], cells[1], cells[2], cells[3]
        labels = sorted({v for k, v in STATUS.items() if k in status}, key=ORDER.index) or ["done"]
        note = re.sub("|".join(map(re.escape, STATUS)), "", status).strip()
        st = ", ".join(labels) + (f" ({esc(note)})" if note else "")
        rows.append(f"{esc(num)} & {esc(step)} & {st} & {esc(where)} \\\\")
    head = (
        r"\begin{longtable}{p{0.05\linewidth}p{0.25\linewidth}p{0.14\linewidth}p{0.46\linewidth}}"
        "\n"
        r"\toprule \# & Step & Status & Where / note \\ \endhead"
        "\n"
    )
    (ROOT / "report" / "chapters" / "plan_table.tex").write_text(
        head + "\n".join(rows) + "\n" + r"\bottomrule\end{longtable}" + "\n", encoding="utf-8"
    )
    print("step rows:", sum(1 for r in rows if not r.startswith(r"\midrule")))


if __name__ == "__main__":
    main()
