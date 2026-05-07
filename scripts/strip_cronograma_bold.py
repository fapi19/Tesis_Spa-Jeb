"""Remove \\textbf{n} (digits only) inside the Cronograma del proyecto longtable."""
import re
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "thesis" / "latex" / "tesis.tex"
t = path.read_text(encoding="utf-8")
start = t.find("\\caption{Cronograma del proyecto}\\label{tab:cronograma}")
if start == -1:
    raise SystemExit("cronograma caption not found")
end = t.find("\\end{longtable}", start) + len("\\end{longtable}")
block = t[start:end]
# Literal backslash before textbf: avoid regex \\t interpretation
# Match literal \\textbf{digits} (avoid \\t in r"\\textbf...")
pat = re.compile(r"[\\]textbf\{(\d+)\}")
new_block, n = pat.subn(r"\1", block)
if n == 0:
    raise SystemExit("no bold numeric cells found")
path.write_text(t[:start] + new_block + t[end:], encoding="utf-8")
print("stripped", n, "cells")
