#!/usr/bin/env python3
"""Rewrite Pandoc longtable blocks in thesis/latex/tesis.tex."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "thesis" / "latex" / "tesis.tex"

TABLE_META = [
    ("Resultados esperados (mapeo de objetivos, MV e IOV)", "tab:mapeo-objetivos"),
    ("Herramientas, métodos y procedimientos a utilizar", "tab:herramientas-metodos"),
    ("Resultados de búsqueda de las cadenas", "tab:busqueda-cadenas"),
    ("Resultados luego de aplicar criterios de exclusión", "tab:exclusion-criterios"),
    ("Formulario de extracción de datos", "tab:formulario-extraccion"),
    ("Resumen de materiales textuales bilingües recopilados", "tab:materiales-bilingues"),
    ("Distribución temática del corpus generado", "tab:distribucion-tematica"),
    ("Etapas del preprocesamiento del corpus bilingüe", "tab:etapas-preprocesamiento"),
    ("Corpus consolidado para modelo de embeddings", "tab:corpus-consolidado"),
    ("Limitaciones del proyecto", "tab:limitaciones"),
    ("Umbral de riesgo (probabilidad e impacto)", "tab:umbral-riesgo"),
    ("Matriz de riesgos del proyecto", "tab:matriz-riesgos"),
    ("Lista de tareas (EDT)", "tab:edt-tareas"),
    ("Cronograma del proyecto", "tab:cronograma"),
    ("Presupuesto del proyecto", "tab:presupuesto"),
    ("Plantilla de presupuesto (ejemplo)", "tab:plantilla-presupuesto"),
]


def remove_noalign(s: str) -> str:
    return (
        s.replace(r"\toprule\noalign{}", r"\toprule")
        .replace(r"\midrule\noalign{}", r"\midrule")
        .replace(r"\bottomrule\noalign{}", r"\bottomrule")
    )


def strip_header_minipages(s: str) -> str:
    """Pandoc ends each header cell with \\end{minipage} & or \\end{minipage} \\\\."""
    pat = re.compile(
        r"\\begin\{minipage\}\[b\]\{\\linewidth\}\\(?:centering|raggedright)\s*\n"
        r"(\\textbf\{[^}]*\})\s*\n"
        r"\\end\{minipage\}(\s*(?:&|\\\\)\s*)",
        re.MULTILINE,
    )
    prev = None
    while prev != s:
        prev = s
        s = pat.sub(r"\1\2", s)
    return s


def insert_caption(inner: str, caption: str, label: str) -> str:
    """Insert caption after column-spec block (everything before the first \\toprule)."""
    m = re.match(
        r"(\\begin\{longtable\}\[\]\{@\{\}.*?\n)(?=\\toprule)",
        inner,
        re.DOTALL,
    )
    if not m:
        raise ValueError("longtable begin/column spec not found")
    return (
        m.group(1)
        + f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n"
        + inner[m.end() :]
    )


def extract_broken_header(rest: str) -> tuple[str, int]:
    """After Pandoc \\endlastfoot, first logical row ends with \\\\ (possibly multiline)."""
    rest2 = rest.lstrip("\n")
    base = len(rest) - len(rest2)
    if rest2.startswith("Ítem"):
        m = re.match(r"(Ítem.*?Total \(S/\.\)\s*\\\\)\s*\n", rest2, re.DOTALL)
        if not m:
            raise ValueError("expected Ítem… header row")
        return m.group(1).strip(), base + m.end()
    m = re.match(r"([\s\S]*?\\\\)\s*\n", rest2)
    if not m:
        raise ValueError(f"no header row after endlastfoot: {rest2[:120]!r}")
    return m.group(1).strip(), base + m.end()


def patch_longtable_heads(inner: str) -> str:
    """Fix Pandoc longtable head/tail; handle broken tables (header after endlastfoot)."""
    m_broken = re.search(
        r"(\\caption\{[^}]*\}\\label\{[^}]*\}\\\\\n)"
        r"\\toprule\s*\\endhead\s*\\bottomrule\s*\\endlastfoot\s*\n",
        inner,
    )
    if m_broken:
        pos = m_broken.end()
        header, endpos = extract_broken_header(inner[pos:])
        cap = m_broken.group(1)
        repl = (
            f"{cap}"
            f"\\toprule\n"
            f"{header}\n"
            f"\\midrule\n"
            f"\\endfirsthead\n"
            f"\\caption*{{\\tablename~\\thetable\\ (continuación)}}\\\\\n"
            f"\\toprule\n"
            f"{header}\n"
            f"\\midrule\n"
            f"\\endhead\n"
            f"\\midrule\n"
            f"\\endfoot\n"
            f"\\bottomrule\n"
            f"\\endlastfoot\n"
        )
        return inner[: m_broken.start()] + repl + inner[pos + endpos :]

    m_std = re.search(
        r"(\\caption\{[^}]*\}\\label\{[^}]*\}\\\\\n)"
        r"\\toprule\s*"
        r"(.*?)"
        r"\\midrule\s*\\endhead\s*"
        r"\\bottomrule\s*\\endlastfoot\s*",
        inner,
        flags=re.DOTALL,
    )
    if not m_std:
        raise ValueError("could not patch longtable heads (not broken, not standard)")
    cap = m_std.group(1)
    header_block = m_std.group(2).strip() + "\n"
    repl = (
        f"{cap}"
        f"\\toprule\n"
        f"{header_block}"
        f"\\midrule\n"
        f"\\endfirsthead\n"
        f"\\caption*{{\\tablename~\\thetable\\ (continuación)}}\\\\\n"
        f"\\toprule\n"
        f"{header_block}"
        f"\\midrule\n"
        f"\\endhead\n"
        f"\\midrule\n"
        f"\\endfoot\n"
        f"\\bottomrule\n"
        f"\\endlastfoot\n"
    )
    return inner[: m_std.start()] + repl + inner[m_std.end() :]


def transform_inner(inner: str, caption: str, label: str) -> str:
    inner = remove_noalign(inner)
    inner = strip_header_minipages(inner)
    inner = insert_caption(inner, caption, label)
    inner = patch_longtable_heads(inner)
    return inner


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    text = remove_noalign(text)

    block_re = re.compile(
        r"\{\s*\\def\\LTcaptype\{none\}\s*%[^\n]*\n"
        r"(\\begin\{longtable\}.*?\\end\{longtable\})\s*\}\s*"
        r"(?:\\textbf\{Tabla[^\n]*\}\s*\n)?",
        re.DOTALL,
    )

    idx = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(TABLE_META):
            raise RuntimeError("too many tables")
        cap, lab = TABLE_META[idx]
        idx += 1
        return transform_inner(m.group(1), cap, lab) + "\n\n"

    new_text, n = block_re.subn(repl, text)
    if n != len(TABLE_META):
        raise RuntimeError(f"expected {len(TABLE_META)} tables, got {n}")
    TEX.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"Wrote {n} tables to {TEX}")


if __name__ == "__main__":
    main()
