"""One-shot analysis: how much of extra.csv is genuinely new vocab vs already
covered by other sources?"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILTERED = PROJECT_ROOT / "data" / "intermediate" / "01_filtrado"


def words(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ'\u2019]+", str(s).lower()))


def main() -> None:
    extra = pd.read_csv(FILTERED / "extra.csv", encoding="utf-8-sig")
    extra = extra.drop_duplicates(subset=["ESP", "SHIWILU"]).reset_index(drop=True)
    extra["ESP_n"] = extra["ESP"].astype(str).str.lower().str.strip()
    extra["SHI_n"] = extra["SHIWILU"].astype(str).str.lower().str.strip()

    others = []
    for src in [
        "flashcards2",
        "flashcards_oraciones",
        "pdf_textos",
        "fidel_lomas",
        "vs_textos_narrativos",
        "cotidianas",
        "el_principito",
    ]:
        p = FILTERED / f"{src}.csv"
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8-sig")
            df["origin"] = src
            others.append(df[["ESP", "SHIWILU", "origin"]])
    other = pd.concat(others, ignore_index=True)
    other["ESP_n"] = other["ESP"].astype(str).str.lower().str.strip()
    other["SHI_n"] = other["SHIWILU"].astype(str).str.lower().str.strip()

    shi_vocab: set[str] = set()
    for s in other["SHIWILU"].astype(str):
        shi_vocab.update(words(s))
    esp_vocab: set[str] = set()
    for s in other["ESP"].astype(str):
        esp_vocab.update(words(s))

    def first_word(s: str) -> str:
        ws = list(words(s))
        return ws[0] if ws else ""

    extra["shi_word"] = extra["SHIWILU"].apply(first_word)
    extra["esp_word"] = extra["ESP"].apply(first_word)
    extra["shi_in_corpus"] = extra["shi_word"].isin(shi_vocab)
    extra["esp_in_corpus"] = extra["esp_word"].isin(esp_vocab)

    other_pairs = set(zip(other["ESP_n"], other["SHI_n"]))
    extra["pair_exact_dup"] = [
        (e, s) in other_pairs for e, s in zip(extra["ESP_n"], extra["SHI_n"])
    ]

    n = len(extra)
    print("=" * 60)
    print(f"TOTAL entradas en extra (post-dedupe interno): {n}")
    print("=" * 60)
    print(
        f"Pares ESP<->SHI EXACTOS ya en otras fuentes:    "
        f"{extra['pair_exact_dup'].sum():>4} ({extra['pair_exact_dup'].mean()*100:5.1f}%)"
    )
    print(
        f"Palabra SHI ya aparece en algun lado del resto: "
        f"{extra['shi_in_corpus'].sum():>4} ({extra['shi_in_corpus'].mean()*100:5.1f}%)"
    )
    new_shi = extra[~extra["shi_in_corpus"]].reset_index(drop=True)
    print(
        f"Palabra SHI NUEVA (no aparece en nada del resto): "
        f"{len(new_shi):>4} ({len(new_shi)/n*100:5.1f}%)"
    )
    print()
    print(">>> Las realmente NUEVAS (vocab shi no presente en el resto del corpus):")
    print(new_shi[["ESP", "SHIWILU"]].to_string())

    out = PROJECT_ROOT / "reports" / "extra_overlap.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    extra.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[wrote] {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
