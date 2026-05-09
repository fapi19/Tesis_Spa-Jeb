from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu


def load_predictions(path: str):
    preds = []
    refs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            preds.append(row["prediction"].strip())
            refs.append(row["reference"].strip())
    return preds, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_jsonl", required=True)
    args = parser.parse_args()

    preds, refs = load_predictions(args.predictions_jsonl)

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])

    print(f"BLEU: {bleu.score:.4f}")
    print(f"chrF: {chrf.score:.4f}")


if __name__ == "__main__":
    main()