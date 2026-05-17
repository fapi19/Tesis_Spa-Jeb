"""Interactive translator for SA-BiNLLB.

Loads a trained NMT checkpoint and lets you type sentences to translate.

Usage:
    .venv-nmt/Scripts/python -m scripts.translate_interactive
    .venv-nmt/Scripts/python -m scripts.translate_interactive --checkpoint models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl
    .venv-nmt/Scripts/python -m scripts.translate_interactive --rerank

Commands inside the prompt:
    > spa: Hola, ¿cómo estás?           # forces spa->shw direction
    > shw: ñapalek wila                 # forces shw->spa direction
    > Hola, ¿cómo estás?                # auto-detect direction (heuristic)
    > /switch                           # toggle default direction
    > /quit                             # exit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.nmt.inference.generate import (  # noqa: E402
    GenerationConfig,
    generate_for_direction,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl",
        help="Path to NMT checkpoint (default: best v2.1b LoRA+ xl)",
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        help="Apply semantic reranking on top-5 hypotheses (slower, usually better)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Reranker alpha (only with --rerank). Default 0.5.",
    )
    p.add_argument(
        "--default-direction",
        choices=["spa2shw", "shw2spa"],
        default="spa2shw",
        help="Default direction when not specified",
    )
    return p.parse_args()


def looks_shiwilu(text: str) -> bool:
    """Heuristic: shiwilu often has apostrophes (laryngealized vowels) or
    consonant clusters atypical in Spanish."""
    t = text.lower().strip()
    if "'" in t:
        return True
    # shiwilu-typical sequences
    shi_markers = ("kk", "tt", "shw", "kh", "kp", "pk", "tk", "ts", "wek", "lek", "tek")
    return any(m in t for m in shi_markers)


def translate(text: str, direction: str, model, tokenizer, lang_code_map, gen_cfg, device,
              rerank: bool, alpha: float, sbert=None):
    """Translate one text."""
    src_plan, tgt_plan = direction.split("2")
    df = pd.DataFrame(
        {
            "id": ["INTERACTIVE__" + direction],
            "pair_id": ["INTERACTIVE"],
            "source": [text],
            "target": [""],
        }
    )
    preds = generate_for_direction(
        model, tokenizer, df,
        src_plan=src_plan, tgt_plan=tgt_plan,
        lang_code_map=lang_code_map, cfg=gen_cfg, device=device,
        return_topk=rerank,
    )
    if not preds:
        return None, []
    pred = preds[0]
    if rerank and sbert is not None and pred.get("candidates"):
        # Apply reranking: final = alpha * p_trad + (1-alpha) * cos(src, candidate)
        import numpy as np
        candidates = pred["candidates"]
        src_emb = sbert.encode([text], normalize_embeddings=True)[0]
        cand_texts = [c["hypothesis"] for c in candidates]
        cand_embs = sbert.encode(cand_texts, normalize_embeddings=True)
        cos_scores = (cand_embs @ src_emb).tolist()
        # softmax over sequence_scores
        seq_scores = np.array([c["sequence_score"] for c in candidates])
        p_trad = np.exp(seq_scores - seq_scores.max())
        p_trad = p_trad / p_trad.sum()
        finals = alpha * p_trad + (1 - alpha) * np.array(cos_scores)
        best = int(finals.argmax())
        return cand_texts[best], list(zip(cand_texts, finals.tolist()))
    return pred.get("hypothesis", ""), []


def main() -> int:
    args = parse_args()

    print(f"[load] loading checkpoint: {args.checkpoint}")
    with (PROJECT_ROOT / "config" / "nmt" / "training.yaml").open(encoding="utf-8") as f:
        training_yaml = yaml.safe_load(f)
    base_model = training_yaml["base_model"]
    lang_code_map = {str(k): str(v) for k, v in training_yaml["data"]["lang_code_map"].items()}

    gen_cfg = GenerationConfig.from_yaml(PROJECT_ROOT / "config" / "nmt" / "inference.yaml")
    model, tokenizer, device = load_checkpoint(
        Path(args.checkpoint).resolve(), base_model=base_model, device="auto"
    )
    print(f"[load] device={device}, beam={gen_cfg.num_beams}")

    sbert = None
    if args.rerank:
        from sentence_transformers import SentenceTransformer
        sbert_path = PROJECT_ROOT / "models" / "sentence_transformers" / "v3_iterative_hn_e5_base_bidirectional_xl"
        print(f"[load] loading reranker: {sbert_path}")
        sbert = SentenceTransformer(str(sbert_path))

    direction = args.default_direction
    print()
    print("=" * 60)
    print("  SA-BiNLLB Interactive Translator")
    print("=" * 60)
    print(f"  Default direction: {direction}")
    print(f"  Reranking: {'ON (alpha=' + str(args.alpha) + ')' if args.rerank else 'OFF'}")
    print()
    print("  Commands:")
    print("    spa: <text>    force Spanish -> Shiwilu")
    print("    shw: <text>    force Shiwilu -> Spanish")
    print("    /switch        toggle default direction")
    print("    /rerank        toggle reranking")
    print("    /quit          exit")
    print()

    while True:
        try:
            line = input(f"[{direction}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit", "q"):
            return 0
        if line == "/switch":
            direction = "shw2spa" if direction == "spa2shw" else "spa2shw"
            print(f"[direction] -> {direction}")
            continue
        if line == "/rerank":
            args.rerank = not args.rerank
            if args.rerank and sbert is None:
                from sentence_transformers import SentenceTransformer
                sbert_path = PROJECT_ROOT / "models" / "sentence_transformers" / "v3_iterative_hn_e5_base_bidirectional_xl"
                sbert = SentenceTransformer(str(sbert_path))
            print(f"[rerank] -> {'ON' if args.rerank else 'OFF'}")
            continue

        forced_dir = None
        if line.lower().startswith("spa:"):
            forced_dir = "spa2shw"; line = line[4:].strip()
        elif line.lower().startswith("shw:"):
            forced_dir = "shw2spa"; line = line[4:].strip()

        use_dir = forced_dir or direction
        translation, alternatives = translate(
            line, use_dir, model, tokenizer, lang_code_map, gen_cfg, device,
            rerank=args.rerank, alpha=args.alpha, sbert=sbert,
        )
        print(f"  {use_dir}: {translation}")
        if alternatives:
            print(f"  alternatives:")
            for cand, score in sorted(alternatives, key=lambda x: -x[1])[:3]:
                print(f"    [{score:.3f}] {cand}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
