"""Prototipo funcional (OE3): interfaz web para probar el traductor neuronal
castellano <-> shiwilu (SA-BiNLLB).

Carga una sola vez el modelo campeon (NLLB-200 + LoRA+ v2.1b, variante xl) y el
reranker semantico (E5-base bidireccional v3, variante xl), y expone una interfaz
Gradio simple: el usuario escribe una oracion, elige la direccion y obtiene la
traduccion. Las opciones tecnicas (reranking, alpha, candidatos alternativos)
estan ocultas tras un panel "Opciones avanzadas" para no recargar la vista.

Cada traduccion se registra en `reports/05_nmt/frontend_logs/session_<fecha>.jsonl`
(registro estructurado de entradas/salidas para analisis posterior, OE3).

Uso:
    .venv-nmt/Scripts/python app.py                # local + enlace publico temporal
    .venv-nmt/Scripts/python app.py --no-share     # solo local (127.0.0.1)
    .venv-nmt/Scripts/python app.py --no-rerank    # arranca con reranking apagado
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.nmt.inference.generate import (  # noqa: E402
    GenerationConfig,
    generate_for_direction,
    load_checkpoint,
)

# --- Rutas de modelos (campeon, variante xl) ---
CHECKPOINT = PROJECT_ROOT / "models" / "nmt" / "nllb_bidi_lora_v2_1b_loraplus_xl"
RERANKER = (
    PROJECT_ROOT
    / "models"
    / "sentence_transformers"
    / "v3_iterative_hn_e5_base_bidirectional_xl"
)
LOG_DIR = PROJECT_ROOT / "reports" / "05_nmt" / "frontend_logs"

DEFAULT_ALPHA = 0.7  # mejor alpha del campeon (ver CLAUDE.md)

# Nombres legibles por codigo de direccion.
LANG_NAMES = {"spa": "Castellano", "shw": "Shiwilu"}


def _dir_names(direction: str) -> tuple[str, str]:
    """('spa2shw') -> ('Castellano', 'Shiwilu')."""
    src, tgt = direction.split("2")
    return LANG_NAMES[src], LANG_NAMES[tgt]


# Ejemplos tomados del test set (data/processed/05_nmt_canonical_xl/test.csv).
# (texto, codigo_de_direccion)
EXAMPLES = [
    ("yo exageré.", "spa2shw"),
    ("me gusta el caldo de gallina.", "spa2shw"),
    ("sólo el pueblo de ellos se reproducirá.", "spa2shw"),
    ("kua a'nadalek", "shw2spa"),
    ("iyatulek wa'dante' dek", "shw2spa"),
    ("madettanna' a'ullinerku.", "shw2spa"),
]


def _build_runtime(load_reranker: bool):
    """Carga (una sola vez) configs, modelo NMT y, opcionalmente, el reranker."""
    with (PROJECT_ROOT / "config" / "nmt" / "training.yaml").open(encoding="utf-8") as f:
        training_yaml = yaml.safe_load(f)
    base_model = training_yaml["base_model"]
    lang_code_map = {str(k): str(v) for k, v in training_yaml["data"]["lang_code_map"].items()}

    gen_cfg = GenerationConfig.from_yaml(PROJECT_ROOT / "config" / "nmt" / "inference.yaml")

    print(f"[load] modelo NMT: {CHECKPOINT}")
    model, tokenizer, device = load_checkpoint(
        CHECKPOINT.resolve(), base_model=base_model, device="auto"
    )
    print(f"[load] device={device}, beam={gen_cfg.num_beams}")

    sbert = None
    if load_reranker:
        sbert = _load_sbert()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "lang_code_map": lang_code_map,
        "gen_cfg": gen_cfg,
        "sbert": sbert,
    }


def _load_sbert():
    from sentence_transformers import SentenceTransformer

    print(f"[load] reranker semantico: {RERANKER}")
    return SentenceTransformer(str(RERANKER))


RUNTIME: dict = {}


def _log_translation(record: dict) -> None:
    """Anexa una linea JSON por traduccion (registro estructurado, OE3)."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = LOG_DIR / f"session_{today}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # el registro nunca debe tumbar la interfaz
        print(f"[log] no se pudo escribir el registro: {exc}")


def translate(text: str, direction: str, use_rerank: bool, alpha: float):
    """Callback de la interfaz. Devuelve (traduccion, tabla_de_alternativas)."""
    text = (text or "").strip()
    if not text:
        return "", []

    src_plan, tgt_plan = direction.split("2")

    # El reranker se carga de forma perezosa si se activa despues del arranque.
    sbert = RUNTIME.get("sbert")
    if use_rerank and sbert is None:
        sbert = _load_sbert()
        RUNTIME["sbert"] = sbert

    df = pd.DataFrame(
        {
            "id": [f"FRONTEND__{direction}"],
            "pair_id": ["FRONTEND"],
            "source": [text],
            "target": [""],
        }
    )

    t0 = time.perf_counter()
    preds = generate_for_direction(
        RUNTIME["model"],
        RUNTIME["tokenizer"],
        df,
        src_plan=src_plan,
        tgt_plan=tgt_plan,
        lang_code_map=RUNTIME["lang_code_map"],
        cfg=RUNTIME["gen_cfg"],
        device=RUNTIME["device"],
        return_topk=use_rerank,
    )

    if not preds:
        return "(sin resultado)", []

    pred = preds[0]
    alternatives_rows: list[list] = []

    if use_rerank and sbert is not None and pred.get("candidates"):
        # final = alpha * p_trad + (1 - alpha) * cos(src, candidato)
        candidates = pred["candidates"]
        cand_texts = [c["hypothesis"] for c in candidates]
        src_emb = sbert.encode([text], normalize_embeddings=True)[0]
        cand_embs = sbert.encode(cand_texts, normalize_embeddings=True)
        cos_scores = (cand_embs @ src_emb)
        seq_scores = np.array([c["sequence_score"] for c in candidates], dtype=np.float64)
        p_trad = np.exp(seq_scores - seq_scores.max())
        p_trad = p_trad / p_trad.sum()
        finals = alpha * p_trad + (1.0 - alpha) * np.asarray(cos_scores)
        order = np.argsort(-finals)
        best = int(order[0])
        translation = cand_texts[best]
        alternatives_rows = [
            [cand_texts[i], round(float(finals[i]), 4), round(float(cos_scores[i]), 4)]
            for i in order
        ]
    else:
        translation = pred.get("hypothesis", "")

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    _log_translation(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "source_text": text,
            "output_text": translation,
            "rerank_on": bool(use_rerank),
            "alpha": float(alpha),
            "candidates": [
                {"hypothesis": r[0], "final_score": r[1]} for r in alternatives_rows
            ],
            "latency_ms": latency_ms,
        }
    )

    return translation, alternatives_rows


def _alt_chip_updates(alternatives_rows: list[list], chosen_text: str):
    """Updates para las 3 burbujas de sugerencia: las mejores alternativas
    distintas de la elegida. Devuelve 3 gr.update (valor + visibilidad)."""
    alts: list[str] = []
    for row in alternatives_rows:
        cand = row[0]
        if cand != chosen_text and cand not in alts:
            alts.append(cand)
        if len(alts) == 3:
            break
    updates = []
    for i in range(3):
        if i < len(alts):
            updates.append(gr.update(value=alts[i], visible=True))
        else:
            updates.append(gr.update(value="", visible=False))
    return updates


def translate_ui(text: str, direction: str, use_rerank: bool, alpha: float):
    """Envoltura para la interfaz: traduce y arma las burbujas de sugerencia + su titulo."""
    translation, rows = translate(text, direction, use_rerank, alpha)
    c1, c2, c3 = _alt_chip_updates(rows, translation)
    has_alts = any(u.get("visible") for u in (c1, c2, c3))
    label = gr.update(visible=has_alts)
    return translation, rows, c1, c2, c3, label


def auto_translate_cb(text: str, direction: str, use_rerank: bool, alpha: float, auto: bool):
    """Traduccion automatica mientras se escribe. Si esta apagada o el texto es
    muy corto, no cambia nada (gr.update() en cada salida)."""
    if not auto or len((text or "").strip()) < 2:
        return (gr.update(),) * 6
    return translate_ui(text, direction, use_rerank, alpha)


def _lang_pill(name: str) -> str:
    return f"<div class='lang-pill'>{name}</div>"


def _char_count(text: str) -> str:
    n = len(text or "")
    return f"<div class='char-count'>{n if n else ''}</div>"


CUSTOM_CSS = """
.gradio-container {max-width: 920px !important; margin: 0 auto !important;}
#app-header {text-align: center; margin: 8px 0 2px 0;}
#app-header h1 {font-size: 1.7rem; margin-bottom: 2px;}
#app-subtitle {text-align: center; color: var(--body-text-color-subdued); margin-bottom: 14px;}
#lang-bar {align-items: center; margin-bottom: -8px;}
.lang-pill {text-align: center; font-weight: 600; font-size: 1.05rem; padding: 8px 0;}
.pane-col {position: relative;}
#char-count {position: absolute; bottom: 8px; right: 0; left: 0; padding-right: 14px; text-align: right; margin: 0; z-index: 2; pointer-events: none;}
.char-count {font-size: 0.78rem; color: var(--body-text-color-subdued);}
#alt-label {font-size: 0.8rem; font-weight: 600; color: var(--body-text-color-subdued); margin: 2px 0 0 4px;}
#swap-btn {min-width: 44px !important; max-width: 56px; border-radius: 999px !important; font-size: 1.2rem; padding: 0;}
#translate-btn {max-width: 280px; margin: 4px auto 0 auto;}
#ex-title {margin: 10px 0 2px 2px; font-weight: 600; color: var(--body-text-color-subdued);}
.ex-chip button {font-size: 0.85rem !important; font-weight: 400 !important; text-align: left !important; white-space: normal !important;}
#footer {text-align: center; color: var(--body-text-color-subdued); font-size: 0.82rem; margin-top: 16px;}
#top-bar {align-items: center; margin-bottom: 4px;}
#theme-btn {min-width: 40px !important; max-width: 48px; border-radius: 999px !important; font-size: 1.15rem; padding: 0;}
#alt-row {gap: 6px; margin-top: 4px; flex-wrap: wrap; justify-content: flex-start;}
.alt-chip button {font-size: 0.8rem !important; font-weight: 400 !important; padding: 4px 12px !important; border-radius: 999px !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%;}
.auto-box {align-self: center;}
"""


def build_demo(default_rerank: bool) -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
        radius_size="lg",
    )

    with gr.Blocks(title="Traductor castellano ↔ shiwilu", theme=theme, css=CUSTOM_CSS) as demo:
        # Direccion actual como estado ('spa2shw' por defecto).
        direction = gr.State("spa2shw")

        with gr.Row(elem_id="top-bar"):
            with gr.Column(scale=20):
                gr.HTML(
                    "<div id='app-header'><h1>Traductor Castellano – Shiwilu</h1></div>"
                    "<div id='app-subtitle'>Traduce oraciones en ambos sentidos</div>"
                )
            with gr.Column(scale=1, min_width=48):
                theme_btn = gr.Button("🌙", elem_id="theme-btn", variant="secondary")

        with gr.Group():
            # Barra de idiomas: origen · ⇄ · destino
            with gr.Row(elem_id="lang-bar"):
                with gr.Column(scale=10, min_width=120):
                    src_lbl = gr.HTML(_lang_pill("Castellano"))
                with gr.Column(scale=1, min_width=56):
                    swap_btn = gr.Button("⇄", elem_id="swap-btn", variant="secondary")
                with gr.Column(scale=10, min_width=120):
                    tgt_lbl = gr.HTML(_lang_pill("Shiwilu"))

            # Paneles de texto, lado a lado (identicos: solo el textbox en cada columna).
            # El contador va dentro del panel de entrada como overlay (no afecta la altura).
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="pane-col"):
                    text_in = gr.Textbox(
                        show_label=False,
                        placeholder="Escribe aqui la oracion a traducir...",
                        lines=6,
                        max_lines=12,
                        autofocus=True,
                    )
                    char_count = gr.HTML(_char_count(""), elem_id="char-count")
                with gr.Column(scale=1):
                    text_out = gr.Textbox(
                        show_label=False,
                        placeholder="La traduccion aparecera aqui...",
                        lines=6,
                        max_lines=12,
                        interactive=False,
                        show_copy_button=True,
                    )

        # Debajo de los paneles: ajuste (bajo la entrada) y sugerencias (bajo la salida).
        with gr.Row():
            with gr.Column(scale=1):
                auto_translate = gr.Checkbox(
                    value=False, label="Traducir mientras escribo", elem_classes="auto-box"
                )
            with gr.Column(scale=1):
                alt_label = gr.HTML("<div id='alt-label'>Otras opciones</div>", visible=False)
                with gr.Row(elem_id="alt-row"):
                    alt1 = gr.Button("", visible=False, size="sm", elem_classes="alt-chip")
                    alt2 = gr.Button("", visible=False, size="sm", elem_classes="alt-chip")
                    alt3 = gr.Button("", visible=False, size="sm", elem_classes="alt-chip")

        with gr.Row():
            clear_btn = gr.Button("Limpiar", variant="secondary", scale=1)
            translate_btn = gr.Button("Traducir", variant="primary", scale=2, elem_id="translate-btn")

        with gr.Accordion("Opciones avanzadas", open=False):
            gr.Markdown(
                "Controles tecnicos. El **reranking semantico** reordena las hipotesis "
                "del modelo combinando la probabilidad de traduccion con la similitud "
                "semantica; **alpha** es el peso de la probabilidad de traduccion."
            )
            with gr.Row():
                use_rerank = gr.Checkbox(value=default_rerank, label="Reranking semantico")
                alpha = gr.Slider(
                    minimum=0.0, maximum=1.0, value=DEFAULT_ALPHA, step=0.05,
                    label="alpha (peso de la probabilidad de traduccion)",
                )
            alternatives = gr.Dataframe(
                headers=["Candidato", "Puntaje final", "Similitud semantica"],
                datatype=["str", "number", "number"],
                label="Hipotesis alternativas (ordenadas por puntaje)",
                wrap=True,
            )

        alt_chips = [alt1, alt2, alt3]
        gr.HTML("<div id='ex-title'>Ejemplos</div>")
        with gr.Row():
            for ex_text, ex_dir in EXAMPLES:
                label = ex_text if len(ex_text) <= 38 else ex_text[:36] + "…"
                tag = "ES→SHW" if ex_dir == "spa2shw" else "SHW→ES"
                chip = gr.Button(f"{label}  ·  {tag}", size="sm", elem_classes="ex-chip")
                chip.click(
                    _make_example_loader(ex_text, ex_dir),
                    inputs=None,
                    outputs=[text_in, direction, src_lbl, tgt_lbl, char_count, text_out, *alt_chips, alt_label],
                )

        gr.HTML(
            "<div id='footer'>Prototipo experimental (SA-BiNLLB) · "
            "NLLB-200 + LoRA+ (v2.1b, xl) · reranker E5-base (v3, xl) · chrF++ 44.99 · "
            "las traducciones pueden contener errores</div>"
        )

        # --- Eventos ---
        gen_inputs = [text_in, direction, use_rerank, alpha]
        gen_outputs = [text_out, alternatives, *alt_chips, alt_label]
        translate_btn.click(translate_ui, inputs=gen_inputs, outputs=gen_outputs)
        text_in.submit(translate_ui, inputs=gen_inputs, outputs=gen_outputs)
        # Contador de caracteres en el navegador (sin viaje al servidor -> sin parpadeo de "cargando").
        text_in.change(
            fn=None,
            inputs=text_in,
            outputs=char_count,
            js="(t) => { const n=(t||'').length; return `<div class='char-count'>${n? n : ''}</div>`; }",
        )
        # Traduccion automatica: coalescida (always_last) y sin spinner de carga mientras se escribe.
        text_in.change(
            auto_translate_cb,
            inputs=[*gen_inputs, auto_translate],
            outputs=gen_outputs,
            show_progress="hidden",
        )

        # Clic en una sugerencia: la pasa al panel de traduccion.
        for chip_btn in alt_chips:
            chip_btn.click(lambda v: v, inputs=chip_btn, outputs=text_out)

        # Arrancar en modo claro (ignorar preferencia del sistema) y poner tooltips.
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js=(
                "() => {"
                " document.body.classList.remove('dark');"
                " const s=document.querySelector('#swap-btn button'); if(s) s.title='Intercambiar idiomas';"
                " const t=document.querySelector('#theme-btn button'); if(t) t.title='Modo claro u oscuro';"
                "}"
            ),
        )
        # Modo oscuro (toggle puro en el navegador, sin viaje a Python).
        theme_btn.click(fn=None, inputs=None, outputs=None, js="() => { document.body.classList.toggle('dark'); }")

        swap_btn.click(
            fn=None,
            inputs=[direction, text_in, text_out],
            outputs=[direction, src_lbl, tgt_lbl, text_in, text_out, char_count],
            js="""(dir, inVal, outVal) => {
                const newDir = dir === 'spa2shw' ? 'shw2spa' : 'spa2shw';
                const newIn = (outVal || '').trim() ? outVal : inVal;
                const src = newDir === 'spa2shw' ? 'Castellano' : 'Shiwilu';
                const tgt = newDir === 'spa2shw' ? 'Shiwilu' : 'Castellano';
                const pill = n => `<div class='lang-pill'>${n}</div>`;
                const count = (newIn || '').length;
                const cc = count ? `<div class='char-count'>${count}</div>` : "<div class='char-count'></div>";
                return [newDir, pill(src), pill(tgt), newIn, '', cc];
            }""",
        )

        clear_btn.click(
            lambda: ("", "", _char_count(""), [], *_hide_chips(), gr.update(visible=False)),
            inputs=None,
            outputs=[text_in, text_out, char_count, alternatives, *alt_chips, alt_label],
        )

    return demo


def _hide_chips():
    return (gr.update(value="", visible=False),) * 3


def _swap_direction(direction: str, in_val: str, out_val: str):
    """Invierte la direccion y lleva la traduccion al panel de entrada (estilo Google)."""
    new_dir = "shw2spa" if direction == "spa2shw" else "spa2shw"
    src_name, tgt_name = _dir_names(new_dir)
    new_in = out_val if (out_val or "").strip() else in_val
    return new_dir, _lang_pill(src_name), _lang_pill(tgt_name), new_in, "", _char_count(new_in)


def _make_example_loader(ex_text: str, ex_dir: str):
    """Devuelve un callback que carga un ejemplo (texto + direccion) en la interfaz."""
    src_name, tgt_name = _dir_names(ex_dir)

    def _loader():
        return (
            ex_text,
            ex_dir,
            _lang_pill(src_name),
            _lang_pill(tgt_name),
            _char_count(ex_text),
            "",
            *_hide_chips(),
            gr.update(visible=False),
        )

    return _loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--no-share",
        action="store_true",
        help="No crear enlace publico temporal; servir solo en 127.0.0.1.",
    )
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Arrancar con el reranking apagado (no carga el reranker hasta activarlo).",
    )
    p.add_argument("--port", type=int, default=7860, help="Puerto local (default 7860).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    default_rerank = not args.no_rerank

    global RUNTIME
    RUNTIME = _build_runtime(load_reranker=default_rerank)

    demo = build_demo(default_rerank=default_rerank)
    demo.launch(share=not args.no_share, server_name="127.0.0.1", server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
