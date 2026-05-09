from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .dataset import ParallelNMTDataset, collate_nmt
from .model import Seq2SeqTransformer
from .utils import pick_device, set_seed


def load_e0_matrix(e0_checkpoint: str) -> torch.Tensor:
    ckpt = torch.load(e0_checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"]

    candidate_keys = [
        "embedding.weight",
        "token_embed.weight",
        "token_embedding.weight",
        "src_embed.weight",
        "embed.weight",
    ]

    for key in candidate_keys:
        if key in state:
            print(f"Usando embedding preentrenado desde key: {key}")
            return state[key]

    available_2d = [k for k, v in state.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
    raise KeyError(
        "No se encontró una matriz de embeddings 2D conocida en el checkpoint E0. "
        f"Claves 2D disponibles: {available_2d}"
    )


# Helper para adaptar la matriz E0 al tamaño de embeddings de NMT
def adapt_e0_matrix_to_nmt(
    e0_matrix: torch.Tensor,
    target_weight: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    adapted = target_weight.detach().clone()

    rows = min(e0_matrix.size(0), adapted.size(0))
    cols = min(e0_matrix.size(1), adapted.size(1))
    adapted[:rows, :cols] = e0_matrix[:rows, :cols]

    if 0 <= pad_id < adapted.size(0):
        adapted[pad_id].zero_()

    return adapted


def shift_tgt_for_teacher_forcing(tgt_ids: torch.Tensor, pad_id: int):
    tgt_input = tgt_ids[:, :-1]
    tgt_output = tgt_ids[:, 1:]
    return tgt_input, tgt_output


def label_smoothed_nll_loss(logits, targets, pad_id: int, smoothing: float = 0.1):
    vocab_size = logits.size(-1)
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    mask = targets.ne(pad_id)
    logits = logits[mask]
    targets = targets[mask]

    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    loss = (1.0 - smoothing) * nll + smoothing * smooth
    return loss.mean()


@torch.no_grad()
def greedy_decode(
    model,
    src_ids: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 120,
    repetition_penalty: float = 1.3,
):
    model.eval()
    device = src_ids.device
    ys = torch.full((src_ids.size(0), 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        logits = model(src_ids, ys)
        next_logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            for i in range(ys.size(0)):
                prev_tokens = ys[i].unique()
                pos_mask = next_logits[i, prev_tokens] > 0
                next_logits[i, prev_tokens[pos_mask]] /= repetition_penalty
                next_logits[i, prev_tokens[~pos_mask]] *= repetition_penalty

        next_token = next_logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == eos_id).all():
            break

    return ys


def compute_mt_metrics(preds: list[str], refs: list[str]) -> tuple[float, float]:
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return bleu, chrf


def save_predictions(model, loader, sp, save_path: str, device: torch.device):
    model.eval()
    preds = []
    refs = []
    for batch in loader:
        src_ids = batch["src_ids"].to(device)
        out_ids = greedy_decode(
            model,
            src_ids,
            bos_id=sp.bos_id(),
            eos_id=sp.eos_id(),
            pad_id=sp.pad_id(),
            max_len=120,
        )
        for pred_ids, ref_text in zip(out_ids.cpu().tolist(), batch["tgt_text"]):
            pred_ids = [x for x in pred_ids if x not in {sp.pad_id(), sp.bos_id(), sp.eos_id()}]
            pred_text = sp.decode(pred_ids)
            preds.append(pred_text)
            refs.append(ref_text)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for pred, ref in zip(preds, refs):
            f.write(json.dumps({"prediction": pred, "reference": ref}, ensure_ascii=False) + "\n")

    bleu, chrf = compute_mt_metrics(preds, refs)
    return bleu, chrf


def train(args):
    set_seed(args.seed)
    device = pick_device()
    print(f"Seed: {args.seed}")
    print(f"Dispositivo: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)

    train_ds = ParallelNMTDataset(args.train_jsonl, args.sp_model, args.src_key, args.tgt_key, max_len=args.max_len)
    val_ds = ParallelNMTDataset(args.val_jsonl, args.sp_model, args.src_key, args.tgt_key, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_nmt(batch, sp.pad_id()),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_nmt(batch, sp.pad_id()),
    )

    model = Seq2SeqTransformer(
        vocab_size=sp.get_piece_size(),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        pad_id=sp.pad_id(),
        share_embeddings=args.share_embeddings,
    ).to(device)

    if args.e0_checkpoint:
        e0_matrix = load_e0_matrix(args.e0_checkpoint)
        print(f"Shape E0: {tuple(e0_matrix.shape)}")
        print(f"Shape NMT src_embed: {tuple(model.src_embed.weight.shape)}")

        adapted_e0 = adapt_e0_matrix_to_nmt(
            e0_matrix=e0_matrix,
            target_weight=model.src_embed.weight,
            pad_id=sp.pad_id(),
        )

        with torch.no_grad():
            model.src_embed.weight.copy_(adapted_e0)
            if model.tgt_embed is not model.src_embed:
                model.tgt_embed.weight.copy_(adapted_e0)

        if e0_matrix.shape == model.src_embed.weight.shape:
            print("Embeddings E0 cargados sin cambios.")
        else:
            print(
                "Embeddings E0 cargados con adaptación de forma "
                f"(checkpoint={tuple(e0_matrix.shape)} -> nmt={tuple(model.src_embed.weight.shape)})."
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = args.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    global_step = 0

    best_chrf = float("-inf")
    epochs_no_improve = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "checkpoint.pt"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {total_params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train = 0.0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)

            tgt_input, tgt_output = shift_tgt_for_teacher_forcing(tgt_ids, sp.pad_id())
            logits = model(src_ids, tgt_input)

            loss = label_smoothed_nll_loss(
                logits,
                tgt_output,
                pad_id=sp.pad_id(),
                smoothing=args.label_smoothing,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_train += float(loss.item())

        train_loss = total_train / max(len(train_loader), 1)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                tgt_input, tgt_output = shift_tgt_for_teacher_forcing(tgt_ids, sp.pad_id())
                logits = model(src_ids, tgt_input)
                loss = label_smoothed_nll_loss(
                    logits,
                    tgt_output,
                    pad_id=sp.pad_id(),
                    smoothing=args.label_smoothing,
                )
                total_val += float(loss.item())

        val_loss = total_val / max(len(val_loader), 1)
        val_pred_path = save_dir / "val_predictions.jsonl"
        bleu, chrf = save_predictions(model, val_loader, sp, str(val_pred_path), device)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} "
            f"| BLEU={bleu:.4f} | chrF={chrf:.4f} | lr={lr_now:.2e}"
        )

        if chrf > best_chrf:
            best_chrf = chrf
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "best_val_loss": val_loss,
                    "best_bleu": bleu,
                    "best_chrf": chrf,
                },
                checkpoint_path,
            )
            print(f"  -> checkpoint guardado en {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping en epoch {epoch} (patience={args.patience})")
                break

    best_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    final_bleu, final_chrf = save_predictions(
        model,
        val_loader,
        sp,
        str(save_dir / "val_predictions.jsonl"),
        device,
    )
    print(
        f"Predicciones del mejor checkpoint guardadas. "
        f"BLEU={final_bleu:.4f} | chrF={final_chrf:.4f}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--sp_model", required=True)
    parser.add_argument("--src_key", required=True, choices=["shiwilu", "spanish"])
    parser.add_argument("--tgt_key", required=True, choices=["shiwilu", "spanish"])
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--e0_checkpoint", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=160)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--share_embeddings", default=True, action="store_true")
    parser.add_argument("--no_share_embeddings", dest="share_embeddings", action="store_false")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()