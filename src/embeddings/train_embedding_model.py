from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from .dataset import ParallelEmbeddingDataset, collate_batch
from .losses import alignment_loss, bidirectional_contrastive_loss
from .model import EmbeddingEncoder


def _evaluate(model, loader, device, use_contrastive: bool) -> float:
    model.eval()
    total = 0.0

    with torch.no_grad():
        for batch in loader:
            shw = batch["shw_ids"].to(device)
            es = batch["es_ids"].to(device)

            z_shw = model.forward_sentence(shw)
            z_es = model.forward_sentence(es)

            loss = 0.0
            if use_contrastive:
                loss += 0.7 * bidirectional_contrastive_loss(z_shw, z_es)
                loss += 0.3 * alignment_loss(z_shw, z_es)
            else:
                loss += alignment_loss(z_shw, z_es)

            total += float(loss.item())

    return total / max(len(loader), 1)


def train(
    train_path: str | Path,
    val_path: str | Path,
    sp_model: str | Path,
    save_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    max_len: int = 64,
    lr: float = 3e-4,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    ff_dim: int = 1024,
    use_contrastive: bool = False,
) -> Path:
    """Run the full training loop and return the path to the best checkpoint."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Dispositivo: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model))
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()

    train_ds = ParallelEmbeddingDataset(str(train_path), str(sp_model), max_len=max_len)
    val_ds = ParallelEmbeddingDataset(str(val_path), str(sp_model), max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
    )

    model = EmbeddingEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        pad_id=pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0

        for batch in train_loader:
            shw = batch["shw_ids"].to(device)
            es = batch["es_ids"].to(device)

            z_shw = model.forward_sentence(shw)
            z_es = model.forward_sentence(es)

            if use_contrastive:
                loss = 0.7 * bidirectional_contrastive_loss(z_shw, z_es)
                loss += 0.3 * alignment_loss(z_shw, z_es)
            else:
                loss = alignment_loss(z_shw, z_es)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train += float(loss.item())

        train_loss = total_train / max(len(train_loader), 1)
        val_loss = _evaluate(model, val_loader, device, use_contrastive)

        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "pad_id": pad_id,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "ff_dim": ff_dim,
                    "use_contrastive": use_contrastive,
                    "best_val_loss": best_val,
                },
                save_path,
            )
            print(f"  -> checkpoint guardado ({save_path})")

    print(f"Mejor val loss: {best_val:.4f}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--sp_model", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--use_contrastive", action="store_true")
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        sp_model=args.sp_model,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        use_contrastive=args.use_contrastive,
    )


if __name__ == "__main__":
    main()
