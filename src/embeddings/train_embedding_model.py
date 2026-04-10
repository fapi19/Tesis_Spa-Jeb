from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from .dataset import ParallelEmbeddingDataset, collate_batch
from .losses import total_embedding_loss
from .model import EmbeddingEncoder
from .utils_seed import set_seed
from .samplers import NoDuplicateBatchSampler


@torch.no_grad()
def _evaluate(
    model: EmbeddingEncoder,
    loader: DataLoader,
    device: torch.device,
    *,
    use_focal: bool,
    use_alignment: bool,
    contrastive_weight: float,
    alignment_weight: float,
    temperature: float,
    gamma: float,
) -> float:
    model.eval()
    total = 0.0

    for batch in loader:
        shw = batch["shw_ids"].to(device)
        es = batch["es_ids"].to(device)

        z_shw = model.forward_sentence(shw)
        z_es = model.forward_sentence(es)

        loss = total_embedding_loss(
            z_shw=z_shw,
            z_es=z_es,
            use_focal=use_focal,
            use_alignment=use_alignment,
            contrastive_weight=contrastive_weight,
            alignment_weight=alignment_weight,
            temperature=temperature,
            gamma=gamma,
        )
        total += float(loss.item())

    return total / max(len(loader), 1)


@torch.no_grad()
def retrieval_at_k(
    model: EmbeddingEncoder,
    loader: DataLoader,
    device: torch.device,
    k: int = 1,
) -> float:
    model.eval()

    shw_embs = []
    es_embs = []

    for batch in loader:
        shw = batch["shw_ids"].to(device)
        es = batch["es_ids"].to(device)

        z_shw = model.forward_sentence(shw)
        z_es = model.forward_sentence(es)

        shw_embs.append(z_shw.cpu())
        es_embs.append(z_es.cpu())

    shw_embs = torch.cat(shw_embs, dim=0)
    es_embs = torch.cat(es_embs, dim=0)

    sims = shw_embs @ es_embs.T
    topk = sims.topk(k=k, dim=1).indices

    correct = 0
    for i in range(topk.size(0)):
        if i in topk[i]:
            correct += 1

    return correct / topk.size(0)


@torch.no_grad()
def mean_reciprocal_rank(
    model: EmbeddingEncoder,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    shw_embs = []
    es_embs = []

    for batch in loader:
        shw = batch["shw_ids"].to(device)
        es = batch["es_ids"].to(device)

        z_shw = model.forward_sentence(shw)
        z_es = model.forward_sentence(es)

        shw_embs.append(z_shw.cpu())
        es_embs.append(z_es.cpu())

    shw_embs = torch.cat(shw_embs, dim=0)
    es_embs = torch.cat(es_embs, dim=0)

    sims = shw_embs @ es_embs.T
    ranks = sims.argsort(dim=1, descending=True)

    rr_sum = 0.0
    for i in range(ranks.size(0)):
        rank_pos = (ranks[i] == i).nonzero(as_tuple=True)[0].item() + 1
        rr_sum += 1.0 / rank_pos

    return rr_sum / ranks.size(0)


def train(
    train_path: str | Path,
    val_path: str | Path,
    sp_model: str | Path,
    save_path: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    max_len: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    d_model: int = 192,
    nhead: int = 4,
    num_layers: int = 2,
    ff_dim: int = 768,
    dropout: float = 0.2,
    use_focal: bool = False,
    use_alignment: bool = False,
    contrastive_weight: float = 1.0,
    alignment_weight: float = 0.3,
    temperature: float = 0.05,
    gamma: float = 2.0,
    seed: int = 42,
    subword_regularization: bool = False,
    nbest_size: int = -1,
    alpha: float = 0.1,
) -> Path:
    """Run the full training loop with no-duplicate training batches and optional subword regularization, then return the best checkpoint path."""
    set_seed(seed)
    print(f"  Seed: {seed}")

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

    train_ds = ParallelEmbeddingDataset(
        str(train_path),
        str(sp_model),
        max_len=max_len,
        subword_regularization=subword_regularization,
        nbest_size=nbest_size,
        alpha=alpha,
    )
    val_ds = ParallelEmbeddingDataset(
        str(val_path),
        str(sp_model),
        max_len=max_len,
        subword_regularization=False,
    )

    train_sampler = NoDuplicateBatchSampler(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
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
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_r1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0

        for batch in train_loader:
            shw = batch["shw_ids"].to(device)
            es = batch["es_ids"].to(device)

            z_shw = model.forward_sentence(shw)
            z_es = model.forward_sentence(es)

            loss = total_embedding_loss(
                z_shw=z_shw,
                z_es=z_es,
                use_focal=use_focal,
                use_alignment=use_alignment,
                contrastive_weight=contrastive_weight,
                alignment_weight=alignment_weight,
                temperature=temperature,
                gamma=gamma,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train += float(loss.item())

        train_loss = total_train / max(len(train_loader), 1)
        val_loss = _evaluate(
            model,
            val_loader,
            device,
            use_focal=use_focal,
            use_alignment=use_alignment,
            contrastive_weight=contrastive_weight,
            alignment_weight=alignment_weight,
            temperature=temperature,
            gamma=gamma,
        )

        r1 = retrieval_at_k(model, val_loader, device, k=1)
        r5 = retrieval_at_k(model, val_loader, device, k=5)
        mrr = mean_reciprocal_rank(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"R@1={r1:.4f} | R@5={r5:.4f} | MRR={mrr:.4f}"
        )

        if r1 > best_r1:
            best_r1 = r1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "pad_id": pad_id,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "ff_dim": ff_dim,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "use_focal": use_focal,
                    "use_alignment": use_alignment,
                    "contrastive_weight": contrastive_weight,
                    "alignment_weight": alignment_weight,
                    "temperature": temperature,
                    "gamma": gamma,
                    "seed": seed,
                    "subword_regularization": subword_regularization,
                    "nbest_size": nbest_size,
                    "alpha": alpha,
                    "best_r1": best_r1,
                    "best_r5": r5,
                    "best_mrr": mrr,
                    "val_loss": val_loss,
                },
                save_path,
            )
            print(f"  -> checkpoint guardado ({save_path})")

    print(f"Mejor R@1: {best_r1:.4f}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--sp_model", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--use_alignment", action="store_true")
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--alignment_weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subword_regularization", action="store_true")
    parser.add_argument("--nbest_size", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.1)
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
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_focal=args.use_focal,
        use_alignment=args.use_alignment,
        contrastive_weight=args.contrastive_weight,
        alignment_weight=args.alignment_weight,
        temperature=args.temperature,
        gamma=args.gamma,
        seed=args.seed,
        subword_regularization=args.subword_regularization,
        nbest_size=args.nbest_size,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()