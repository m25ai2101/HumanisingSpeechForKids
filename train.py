"""
Training Entry Point

Trains the ProsodyPredictor model on paired text+prosody data produced by
data_pipeline.py.

Usage:
    python train.py --data data/train.jsonl --val data/val.jsonl \
                    --epochs 30 --output checkpoints/

The best checkpoint (lowest val loss) is saved to:
    checkpoints/best.pt

Training progress is logged to:
    checkpoints/loss_log.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import prosody_model as pm
from prosody_model import ProsodyDataset, ProsodyPredictor


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def prosody_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    L1 loss on all four heads, masked to real (non-padding) tokens only.
    mask: (B, T) boolean tensor, True for real tokens.
    """
    loss = torch.tensor(0.0, device=mask.device)
    pairs = [
        (pred["pitch_shift"],    batch["pitch_targets"]),
        (pred["duration_ratio"], batch["duration_targets"]),
        (pred["volume"],         batch["volume_targets"]),
        (pred["pause_after"],    batch["pause_targets"]),
    ]
    for p, t in pairs:
        # Only compute loss on non-padding positions
        loss = loss + (p[mask] - t[mask]).abs().mean()
    return loss / len(pairs)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    train_path: str,
    val_path:   str,
    output_dir: str = "checkpoints",
    bert_model_id: str = "bert-base-uncased",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    resume: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ---- Data ---------------------------------------------------------------
    train_ds = ProsodyDataset(train_path, bert_model_id)
    val_ds   = ProsodyDataset(val_path,   bert_model_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Train] {len(train_ds)} train / {len(val_ds)} val utterances")

    # ---- Model --------------------------------------------------------------
    model = ProsodyPredictor(bert_model_id).to(device)

    start_epoch = 0
    best_val_loss = float("inf")

    if resume and Path(resume).exists():
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch   = state.get("epoch", 0) + 1
        best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"[Train] Resumed from '{resume}' (epoch {start_epoch}, best_val={best_val_loss:.4f})")

    # ---- Optimiser + scheduler ----------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    total_steps   = len(train_loader) * epochs
    warmup_steps  = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Output dir + CSV log -----------------------------------------------
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "loss_log.csv"

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    global_step = start_epoch * len(train_loader)

    # ---- Main loop ----------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mask  = batch["attention_mask"].bool()

            optimizer.zero_grad()
            pred = model(batch["input_ids"], batch["attention_mask"], batch["emotion_vec"])
            loss = prosody_loss(pred, batch, mask)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_loss_sum += loss.item()

        avg_train = train_loss_sum / len(train_loader)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mask  = batch["attention_mask"].bool()
                pred  = model(batch["input_ids"], batch["attention_mask"], batch["emotion_vec"])
                val_loss_sum += prosody_loss(pred, batch, mask).item()

        avg_val = val_loss_sum / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"[Train] Epoch {epoch + 1}/{epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  lr={current_lr:.2e}")

        # Log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_train:.6f}", f"{avg_val:.6f}", f"{current_lr:.2e}"])

        # Checkpoint every epoch
        state = {
            "epoch":          epoch,
            "model":          model.state_dict(),
            "best_val_loss":  best_val_loss,
        }
        torch.save(state, out / f"epoch_{epoch + 1:03d}.pt")

        # Best checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(state | {"best_val_loss": best_val_loss}, out / "best.pt")
            print(f"[Train] ✓ New best val_loss={best_val_loss:.4f}  → checkpoints/best.pt")

    print(f"\n[Train] Done. Best val_loss={best_val_loss:.4f}")
    print(f"[Train] Checkpoint: {out / 'best.pt'}")
    print(f"[Train] Loss log:   {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProsodyPredictor on kids story data")
    parser.add_argument("--data",    required=True, help="Path to train.jsonl")
    parser.add_argument("--val",     required=True, help="Path to val.jsonl")
    parser.add_argument("--output",  default="checkpoints", help="Output dir for checkpoints")
    parser.add_argument("--bert",    default="bert-base-uncased", help="BERT model id")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--warmup",  type=float, default=0.1,  help="Warmup fraction of total steps")
    parser.add_argument("--resume",  default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    train(
        train_path=args.data,
        val_path=args.val,
        output_dir=args.output,
        bert_model_id=args.bert,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        warmup_ratio=args.warmup,
        resume=args.resume,
    )
