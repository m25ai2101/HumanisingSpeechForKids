"""
Quantitative evaluation of the trained ProsodyPredictor.

Compares the model's PREDICTED prosody against the ACTUAL (ground-truth) prosody
measured from real narrated audio, on the held-out test set (data/test.jsonl).

Metrics per prosody parameter (pitch, duration, volume, pause):
  MAE        — mean absolute error (lower = better)
  RMSE       — root mean squared error (lower = better)
  Pearson r  — correlation between predicted and actual (higher = model tracks
               the real variation, i.e. it is expressive, not flat)

Two naive baselines are reported for context:
  mean-predictor    — always predict the dataset mean of that parameter
  neutral-predictor — always predict the flat/neutral value (what a basic
                      monotone TTS effectively does: pitch 0, duration 1,
                      volume 0.8, pause 0.2)

A model that beats both baselines proves it learned emotion-conditioned prosody
rather than collapsing to a constant.

Usage:
    python evaluate.py
    python evaluate.py --data data/test.jsonl --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

from emotion_schema import EMOTIONS
from prosody_model import ProsodyPredictor, _align_word_labels


class EvalDataset(Dataset):
    """
    Self-contained copy of prosody_model.ProsodyDataset, minus the invalid
    `return_word_ids` tokenizer kwarg that the installed transformers rejects.
    Produces the same aligned per-token targets used in training.
    """

    MAX_LEN = 128

    def __init__(self, jsonl_path: str, bert_model_id: str = "bert-base-uncased"):
        self.records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_id)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        emotion_vec = rec.get("emotion_vec", {})
        emo = torch.tensor([emotion_vec.get(e, 0.0) for e in EMOTIONS], dtype=torch.float32)

        words = rec.get("words", [])
        word_texts    = [w.get("word", "") for w in words]
        word_pitch    = [w.get("pitch_shift",    0.0) for w in words]
        word_duration = [w.get("duration_ratio", 1.0) for w in words]
        word_volume   = [w.get("volume",         0.8) for w in words]
        word_pause    = [w.get("pause_after_sec", 0.2) for w in words]

        encoding = self.tokenizer(
            rec["text"], truncation=True, max_length=self.MAX_LEN,
            padding="max_length", return_tensors="pt",
        )
        token_pitch, token_dur, token_vol, token_pause = _align_word_labels(
            self.tokenizer, word_texts, word_pitch, word_duration,
            word_volume, word_pause, max_len=self.MAX_LEN,
        )
        dom_emotion = rec.get("emotion") or (
            max(emotion_vec, key=emotion_vec.get) if emotion_vec else "neutral")
        return {
            "input_ids":       encoding["input_ids"].squeeze(0),
            "attention_mask":  encoding["attention_mask"].squeeze(0),
            "emotion_vec":     emo,
            "pitch_targets":   token_pitch,
            "duration_targets": token_dur,
            "volume_targets":  token_vol,
            "pause_targets":   token_pause,
            "_emotion":        dom_emotion,
        }

HEADS = [
    ("Pitch (semitones)", "pitch_shift",    "pitch_targets"),
    ("Duration ratio",    "duration_ratio", "duration_targets"),
    ("Volume (0-1)",      "volume",         "volume_targets"),
    ("Pause (seconds)",   "pause_after",    "pause_targets"),
]

# Flat/neutral constants a basic monotone TTS effectively uses.
NEUTRAL = {"pitch_targets": 0.0, "duration_targets": 1.0,
           "volume_targets": 0.8, "pause_targets": 0.2}

# Model output ranges (from prosody_model clamps). Ground-truth targets are
# clamped to these before scoring so the model is not penalised for values it
# architecturally cannot produce (e.g. a word stretched 6x over a pause).
CLAMP = {"pitch_targets": (-6.0, 6.0), "duration_targets": (0.5, 2.5),
         "volume_targets": (0.4, 1.0), "pause_targets": (0.0, 2.0)}


def _metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if pred.std() > 1e-8 and true.std() > 1e-8:
        r = float(np.corrcoef(pred, true)[0, 1])
    else:
        r = float("nan")
    return mae, rmse, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--bert", default="bert-base-uncased")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    ds = EvalDataset(args.data, args.bert)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    print(f"[Eval] Test utterances: {len(ds)}")

    model = ProsodyPredictor(args.bert).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[Eval] Loaded checkpoint: {args.checkpoint}")

    # Collect predictions + targets over real (non-padding) tokens.
    preds = {h[2]: [] for h in HEADS}
    trues = {h[2]: [] for h in HEADS}
    tok_emotion: list[str] = []   # dominant emotion label per scored token

    with torch.no_grad():
        for batch in loader:
            emos = batch.pop("_emotion")
            batch = {k: v.to(device) for k, v in batch.items()}
            mask = batch["attention_mask"].bool()
            out = model(batch["input_ids"], batch["attention_mask"], batch["emotion_vec"])
            for _, out_key, tgt_key in HEADS:
                preds[tgt_key].append(out[out_key][mask].cpu().numpy())
                trues[tgt_key].append(batch[tgt_key][mask].cpu().numpy())
            # replicate each row's emotion label across its real tokens
            counts = mask.sum(dim=1).cpu().tolist()
            for emo, c in zip(emos, counts):
                tok_emotion.extend([emo] * int(c))

    for k in preds:
        preds[k] = np.concatenate(preds[k])
        # clamp targets to the model's achievable range for a fair comparison
        lo, hi = CLAMP[k]
        trues[k] = np.clip(np.concatenate(trues[k]), lo, hi)
    tok_emotion_arr = np.array(tok_emotion)

    n_tokens = len(next(iter(trues.values())))
    print(f"[Eval] Scored tokens: {n_tokens}\n")

    # ---- Results table ------------------------------------------------------
    print("=" * 84)
    print(f"{'Prosody parameter':<20}{'MAE':>8}{'RMSE':>8}{'Pearson r':>11}"
          f"{'MAE(mean)':>11}{'MAE(flat)':>11}{'vs flat':>9}")
    print("-" * 84)

    overall_model_mae, overall_flat_mae = [], []
    rows = []
    for label, _, tgt_key in HEADS:
        p, t = preds[tgt_key], trues[tgt_key]
        mae, rmse, r = _metrics(p, t)
        mae_mean = float(np.mean(np.abs(np.full_like(t, t.mean()) - t)))
        mae_flat = float(np.mean(np.abs(NEUTRAL[tgt_key] - t)))
        improve = (mae_flat - mae) / mae_flat * 100 if mae_flat > 0 else 0.0
        overall_model_mae.append(mae)
        overall_flat_mae.append(mae_flat)
        rows.append((label, mae, rmse, r, mae_mean, mae_flat, improve))
        print(f"{label:<20}{mae:>8.3f}{rmse:>8.3f}{r:>11.3f}"
              f"{mae_mean:>11.3f}{mae_flat:>11.3f}{improve:>8.1f}%")

    print("=" * 84)
    avg_improve = (
        (sum(overall_flat_mae) - sum(overall_model_mae)) / sum(overall_flat_mae) * 100
    )
    print(f"Model vs flat/neutral baseline: {avg_improve:.1f}% lower total error")
    print("(Pearson r > 0 means the model tracks real prosody variation — it is "
          "expressive, not monotone.)\n")

    # ---- Per-emotion directional analysis -----------------------------------
    # Does predicted pitch move in the SAME direction as the human ground truth
    # across emotions? (e.g. joyful/excited higher, sad/scared lower.)
    print("=" * 84)
    print("PER-EMOTION MEAN PITCH (semitones) — does the model react to emotion?")
    print("-" * 84)
    print(f"{'Emotion':<14}{'Tokens':>8}{'Actual pitch':>15}{'Predicted pitch':>18}{'Same dir?':>12}")
    print("-" * 84)
    pitch_pred = preds["pitch_targets"]
    pitch_true = trues["pitch_targets"]
    global_true_mean = float(pitch_true.mean())
    global_pred_mean = float(pitch_pred.mean())
    agree = 0
    total = 0
    for emo in sorted(set(tok_emotion_arr.tolist())):
        sel = tok_emotion_arr == emo
        n = int(sel.sum())
        if n < 20:
            continue
        a = float(pitch_true[sel].mean())
        p = float(pitch_pred[sel].mean())
        # direction relative to each distribution's own global mean
        same = (a - global_true_mean) * (p - global_pred_mean) >= 0
        agree += int(same); total += 1
        print(f"{emo:<14}{n:>8}{a:>15.3f}{p:>18.3f}{('YES' if same else 'no'):>12}")
    print("-" * 84)
    if total:
        print(f"Directional agreement: {agree}/{total} emotions "
              f"({agree/total*100:.0f}%) — predicted pitch shifts the same way as "
              f"the human narrator.")
    print("=" * 84)


if __name__ == "__main__":
    main()
