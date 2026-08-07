"""
Prosody Prediction Model

Architecture:
  Encoder : bert-base-uncased — produces per-token contextual embeddings (768-dim)
  Conditioning : fused emotion vector (8-dim) concatenated to every token embedding
  Heads (4 × independent linear layers, one per prosody parameter):
    - pitch_shift   : float, ±6 semitones
    - duration_ratio: float, 0.5 – 2.5
    - volume        : float, 0.4 – 1.0
    - pause_after   : float, 0.0 – 2.0 seconds

At inference: text + emotion vector → list[AnnotatedSegment] (source of truth for TTS)
At training : compare predicted prosody params against ground truth from audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

from emotion_schema import EMOTIONS
from prosody import AnnotatedSegment, NEUTRAL_RATE, NEUTRAL_PITCH, NEUTRAL_VOLUME, PAUSE


# ---------------------------------------------------------------------------
# Constants for output clamping
# ---------------------------------------------------------------------------
PITCH_MIN,    PITCH_MAX    = -6.0, 6.0
DURATION_MIN, DURATION_MAX =  0.5, 2.5
VOLUME_MIN,   VOLUME_MAX   =  0.4, 1.0
PAUSE_MIN,    PAUSE_MAX    =  0.0, 2.0
BASE_RATE                  = 150   # wpm at duration_ratio = 1.0


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class ProsodyPredictor(nn.Module):
    """
    BERT-based prosody predictor.

    Input:
        input_ids      : (B, T) — BERT token ids
        attention_mask : (B, T)
        emotion_vec    : (B, 8) — fused canonical emotion probability vector

    Output (all are (B, T) float tensors — one value per token):
        pitch_shift    : semitone shift relative to neutral
        duration_ratio : how much to stretch/compress each token's timing
        volume         : amplitude gain
        pause_after    : silence to insert after each token
    """

    def __init__(self, bert_model_id: str = "bert-base-uncased", n_emotions: int = 8):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_id)
        hidden = self.bert.config.hidden_size   # 768

        # emotion conditioning: project 8-dim → 768-dim and add to token repr
        self.emotion_proj = nn.Linear(n_emotions, hidden)

        in_dim = hidden

        # 4 regression heads — each outputs one scalar per token
        self.head_pitch    = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(),  nn.Linear(64, 1))
        self.head_duration = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(),  nn.Linear(64, 1))
        self.head_volume   = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(),  nn.Linear(64, 1))
        self.head_pause    = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(),  nn.Linear(64, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_vec: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # BERT: (B, T, 768)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_repr = outputs.last_hidden_state

        # Emotion conditioning: broadcast emotion projection to every token
        emo_proj = self.emotion_proj(emotion_vec).unsqueeze(1)   # (B, 1, 768)
        token_repr = token_repr + emo_proj                        # (B, T, 768)

        # Run heads, squeeze last dim → (B, T)
        pitch    = self.head_pitch(token_repr).squeeze(-1)
        duration = self.head_duration(token_repr).squeeze(-1)
        volume   = self.head_volume(token_repr).squeeze(-1)
        pause    = self.head_pause(token_repr).squeeze(-1)

        return {
            "pitch_shift":    pitch.clamp(PITCH_MIN,    PITCH_MAX),
            "duration_ratio": duration.clamp(DURATION_MIN, DURATION_MAX),
            "volume":         volume.clamp(VOLUME_MIN,   VOLUME_MAX),
            "pause_after":    pause.clamp(PAUSE_MIN,     PAUSE_MAX),
        }


# ---------------------------------------------------------------------------
# Singleton for inference
# ---------------------------------------------------------------------------

_model:     Optional[ProsodyPredictor]    = None
_tokenizer: Optional[BertTokenizerFast]   = None
_device:    Optional[torch.device]        = None


def load(checkpoint_path: str | None = None, bert_model_id: str = "bert-base-uncased") -> None:
    """Load model and tokenizer into module-level singletons."""
    global _model, _tokenizer, _device

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = BertTokenizerFast.from_pretrained(bert_model_id)
    _model     = ProsodyPredictor(bert_model_id)

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(state["model"])
        print(f"[ProsodyModel] Loaded checkpoint '{checkpoint_path}'")
    else:
        print("[ProsodyModel] No checkpoint — running with random weights (for testing only).")

    _model.to(_device)
    _model.eval()


def predict(
    text: str,
    emotion_vec: dict[str, float],
    content_type: str = "fiction",
) -> list[AnnotatedSegment]:
    """
    Predict per-segment prosody from text + emotion vector.

    Args:
        text:         Story sentence or paragraph.
        emotion_vec:  Canonical emotion probability dict from fusion.fuse().
        content_type: "fiction" or "non_fiction" (scales pitch range).

    Returns:
        List of AnnotatedSegment — the source-of-truth intermediate
        representation consumed by tts.speak_prosody().
    """
    if _model is None:
        raise RuntimeError("Call prosody_model.load() before predict().")

    from fusion import dominant_label

    dominant, confidence = dominant_label(emotion_vec)

    # Build emotion tensor: ordered by EMOTIONS list
    emo_tensor = torch.tensor(
        [[emotion_vec.get(e, 0.0) for e in EMOTIONS]],
        dtype=torch.float32,
    ).to(_device)

    # Tokenize — word-level alignment via return_offsets_mapping
    encoding = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    with torch.no_grad():
        out = _model(input_ids, attention_mask, emo_tensor)

    # Extract per-token predictions (drop [CLS] and [SEP])
    pitch_shifts    = out["pitch_shift"][0, 1:-1].cpu().tolist()
    duration_ratios = out["duration_ratio"][0, 1:-1].cpu().tolist()
    volumes         = out["volume"][0, 1:-1].cpu().tolist()
    pauses          = out["pause_after"][0, 1:-1].cpu().tolist()

    # Convert token-level predictions to word-level segments
    tokens  = _tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())[1:-1]
    segments = _tokens_to_segments(
        tokens, pitch_shifts, duration_ratios, volumes, pauses,
        dominant, confidence, content_type,
    )

    return segments


def _tokens_to_segments(
    tokens: list[str],
    pitch_shifts: list[float],
    duration_ratios: list[float],
    volumes: list[float],
    pauses: list[float],
    emotion: str,
    emotion_confidence: float,
    content_type: str,
) -> list[AnnotatedSegment]:
    """
    Merge sub-word tokens (## prefix) into whole words and average their
    prosody values, then emit one AnnotatedSegment per phrase boundary.
    A phrase boundary is a pause_after > 0.15s or end-of-sequence.
    """
    PHRASE_GAP = 0.15
    is_fiction = content_type == "fiction"
    pitch_scale = 1.0 if is_fiction else 0.4

    words: list[dict] = []
    buf_text, buf_pitch, buf_dur, buf_vol, buf_pause = "", [], [], [], []

    for tok, p, d, v, pa in zip(tokens, pitch_shifts, duration_ratios, volumes, pauses):
        if tok.startswith("##"):
            buf_text += tok[2:]
        else:
            if buf_text:
                words.append({
                    "text": buf_text,
                    "pitch":    float(sum(buf_pitch)  / len(buf_pitch)),
                    "duration": float(sum(buf_dur)    / len(buf_dur)),
                    "volume":   float(sum(buf_vol)    / len(buf_vol)),
                    "pause":    float(sum(buf_pause)  / len(buf_pause)),
                })
            buf_text  = tok
            buf_pitch = [p]
            buf_dur   = [d]
            buf_vol   = [v]
            buf_pause = [pa]

    if buf_text:
        words.append({
            "text": buf_text,
            "pitch":    float(sum(buf_pitch)  / len(buf_pitch)),
            "duration": float(sum(buf_dur)    / len(buf_dur)),
            "volume":   float(sum(buf_vol)    / len(buf_vol)),
            "pause":    float(sum(buf_pause)  / len(buf_pause)),
        })

    segments: list[AnnotatedSegment] = []
    grp_words: list[dict] = []

    def _flush(group: list[dict], pause_after: float) -> AnnotatedSegment:
        seg_text   = " ".join(w["text"] for w in group)
        avg_pitch  = float(sum(w["pitch"]    for w in group) / len(group)) * pitch_scale
        avg_dur    = float(sum(w["duration"] for w in group) / len(group))
        avg_vol    = float(sum(w["volume"]   for w in group) / len(group))
        rate       = int(max(80, min(220, BASE_RATE / max(avg_dur, 0.4))))
        return AnnotatedSegment(
            text=seg_text,
            rate=rate,
            pitch=round(avg_pitch, 2),
            volume=round(avg_vol, 2),
            pause_after=round(pause_after, 2),
            emotion=emotion,
            emotion_confidence=round(emotion_confidence, 3),
            source="model",
        )

    for w in words:
        grp_words.append(w)
        if w["pause"] >= PHRASE_GAP:
            segments.append(_flush(grp_words, w["pause"]))
            grp_words = []

    if grp_words:
        segments.append(_flush(grp_words, pauses[-1] if pauses else 0.35))

    return segments


# ---------------------------------------------------------------------------
# Dataset helpers (used by train.py)
# ---------------------------------------------------------------------------

class ProsodyDataset(torch.utils.data.Dataset):
    """
    Loads utterances from a jsonl file produced by data_pipeline.py.

    Each item:
        input_ids      : (T,)
        attention_mask : (T,)
        emotion_vec    : (8,)
        pitch_targets  : (T,)   — per-token (word-level values broadcast to tokens)
        duration_targets: (T,)
        volume_targets : (T,)
        pause_targets  : (T,)
    """

    MAX_LEN = 128

    def __init__(self, jsonl_path: str, bert_model_id: str = "bert-base-uncased"):
        self.records: list[dict] = []
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
        text = rec["text"]

        # Emotion vector — ordered by canonical EMOTIONS list
        emotion_vec = rec.get("emotion_vec", {})
        emo = torch.tensor([emotion_vec.get(e, 0.0) for e in EMOTIONS], dtype=torch.float32)

        # Per-word targets
        words = rec.get("words", [])
        word_pitch    = [w.get("pitch_shift",    0.0) for w in words]
        word_duration = [w.get("duration_ratio", 1.0) for w in words]
        word_volume   = [w.get("volume",         0.8) for w in words]
        word_pause    = [w.get("pause_after_sec", 0.2) for w in words]
        word_texts    = [w.get("word", "") for w in words]

        # Tokenize and align word-level labels to tokens
        encoding = self.tokenizer(
            text,
            is_split_into_words=False,
            truncation=True,
            max_length=self.MAX_LEN,
            padding="max_length",
            return_tensors="pt",
            return_word_ids=False,
        )

        # Simple alignment: tokenize each word individually to get token counts,
        # then broadcast that word's prosody value to all its sub-tokens.
        token_pitch, token_dur, token_vol, token_pause = _align_word_labels(
            self.tokenizer, word_texts, word_pitch, word_duration, word_volume, word_pause,
            max_len=self.MAX_LEN,
        )

        return {
            "input_ids":       encoding["input_ids"].squeeze(0),
            "attention_mask":  encoding["attention_mask"].squeeze(0),
            "emotion_vec":     emo,
            "pitch_targets":   token_pitch,
            "duration_targets": token_dur,
            "volume_targets":  token_vol,
            "pause_targets":   token_pause,
        }


def _align_word_labels(
    tokenizer: BertTokenizerFast,
    words: list[str],
    pitch: list[float],
    duration: list[float],
    volume: list[float],
    pause: list[float],
    max_len: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Broadcast word-level prosody values to BERT sub-word tokens.
    Positions for [CLS], [SEP], and padding receive a neutral 0/1/0.8/0 value.
    """
    p_toks, d_toks, v_toks, pa_toks = [0.0], [1.0], [0.8], [0.0]  # CLS

    for w, p_val, d_val, v_val, pa_val in zip(words, pitch, duration, volume, pause):
        n_toks = len(tokenizer.tokenize(w)) or 1
        p_toks.extend([p_val]  * n_toks)
        d_toks.extend([d_val]  * n_toks)
        v_toks.extend([v_val]  * n_toks)
        pa_toks.extend([pa_val] * n_toks)

    p_toks.append(0.0);  d_toks.append(1.0);  v_toks.append(0.8);  pa_toks.append(0.0)  # SEP

    def _pad(lst: list[float], val: float) -> torch.Tensor:
        lst = lst[:max_len]
        lst += [val] * (max_len - len(lst))
        return torch.tensor(lst, dtype=torch.float32)

    return _pad(p_toks, 0.0), _pad(d_toks, 1.0), _pad(v_toks, 0.8), _pad(pa_toks, 0.0)
