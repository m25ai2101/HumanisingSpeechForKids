"""
Multimodal Emotion Fusion

Combines text-emotion and speech-emotion probability vectors into a single
fused vector that downstream components (prosody prediction, TTS) consume.

Two modes:
  - text_only : only the text vector is available (inference on text input)
  - weighted  : weighted average of text + speech vectors, with configurable
                per-modality weights. Weights can be learned during training
                and stored in a checkpoint; at inference they are loaded from
                that checkpoint or fall back to the hard-coded defaults below.
"""

from __future__ import annotations
from emotion_schema import EMOTIONS, zero_vector

# Default modality weights (text, speech).  These are overridden if a trained
# checkpoint supplies its own weights via load_weights().
_DEFAULT_TEXT_WEIGHT   = 0.45
_DEFAULT_SPEECH_WEIGHT = 0.55  # speech emotion is grounded in actual audio

_text_w:   float = _DEFAULT_TEXT_WEIGHT
_speech_w: float = _DEFAULT_SPEECH_WEIGHT


def load_weights(text_weight: float, speech_weight: float) -> None:
    """Override default fusion weights (called by train.py after training)."""
    global _text_w, _speech_w
    total = text_weight + speech_weight
    _text_w   = text_weight   / total
    _speech_w = speech_weight / total


def fuse(
    text_vec: dict[str, float],
    speech_vec: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Fuse text and speech emotion probability vectors.

    Args:
        text_vec:   Canonical emotion probability dict from text_emotion.detect().
        speech_vec: Canonical emotion probability dict from emotion.detect_vector().
                    Pass None (or omit) when only text is available.

    Returns:
        Fused probability dict over canonical emotion labels (sums to 1.0).
    """
    if speech_vec is None:
        return _normalise(text_vec)

    fused = zero_vector()
    for e in EMOTIONS:
        t = text_vec.get(e, 0.0)
        s = speech_vec.get(e, 0.0)
        fused[e] = _text_w * t + _speech_w * s

    return _normalise(fused)


def dominant_label(vec: dict[str, float]) -> tuple[str, float]:
    """Return (label, confidence) for the highest-scoring emotion in a vector."""
    label = max(vec, key=vec.get)
    return label, vec[label]


def _normalise(vec: dict[str, float]) -> dict[str, float]:
    total = sum(vec.values())
    if total <= 0:
        n = len(EMOTIONS)
        return {e: 1.0 / n for e in EMOTIONS}
    return {k: v / total for k, v in vec.items()}
