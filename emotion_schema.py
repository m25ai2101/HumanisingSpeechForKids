"""
Canonical emotion label schema for kids story narration.

All emotion-detection modules (text, speech, fusion) map their outputs
into this shared 8-class space so downstream components speak one language.
"""

# Ordered list of canonical emotion labels.
EMOTIONS = ["excited", "joyful", "scared", "sad", "calm", "dramatic", "surprised", "neutral"]

# -------------------------------------------------------------------------
# Cross-dataset mappings
# -------------------------------------------------------------------------

# IEMOCAP 4-class (Wav2Vec2-SUPERB output codes and decoded labels)
IEMOCAP_TO_CANONICAL = {
    "ang": "dramatic",
    "angry": "dramatic",
    "hap": "joyful",
    "happy": "joyful",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sad",
}

# j-hartmann/emotion-english-distilroberta-base 7-class labels
DISTILROBERTA_TO_CANONICAL = {
    "anger": "dramatic",
    "disgust": "dramatic",
    "fear": "scared",
    "joy": "joyful",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised",
}

# eGeMAPSv02 heuristic labels produced by the fallback path in emotion.py
HEURISTIC_TO_CANONICAL = {
    "dramatic": "dramatic",
    "angry": "dramatic",
    "surprised": "surprised",
    "happy": "joyful",
    "sad": "sad",
    "fearful": "scared",
    "calm": "calm",
    "neutral": "neutral",
}

# -------------------------------------------------------------------------
# Prosody defaults per canonical emotion
# (rate wpm, pitch_shift semitones, volume 0-1.5, pause_after seconds)
# -------------------------------------------------------------------------
PROSODY_DEFAULTS: dict[str, dict] = {
    "excited":   {"rate": 195, "pitch_shift":  2.0, "volume": 1.20, "pause_after": 0.1},
    "joyful":    {"rate": 180, "pitch_shift":  1.5, "volume": 1.10, "pause_after": 0.1},
    "scared":    {"rate": 185, "pitch_shift":  1.0, "volume": 0.85, "pause_after": 0.2},
    "sad":       {"rate": 135, "pitch_shift": -1.5, "volume": 0.75, "pause_after": 0.4},
    "calm":      {"rate": 145, "pitch_shift":  0.0, "volume": 0.90, "pause_after": 0.3},
    "dramatic":  {"rate": 160, "pitch_shift": -0.5, "volume": 1.15, "pause_after": 0.5},
    "surprised": {"rate": 190, "pitch_shift":  2.5, "volume": 1.05, "pause_after": 0.2},
    "neutral":   {"rate": 160, "pitch_shift":  0.0, "volume": 1.00, "pause_after": 0.2},
}


def map_to_canonical(label: str, source: str = "heuristic") -> str:
    """
    Map any source-specific emotion label to the canonical schema.

    Args:
        label:  Raw label from a model or heuristic.
        source: One of "iemocap", "distilroberta", "heuristic".
    Returns:
        Canonical label string. Falls back to "neutral" if unknown.
    """
    mapping = {
        "iemocap": IEMOCAP_TO_CANONICAL,
        "distilroberta": DISTILROBERTA_TO_CANONICAL,
        "heuristic": HEURISTIC_TO_CANONICAL,
    }.get(source, HEURISTIC_TO_CANONICAL)

    return mapping.get(label.lower(), "neutral")


def zero_vector() -> dict[str, float]:
    """Return a zeroed probability vector over canonical emotions."""
    return {e: 0.0 for e in EMOTIONS}


def uniform_vector() -> dict[str, float]:
    """Return a uniform probability vector over canonical emotions."""
    p = 1.0 / len(EMOTIONS)
    return {e: p for e in EMOTIONS}


def label_to_vector(label: str) -> dict[str, float]:
    """
    Convert a single canonical label to a one-hot probability vector.
    Unknown labels map to 'neutral'.
    """
    vec = zero_vector()
    canonical = label if label in vec else "neutral"
    vec[canonical] = 1.0
    return vec
