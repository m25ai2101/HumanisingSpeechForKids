"""
Dynamic Text Emotion Detection

Uses j-hartmann/emotion-english-distilroberta-base (HuggingFace) to detect
emotion from text. Returns a full probability distribution over the canonical
8-class kids-story emotion schema — not a single collapsed label — so that the
fusion layer can combine it with speech emotion scores without losing nuance.

Downloads ~300 MB on first run, then cached locally.
"""

from emotion_schema import DISTILROBERTA_TO_CANONICAL, EMOTIONS, zero_vector

_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
_text_pipe = None


def _load_model():
    global _text_pipe
    if _text_pipe is None:
        from transformers import pipeline as hf_pipeline
        print(f"[TextEmotion] Loading '{_MODEL_ID}' (~300 MB on first run)...")
        _text_pipe = hf_pipeline(
            "text-classification",
            model=_MODEL_ID,
            top_k=None,     # return scores for all labels
            device=-1,      # CPU
        )
        print("[TextEmotion] Model ready.")
    return _text_pipe


def _aggregate_to_canonical(raw_scores: list[dict]) -> dict[str, float]:
    """
    Map distilroberta label scores into canonical emotion space.

    Multiple source labels can map to the same canonical label (e.g. both
    'anger' and 'disgust' → 'dramatic'); their scores are summed and the
    result is re-normalised to sum to 1.0.
    """
    vec = zero_vector()
    for item in raw_scores:
        canonical = DISTILROBERTA_TO_CANONICAL.get(item["label"].lower(), "neutral")
        vec[canonical] = vec.get(canonical, 0.0) + item["score"]

    total = sum(vec.values())
    if total > 0:
        vec = {k: v / total for k, v in vec.items()}
    return vec


def detect(text: str) -> dict[str, float]:
    """
    Detect emotion from text.

    Args:
        text: Raw story text (sentence or paragraph).

    Returns:
        Dict mapping each canonical emotion label to a probability (sums to 1).
        Example: {"excited": 0.05, "joyful": 0.60, "scared": 0.02, ...}
    """
    text = text.strip()
    if not text:
        from emotion_schema import uniform_vector
        return uniform_vector()

    try:
        pipe = _load_model()
        # Truncate to 512 tokens silently — the model's max context
        raw = pipe(text[:1024], truncation=True)[0]
        vec = _aggregate_to_canonical(raw)
        dominant = max(vec, key=vec.get)
        print(f"[TextEmotion] → {dominant} ({vec[dominant]:.0%})  [distilroberta]")
        return vec

    except Exception as e:
        print(f"[TextEmotion] Model unavailable ({e}), using keyword fallback.")
        return _keyword_fallback(text)


def dominant(text: str) -> tuple[str, float]:
    """
    Convenience wrapper — returns (label, confidence) instead of full vector.
    """
    vec = detect(text)
    label = max(vec, key=vec.get)
    return label, vec[label]


# ---------------------------------------------------------------------------
# Lightweight keyword fallback (used only when transformer is unavailable)
# ---------------------------------------------------------------------------

_KEYWORD_MAP: dict[str, list[str]] = {
    "excited":   ["exciting", "amazing", "adventure", "thrilling", "wow", "incredible", "fantastic"],
    "joyful":    ["happy", "joy", "laugh", "smile", "fun", "love", "wonderful", "delightful", "cheerful"],
    "scared":    ["scared", "afraid", "frightened", "terrified", "dark", "shadow", "monster", "danger", "trembl"],
    "sad":       ["sad", "cry", "tear", "lonely", "miss", "lost", "sorrow", "mourn", "weep"],
    "calm":      ["quiet", "still", "peaceful", "gentle", "soft", "serene", "slow", "whisper"],
    "dramatic":  ["suddenly", "crash", "roar", "storm", "furious", "rage", "shout", "thunder", "fierce"],
    "surprised": ["surprised", "gasped", "shocked", "unbelievable", "unexpected", "suddenly realized"],
}


def _keyword_fallback(text: str) -> dict[str, float]:
    lower = text.lower()
    vec = zero_vector()
    for label, keywords in _KEYWORD_MAP.items():
        hits = sum(1 for kw in keywords if kw in lower)
        vec[label] = float(hits)

    total = sum(vec.values())
    if total > 0:
        vec = {k: v / total for k, v in vec.items()}
    else:
        vec["neutral"] = 1.0

    dominant_label = max(vec, key=vec.get)
    print(f"[TextEmotion] → {dominant_label} ({vec[dominant_label]:.0%})  [keyword fallback]")
    return vec
