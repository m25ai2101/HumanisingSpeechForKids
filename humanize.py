"""
Text Humanization Layer

Implements three stacked ideas from the surveyed papers to transform flat
ASR output into expressive, natural-sounding speech text before TTS:

  Layer 1 — DISCOURSE DETECTOR (Storytelling TTS Survey)
    The survey stresses that storytelling involves distinct discourse
    modes: narrative, dialogue, and descriptive. Each mode requires
    different pacing and delivery. We detect the mode from surface cues.

  Layer 2 — PROSODY PROFILES / STYLE PRESETS (Global Style Tokens)
    GST proposes learnable style embeddings that modulate speech.
    We implement the same idea as fixed rule profiles: each emotion maps
    to a named style preset (rate, volume) that drives pyttsx3 parameters.
    This is the interpretable, label-aware version of what GST learns.

  Layer 3 — EMOTION-AWARE TEXT STYLISATION (FastSpeech2 + Prosody Transfer)
    FastSpeech2 shows that explicitly conditioning on pitch and duration
    improves expressiveness. Since pyttsx3 only exposes rate and volume,
    we simulate prosodic variation through punctuation: commas add micro-
    pauses, ellipses slow the reader, exclamation marks raise energy.
    This mirrors the Prosody Transfer idea of encoding speaking style.
"""

from __future__ import annotations
import re
from emotion_schema import PROSODY_DEFAULTS


# ---------------------------------------------------------------------------
# Layer 1: Discourse type detection  (Storytelling TTS Survey)
# ---------------------------------------------------------------------------
# The survey identifies three main discourse styles in storytelling:
#   - Dialogue    : direct speech between characters  → conversational pace
#   - Narrative   : past-tense story progression      → measured, storytelling pace
#   - Descriptive : present-tense scene/state setting → calm, slower pace

_NARRATIVE_VERBS   = re.compile(
    r'\b(said|asked|replied|whispered|shouted|ran|walked|turned|looked|'
    r'smiled|laughed|cried|thought|felt|saw|heard|became|went|came)\b',
    re.IGNORECASE,
)
_DIALOGUE_PATTERN  = re.compile(r'[""\'\'](.*?)[""\'\'"]', re.DOTALL)
_DESCRIPTIVE_VERBS = re.compile(
    r'\b(is|are|was|were|appears?|seems?|looks?|remains?|feels?)\b',
    re.IGNORECASE,
)


def detect_discourse(text: str) -> str:
    """
    Return one of: 'dialogue', 'narrative', 'descriptive', 'neutral'.

    Heuristic priority order: dialogue > narrative > descriptive > neutral.
    """
    if _DIALOGUE_PATTERN.search(text):
        return "dialogue"
    if len(_NARRATIVE_VERBS.findall(text)) >= 2:
        return "narrative"
    if _DESCRIPTIVE_VERBS.search(text):
        return "descriptive"
    return "neutral"


# ---------------------------------------------------------------------------
# Layer 2: GST-inspired style presets  (Wang et al., 2018)
# ---------------------------------------------------------------------------
# Profiles are sourced from emotion_schema.PROSODY_DEFAULTS (canonical 8-class
# kids-story labels) rather than hardcoded here. This keeps rate/volume in sync
# with every other module that reads from the shared schema.
#
# Discourse-specific adjustments are still applied on top.

_DISCOURSE_ADJUSTMENTS: dict[str, dict] = {
    "dialogue":    {"rate_delta": +12, "volume_delta": +0.03},
    "narrative":   {"rate_delta":  -8, "volume_delta": -0.02},
    "descriptive": {"rate_delta": -12, "volume_delta": -0.04},
    "neutral":     {"rate_delta":   0, "volume_delta":  0.00},
}


def get_prosody(emotion: str, discourse: str) -> dict[str, int | float]:
    """
    Return the final {rate, volume} for this (emotion, discourse) pair.
    Reads base rates/volumes from emotion_schema.PROSODY_DEFAULTS so all
    modules share the same canonical prosody values.
    """
    defaults = PROSODY_DEFAULTS.get(emotion, PROSODY_DEFAULTS["neutral"])
    adj      = _DISCOURSE_ADJUSTMENTS.get(discourse, _DISCOURSE_ADJUSTMENTS["neutral"])

    rate   = max(100, min(300, defaults["rate"]   + adj["rate_delta"]))
    volume = max(0.5, min(1.0, defaults["volume"] + adj["volume_delta"]))

    return {"rate": rate, "volume": volume}


# ---------------------------------------------------------------------------
# Layer 3: Emotion-aware text stylisation  (FastSpeech2 + Prosody Transfer)
# ---------------------------------------------------------------------------
# We add punctuation cues that drive pyttsx3's prosody engine.
# Comma  → ~80 ms pause  |  Period  → ~200 ms pause  |  Ellipsis → ~500 ms pause
# This approximates FastSpeech2's duration modelling with plain text.

def _stylise(text: str, emotion: str, discourse: str) -> str:
    """Apply punctuation-based prosody cues to `text` using canonical emotion labels."""
    text = text.strip()

    if emotion in ("joyful", "excited"):
        # Upbeat: exclamation, double key intensifiers
        text = re.sub(r'\.\s*$', '!', text)
        text = re.sub(
            r'\b(really|very|so|absolutely|totally)\b',
            r'\1, \1', text, flags=re.IGNORECASE,
        )
        if not text.endswith(('!', '?')):
            text += '!'

    elif emotion == "sad":
        # Slow delivery: ellipsis between sentences, trailing off
        text = re.sub(r'\.\s+', '... ', text)
        text = text.rstrip('.')
        if not text.endswith(('...', '?', '!')):
            text += '...'

    elif emotion == "dramatic":
        # Storytelling style: phrase-boundary commas + ellipsis before climactic words
        text = re.sub(
            r'\s+(and|but|then|so|when|as|until)\s+',
            r', \1 ', text, flags=re.IGNORECASE,
        )
        text = re.sub(
            r'\b(suddenly|finally|then|slowly|quickly|never|always|only)\b',
            r'... \1', text, flags=re.IGNORECASE,
        )
        text = re.sub(r'(\.\.\.\s*){2,}', '... ', text)
        if not text.endswith(('!', '?', '...', '.')):
            text += '.'

    elif emotion == "scared":
        # Hesitant: ellipsis at comma positions, nervous opener
        text = re.sub(r'^([A-Z])', lambda m: 'I... ' + m.group(1).lower(), text, count=1)
        text = re.sub(r',\s+', '... ', text)

    elif emotion == "surprised":
        prefix_words = text.lower()[:5]
        if not any(w in prefix_words for w in ("oh", "wow", "what", "no ")):
            text = "Oh! " + text
        text = re.sub(r'\.\s*$', '!', text)
        if not text.endswith(('!', '?')):
            text += '!'

    elif emotion == "calm":
        # Smooth conjunctions with commas for measured, gentle delivery
        text = re.sub(r'\s+(and|but|so)\s+', r', \1 ', text, flags=re.IGNORECASE)

    # Discourse overlay
    if discourse == "narrative":
        text = "...  " + text
    elif discourse == "dialogue":
        text = re.sub(r'([a-z])\s+(")', r'\1, \2', text)

    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enhance(text: str, emotion: str, content_type: str = "non_fiction") -> dict:
    """
    Humanize `text` based on `emotion` and `content_type`.

    For fiction, the full punctuation-based stylisation runs so storytelling
    sounds expressive. For non_fiction the text is kept verbatim — only the
    rate and volume are adjusted — so factual speech isn't over-dramatised.

    Returns a dict with:
      text      — (stylised or plain) text for the TTS engine
      rate      — recommended pyttsx3 rate (words per minute)
      volume    — recommended pyttsx3 volume (0.0 – 1.0)
      discourse — detected discourse type for logging
    """
    discourse = detect_discourse(text)
    prosody   = get_prosody(emotion, discourse)

    if content_type == "fiction":
        styled_text = _stylise(text, emotion, discourse)
    else:
        # Non-fiction: keep the original text untouched; only carry rate/volume
        styled_text = text.strip()

    print(
        f"[Humanize] content_type={content_type}  emotion={emotion}  "
        f"discourse={discourse}  rate={prosody['rate']}wpm  volume={prosody['volume']:.2f}"
    )
    print(f"[Humanize] text: {styled_text!r}")

    return {
        "text":      styled_text,
        "rate":      prosody["rate"],
        "volume":    prosody["volume"],
        "discourse": discourse,
    }
