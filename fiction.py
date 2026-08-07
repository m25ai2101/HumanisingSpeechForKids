"""
Fiction vs Non-Fiction Classifier
==================================
Primary path  — Zero-shot NLI classification via DistilBERT fine-tuned on
               MNLI (typeform/distilbert-base-uncased-mnli, ~268 MB).
               Downloads on first run, then cached locally.
               Understands context — not just keywords — so "I ran to the
               store" is correctly classified as non-fiction even though
               "ran" is a narrative verb.

Fallback path — Rule-based keyword scoring (no download needed).
               Used automatically if the model fails to load.

Both paths operate on the ASR transcript (text only, no audio).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# NLI model (primary)
# ---------------------------------------------------------------------------

_MODEL_ID = "typeform/distilbert-base-uncased-mnli"

# Descriptive label strings fed to the NLI model as candidate hypotheses.
# More specific labels give NLI models better signal than bare "fiction".
_CANDIDATE_LABELS = [
    "a fictional story or narrative",
    "factual or real-world speech",
]

_classifier = None


def _load_model():
    global _classifier
    if _classifier is None:
        from transformers import pipeline as hf_pipeline
        print(
            f"[Fiction] Loading NLI classifier '{_MODEL_ID}' "
            "(downloads ~268 MB on first run)..."
        )
        _classifier = hf_pipeline(
            "zero-shot-classification",
            model=_MODEL_ID,
            device=-1,      # CPU
        )
        print("[Fiction] Classifier ready.")
    return _classifier


def _detect_with_model(text: str) -> tuple[str, float]:
    """Zero-shot NLI: classify text against fiction vs factual hypotheses."""
    clf    = _load_model()
    result = clf(
        text,
        candidate_labels=_CANDIDATE_LABELS,
        hypothesis_template="This is {}.",
    )
    top_label = result["labels"][0]
    top_score = float(result["scores"][0])

    label = "fiction" if "fictional" in top_label else "non_fiction"
    print(f"[Fiction] → {label} ({top_score:.0%})  [NLI model]")
    return label, top_score


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_NARRATIVE_VERBS = re.compile(
    r"\b(said|replied|asked|answered|whispered|shouted|yelled|murmured|cried|"
    r"laughed|smiled|frowned|sighed|gasped|nodded|shook|ran|walked|jumped|"
    r"climbed|fled|rushed|charged|turned|looked|glanced|stared|reached|grabbed|"
    r"pulled|pushed|threw|fell|rose|stood|sat|lay|knelt|leaned|stepped|"
    r"thought|felt|knew|realized|remembered|wondered|imagined|believed|feared|"
    r"heard|saw|noticed|watched|followed|led|brought|carried|held)\b",
    re.IGNORECASE,
)
_STORY_STRUCTURE = re.compile(
    r"\b(once upon a time|once there was|one day|long ago|in a land|"
    r"at that moment|in those days|legend has it|story goes|"
    r"suddenly|meanwhile|finally|thereafter|eventually|at last|"
    r"little did|just then|before long|not long after|"
    r"to his surprise|to her surprise|chapter|the end|happily ever after)\b",
    re.IGNORECASE,
)
_THIRD_PERSON = re.compile(
    r"\b(he|she|they|it)\s+"
    r"(said|ran|walked|felt|thought|knew|saw|heard|became|went|came|"
    r"took|made|found|told|asked|replied|looked|grabbed|turned|fell|rose)\b",
    re.IGNORECASE,
)
_DIALOGUE_MARKERS = re.compile(r'["""].*?["""]', re.DOTALL)
_PAST_TENSE_ED   = re.compile(r"\b\w+ed\b", re.IGNORECASE)

_FIRST_PERSON_PRESENT = re.compile(
    r"\b(I am|I'm|I think|I believe|I feel|I know|I need|I want|I have|"
    r"I can|I will|I would|I should|I must|I did|I was|"
    r"we are|we're|we need|we have|we can|we will|"
    r"you should|you can|you need|you have|you are|you're)\b",
    re.IGNORECASE,
)
_FACTUAL_MARKERS = re.compile(
    r"\b(according to|research shows|studies indicate|the fact is|"
    r"statistics show|data suggests|experts say|scientists found|"
    r"in reality|in fact|actually|technically|specifically|"
    r"for example|for instance|as a result|therefore|consequently|"
    r"however|nevertheless|it is|this is|there are|there is)\b",
    re.IGNORECASE,
)
_QUESTIONS    = re.compile(r"\?")
_INSTRUCTIONAL = re.compile(
    r"\b(please|make sure|remember to|keep in mind|note that|"
    r"first|second|third|next|step|steps|in order to|"
    r"you need to|you have to|don't forget|be sure to)\b",
    re.IGNORECASE,
)
_REAL_WORLD_REFS = re.compile(
    r"\b\d+\s*(%|percent|km|kg|mb|gb|tb|ms|hz|°c|°f|dollars?|\$|€|£)\b"
    r"|\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)


def _detect_heuristic(text: str) -> tuple[str, float]:
    """Rule-based fallback when the NLI model is unavailable."""
    words = max(1, len(text.split()))

    f_score  = 0
    f_score += len(_NARRATIVE_VERBS.findall(text)) * 2
    f_score += len(_STORY_STRUCTURE.findall(text)) * 3
    f_score += len(_THIRD_PERSON.findall(text))    * 2
    f_score += len(_DIALOGUE_MARKERS.findall(text))
    if len(_PAST_TENSE_ED.findall(text)) / words > 0.10:
        f_score += 3

    nf_score  = 0
    nf_score += len(_FIRST_PERSON_PRESENT.findall(text)) * 2
    nf_score += len(_FACTUAL_MARKERS.findall(text))      * 3
    nf_score += len(_QUESTIONS.findall(text))
    nf_score += len(_INSTRUCTIONAL.findall(text))        * 2
    nf_score += len(_REAL_WORLD_REFS.findall(text))      * 2

    total = f_score + nf_score
    if total == 0 or nf_score >= f_score:
        label     = "non_fiction"
        margin    = (nf_score - f_score) / max(total, 1)
    else:
        label     = "fiction"
        margin    = (f_score - nf_score) / max(total, 1)

    confidence = round(0.50 + margin * 0.45, 2)
    print(f"[Fiction] → {label} ({confidence:.0%})  [heuristic fallback]")
    return label, confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(text: str, verbose: bool = True) -> tuple[str, float]:
    """
    Classify `text` as 'fiction' or 'non_fiction'.

    Tries the NLI model first; falls back to keyword heuristics if the model
    is unavailable.  Returns (label, confidence) where confidence ∈ [0.5, 1.0].

    The label gates the pipeline:
      fiction     → emotion detection + prosody + TTS proceed
      non_fiction → pipeline exits early (no emotion, no TTS)
    """
    try:
        return _detect_with_model(text)
    except Exception as e:
        print(f"[Fiction] Model unavailable ({e}), using heuristics.")
        return _detect_heuristic(text)
