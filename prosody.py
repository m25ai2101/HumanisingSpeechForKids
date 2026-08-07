"""
Rule-Based Prosody Control
==========================
Sits between the ASR text output and the TTS backend. It analyses the *text*
(not the audio) and produces a per-segment **prosody plan** — for every chunk
of speech it decides a rate, pitch, volume and the pause that follows. The
renderer (`tts.speak_prosody`) turns that plan into expressive audio.

This is the rule-based prototype. The only "learned-looking" step is emotion
detection, which here is plain keyword spotting (`_score_emotion_words`). That
step is deliberately isolated so it can later be swapped for — or fused with —
a model such as a BERT emotion classifier: the classifier would emit the same
kind of emotion label that feeds `EMOTION_PROSODY`, and every other rule below
stays exactly the same. See the `emotion_hint` argument of `analyze()` for the
integration seam (the acoustic `emotion.detect` result is already fed in there).

Implemented rule groups (see the task spec):
  1. Sentence type        (? ! ... . !!!)
  2. Punctuation pauses    (, ; — ...)
  3. Discourse mode        (dialogue / descriptive / narrative)
  4. Emotion words         (happy / sad / angry / scared / surprised / calm)
  5. Emphasis             (ALL CAPS words, repeated words)
  6. Dialogue tags         (whispered / shouted / asked / laughed)
  7. Story structure       (first / last sentence of a paragraph)

Conflict resolution: when several rules touch the same parameter we keep the
value that deviates *furthest from neutral* (sign preserved) — i.e. the
"stronger" expression wins. Pauses always take the longest.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Parameter vocabularies  (from the spec's "Prosody Parameter Mapping")
# ---------------------------------------------------------------------------
RATE = {"slow": 110, "default": 150, "fast": 190}
RATE_MIN, RATE_MAX = 100, 215          # allow stacked intensifiers to exceed "fast"

# Pitch is expressed in semitones for librosa's pitch-shift (low/medium/high).
PITCH = {"low": -3.0, "medium": 0.0, "high": +3.0}
PITCH_MIN, PITCH_MAX = -6.0, +6.0

PAUSE = {
    "none": 0.0, "very_short": 0.2, "short": 0.3,
    "medium": 0.6, "long": 1.0, "dramatic": 1.5,
}

VOLUME = {"soft": 0.5, "normal": 0.8, "loud": 1.0}
VOLUME_MIN, VOLUME_MAX = 0.4, 1.0

# Neutral baselines used for conflict resolution ("furthest from neutral wins").
NEUTRAL_RATE = RATE["default"]
NEUTRAL_PITCH = PITCH["medium"]
NEUTRAL_VOLUME = VOLUME["normal"]


# ---------------------------------------------------------------------------
# Output types: a flat list of Segment (or AnnotatedSegment) is what the
# renderer consumes.
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """One renderable chunk of speech with its own prosody."""
    text: str
    rate: int = NEUTRAL_RATE          # words per minute
    pitch: float = NEUTRAL_PITCH      # semitones (+ up / - down)
    volume: float = NEUTRAL_VOLUME    # 0.0 – 1.0
    pause_after: float = PAUSE["short"]  # seconds of silence after this chunk


@dataclass
class AnnotatedSegment(Segment):
    """
    Extended Segment that carries the emotion context driving its prosody.

    Produced by prosody_model.predict() at inference time (source="model") or
    by the rule-based analyze() path as a fallback (source="rule_fallback").
    The renderer (tts.speak_prosody) accepts both Segment and AnnotatedSegment
    transparently since AnnotatedSegment inherits all Segment fields.
    """
    emotion: str = "neutral"           # dominant canonical emotion label
    emotion_confidence: float = 0.0    # 0.0 – 1.0
    source: str = "rule_fallback"      # "model" | "rule_fallback"


# ---------------------------------------------------------------------------
# Rule 4 data — emotion keyword lexicon and its prosody mapping
# ---------------------------------------------------------------------------
EMOTION_WORDS: dict[str, list[str]] = {
    "happy":     ["happy", "yay", "wonderful", "excited", "joy", "joyful",
                  "delighted", "glad", "cheerful"],
    "sad":       ["sad", "cried", "crying", "alone", "tears", "gloomy",
                  "sorrow", "weeping", "miserable", "unhappy"],
    "angry":     ["angry", "furious", "shouted", "rage", "mad",
                  "irritated", "annoyed", "fuming"],
    "scared":    ["scared", "whispered", "dark", "terrified", "afraid",
                  "frightened", "trembling", "fear"],
    "surprised": ["suddenly", "gasped", "shocked", "amazed",
                  "astonished", "stunned", "unexpected"],
    "calm":      ["peaceful", "quietly", "gentle", "soft", "serene",
                  "tranquil", "calm", "slowly"],
}

# emotion -> (pitch level, speed level, volume level)
EMOTION_PROSODY: dict[str, tuple[str, str, str]] = {
    "happy":     ("high", "fast", "loud"),
    "sad":       ("low",  "slow", "soft"),
    "angry":     ("high", "fast", "loud"),
    "scared":    ("low",  "slow", "soft"),
    "surprised": ("high", "fast", "normal"),
    "calm":      ("low",  "slow", "soft"),
    "dramatic":  ("high", "slow", "loud"),   # wide pitch + measured pace + strong volume
}

# Map the acoustic emotion labels (from emotion.detect) onto our six text
# categories so an acoustic/text fusion is possible. Labels with no clean
# counterpart (dramatic, neutral, disgust) are intentionally left out — the
# text rules then drive prosody on their own.
_ACOUSTIC_MAP = {
    "happy": "happy", "sad": "sad", "angry": "angry",
    "surprised": "surprised", "calm": "calm", "fearful": "scared",
    "dramatic": "dramatic",
}

_EMO_RE = {
    emo: re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
    for emo, words in EMOTION_WORDS.items()
}


def _score_emotion_words(text: str) -> dict[str, int]:
    """Return {emotion: match_count} for every emotion with at least one hit."""
    return {emo: len(rx.findall(text)) for emo, rx in _EMO_RE.items() if rx.search(text)}


def _dominant_emotion(text: str, hint: str | None) -> str | None:
    """
    Pick the emotion driving this sentence.

    Acoustic signal (hint) takes priority: it captures HOW something is said
    — pitch, energy, voice quality — not just WHAT words appear.  This avoids
    the 'affective underspecification' problem where identical text ("I'm fine")
    maps to different emotions depending on delivery.

    Text keyword spotting only fires when no acoustic hint is available (e.g.
    TTS-only mode with no input audio), acting as a fallback rather than an
    override.
    """
    acoustic = _ACOUSTIC_MAP.get(hint) if hint else None
    if acoustic:
        return acoustic
    scores = _score_emotion_words(text)
    return max(scores, key=scores.get) if scores else None


# ---------------------------------------------------------------------------
# Rule 3 data — discourse mode detection
# ---------------------------------------------------------------------------
_QUOTE_RE = re.compile(r'["“”].*?["“”]', re.DOTALL)
_ACTION_VERBS = re.compile(
    r"\b(ran|jumped|grabbed|rushed|threw|leapt|dashed|burst|slammed|kicked|"
    r"punched|chased|fled|charged|swung|hit|fell|climbed|crawled|raced)\b",
    re.IGNORECASE,
)


def _discourse(text: str) -> str:
    """dialogue > descriptive > narrative > neutral."""
    if _QUOTE_RE.search(text):
        return "dialogue"
    if text.count(",") >= 2 and len(text.split()) >= 12:
        return "descriptive"
    if _ACTION_VERBS.search(text):
        return "narrative"
    return "neutral"


# ---------------------------------------------------------------------------
# Per-sentence analysis target
# ---------------------------------------------------------------------------
@dataclass
class _Target:
    rate: float = NEUTRAL_RATE
    pitch: float = NEUTRAL_PITCH
    volume: float = NEUTRAL_VOLUME
    pre_pause: float = 0.0                  # silence *before* the sentence
    post_pause: float = PAUSE["short"]      # silence *after* the sentence
    comma_pause: float = PAUSE["very_short"]  # baseline pause for inner commas
    end_rise: bool = False                  # extra pitch lift on final segment
    rules: list[str] = field(default_factory=list)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _stronger(current: float, candidate: float, neutral: float) -> float:
    """Keep whichever value deviates further from neutral (sign preserved)."""
    return candidate if abs(candidate - neutral) > abs(current - neutral) else current


def _apply(t: _Target, *, rate=None, pitch=None, volume=None, label: str = "") -> None:
    """Combine a rule's candidate values into the target via conflict resolution."""
    if rate is not None:
        t.rate = _stronger(t.rate, rate, NEUTRAL_RATE)
    if pitch is not None:
        t.pitch = _stronger(t.pitch, pitch, NEUTRAL_PITCH)
    if volume is not None:
        t.volume = _stronger(t.volume, volume, NEUTRAL_VOLUME)
    if label:
        t.rules.append(label)


def _speed_to_rate(level: str) -> int:
    return {"fast": RATE["fast"], "slow": RATE["slow"]}.get(level, RATE["default"])


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
@dataclass
class _Sentence:
    text: str    # body without the terminator
    term: str    # raw terminator, e.g. "?", "!!!", "..."
    kind: str    # question / exclaim / exclaim_multi / interrobang / ellipsis / period / none


_TERM_RE = re.compile(r"(\.\.\.|…|[.!?]+)")


def _classify(term: str) -> str:
    if term in ("...", "…"):
        return "ellipsis"
    has_q, has_excl = "?" in term, "!" in term
    if has_q and has_excl:
        return "interrobang"
    if has_excl:
        return "exclaim_multi" if len(term) >= 2 else "exclaim"
    if has_q:
        return "question"
    if term.strip() == "":
        return "none"
    return "period"


def _split_sentences(paragraph: str) -> list[_Sentence]:
    tokens = _TERM_RE.split(paragraph)
    out: list[_Sentence] = []
    buf = ""
    for tok in tokens:
        if tok == "":
            continue
        if _TERM_RE.fullmatch(tok):
            if buf.strip():
                out.append(_Sentence(buf.strip(), tok, _classify(tok)))
            buf = ""
        else:
            buf += tok
    if buf.strip():
        out.append(_Sentence(buf.strip(), "", "none"))
    return out


# ---------------------------------------------------------------------------
# Rule application — one sentence -> a _Target
# ---------------------------------------------------------------------------
def _analyze_sentence(s: _Sentence, is_first: bool, is_last: bool,
                      emotion_hint: str | None) -> _Target:
    t = _Target()

    # Rule 1: sentence type
    if s.kind == "question":
        _apply(t, pitch=PITCH["high"], label="type:?")
        t.pre_pause = max(t.pre_pause, PAUSE["short"])   # slight pause before
        t.end_rise = True                                # pitch rise at the end
    elif s.kind == "exclaim":
        _apply(t, rate=RATE["fast"], volume=VOLUME["loud"], label="type:!")
    elif s.kind == "exclaim_multi":
        # even faster and louder than a single "!"
        _apply(t, rate=min(RATE_MAX, RATE["fast"] + 18),
               volume=VOLUME["loud"], label="type:!!!")
    elif s.kind == "interrobang":
        _apply(t, rate=RATE["fast"], volume=VOLUME["loud"],
               pitch=PITCH["high"], label="type:?!")
        t.end_rise = True
    elif s.kind == "ellipsis":
        _apply(t, rate=RATE["slow"], label="type:...")
        t.post_pause = max(t.post_pause, PAUSE["long"])
    else:  # period / none
        t.post_pause = max(t.post_pause, PAUSE["short"])
        t.rules.append("type:.")

    # Rule 4: emotion words (or acoustic hint fallback)
    emo = _dominant_emotion(s.text, emotion_hint)
    if emo:
        pitch_lvl, speed_lvl, vol_lvl = EMOTION_PROSODY[emo]
        _apply(t, pitch=PITCH[pitch_lvl], rate=_speed_to_rate(speed_lvl),
               volume=VOLUME[vol_lvl], label=f"emotion:{emo}")

    # Rule 3: discourse mode
    disc = _discourse(s.text)
    if disc == "dialogue":
        # exaggerated pitch + faster pace
        _apply(t, pitch=PITCH["high"] + 1.5, rate=RATE["fast"] - 15,
               label="discourse:dialogue")
    elif disc == "descriptive":
        _apply(t, rate=RATE["slow"], label="discourse:descriptive")
        t.comma_pause = PAUSE["medium"]    # medium pauses for the comma-heavy clauses
    elif disc == "narrative":
        t.rules.append("discourse:narrative")  # normal pace, short pauses (defaults)

    # Rule 6: dialogue tags
    _apply_dialogue_tags(t, s.text)

    # Rule 7: story structure
    if is_first:
        _apply(t, rate=130, label="story:first(slower)")   # scene-setting tone
    if is_last:
        t.pre_pause = max(t.pre_pause, PAUSE["medium"])     # dramatic pause before
        t.rules.append("story:last(pause-before)")

    # Clamp to valid ranges
    t.rate = int(_clamp(t.rate, RATE_MIN, RATE_MAX))
    t.pitch = _clamp(t.pitch, PITCH_MIN, PITCH_MAX)
    t.volume = _clamp(t.volume, VOLUME_MIN, VOLUME_MAX)
    return t


_TAG_WHISPER = re.compile(r"\b(whispered|murmured|softly|quietly)\b", re.IGNORECASE)
_TAG_SHOUT = re.compile(r"\b(shouted|yelled|screamed|bellowed)\b", re.IGNORECASE)
_TAG_ASKED = re.compile(r"\b(asked|questioned|wondered)\b", re.IGNORECASE)
_TAG_LAUGH = re.compile(r"\b(laughed|giggled|chuckled)\b", re.IGNORECASE)


def _apply_dialogue_tags(t: _Target, text: str) -> None:
    low = text.lower()
    if _TAG_WHISPER.search(low) or "said quietly" in low:
        _apply(t, volume=VOLUME["soft"], rate=RATE["slow"], label="tag:whisper")
    if _TAG_SHOUT.search(low):
        _apply(t, volume=VOLUME["loud"], rate=RATE["fast"], label="tag:shout")
    if _TAG_ASKED.search(low):
        t.end_rise = True
        t.rules.append("tag:asked")
    if _TAG_LAUGH.search(low):
        _apply(t, rate=RATE["fast"], pitch=PITCH["high"], label="tag:laughed")


# ---------------------------------------------------------------------------
# Segment building — split a sentence into clauses / words and attach pauses
# ---------------------------------------------------------------------------
# Delimiters that create an inner pause. Dashes only count when whitespace-
# separated so we don't break hyphenated words.
_CLAUSE_RE = re.compile(r"(\.\.\.|…|;|,|\s+—\s+|\s+--\s+|\s+-\s+)")
_WORD_CLEAN = re.compile(r"^\W+|\W+$")


def _delim_pause(delim: str, comma_pause: float) -> tuple[float, bool]:
    """Return (pause_seconds, slows_next_clause) for a clause delimiter."""
    d = delim.strip()
    if d == ",":
        return comma_pause, False
    if d == ";":
        return 0.5, False                # medium pause
    if d in ("—", "--", "-"):
        return 0.4, False                # dramatic pause
    if d in ("...", "…"):
        return PAUSE["long"], True       # long pause + slow down the next clause
    return 0.0, False


def _split_clauses(text: str, comma_pause: float) -> list[tuple[str, float, bool]]:
    """Split into (clause_text, pause_after, is_slowed) triples."""
    parts = _CLAUSE_RE.split(text)
    clauses: list[tuple[str, float, bool]] = []
    slow_this = False
    for j in range(0, len(parts), 2):
        body = parts[j].strip()
        delim = parts[j + 1] if j + 1 < len(parts) else ""
        pause, sets_slow_next = _delim_pause(delim, comma_pause)
        if body:
            clauses.append((body, pause, slow_this))
        slow_this = sets_slow_next
    return clauses


def _clean(word: str) -> str:
    return _WORD_CLEAN.sub("", word)


def _is_caps(word: str) -> bool:
    c = _clean(word)
    return len(c) >= 2 and c.isalpha() and c.isupper()


def _split_words(text: str, rate: int, pitch: float, volume: float) -> list[Segment]:
    """
    Rule 5: emphasis. Most clauses render as a single segment (smoother audio);
    only clauses containing ALL-CAPS or repeated words are broken into per-word
    segments so the emphasis is actually audible.
    """
    words = text.split()
    if not words:
        return []

    cleaned = [_clean(w).lower() for w in words]
    counts = Counter(c for c in cleaned if len(c) >= 2)
    has_caps = any(_is_caps(w) for w in words)
    has_repeat = any(counts[c] >= 2 for c in cleaned if len(c) >= 2)

    if not (has_caps or has_repeat):
        return [Segment(text, int(rate), pitch, volume, 0.0)]

    segs: list[Segment] = []
    seen: dict[str, int] = {}
    for w, c in zip(words, cleaned):
        r, p, v = float(rate), pitch, volume
        # Repeated word -> progressively faster on each occurrence
        if c and counts.get(c, 0) >= 2:
            idx = seen.get(c, 0)
            seen[c] = idx + 1
            r = rate + 18 * idx
        # ALL CAPS -> louder + slower (applied last so it wins over the ramp)
        if _is_caps(w):
            v = VOLUME["loud"]
            r = min(r, RATE["slow"])
            p = pitch + 1.0
        segs.append(Segment(
            w,
            int(_clamp(r, RATE_MIN, RATE_MAX)),
            _clamp(p, PITCH_MIN, PITCH_MAX),
            _clamp(v, VOLUME_MIN, VOLUME_MAX),
            0.0,
        ))
    return segs


def _build_segments(s: _Sentence, t: _Target) -> tuple[list[Segment], float]:
    """Turn a sentence + its target prosody into renderable segments.

    Returns (segments, pre_pause) — pre_pause is folded into the previous
    sentence by the caller.
    """
    segs: list[Segment] = []
    for ctext, cpause, is_slowed in _split_clauses(s.text, t.comma_pause):
        rate = min(t.rate, RATE["slow"]) if is_slowed else t.rate  # ellipsis slows next
        wsegs = _split_words(ctext, rate, t.pitch, t.volume)
        if wsegs:
            wsegs[-1].pause_after = cpause
            segs.extend(wsegs)

    if segs:
        segs[-1].pause_after = max(segs[-1].pause_after, t.post_pause)
        if t.end_rise:
            segs[-1].pitch = _clamp(segs[-1].pitch + 2.0, PITCH_MIN, PITCH_MAX)
    return segs, t.pre_pause


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze(
    text: str,
    emotion_hint: str | None = None,
    verbose: bool = True,
    content_type: str = "non_fiction",
) -> list[Segment]:
    """
    Analyse `text` and return a flat list of `Segment`s for the renderer.

    Args:
        text:         The (ASR) text to speak.
        emotion_hint: Optional acoustic emotion label (from emotion.detect).
        verbose:      Print the per-sentence prosody plan for inspection.
        content_type: 'fiction' or 'non_fiction' (from fiction.detect).
                      Fiction allows stronger story-structure pauses and full
                      emotion-driven rate/pitch ranges. Non-fiction tones them
                      down so factual speech sounds natural.
    """
    is_fiction = content_type == "fiction"
    segments: list[Segment] = []
    debug: list[tuple[_Sentence, _Target, list[Segment]]] = []

    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()] or [text]
    for para in paragraphs:
        sentences = _split_sentences(para.strip())
        for i, s in enumerate(sentences):
            t = _analyze_sentence(s, i == 0, i == len(sentences) - 1, emotion_hint)

            if not is_fiction:
                # Non-fiction: pull dramatic extremes back toward neutral
                t.rate   = int(NEUTRAL_RATE + (t.rate - NEUTRAL_RATE) * 0.55)
                t.pitch  = NEUTRAL_PITCH + (t.pitch - NEUTRAL_PITCH) * 0.40
                t.pre_pause  = min(t.pre_pause,  PAUSE["very_short"])
                t.post_pause = min(t.post_pause, PAUSE["short"])

            segs, pre = _build_segments(s, t)
            if pre > 0 and segments:
                segments[-1].pause_after = max(segments[-1].pause_after, pre)
            segments.extend(segs)
            debug.append((s, t, segs))

    if verbose:
        mode = "fiction" if is_fiction else "non_fiction"
        print(f"[Prosody] content_type={mode}")
        _print_plan(debug)
    return segments


def _print_plan(debug) -> None:
    print(f"[Prosody] Analysed {len(debug)} sentence(s):")
    for s, t, segs in debug:
        print(f"  - {s.text!r}  [{s.kind}]")
        print(f"      rate={t.rate}wpm  pitch={t.pitch:+.1f}st  vol={t.volume:.2f}  "
              f"pre={t.pre_pause:.1f}s  post={t.post_pause:.1f}s  end_rise={t.end_rise}")
        print(f"      rules: {', '.join(t.rules) if t.rules else 'none'}  "
              f"-> {len(segs)} segment(s)")


# ---------------------------------------------------------------------------
# Acoustic prosody — per-word feature extraction directly from audio
# ---------------------------------------------------------------------------

def analyze_acoustic(
    audio_path: str,
    words: list[dict],
    verbose: bool = True,
    content_type: str = "non_fiction",
) -> list[Segment]:
    """
    Build TTS segments by extracting acoustic features per word DIRECTLY from
    the audio signal, using Whisper word timestamps.

    This replaces text-based rule analysis entirely.  Instead of guessing
    prosody from punctuation or emotion keywords, we measure HOW the speaker
    actually said each word:

      Duration in audio  →  TTS speaking rate   (stretched word → slow TTS)
      RMS energy         →  TTS volume           (loud word → loud TTS)
      F0 (pitch)         →  TTS pitch shift      (high pitch → raised TTS pitch)

    All three features are computed from the raw waveform slice for each word,
    then normalised relative to the speaker's own baseline (median across the
    whole recording) so the mapping is speaker-independent.

    Args:
        audio_path:  Path to the original WAV recording.
        words:       List of {"word", "start", "end"} from asr.transcribe_with_timestamps.
        verbose:     Print per-word feature table.

    Returns:
        Flat list of Segment objects ready for tts.speak_prosody().
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=16_000, mono=True)

    # --- Step 1: extract raw acoustic features for each word's audio slice ---
    raw: list[dict] = []
    for w in words:
        start = int(w["start"] * sr)
        end   = int(w["end"]   * sr)
        chunk = y[start:end]

        if len(chunk) < 320:          # < 20 ms — too short for reliable F0
            raw.append({"f0": 0.0, "energy": 0.0, "duration": w["end"] - w["start"]})
            continue

        # Size-adaptive FFT so librosa never warns about n_fft > signal length
        n_fft = max(128, min(2048, 2 ** int(np.log2(max(len(chunk), 128)))))

        # F0 directly from this word's audio window
        pitches, magnitudes = librosa.piptrack(y=chunk, sr=sr, fmin=75, fmax=400, n_fft=n_fft)
        voiced = (
            pitches[magnitudes > magnitudes.max() * 0.1]
            if magnitudes.max() > 0 else np.array([])
        )
        f0 = float(np.mean(voiced)) if len(voiced) > 0 else 0.0

        # RMS energy directly from this word's audio window
        energy = float(np.sqrt(np.mean(chunk ** 2)))

        raw.append({"f0": f0, "energy": energy, "duration": w["end"] - w["start"]})

    if not raw:
        return []

    # --- Step 2: compute speaker baseline (median = typical/neutral delivery) ---
    valid_f0      = [r["f0"]     for r in raw if r["f0"] > 50]
    valid_energy  = [r["energy"] for r in raw]
    dur_per_char  = [
        r["duration"] / max(len(w["word"]), 1)
        for r, w in zip(raw, words)
        if r["duration"] > 0
    ]

    baseline_f0     = float(np.median(valid_f0))     if valid_f0     else 150.0
    baseline_energy = float(np.median(valid_energy)) if valid_energy else 0.01
    baseline_dpc    = float(np.median(dur_per_char)) if dur_per_char else 0.07  # s/char

    if verbose:
        print(
            f"[AcousticProsody] baseline — F0={baseline_f0:.0f}Hz  "
            f"energy={baseline_energy:.4f}  dur/char={baseline_dpc:.3f}s"
        )

    is_fiction = content_type == "fiction"

    BASE_RATE         = 150
    BASE_VOLUME       = 0.80
    # Fiction lowers the stretch bar (more expressive delivery expected) and
    # uses a wider pitch range. Non-fiction keeps things neutral and clipped.
    PHRASE_GAP        = 0.15
    STRETCH_THRESHOLD = 1.5 if is_fiction else 1.8
    PITCH_SCALE       = 1.0 if is_fiction else 0.0   # 1.0 = full F0-derived shift; 0 = flat

    # Per-word duration ratios and inter-word gaps
    dur_ratios: list[float] = []
    for feat, w in zip(raw, words):
        expected = max(len(w["word"]), 1) * baseline_dpc
        dur_ratios.append(feat["duration"] / max(expected, 0.03))

    gap_after: list[float] = []
    for i in range(len(words)):
        gap = float(max(0.0, words[i + 1]["start"] - words[i]["end"])) \
              if i < len(words) - 1 else 0.35
        gap_after.append(gap)

    def _vol(feats: list[dict]) -> float:
        avg = float(np.mean([f["energy"] for f in feats]))
        ratio = avg / max(baseline_energy, 1e-6)
        return float(max(0.4, min(1.0, BASE_VOLUME * min(ratio * 1.5, 2.0))))

    def _f0_to_semitones(feats: list[dict]) -> float:
        """Convert mean group F0 to a semitone pitch shift relative to baseline.
        Only used in fiction mode (PITCH_SCALE=1.0); returns 0.0 otherwise."""
        if PITCH_SCALE == 0.0 or baseline_f0 <= 0:
            return 0.0
        voiced = [f["f0"] for f in feats if f["f0"] > 50]
        if not voiced:
            return 0.0
        mean_f0 = float(np.mean(voiced))
        import math
        raw = math.log2(mean_f0 / baseline_f0) * 12 * PITCH_SCALE
        return float(max(-4.0, min(4.0, raw)))   # clamp to ±4 semitones

    def _flush(grp_w: list[dict], grp_f: list[dict], pause: float) -> Segment:
        text    = " ".join(w["word"] for w in grp_w)
        n_chars = max(sum(len(w["word"]) for w in grp_w), 1)
        actual  = sum(f["duration"] for f in grp_f)
        dr      = actual / max(n_chars * baseline_dpc, 0.03)
        rate    = int(max(80, min(220, BASE_RATE / max(dr, 0.4))))
        pitch   = _f0_to_semitones(grp_f)
        return Segment(text=text, rate=rate, pitch=pitch, volume=_vol(grp_f), pause_after=pause)

    if verbose:
        mode = "fiction" if is_fiction else "non_fiction"
        print(f"[AcousticProsody] content_type={mode}  "
              f"stretch_threshold={STRETCH_THRESHOLD}  pitch_scale={PITCH_SCALE}")
        print(f"[AcousticProsody] segments  (★ = stretched word):")
        print(f"  {'Text':<35} {'dur':>5} {'rate':>5} {'vol':>5} {'pause':>6}")
        print("  " + "-" * 57)

    segments: list[Segment] = []
    grp_w: list[dict] = []
    grp_f: list[dict] = []
    i = 0

    while i < len(words):
        if dur_ratios[i] >= STRETCH_THRESHOLD:
            # Flush any accumulated normal words first
            if grp_w:
                seg = _flush(grp_w, grp_f, 0.06)
                if verbose:
                    lbl = (seg.text[:32] + "…") if len(seg.text) > 33 else seg.text
                    print(f"  {lbl!r:<35} {sum(f['duration'] for f in grp_f):>5.2f}s  "
                          f"{seg.rate:>4}wpm  {seg.volume:>4.2f}  {seg.pause_after:>5.2f}s")
                segments.append(seg)
                grp_w, grp_f = [], []

            # Stretched word gets its own slow segment
            rate  = int(max(70, min(140, BASE_RATE / max(dur_ratios[i], 0.4))))
            pitch = _f0_to_semitones([raw[i]])
            seg   = Segment(
                text=words[i]["word"], rate=rate, pitch=pitch,
                volume=_vol([raw[i]]), pause_after=gap_after[i],
            )
            if verbose:
                print(f"  {words[i]['word']!r:<35}★{raw[i]['duration']:>4.2f}s  "
                      f"{rate:>4}wpm  {seg.volume:>4.2f}  {gap_after[i]:>5.2f}s")
            segments.append(seg)
            i += 1

        else:
            grp_w.append(words[i])
            grp_f.append(raw[i])
            pause = gap_after[i]
            i += 1

            # Flush on phrase boundary
            if pause >= PHRASE_GAP:
                seg = _flush(grp_w, grp_f, pause)
                if verbose:
                    lbl = (seg.text[:32] + "…") if len(seg.text) > 33 else seg.text
                    print(f"  {lbl!r:<35} {sum(f['duration'] for f in grp_f):>5.2f}s  "
                          f"{seg.rate:>4}wpm  {seg.volume:>4.2f}  {seg.pause_after:>5.2f}s")
                segments.append(seg)
                grp_w, grp_f = [], []

    # Flush any remaining words
    if grp_w:
        seg = _flush(grp_w, grp_f, 0.35)
        if verbose:
            lbl = (seg.text[:32] + "…") if len(seg.text) > 33 else seg.text
            print(f"  {lbl!r:<35} {sum(f['duration'] for f in grp_f):>5.2f}s  "
                  f"{seg.rate:>4}wpm  {seg.volume:>4.2f}  {seg.pause_after:>5.2f}s")
        segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Standalone demo — `python prosody.py` prints plans without needing audio.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        "Are you really coming to the party?",
        "I can't believe it!!!",
        "And then... everything went quiet.",
        '"Get out of here!" she shouted.',
        "He whispered that the dark room terrified him.",
        "The old, dusty, forgotten library, full of crumbling books, smelled of rain.",
        "He ran and ran and ran until he could run no more.",
        "STOP right there.",
        "She asked if I was okay.",
    ]
    for ex in examples:
        print("\n" + "=" * 70)
        print(f"INPUT: {ex}")
        analyze(ex, verbose=True)
