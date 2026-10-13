"""
Fiction vs Non-Fiction Classifier
==================================
POS-ratio classifier grounded in two published studies:

Qureshi et al. (2019) — "A Simple Approach to Classify Fictional and
    Non-Fictional Genres". Just two POS ratios (adverb/adjective and
    adjective/pronoun) reach ~96% accuracy on the Brown corpus.

Kazmi et al. (2022)  — "Linguistically Motivated Features for Classifying
    Shorter Text into Fiction and Non-Fiction Genre". Adds lexical
    density (content/function words) and character diversity (TTR) which
    are the strongest single features for paragraph-level text.

Fiction indicators (positive logit):
    adverb/adjective ratio     — fiction uses more descriptive adverbs
    character diversity (TTR)  — fiction has richer vocabulary
    strong fiction signals     — dialogue quotes, narrative verbs, story phrases

Non-fiction indicators (negative logit):
    adjective/pronoun ratio    — non-fiction packs noun-modifying adjectives
    noun/verb ratio            — non-fiction is noun-heavy
    lexical density            — non-fiction is more information-dense
    adverb/pronoun ratio       — Kazmi Table 5 coefficient

POS tagging uses NLTK's averaged perceptron tagger (Penn Treebank tagset).
Both papers' tag-to-category mappings (Kazmi Table 3) are Penn Treebank tags,
so NLTK's output plugs in directly.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# POS tag categories — Penn Treebank tagset (Kazmi et al. 2022, Table 3)
# ---------------------------------------------------------------------------

_ADJ_TAGS   = {"JJ", "JJR", "JJS"}
_ADV_TAGS   = {"RB", "RBR", "RBS", "WRB"}
_NOUN_TAGS  = {"NN", "NNS"}
_PROPN_TAGS = {"NNP", "NNPS"}
_VERB_TAGS  = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
_PRON_TAGS  = {"PRP", "PRP$", "WP", "WP$"}
_POSS_TAGS  = {"PRP$", "WP$"}  # possessive determiners: her, his, my, our…

# Content words = noun ∪ verb ∪ adjective ∪ adverb (Kazmi §3.2). We include
# proper nouns with common nouns for the content/function distinction.
_CONTENT_TAGS = _NOUN_TAGS | _PROPN_TAGS | _VERB_TAGS | _ADJ_TAGS | _ADV_TAGS


# ---------------------------------------------------------------------------
# NLTK setup — download resources on first use, then cache
# ---------------------------------------------------------------------------

_tagger_ready = False


def _ensure_nltk() -> None:
    """Download NLTK POS-tagger + tokenizer resources on first call."""
    global _tagger_ready
    if _tagger_ready:
        return
    import nltk
    resources = [
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("tokenizers/punkt_tab",                   "punkt_tab"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[Fiction] Downloading NLTK resource '{pkg}' (~few MB)...")
            nltk.download(pkg, quiet=True)
    _tagger_ready = True


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _pos_counts(text: str) -> dict[str, int | float]:
    """
    Tokenize `text`, POS-tag it, and count everything the scorer needs:
    POS categories, dialogue markers, proper-noun→verb bigrams, possessive
    determiners, and a length-robust type-token ratio (MATTR-30).
    """
    _ensure_nltk()
    import nltk

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if any(c.isalpha() for c in t)]  # drop pure-punct

    empty_keys = ("adj", "adv", "noun", "propn", "verb", "pron", "poss",
                  "content", "function", "tokens", "types",
                  "dialogue", "propn_verb")
    if not tokens:
        return {k: 0 for k in empty_keys} | {"mattr": 0.0}

    tagged = nltk.pos_tag(tokens)
    c: Counter = Counter()
    for _, tag in tagged:
        if   tag in _ADJ_TAGS:   c["adj"]   += 1
        elif tag in _ADV_TAGS:   c["adv"]   += 1
        elif tag in _NOUN_TAGS:  c["noun"]  += 1
        elif tag in _PROPN_TAGS: c["propn"] += 1
        elif tag in _VERB_TAGS:  c["verb"]  += 1
        elif tag in _PRON_TAGS:  c["pron"]  += 1
        if tag in _POSS_TAGS:    c["poss"]  += 1
        if tag in _CONTENT_TAGS:
            c["content"]  += 1
        else:
            c["function"] += 1

    # Kazmi Table 5: (VERB, PROPN, after) has coef +0.59 — the "said Lily"
    # word-order pattern. We count adjacent PROPN→VERB bigrams either way
    # around ("Leo nodded", "said Lily") as a proxy without a parser.
    propn_verb = 0
    for i in range(len(tagged) - 1):
        t1, t2 = tagged[i][1], tagged[i + 1][1]
        if (t1 in _PROPN_TAGS and t2 in _VERB_TAGS) or \
           (t1 in _VERB_TAGS  and t2 in _PROPN_TAGS):
            propn_verb += 1

    # MATTR — moving-average type-token ratio at word level, window 30.
    # Length-robust proxy for Kazmi's character-diversity feature (+2.31).
    window = 30
    if len(tokens) <= window:
        mattr = len({t.lower() for t in tokens}) / len(tokens)
    else:
        ratios = []
        lowers = [t.lower() for t in tokens]
        for i in range(len(lowers) - window + 1):
            seg = lowers[i:i + window]
            ratios.append(len(set(seg)) / window)
        mattr = sum(ratios) / len(ratios)

    # Discourse marker proxy — dialogue quote count (Kazmi discourse coef +1.17).
    c["dialogue"]    = len(_DIALOGUE_MARKERS.findall(text))
    c["propn_verb"]  = propn_verb
    c["tokens"]      = len(tokens)
    c["types"]       = len({t.lower() for t in tokens})
    out: dict[str, int | float] = dict(c)
    out["mattr"] = mattr
    return out


def _safe(a: int, b: int, default: float = 0.0) -> float:
    return a / b if b > 0 else default


def _compute_features(text: str) -> dict[str, float]:
    """
    All features derived from POS counts. See module docstring for citations.
    """
    c = _pos_counts(text)
    g = lambda k: c.get(k, 0)
    tokens = max(1, g("tokens"))
    # Include proper nouns in noun-related ratios so fiction with named
    # characters (Maya, Leo) doesn't get penalised as low-noun text.
    noun_all = g("noun") + g("propn")
    return {
        # Paper 1 (Qureshi 2019) — the two strongest features
        "adv_adj":   _safe(g("adv"),  g("adj")),
        "adj_pro":   _safe(g("adj"),  g("pron")),
        # Paper 2 (Kazmi 2022) — POS-ratio discriminators
        "adv_pro":   _safe(g("adv"),  g("pron")),
        "noun_verb": _safe(noun_all,  g("verb")),
        "lex_den":   _safe(g("content"), g("function")),
        # Character diversity — MATTR-30 at word level (Kazmi Table 5, +2.31)
        "mattr":     float(c.get("mattr", 0.0)),
        # Narrative-specific signals from Kazmi Table 5 (approximated
        # without a dependency parser, all densities per 100 tokens)
        "dial_pct":  100.0 * g("dialogue")   / tokens,   # discourse +1.17
        "propn_v":   100.0 * g("propn_verb") / tokens,   # (VERB,PROPN) +0.59
        "poss_pct":  100.0 * g("poss")       / tokens,   # nmod:poss +0.19
    }


# ---------------------------------------------------------------------------
# Logistic-regression scorer
#
# Weights follow the SIGN of each paper's regression coefficient and
# preserve relative magnitude, but are re-scaled around empirical
# baselines because both papers fit their models on standardized inputs
# and we have no access to their training data's mean/std.
#
# Empirical means below are approximate mid-points between Brown fiction
# and non-fiction values reported in the papers (e.g. Kazmi §5, Fig 1-2).
# ---------------------------------------------------------------------------

_INTERCEPT = 0.0

_WEIGHTS = {
    # POS ratios (Qureshi 2019 + Kazmi 2022)
    "adv_adj":   +2.73,   # Qureshi Table 4
    "adj_pro":   -1.20,   # Qureshi -2.90, tempered
    "adv_pro":   -0.19,   # Kazmi Table 5
    "noun_verb": -0.30,   # Kazmi -0.68, tempered
    "lex_den":   -2.40,   # Kazmi -5.83, tempered
    "mattr":     +2.30,   # Kazmi TTR +2.31
    # Narrative-specific signals (Kazmi Table 5)
    "dial_pct":  +0.35,   # discourse +1.17 — dialogue quotes per 100 tokens
    "propn_v":   +0.40,   # (VERB,PROPN) +0.59 — "said Lily" / "Leo nodded"
    "poss_pct":  +0.15,   # nmod:poss +0.19 — "her heart", "his eyes"
}

_MEANS = {
    "adv_adj":   0.75,
    "adj_pro":   2.20,
    "adv_pro":   1.00,
    "noun_verb": 1.60,
    "lex_den":   0.80,    # real prose is content-word rich; midpoint fic/non-fic
    "mattr":     0.65,    # MATTR-30 baseline for English prose
    "dial_pct":  1.00,    # ~1 quote per 100 words is neutral
    "propn_v":   1.50,
    "poss_pct":  2.00,
}


# Clip ratios so a single outlier (e.g. formal science prose with lots of
# hedge adverbs and no adjectives) can't swamp the whole logit. Ceilings
# reflect the paper's observed distributional range.
_CLIPS = {
    "adv_adj":   (0.0, 1.5),
    "adj_pro":   (0.0, 6.0),
    "adv_pro":   (0.0, 4.0),
    "noun_verb": (0.0, 4.0),
    "lex_den":   (0.0, 3.0),
}


def _score(features: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Return (fiction_probability, per-feature contribution dict)."""
    logit = _INTERCEPT
    contribs: dict[str, float] = {}
    for name, w in _WEIGHTS.items():
        raw = features[name]
        if name in _CLIPS:
            lo, hi = _CLIPS[name]
            raw = max(lo, min(hi, raw))
        centred = raw - _MEANS[name]
        contrib = w * centred
        contribs[name] = contrib
        logit += contrib
    prob_fiction = 1.0 / (1.0 + math.exp(-logit))
    return prob_fiction, contribs


# ---------------------------------------------------------------------------
# Strong-signal fast-path — cheap regex checks for unmistakable fiction
# ---------------------------------------------------------------------------

_NARRATIVE_VERBS = re.compile(
    r"\b(said|replied|asked|answered|whispered|shouted|yelled|murmured|cried|"
    r"laughed|smiled|frowned|sighed|gasped|nodded|shook|ran|walked|jumped|"
    r"climbed|fled|rushed|charged|turned|looked|glanced|stared|reached|grabbed|"
    r"pulled|pushed|threw|fell|rose|stood|sat|lay|knelt|leaned|stepped|"
    r"thought|felt|knew|realized|remembered|wondered|imagined|believed|feared|"
    r"heard|saw|noticed|watched|followed|led|brought|carried|held|"
    r"crouched|crept|tiptoed|dashed|darted|bolted|scrambled|stumbled|"
    r"tumbled|slipped|leapt|leaped|hopped|skipped|swayed|swung|swept|"
    r"gripped|clutched|clasped|snatched|dropped|tossed|hurled|flung|"
    r"tapped|patted|stroked|touched|kissed|hugged|shoved|dragged|"
    r"snapped|cracked|slammed|banged|thumped|crashed|smashed|shattered|"
    r"whimpered|sobbed|moaned|groaned|hissed|growled|barked|snarled|"
    r"chuckled|giggled|grinned|beamed|winked|blinked|peeked|peered|"
    r"squinted|gaped|scowled|glared|smirked|pouted|shrugged|shivered|"
    r"trembled|shuddered|flinched|jerked|jolted|stiffened|slumped|"
    r"breathed|panted|gasped|choked|coughed|yawned|sniffed|snorted|"
    r"tilted|craned|cocked|turned|twisted|spun|whirled|hovered|floated|"
    r"drifted|glided|soared|plunged|dove|dived|dipped|swam|paddled|"
    r"bounded|scampered|scurried|trotted|galloped|prowled|padded|"
    r"emerged|appeared|vanished|disappeared|arrived|departed|entered|"
    r"exited|greeted|waved|beckoned|pointed|gestured|shouted|whispered)\b",
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
# "she whispered", "he nodded", "they ran" — any past-tense verb after a
# third-person pronoun. Broad by design; false positives are cheap because
# this is only one of several fast-path signals.
_THIRD_PERSON = re.compile(
    r"\b(he|she|they|it)\s+\w+ed\b",
    re.IGNORECASE,
)
_DIALOGUE_MARKERS = re.compile(r'["“”].*?["“”]', re.DOTALL)


def _strong_fiction_signals(text: str) -> float | None:
    """Return a high fiction confidence when signals are unmistakable."""
    dialogue_count  = len(_DIALOGUE_MARKERS.findall(text))
    narrative_count = len(_NARRATIVE_VERBS.findall(text))
    third_person    = len(_THIRD_PERSON.findall(text))
    story_phrases   = len(_STORY_STRUCTURE.findall(text))

    # Strong: dialogue quotes + narrative verbs → almost certainly fiction
    if dialogue_count >= 2 and narrative_count >= 5:
        conf = min(0.97, 0.80 + dialogue_count * 0.02 + third_person * 0.01)
        print(f"[Fiction] → fiction ({conf:.0%})  [strong signals: "
              f"{dialogue_count} dialogue blocks, {narrative_count} narrative verbs]")
        return conf

    # Moderate: story-structure phrases OR heavy third-person narrative
    if story_phrases >= 2 or (third_person >= 3 and narrative_count >= 5):
        conf = min(0.92, 0.75 + story_phrases * 0.03 + third_person * 0.01)
        print(f"[Fiction] → fiction ({conf:.0%})  [story signals: "
              f"{story_phrases} structure phrases, {third_person} third-person verbs]")
        return conf

    return None


# ---------------------------------------------------------------------------
# Fallback — if NLTK is unavailable, fall back to the strong-signal check
# alone; when even that is silent, default to non_fiction with low
# confidence so callers can decide.
# ---------------------------------------------------------------------------

def _fallback(text: str) -> tuple[str, float]:
    fast = _strong_fiction_signals(text)
    if fast is not None:
        return "fiction", fast
    print("[Fiction] → non_fiction (55%)  [no POS tagger, no strong signals]")
    return "non_fiction", 0.55


# ---------------------------------------------------------------------------
# Human-readable feature descriptions for the /explain endpoint
# ---------------------------------------------------------------------------

# For each feature: how to describe it when its contribution is fiction-positive
# vs non-fiction-positive, and a citation to the paper it comes from.
_FEATURE_INFO = {
    "adv_adj": {
        "label":     "Adverb-to-adjective ratio",
        "fiction":   "high — fiction uses more descriptive adverbs "
                     "(e.g. \"walked slowly\", \"whispered softly\")",
        "nonfic":    "low — non-fiction packs more noun-modifying adjectives "
                     "than adverbs",
        "cite":      "Qureshi et al. 2019, Table 4 (coef +2.73)",
    },
    "adj_pro": {
        "label":     "Adjective-to-pronoun ratio",
        "fiction":   "low — fiction is pronoun-heavy (characters referred to "
                     "as he/she/they)",
        "nonfic":    "high — non-fiction stacks adjectives with few pronouns "
                     "(\"green pigment chlorophyll\")",
        "cite":      "Qureshi et al. 2019, Table 4 (coef −2.90)",
    },
    "adv_pro": {
        "label":     "Adverb-to-pronoun ratio",
        "fiction":   "low — many pronouns, few free-standing adverbs",
        "nonfic":    "high — formal prose uses hedge adverbs "
                     "(\"generally\", \"approximately\") with no characters",
        "cite":      "Kazmi et al. 2022, Table 5 (coef −0.19)",
    },
    "noun_verb": {
        "label":     "Noun-to-verb ratio",
        "fiction":   "low — action-driven prose has more verbs relative to nouns",
        "nonfic":    "high — non-fiction is noun-heavy (concepts, entities)",
        "cite":      "Kazmi et al. 2022, Table 5 (coef −0.68)",
    },
    "lex_den": {
        "label":     "Lexical density (content/function words)",
        "fiction":   "low — fiction has more function words (pronouns, articles, "
                     "auxiliaries) around narrative",
        "nonfic":    "high — non-fiction is information-dense, packed with "
                     "content words",
        "cite":      "Kazmi et al. 2022, Table 5 (coef −5.83, strongest signal)",
    },
    "mattr": {
        "label":     "Vocabulary diversity (MATTR-30)",
        "fiction":   "high — fiction uses richer, more varied vocabulary",
        "nonfic":    "low — non-fiction repeats key terms (subject-matter jargon)",
        "cite":      "Kazmi et al. 2022, Table 5 TTR (coef +2.31)",
    },
    "dial_pct": {
        "label":     "Dialogue density",
        "fiction":   "high — quoted speech is a hallmark of fiction narrative",
        "nonfic":    "low — no dialogue markers found",
        "cite":      "Kazmi et al. 2022, Table 5 discourse (coef +1.17)",
    },
    "propn_v": {
        "label":     "Named-character + verb pairs",
        "fiction":   "high — named characters performing actions "
                     "(e.g. \"Leo nodded\", \"Maya laughed\")",
        "nonfic":    "low — few named agents acting in the text",
        "cite":      "Kazmi et al. 2022, Table 5 (VERB,PROPN) (coef +0.59)",
    },
    "poss_pct": {
        "label":     "Possessive-determiner density",
        "fiction":   "high — narrative uses many possessives "
                     "(\"her heart\", \"his eyes\", \"their friend\")",
        "nonfic":    "low — non-fiction uses full noun references instead",
        "cite":      "Kazmi et al. 2022, Table 5 nmod:poss (coef +0.19)",
    },
}


def _humanize_features(features: dict[str, float],
                       contribs: dict[str, float],
                       top_n: int = 4) -> list[dict]:
    """
    Rank features by |contribution| and return a list of plain-English
    explanations for the top ones. Each entry describes what the feature
    measured and which direction it pushed.
    """
    ranked = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))
    reasons: list[dict] = []
    for name, contrib in ranked[:top_n]:
        if abs(contrib) < 0.05:
            continue  # skip trivial contributions
        info = _FEATURE_INFO.get(name, {})
        direction = "fiction" if contrib > 0 else "non-fiction"
        message   = info.get("fiction" if contrib > 0 else "nonfic", "")
        reasons.append({
            "feature":      info.get("label", name),
            "value":        round(features[name], 3),
            "contribution": round(contrib, 3),
            "direction":    direction,
            "message":      message,
            "cite":         info.get("cite", ""),
        })
    return reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _explain_fast_path(text: str) -> dict | None:
    """Rebuild the fast-path decision as an explanation dict."""
    dialogue_count  = len(_DIALOGUE_MARKERS.findall(text))
    narrative_count = len(_NARRATIVE_VERBS.findall(text))
    third_person    = len(_THIRD_PERSON.findall(text))
    story_phrases   = len(_STORY_STRUCTURE.findall(text))

    if dialogue_count >= 2 and narrative_count >= 5:
        return {
            "method": "fast-path",
            "trigger": "strong",
            "signals": {
                "dialogue_blocks":    dialogue_count,
                "narrative_verbs":    narrative_count,
                "third_person_verbs": third_person,
                "story_phrases":      story_phrases,
            },
            "summary": (
                f"The text contains {dialogue_count} dialogue block(s) and "
                f"{narrative_count} past-tense narrative verbs — classic markers "
                f"of storytelling that non-fiction almost never displays together."
            ),
        }

    if story_phrases >= 2 or (third_person >= 3 and narrative_count >= 5):
        return {
            "method": "fast-path",
            "trigger": "moderate",
            "signals": {
                "dialogue_blocks":    dialogue_count,
                "narrative_verbs":    narrative_count,
                "third_person_verbs": third_person,
                "story_phrases":      story_phrases,
            },
            "summary": (
                f"Detected {story_phrases} story-structure phrase(s) "
                f"(e.g. \"once upon a time\", \"suddenly\") and "
                f"{third_person} third-person past-tense clause(s) — hallmarks "
                f"of narrative fiction."
            ),
        }

    return None


def explain(text: str, verbose: bool = False) -> tuple[str, float, dict]:
    """
    Same classification as `detect()`, but also returns a full explanation
    dict suitable for rendering in a UI. The dict contains the method
    used (fast-path or POS-ratios), the numeric evidence, and a ranked
    list of plain-English reasons with paper citations.
    """
    fast_expl = _explain_fast_path(text)
    if fast_expl is not None:
        # Fast-path fired — reuse its confidence formula
        d = fast_expl["signals"]
        if fast_expl["trigger"] == "strong":
            conf = min(0.97, 0.80 + d["dialogue_blocks"] * 0.02
                              + d["third_person_verbs"] * 0.01)
        else:
            conf = min(0.92, 0.75 + d["story_phrases"] * 0.03
                              + d["third_person_verbs"] * 0.01)
        return "fiction", conf, fast_expl

    try:
        feats = _compute_features(text)
        prob_fiction, contribs = _score(feats)
    except Exception as e:
        print(f"[Fiction] POS tagger unavailable ({e}), using fallback.")
        label, conf = _fallback(text)
        return label, conf, {"method": "fallback",
                             "summary": "POS tagger unavailable; used keyword fallback."}

    label = "fiction" if prob_fiction >= 0.5 else "non_fiction"
    conf  = prob_fiction if label == "fiction" else 1.0 - prob_fiction

    reasons = _humanize_features(feats, contribs)
    total_pos = sum(v for v in contribs.values() if v > 0)
    total_neg = sum(v for v in contribs.values() if v < 0)

    expl = {
        "method":         "pos-ratios",
        "fiction_prob":   round(prob_fiction, 3),
        "reasons":        reasons,
        "total_fiction_push":     round(total_pos, 3),
        "total_nonfiction_push":  round(total_neg, 3),
        "summary": (
            f"Weighted logistic-regression over {len(_WEIGHTS)} POS-based "
            f"features. Fiction-pushing evidence totals {total_pos:+.2f}; "
            f"non-fiction-pushing evidence totals {total_neg:+.2f}. "
            f"After sigmoid, fiction probability = {prob_fiction:.0%}."
        ),
    }
    return label, conf, expl


def detect(text: str, verbose: bool = True) -> tuple[str, float]:
    """
    Classify `text` as 'fiction' or 'non_fiction'.

    Returns (label, confidence). For the full reasoning (feature values,
    contributions, plain-English justifications, paper citations) use
    `explain()` instead.
    """
    label, conf, expl = explain(text, verbose=verbose)

    if verbose:
        if expl.get("method") == "fast-path":
            d = expl["signals"]
            tag = "strong signals" if expl["trigger"] == "strong" else "story signals"
            print(f"[Fiction] → {label} ({conf:.0%})  [{tag}: "
                  f"{d['dialogue_blocks']} dialogue blocks, "
                  f"{d['narrative_verbs']} narrative verbs]")
        elif expl.get("method") == "pos-ratios":
            top = expl["reasons"][:3]
            top_str = ", ".join(f"{r['feature'].split()[0].lower()}"
                                f"={r['contribution']:+.2f}" for r in top)
            print(f"[Fiction] → {label} ({conf:.0%})  [POS ratios; top: {top_str}]")

    return label, conf
