"""
HumanizingSpeech — Streamlit UI
================================
Upload a .txt story file to:
  1. Detect fiction vs non-fiction (with confidence score)
  2. Analyse the emotion of every paragraph
  3. Generate expressive audio via the trained BERT prosody model + Edge Neural TTS
"""

import os
import sys
import tempfile

import streamlit as st

# ── ensure local modules are importable ──────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HumanizingSpeech",
    page_icon="🎙️",
    layout="wide",
)

# ── emotion styling ───────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "neutral":   "😐",
    "joyful":    "😄",
    "scared":    "😨",
    "dramatic":  "🎭",
    "sad":       "😢",
    "surprised": "😮",
    "angry":     "😠",
    "fearful":   "😱",
}

EMOTION_COLOR = {
    "neutral":   "#6B7280",
    "joyful":    "#B45309",
    "scared":    "#1D4ED8",
    "dramatic":  "#7C3AED",
    "sad":       "#0369A1",
    "surprised": "#B45309",
    "angry":     "#B91C1C",
    "fearful":   "#1E3A5F",
}

EMOTION_BG = {
    "neutral":   "#F3F4F6",
    "joyful":    "#FEF3C7",
    "scared":    "#DBEAFE",
    "dramatic":  "#EDE9FE",
    "sad":       "#E0F2FE",
    "surprised": "#FFF7ED",
    "angry":     "#FEE2E2",
    "fearful":   "#EFF6FF",
}


def _badge(emotion: str, confidence: float) -> str:
    fg  = EMOTION_COLOR.get(emotion, "#374151")
    bg  = EMOTION_BG.get(emotion, "#F3F4F6")
    emoji = EMOTION_EMOJI.get(emotion, "❓")
    return (
        f'<span style="background:{bg};color:{fg};padding:4px 12px;'
        f'border-radius:999px;font-size:0.82em;font-weight:700;'
        f'border:1px solid {fg}30;">'
        f'{emoji}&nbsp;{emotion}</span>'
    )


# ── cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading text-emotion model (first run downloads ~300 MB)…")
def _get_text_emotion():
    import text_emotion as te
    te.detect("warm up call")   # triggers download + caching
    return te


@st.cache_resource(show_spinner="Loading fiction classifier…")
def _get_fiction():
    import fiction as fc
    return fc


@st.cache_resource(show_spinner="Loading BERT prosody model (418 MB checkpoint)…")
def _get_prosody_model():
    import prosody_model as pm
    ckpt = os.path.join(_ROOT, "checkpoints", "best.pt")
    pm.load(ckpt if os.path.exists(ckpt) else None)
    return pm


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_paragraphs(text: str) -> list[str]:
    """
    Split story text into paragraphs, handling multiple formatting styles:
      1. Double newline  (standard — most Storynory files)
      2. Single newline  (each line is a paragraph)
      3. No newlines     (continuous prose — split into sentence clusters)
    Returns at least 1 non-empty paragraph.
    """
    import re

    # Strategy 1: double newline
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) >= 2:
        return paras

    # Strategy 2: single newline (lines ARE paragraphs)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        # Group short consecutive lines into one paragraph (handles soft wraps)
        groups, buf = [], []
        for ln in lines:
            buf.append(ln)
            # Flush when we hit what looks like a sentence end or a long line
            if ln[-1] in ".!?" or len(" ".join(buf)) > 300:
                groups.append(" ".join(buf))
                buf = []
        if buf:
            groups.append(" ".join(buf))
        if len(groups) >= 2:
            return groups

    # Strategy 3: no newlines — cluster sentences into groups of 4
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunk_size = 4
    chunks = [
        " ".join(sentences[i : i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]
    return [c for c in chunks if c] or [text.strip()]


def _detect_paragraph_emotions(paragraphs: list[str]) -> list[dict]:
    """Return full emotion vector dict for each paragraph."""
    te = _get_text_emotion()
    return [te.detect(p) for p in paragraphs]


def _generate_audio(
    paragraphs: list[str],
    para_vecs: list[dict],
    content_type: str,
    use_va: bool = True,
) -> bytes:
    """
    Run prosody prediction + Edge Neural TTS for every paragraph.
    Returns WAV bytes (browser-playable).
    use_va=True  → V-A continuous formula
    use_va=False → legacy discrete label lookup
    """
    import fusion
    import tts

    pm = _get_prosody_model()
    all_segments = []

    for para, para_vec in zip(paragraphs, para_vecs):
        fused = fusion.fuse(para_vec, speech_vec=None)
        segs  = pm.predict(para, fused, content_type=content_type)
        all_segments.extend(segs)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    tts.speak_prosody_edge(all_segments, save_path=out_path, play=False, use_va=use_va)

    with open(out_path, "rb") as fh:
        wav_bytes = fh.read()
    os.remove(out_path)
    return wav_bytes


def _prosody_diff_table(paragraphs: list[str], para_vecs: list[dict]) -> list[dict]:
    """
    Compute per-paragraph prosody parameters for both standard and V-A modes.
    Returns list of dicts with keys: preview, emotion, conf, std_rate, std_pitch, va_rate, va_pitch, valence, arousal.
    """
    import fusion

    _LEG_RATE  = {"scared": 18, "fearful": 18, "sad": 15, "dramatic": 12,
                  "joyful":  8, "surprised": 5, "angry": 6}
    _NEG_RATE  = {"scared", "fearful", "sad", "dramatic"}

    rows = []
    for para, pvec in zip(paragraphs, para_vecs):
        dom    = max(pvec, key=pvec.get)
        conf   = pvec[dom]
        v, a   = fusion.derive_va(pvec)
        leg_r  = int(conf * _LEG_RATE.get(dom, 0)) * (-1 if dom in _NEG_RATE else 1)
        leg_p_map = {"scared": 30, "fearful": 30, "sad": 22, "dramatic": 18,
                     "joyful": 20, "surprised": 28, "angry": 12}
        leg_p  = int(conf * leg_p_map.get(dom, 0)) * (-1 if dom in _NEG_RATE else 1)
        va_r   = max(-48, min(22, int(v * abs(a) * conf * 45)))
        va_p   = max(-60, min(60, int(v * conf * 100)))
        rows.append({
            "preview":   para[:80].replace("\n", " ") + ("…" if len(para) > 80 else ""),
            "emotion":   dom,
            "conf":      conf,
            "std_rate":  leg_r,
            "std_pitch": leg_p,
            "va_rate":   va_r,
            "va_pitch":  va_p,
            "valence":   round(v, 3),
            "arousal":   round(a, 3),
        })
    return rows


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎙️ HumanizingSpeech")
st.caption(
    "Upload a children's story, detect its emotion paragraph-by-paragraph, "
    "and generate expressive audio using a BERT prosody model trained on Storynory."
)

uploaded = st.file_uploader("Upload a story (.txt)", type=["txt"])

if not uploaded:
    st.info("Upload a .txt story file to begin.")
    st.stop()

# Read text — reset state when a new file is uploaded
file_key = uploaded.name + str(uploaded.size)
if st.session_state.get("_file_key") != file_key:
    for k in ("para_vecs", "wav_bytes"):
        st.session_state.pop(k, None)
    st.session_state["_file_key"] = file_key

text = uploaded.read().decode("utf-8")
paragraphs = _split_paragraphs(text)

with st.expander("Story text", expanded=False):
    st.text_area("", text, height=260, label_visibility="collapsed")

st.divider()

# ── Fiction gate ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Fiction Gate")
    fc = _get_fiction()
    content_type, confidence, explanation = fc.explain(text)

    if content_type == "fiction":
        st.success(f"**Fiction** — {confidence:.0%} confident")
        fiction_emoji = "📖"
    elif confidence >= 0.70:
        st.error(f"**Non-fiction** — {confidence:.0%} confident. Audio skipped.")
        # Show why before we bail
        st.markdown("**Why this verdict:**")
        st.markdown(explanation.get("summary", ""))
        for r in explanation.get("reasons", [])[:4]:
            st.markdown(
                f"- **{r['feature']}** = {r['value']}  →  pushes toward "
                f"*{r['direction']}* ({r['contribution']:+.2f}). {r['message']}  \n"
                f"  <small><i>{r['cite']}</i></small>",
                unsafe_allow_html=True,
            )
        st.stop()
    else:
        st.warning(f"Borderline ({confidence:.0%}) — treating as fiction.")
        content_type = "fiction"

    # Confidence gauge
    st.progress(int(confidence * 100), text=f"{confidence:.0%} confidence")

with col_right:
    st.subheader("What this means")

    method = explanation.get("method", "unknown")

    # Method badge + one-line summary
    method_label = {
        "fast-path":  "Fast-path (regex signals)",
        "pos-ratios": "POS-ratio logistic regression",
        "fallback":   "Keyword fallback",
    }.get(method, method)
    st.caption(f"Decision method: **{method_label}**")
    st.markdown(explanation.get("summary", ""))

    # Detailed breakdown
    if method == "fast-path":
        sig = explanation["signals"]
        st.markdown("**Evidence found in the text:**")
        st.markdown(
            f"- {sig['dialogue_blocks']} dialogue block(s) — quoted speech\n"
            f"- {sig['narrative_verbs']} past-tense narrative verb(s) "
            f"(said, whispered, ran, nodded, …)\n"
            f"- {sig['third_person_verbs']} third-person past-tense clause(s) "
            f"(she ran, he said, they went)\n"
            f"- {sig['story_phrases']} story-structure phrase(s) "
            f"(once upon a time, suddenly, …)"
        )
        st.caption(
            "These regex-based signals are unmistakable fiction markers — "
            "non-fiction almost never combines them at these densities, so "
            "the classifier skips the POS-ratio model and returns immediately."
        )

    elif method == "pos-ratios":
        st.markdown(
            f"**Fiction-pushing evidence** totalled `{explanation['total_fiction_push']:+.2f}`, "
            f"**non-fiction-pushing evidence** totalled `{explanation['total_nonfiction_push']:+.2f}`. "
            "Applying the sigmoid function to the sum gives the confidence above."
        )
        st.markdown("**Top features driving this decision:**")
        for r in explanation["reasons"]:
            arrow = "📖" if r["direction"] == "fiction" else "📄"
            st.markdown(
                f"{arrow} **{r['feature']}** = `{r['value']}`  →  "
                f"pushes toward *{r['direction']}* by `{r['contribution']:+.2f}`  \n"
                f"&nbsp;&nbsp;&nbsp; _{r['message']}_  \n"
                f"&nbsp;&nbsp;&nbsp; <small style='color:#6B7280'>"
                f"Source: {r['cite']}</small>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Method: logistic-regression scorer based on Qureshi et al. (2019) "
            "\"A Simple Approach to Classify Fictional and Non-Fictional Genres\" "
            "and Kazmi et al. (2022) \"Linguistically Motivated Features for "
            "Classifying Shorter Text into Fiction and Non-Fiction Genre\". "
            "POS tags come from NLTK's Penn Treebank tagger."
        )

st.divider()

# ── Per-paragraph emotions ────────────────────────────────────────────────────
st.subheader(f"Paragraph Emotions  —  {len(paragraphs)} paragraphs detected")

if "para_vecs" not in st.session_state:
    progress_bar = st.progress(0, text="Analysing paragraphs…")
    vecs = []
    te = _get_text_emotion()
    for i, para in enumerate(paragraphs):
        vecs.append(te.detect(para))
        progress_bar.progress((i + 1) / len(paragraphs), text=f"Paragraph {i+1} / {len(paragraphs)}")
    progress_bar.empty()
    st.session_state["para_vecs"] = vecs

para_vecs = st.session_state["para_vecs"]

# Summary emotion bar chart at the top
dominant_counts: dict[str, int] = {}
for vec in para_vecs:
    label = max(vec, key=vec.get)
    dominant_counts[label] = dominant_counts.get(label, 0) + 1

summary_cols = st.columns(len(dominant_counts))
for col, (label, count) in zip(summary_cols, sorted(dominant_counts.items(), key=lambda x: -x[1])):
    emoji = EMOTION_EMOJI.get(label, "❓")
    col.metric(f"{emoji} {label}", f"{count} para{'s' if count != 1 else ''}")

st.markdown("---")

# Paragraph-level table
header = st.columns([6, 2, 1.5])
header[0].markdown("**Paragraph**")
header[1].markdown("**Detected Emotion**")
header[2].markdown("**Confidence**")

for para, vec in zip(paragraphs, para_vecs):
    label = max(vec, key=vec.get)
    conf  = vec[label]
    preview = para[:140].replace("\n", " ") + ("…" if len(para) > 140 else "")

    c1, c2, c3 = st.columns([6, 2, 1.5])
    c1.markdown(f"<small style='color:#374151'>{preview}</small>", unsafe_allow_html=True)
    c2.markdown(_badge(label, conf), unsafe_allow_html=True)
    c3.markdown(f"<span style='font-weight:700'>{conf:.0%}</span>", unsafe_allow_html=True)

st.divider()

# ── Sidebar: Compare mode toggle ──────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    compare_mode = st.toggle(
        "Compare Mode",
        value=False,
        help="Generate both Standard (label-based) and V-A Enhanced audio side-by-side "
             "so you can hear the difference. Doubles generation time.",
    )

# ── Audio generation ──────────────────────────────────────────────────────────
st.subheader("Generate Expressive Audio")

if compare_mode:
    st.markdown(
        "**Compare Mode** — generates two versions of the audio:\n\n"
        "- **Standard**: discrete label-based prosody (fixed rate/pitch per emotion class)\n"
        "- **V-A Enhanced**: continuous Valence–Arousal formula (magnitude scales with "
        "emotional intensity and confidence; blended emotions interpolate smoothly)"
    )
    if st.button("🎵 Generate Both Audio Versions", type="primary", use_container_width=True):
        col_std, col_va = st.columns(2)
        with col_std:
            with st.spinner("Generating Standard audio…"):
                wav_std = _generate_audio(paragraphs, para_vecs, content_type, use_va=False)
            st.session_state["wav_std"] = wav_std
        with col_va:
            with st.spinner("Generating V-A Enhanced audio…"):
                wav_va = _generate_audio(paragraphs, para_vecs, content_type, use_va=True)
            st.session_state["wav_va"] = wav_va
        st.success("Both versions ready — listen and compare!")

    if "wav_std" in st.session_state and "wav_va" in st.session_state:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Standard (label-based)")
            st.audio(st.session_state["wav_std"], format="audio/wav")
            st.download_button(
                "⬇️ Download Standard",
                data=st.session_state["wav_std"],
                file_name=f"{uploaded.name.replace('.txt','')}_standard.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        with col_r:
            st.markdown("#### V-A Enhanced")
            st.audio(st.session_state["wav_va"], format="audio/wav")
            st.download_button(
                "⬇️ Download V-A Enhanced",
                data=st.session_state["wav_va"],
                file_name=f"{uploaded.name.replace('.txt','')}_va.wav",
                mime="audio/wav",
                use_container_width=True,
            )

        # ── Prosody diff table ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Prosody Diff — paragraph by paragraph")
        st.caption(
            "Shows the rate (%) and pitch (Hz) adjustments each system applies on top of "
            "the BERT prosody model's baseline per paragraph. "
            "V-A scales with emotional intensity (valence × arousal × confidence) — "
            "high-confidence peaks get amplified; low-confidence blends stay near neutral."
        )
        diff_rows = _prosody_diff_table(paragraphs, para_vecs)

        hdr = st.columns([4, 1.5, 1, 1.2, 1.2, 1.2, 1.2])
        for col, label in zip(hdr, ["Paragraph", "Emotion", "Conf",
                                     "Std rate", "Std pitch", "VA rate", "VA pitch"]):
            col.markdown(f"**{label}**")

        for row in diff_rows:
            c1, c2, c3, c4, c5, c6, c7 = st.columns([4, 1.5, 1, 1.2, 1.2, 1.2, 1.2])
            c1.markdown(f"<small>{row['preview']}</small>", unsafe_allow_html=True)
            c2.markdown(_badge(row["emotion"], row["conf"]), unsafe_allow_html=True)
            c3.markdown(f"**{row['conf']:.0%}**")
            c4.markdown(f"`{row['std_rate']:+d}%`")
            c5.markdown(f"`{row['std_pitch']:+d}Hz`")
            c6.markdown(f"`{row['va_rate']:+d}%`")
            c7.markdown(f"`{row['va_pitch']:+d}Hz`")

        # ── Circumplex scatter plot ────────────────────────────────────────
        st.markdown("---")
        st.subheader("Valence–Arousal Map")
        st.caption(
            "Each dot is one paragraph. Position shows where it falls in emotional "
            "space. V-A Enhanced delivery is directly proportional to this position — "
            "dots further from centre get more expressive treatment."
        )
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.axhline(0, color="#D1D5DB", linewidth=0.8)
            ax.axvline(0, color="#D1D5DB", linewidth=0.8)
            ax.set_xlabel("Valence  (negative ← → positive)", fontsize=10)
            ax.set_ylabel("Arousal  (calm ← → excited)", fontsize=10)
            ax.set_title("Paragraph positions in V-A space", fontsize=11)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

            _CMAP = {
                "joyful": "#F59E0B", "excited": "#10B981", "surprised": "#6366F1",
                "scared": "#1D4ED8", "dramatic": "#7C3AED", "sad": "#0369A1",
                "calm": "#059669", "neutral": "#9CA3AF",
            }
            for i, row in enumerate(diff_rows):
                color = _CMAP.get(row["emotion"], "#9CA3AF")
                ax.scatter(row["valence"], row["arousal"], color=color, s=90, alpha=0.85, zorder=3)
                ax.annotate(f"¶{i+1}", (row["valence"], row["arousal"]),
                            textcoords="offset points", xytext=(5, 3), fontsize=7, color="#374151")

            patches = [mpatches.Patch(color=c, label=e) for e, c in _CMAP.items()
                       if any(r["emotion"] == e for r in diff_rows)]
            ax.legend(handles=patches, fontsize=7, loc="lower right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as _plot_err:
            st.info(f"Scatter plot unavailable: {_plot_err}")

else:
    st.markdown(
        "Runs the trained **BERT prosody model** on each paragraph using its detected emotion, "
        "then synthesises with **Microsoft Edge Neural TTS** (en-GB-SoniaNeural) using the "
        "**V-A Enhanced** formula — rate and pitch scale continuously with emotional intensity. "
        "Enable **Compare Mode** in the sidebar to hear the difference vs the standard approach."
    )
    if st.button("🎵 Generate Audio", type="primary", use_container_width=True):
        with st.spinner("Generating expressive audio… (30–90 seconds depending on story length)"):
            wav_bytes = _generate_audio(paragraphs, para_vecs, content_type, use_va=True)
        st.session_state["wav_bytes"] = wav_bytes
        st.success("Audio ready!")

    if "wav_bytes" in st.session_state:
        st.audio(st.session_state["wav_bytes"], format="audio/wav")
        st.download_button(
            label="⬇️ Download WAV",
            data=st.session_state["wav_bytes"],
            file_name=f"{uploaded.name.replace('.txt', '')}_humanized.wav",
            mime="audio/wav",
            use_container_width=True,
        )
