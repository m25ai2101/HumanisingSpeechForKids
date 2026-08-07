"""
Acoustic Emotion Detection

Primary path  — Wav2Vec2 fine-tuned on IEMOCAP (superb/wav2vec2-base-superb-er).
               Downloads ~380 MB on first run, then cached locally.
               Directly aligns with the reference paper's use of Wav2Vec 2.0 for
               speech representation learning on the IEMOCAP dataset.

Fallback path — eGeMAPSv02-aligned heuristics via librosa (no download needed).
               Used automatically if the model fails to load.

Both paths extract features DIRECTLY from raw audio — no transcription involved.
All outputs are mapped to the canonical 8-class kids-story emotion schema defined
in emotion_schema.py before being returned.
"""

import numpy as np
from emotion_schema import IEMOCAP_TO_CANONICAL, HEURISTIC_TO_CANONICAL

# ---------------------------------------------------------------------------
# Wav2Vec2 model (primary)
# ---------------------------------------------------------------------------

_MODEL_ID   = "superb/wav2vec2-base-superb-er"
_emotion_pipe = None


def _load_model():
    global _emotion_pipe
    if _emotion_pipe is None:
        from transformers import pipeline as hf_pipeline
        print(f"[Emotion] Loading Wav2Vec2 model '{_MODEL_ID}' "
              "(downloads ~380 MB on first run)...")
        _emotion_pipe = hf_pipeline(
            "audio-classification",
            model=_MODEL_ID,
            device=-1,          # CPU
        )
        print("[Emotion] Wav2Vec2 model ready.")
    return _emotion_pipe


def _detect_with_model(audio_path: str) -> tuple[str, float]:
    """Run Wav2Vec2 emotion classifier directly on raw audio."""
    import librosa
    pipe = _load_model()

    # Cap at 60 s — sufficient to characterise story emotion, avoids loading
    # entire 15-20 min audio files which makes CPU inference extremely slow.
    y, sr = librosa.load(audio_path, sr=16_000, mono=True, duration=60.0)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) > sr * 0.2:
        y = y_trimmed

    preds = pipe({"array": y, "sampling_rate": sr}, top_k=1)
    raw_label = preds[0]["label"]
    score     = float(preds[0]["score"])
    label     = IEMOCAP_TO_CANONICAL.get(raw_label, raw_label)

    print(f"[Emotion] → {label} ({score:.0%})  [Wav2Vec2]")
    return label, score


# ---------------------------------------------------------------------------
# eGeMAPSv02 heuristic fallback
# ---------------------------------------------------------------------------

def _extract_features(audio_path: str) -> dict:
    """
    Extract eGeMAPSv02-aligned acoustic features directly from a WAV file.
    Used by the heuristic fallback and optionally for prosody analysis.
    """
    import librosa
    import scipy.signal

    y, sr = librosa.load(audio_path, sr=16_000, mono=True, duration=60.0)

    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) > sr * 0.2:
        y = y_trimmed

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
    voiced      = pitches[magnitudes > magnitudes.max() * 0.15]
    mean_pitch  = float(np.mean(voiced))           if len(voiced) > 0 else 0.0
    pitch_std   = float(np.std(voiced))            if len(voiced) > 0 else 0.0
    pitch_range = float(np.ptp(voiced))            if len(voiced) > 0 else 0.0

    rms         = librosa.feature.rms(y=y)[0]
    mean_energy = float(np.mean(rms))
    energy_cv   = float(np.std(rms) / mean_energy) if mean_energy > 0 else 0.0

    threshold      = mean_energy * 1.6
    peaks, _       = scipy.signal.find_peaks(rms, height=threshold, distance=4)
    duration_sec   = len(y) / sr
    stress_per_sec = len(peaks) / max(duration_sec, 1.0)

    zcr      = librosa.feature.zero_crossing_rate(y)
    mean_zcr = float(np.mean(zcr))

    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)
    mfcc1_mean = float(np.mean(mfccs[0]))
    mfcc1_std  = float(np.std(mfccs[0]))

    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    y_harm, _    = librosa.effects.hpss(y)
    harm_energy  = float(np.mean(y_harm ** 2))
    total_energy = float(np.mean(y ** 2))
    hnr = harm_energy / total_energy if total_energy > 0 else 0.0

    return {
        "mean_pitch": mean_pitch, "pitch_std": pitch_std,
        "pitch_range": pitch_range, "mean_energy": mean_energy,
        "energy_cv": energy_cv, "stress_per_sec": stress_per_sec,
        "mean_zcr": mean_zcr, "mfcc1_mean": mfcc1_mean,
        "mfcc1_std": mfcc1_std, "spec_centroid": spec_centroid, "hnr": hnr,
    }


def _detect_heuristic(audio_path: str) -> tuple[str, float]:
    """eGeMAPSv02 heuristic emotion detection — fallback when model unavailable."""
    f = _extract_features(audio_path)

    HIGH_PITCH     = 200.0
    HIGH_STD       = 60.0
    WIDE_RANGE     = 250.0
    HIGH_ENERGY    = 0.030
    HIGH_ENERGY_CV = 1.10
    HIGH_STRESS    = 2.0
    HIGH_ZCR       = 0.12
    HIGH_CENTROID  = 2500.0
    LOW_HNR        = 0.20
    HIGH_HNR       = 0.50

    high_pitch   = f["mean_pitch"]    > HIGH_PITCH
    high_std     = f["pitch_std"]     > HIGH_STD
    wide_range   = f["pitch_range"]   > WIDE_RANGE
    high_energy  = f["mean_energy"]   > HIGH_ENERGY
    spiky_energy = f["energy_cv"]     > HIGH_ENERGY_CV
    emphatic     = f["stress_per_sec"] > HIGH_STRESS
    fast_rate    = f["mean_zcr"]      > HIGH_ZCR
    bright_voice = f["spec_centroid"] > HIGH_CENTROID
    breathy      = f["hnr"]           < LOW_HNR
    modal_voice  = f["hnr"]           > HIGH_HNR

    print(
        f"[Emotion] F0={f['mean_pitch']:.0f}Hz  F0_std={f['pitch_std']:.0f}  "
        f"F0_range={f['pitch_range']:.0f}  energy={f['mean_energy']:.4f}  "
        f"energy_cv={f['energy_cv']:.2f}  stress/s={f['stress_per_sec']:.1f}  "
        f"HNR={f['hnr']:.2f}  centroid={f['spec_centroid']:.0f}Hz"
    )

    if wide_range and emphatic and spiky_energy:
        raw_label, conf = "dramatic", 0.80
    elif high_energy and high_std and high_pitch and fast_rate and (bright_voice or modal_voice):
        raw_label, conf = "angry",    0.75
    elif high_energy and high_std and high_pitch and bright_voice and not fast_rate:
        raw_label, conf = "surprised", 0.72
    elif high_energy and high_pitch and modal_voice and not high_std:
        raw_label, conf = "happy",    0.78
    elif not high_energy and not high_pitch and breathy:
        raw_label, conf = "sad",      0.76
    elif not high_energy and not high_pitch and not high_std:
        raw_label, conf = "sad",      0.72
    elif high_std and not high_energy and breathy:
        raw_label, conf = "fearful",  0.70
    elif high_std and not high_energy:
        raw_label, conf = "fearful",  0.65
    elif high_energy and not high_pitch and fast_rate:
        raw_label, conf = "angry",    0.68
    elif not high_std and not high_energy and not fast_rate and modal_voice:
        raw_label, conf = "calm",     0.76
    else:
        raw_label, conf = "neutral",  0.60

    label = HEURISTIC_TO_CANONICAL.get(raw_label, raw_label)
    print(f"[Emotion] → {label} ({conf:.0%})  [heuristic]")
    return label, conf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(audio_path: str) -> tuple[str, float]:
    """
    Detect emotion from raw audio. Returns (canonical_label, confidence).

    Tries Wav2Vec2 (superb/wav2vec2-base-superb-er, trained on IEMOCAP) first.
    Falls back to eGeMAPSv02 heuristics if the model is unavailable.
    Both paths operate directly on the audio signal — no transcription.
    Labels are mapped to the canonical 8-class kids-story schema.
    """
    try:
        return _detect_with_model(audio_path)
    except Exception as e:
        print(f"[Emotion] Model unavailable ({e}), using heuristics.")
        return _detect_heuristic(audio_path)


def detect_vector(audio_path: str) -> dict[str, float]:
    """
    Detect emotion from raw audio. Returns a probability vector over all
    canonical emotion labels (for use by fusion.py).

    The detected label receives weight equal to the model's confidence; the
    remaining probability mass is spread uniformly across the other labels.
    """
    label, conf = detect(audio_path)
    from emotion_schema import EMOTIONS, zero_vector
    vec = zero_vector()
    n = len(EMOTIONS)
    remainder = (1.0 - conf) / (n - 1)
    for e in EMOTIONS:
        vec[e] = conf if e == label else remainder
    return vec
