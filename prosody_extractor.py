"""
Prosody Feature Extractor for Dataset Export

Wraps prosody.analyze_acoustic() to produce per-word prosody feature dicts
suitable for writing to the training jsonl. Each record captures the acoustic
ground truth (F0, energy, duration, pause) that the prosody prediction model
will be trained to replicate from text alone.
"""

from __future__ import annotations
import numpy as np


def extract(
    audio_path: str,
    words: list[dict],
    content_type: str = "fiction",
    verbose: bool = False,
) -> list[dict]:
    """
    Extract per-word prosody features from a speech recording.

    Args:
        audio_path:   Path to audio file (WAV or MP3).
        words:        List of {"word", "start", "end"} from asr.transcribe_with_timestamps.
        content_type: "fiction" or "non_fiction" — controls pitch scaling.
        verbose:      Pass through verbosity to analyze_acoustic.

    Returns:
        List of per-word dicts:
        {
            "word":           str,
            "start":          float,   # seconds
            "end":            float,
            "f0_hz":          float,   # mean F0 in Hz (0 if unvoiced)
            "f0_semitone_shift": float, # relative to speaker median, ±6 st
            "rms_energy":     float,
            "duration_ratio": float,   # actual duration / expected duration
            "pause_after_sec": float,  # silence gap to next word
            "rate_wpm":       int,     # TTS rate that would reproduce this timing
            "pitch_shift":    float,   # semitones (same as f0_semitone_shift)
            "volume":         float,   # normalised 0.0–1.0
        }
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=16_000, mono=True)

    # ------------------------------------------------------------------ #
    # 1. Raw per-word acoustic features                                   #
    # ------------------------------------------------------------------ #
    raw: list[dict] = []
    for w in words:
        start = int(w["start"] * sr)
        end   = int(w["end"]   * sr)
        chunk = y[start:end]

        if len(chunk) < 320:
            raw.append({"f0": 0.0, "energy": 0.0, "duration": w["end"] - w["start"]})
            continue

        n_fft = max(128, min(2048, 2 ** int(np.log2(max(len(chunk), 128)))))
        pitches, magnitudes = librosa.piptrack(
            y=chunk, sr=sr, fmin=75, fmax=400, n_fft=n_fft
        )
        voiced = (
            pitches[magnitudes > magnitudes.max() * 0.1]
            if magnitudes.max() > 0 else np.array([])
        )
        f0     = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        raw.append({"f0": f0, "energy": energy, "duration": w["end"] - w["start"]})

    # ------------------------------------------------------------------ #
    # 2. Speaker baselines                                                #
    # ------------------------------------------------------------------ #
    valid_f0      = [r["f0"] for r in raw if r["f0"] > 50]
    valid_energy  = [r["energy"] for r in raw]
    dur_per_char  = [
        r["duration"] / max(len(w["word"]), 1)
        for r, w in zip(raw, words)
        if r["duration"] > 0
    ]

    baseline_f0     = float(np.median(valid_f0))     if valid_f0     else 150.0
    baseline_energy = float(np.median(valid_energy)) if valid_energy else 0.01
    baseline_dpc    = float(np.median(dur_per_char)) if dur_per_char else 0.07

    is_fiction = content_type == "fiction"
    BASE_RATE  = 150
    BASE_VOL   = 0.80

    # ------------------------------------------------------------------ #
    # 3. Per-word derived features                                        #
    # ------------------------------------------------------------------ #
    import math

    result: list[dict] = []
    for i, (w, feat) in enumerate(zip(words, raw)):
        # Duration ratio
        expected       = max(len(w["word"]), 1) * baseline_dpc
        duration_ratio = feat["duration"] / max(expected, 0.03)

        # F0 → semitone shift relative to speaker median
        if is_fiction and feat["f0"] > 50 and baseline_f0 > 0:
            f0_shift = math.log2(feat["f0"] / baseline_f0) * 12
            f0_shift = float(max(-6.0, min(6.0, f0_shift)))
        else:
            f0_shift = 0.0

        # Volume from energy ratio
        energy_ratio = feat["energy"] / max(baseline_energy, 1e-6)
        volume       = float(max(0.4, min(1.0, BASE_VOL * min(energy_ratio * 1.5, 2.0))))

        # Rate from duration ratio
        rate = int(max(70, min(220, BASE_RATE / max(duration_ratio, 0.4))))

        # Gap to next word
        if i < len(words) - 1:
            pause_after = float(max(0.0, words[i + 1]["start"] - w["end"]))
        else:
            pause_after = 0.35

        result.append({
            "word":               w["word"],
            "start":              w["start"],
            "end":                w["end"],
            "f0_hz":              feat["f0"],
            "f0_semitone_shift":  f0_shift,
            "rms_energy":         feat["energy"],
            "duration_ratio":     duration_ratio,
            "pause_after_sec":    pause_after,
            "rate_wpm":           rate,
            "pitch_shift":        f0_shift,
            "volume":             volume,
        })

    if verbose:
        print(f"[ProsodyExtractor] {len(result)} words  "
              f"baseline F0={baseline_f0:.0f}Hz  energy={baseline_energy:.4f}  "
              f"dur/char={baseline_dpc:.3f}s")

    return result
