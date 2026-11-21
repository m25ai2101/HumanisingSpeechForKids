"""
System-level (audio) evaluation: Basic TTS vs the full emotion pipeline.

Generates narration of the same text three ways and measures objective
expressiveness + predicted naturalness (UTMOS) for each:

  1. Basic TTS            — pyttsx3, raw text, no emotion (monotone baseline)
  2. Pipeline (label)     — emotion -> BERT prosody -> Edge TTS, discrete labels
  3. Pipeline (V-A)       — same, with continuous valence-arousal overlay

Expressiveness metrics (higher spread = more human-like, less robotic):
  F0 std / range   — pitch variation (monotone ~ 0)
  RMS std          — loudness dynamics
  silence ratio    — fraction of time that is pauses (phrasing)
  UTMOS            — predicted mean opinion score 1-5 (naturalness), if available

Usage:  python evaluate_audio.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

# Windows consoles default to cp1252 and crash on the → characters some modules
# print. Force UTF-8 so the script runs cleanly everywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import librosa

import text_emotion
import fusion
import prosody_model
import tts

# A few emotionally varied paragraphs from the demo story.
PARAS = [
    "Once upon a time, in a cosy burrow beneath an old oak tree, there lived a "
    "little fox named Ember. The world felt warm and safe, and everything was "
    "just as it should be.",
    "She leapt and tumbled and spun with pure, unbounded joy. She laughed her "
    "high little fox laugh and chased fireflies until her heart was so full she "
    "thought it might burst.",
    "Suddenly, a twig snapped in the darkness. A huge shadow rose up between the "
    "trees. Ember froze, her heart pounding. She was trembling, terrified, alone "
    "in the cold black night.",
]


def _metrics(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path, sr=22_050, mono=True)
    dur = len(y) / sr
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=2048)
    voiced = f0[~np.isnan(f0)]
    f0_std = float(np.std(voiced)) if voiced.size else 0.0
    f0_rng = float(np.percentile(voiced, 95) - np.percentile(voiced, 5)) if voiced.size else 0.0
    rms = librosa.feature.rms(y=y)[0]
    rms_std = float(np.std(rms))
    # silence ratio: frames below 15% of median RMS
    thr = np.median(rms) * 0.15
    silence_ratio = float(np.mean(rms < thr))
    return {"dur": dur, "f0_std": f0_std, "f0_range": f0_rng,
            "rms_std": rms_std, "silence_ratio": silence_ratio}


def _basic(text: str, path: str) -> None:
    tts.save(text, path, rate=175, volume=1.0, voice_index=0)


def _pipeline(text: str, path: str, use_va: bool) -> None:
    all_segs = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        vec = text_emotion.detect(para)
        fused = fusion.fuse(vec, speech_vec=None)
        all_segs.extend(prosody_model.predict(para, fused, content_type="fiction"))
    tts.speak_prosody_edge(all_segs, save_path=path, play=False, use_va=use_va)


def main() -> None:
    scratch = tempfile.mkdtemp(prefix="humspeech_eval_")
    text = "\n".join(PARAS)
    print(f"[AudioEval] Text: {len(text.split())} words, {len(PARAS)} paragraphs")
    print(f"[AudioEval] Temp dir: {scratch}\n")

    prosody_model.load("checkpoints/best.pt" if os.path.exists("checkpoints/best.pt") else None)

    jobs = {
        "Basic TTS (no emotion)": (os.path.join(scratch, "basic.wav"), lambda p: _basic(text, p)),
        "Pipeline (label-based)": (os.path.join(scratch, "label.wav"), lambda p: _pipeline(text, p, use_va=False)),
        "Pipeline (V-A)":         (os.path.join(scratch, "va.wav"),    lambda p: _pipeline(text, p, use_va=True)),
    }

    results = {}
    for name, (path, fn) in jobs.items():
        print(f"[AudioEval] Generating: {name} ...")
        fn(path)
        results[name] = (path, _metrics(path))

    # Optional UTMOS
    mos = {}
    try:
        import mos_eval
        for name, (path, _) in results.items():
            mos[name] = mos_eval.score_file(path)
    except Exception as e:
        print(f"[AudioEval] UTMOS unavailable ({str(e)[:80]}) — skipping MOS.")

    # ---- Table --------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"{'System':<26}{'F0 std':>9}{'F0 range':>10}{'RMS std':>10}"
          f"{'Silence%':>10}{'Dur(s)':>9}{'UTMOS':>8}")
    print("-" * 92)
    for name, (path, m) in results.items():
        mos_str = f"{mos[name]:.2f}" if name in mos else "  -"
        print(f"{name:<26}{m['f0_std']:>9.1f}{m['f0_range']:>10.1f}"
              f"{m['rms_std']:>10.4f}{m['silence_ratio']*100:>9.1f}%{m['dur']:>9.1f}{mos_str:>8}")
    print("=" * 92)

    base = results["Basic TTS (no emotion)"][1]
    va = results["Pipeline (V-A)"][1]
    if base["f0_std"] > 0:
        print(f"Pipeline vs Basic — pitch variation: {va['f0_std']/base['f0_std']:.1f}x,  "
              f"pitch range: {va['f0_range']/max(base['f0_range'],1e-9):.1f}x,  "
              f"loudness dynamics: {va['rms_std']/max(base['rms_std'],1e-9):.1f}x")
    print(f"\n[AudioEval] WAVs saved in: {scratch}")


if __name__ == "__main__":
    main()
