"""Automated MOS (Mean Opinion Score) estimation for TTS output.

Uses UTMOS (UTokyo-SaruLab, VoiceMOS 2022 winner), a neural predictor trained
on thousands of human naturalness ratings. It outputs a 1-5 score per clip
WITHOUT needing a reference recording, so it works directly on pipeline output.

This is a *predicted* MOS: best used for relative comparison (version A vs B,
catching regressions) and as a confidence sanity-check — not as a substitute
for a real human listening study. Always average over many clips.

Usage:
    # Score a single file
    python mos_eval.py output.wav

    # Score every .wav in a folder and report mean +/- std
    python mos_eval.py path/to/wavs/

    # Score several files/folders at once
    python mos_eval.py a.wav b.wav outputs/

Model is downloaded once via torch.hub and cached (~/.cache/torch/hub).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import librosa
import torch

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
_UTMOS_SR = 16_000  # UTMOS expects 16 kHz mono

_predictor = None


def _load_predictor():
    """Load (and cache) the UTMOS predictor from torch.hub."""
    global _predictor
    if _predictor is None:
        print("[MOS] Loading UTMOS predictor (first run downloads ~100 MB)...")
        _predictor = torch.hub.load(
            "tarepan/SpeechMOS", "utmos22_strong", trust_repo=True
        )
        _predictor.eval()
    return _predictor


def score_file(path: str | Path) -> float:
    """Return the predicted MOS (1-5) for a single audio file."""
    predictor = _load_predictor()
    wave, _ = librosa.load(str(path), sr=_UTMOS_SR, mono=True)
    tensor = torch.from_numpy(wave).unsqueeze(0)  # (1, num_samples)
    with torch.no_grad():
        score = predictor(tensor, _UTMOS_SR)
    return float(score.item())


def print_mos(path: str | Path, label: str = "MOS") -> float | None:
    """Score a file and print it. Safe to call inline from the pipeline.

    Never raises — if UTMOS can't load or the file can't be read, it prints a
    warning and returns None so audio generation is never interrupted.
    """
    try:
        mos = score_file(path)
    except Exception as exc:  # noqa: BLE001 - scoring must never break synthesis
        print(f"[{label}] Could not estimate MOS: {exc}")
        return None
    print(f"[{label}] Estimated MOS (UTMOS): {mos:.2f} / 5")
    return mos


def _collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(
                sorted(f for f in path.rglob("*") if f.suffix.lower() in _AUDIO_EXTS)
            )
        elif path.is_file():
            files.append(path)
        else:
            print(f"[MOS] Warning: path not found, skipping: {p}", file=sys.stderr)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate MOS for TTS output using UTMOS (no reference needed)."
    )
    parser.add_argument("paths", nargs="+", help="Audio file(s) and/or folder(s).")
    args = parser.parse_args()

    files = _collect_files(args.paths)
    if not files:
        print("[MOS] No audio files found.", file=sys.stderr)
        return 1

    scores: list[float] = []
    for f in files:
        try:
            mos = score_file(f)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[MOS] Failed on {f}: {exc}", file=sys.stderr)
            continue
        scores.append(mos)
        print(f"{mos:5.2f}  {f}")

    if not scores:
        return 1

    print("-" * 40)
    print(f"Clips scored : {len(scores)}")
    print(f"Mean MOS     : {statistics.mean(scores):.2f} / 5")
    if len(scores) > 1:
        print(f"Std dev      : {statistics.pstdev(scores):.2f}")
        print(f"Min / Max    : {min(scores):.2f} / {max(scores):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
