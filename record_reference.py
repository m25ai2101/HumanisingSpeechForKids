"""
Record a reference voice clip for XTTS-v2 voice cloning.

Capture ~6-30 seconds of your own clean speech (quiet room, no music, steady
pace) and save it as a mono WAV. Feed the result to the pipeline with:

    python pipeline.py --text-file demo_story.txt --model-prosody \
        --clone-ref my_voice.wav --save output.wav

Usage
-----
    python record_reference.py                 # saves to my_voice.wav
    python record_reference.py --out me.wav    # custom path
    python record_reference.py --seconds 15    # fixed-duration capture

With no --seconds, press Enter to start and Enter again to stop.

Tip: read a couple of expressive sentences, as if narrating a story — XTTS
clones not just your timbre but some of your delivery style.
"""

import argparse
import threading

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

SAMPLE_RATE = 24_000  # XTTS-v2 works natively at 24 kHz


def record_fixed(seconds: float) -> np.ndarray:
    print(f"[REC] Recording {seconds:.0f}s... speak now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    return audio


def record_toggle() -> np.ndarray:
    input("[REC] Press Enter to START recording...")
    print("[REC] Recording... Press Enter to STOP.")
    chunks = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", callback=callback):
        input()

    print("[REC] Recording complete.")
    return np.concatenate(chunks, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Record a reference voice clip for cloning.")
    p.add_argument("--out", default="my_voice.wav", help="Output WAV path (default: my_voice.wav).")
    p.add_argument("--seconds", type=float, default=None,
                   help="Fixed recording duration. Omit for press-Enter start/stop.")
    args = p.parse_args()

    audio = record_fixed(args.seconds) if args.seconds else record_toggle()

    # Trim tiny DC offset and normalise to a safe peak.
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak * 0.97 * 32767).astype(np.int16)

    wav.write(args.out, SAMPLE_RATE, audio)
    dur = len(audio) / SAMPLE_RATE
    print(f"[REC] Saved {dur:.1f}s to {args.out}")
    if dur < 5:
        print("[REC] Warning: under 5s. Aim for 6-30s of clean speech for best cloning.")


if __name__ == "__main__":
    main()
