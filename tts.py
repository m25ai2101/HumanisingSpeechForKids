"""
TTS module with two backends:

  1. pyttsx3        — fast, offline, robotic-sounding (speak / save)
  2. XTTS-v2        — Coqui neural TTS with voice cloning (speak_cloned)
                      Clones the speaker's voice from a reference WAV and
                      generates natural, expressive speech.
                      First run downloads the model (~1.9 GB).
"""

import glob
import os
import shutil
import tempfile

import pyttsx3


def _register_ffmpeg_dll_dir() -> str | None:
    """
    Make FFmpeg's *shared* DLLs discoverable to torchcodec.

    torchcodec (pulled in by ``coqui-tts[codec]`` and used by torchaudio.load)
    links against FFmpeg's shared libraries. Since Python 3.8 on Windows, the
    OS does NOT search PATH when resolving a DLL's *dependencies*, so having
    ffmpeg on PATH is not enough — the directory must be registered explicitly
    via os.add_dll_directory(). We look for it on PATH first, then fall back to
    the default winget (Gyan.FFmpeg.Shared) install location.
    """
    candidates = []
    ff = shutil.which("ffmpeg")
    if ff:
        candidates.append(os.path.dirname(ff))
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        candidates += glob.glob(os.path.join(
            local, "Microsoft", "WinGet", "Packages",
            "Gyan.FFmpeg.Shared*", "ffmpeg-*-shared", "bin"))
    for d in candidates:
        if d and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                continue
            return d
    return None


# Register FFmpeg DLLs at import time, before torchaudio/torchcodec load.
_register_ffmpeg_dll_dir()

# Lazy-loaded Coqui XTTS-v2 engine (avoids loading the large model at import time)
_coqui_tts = None


def _get_engine(rate: int = 175, volume: float = 1.0, voice_index: int = 0) -> pyttsx3.Engine:
    """Create and configure a pyttsx3 engine instance."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)        # words per minute
    engine.setProperty("volume", volume)    # 0.0 – 1.0

    voices = engine.getProperty("voices")
    if voices and voice_index < len(voices):
        engine.setProperty("voice", voices[voice_index].id)

    return engine


def speak(text: str, rate: int = 175, volume: float = 1.0, voice_index: int = 0) -> None:
    """
    Speak text aloud through the default audio output.

    Args:
        text:        The text to synthesize and play.
        rate:        Speech rate in words per minute (default 175).
        volume:      Volume from 0.0 (silent) to 1.0 (full, default).
        voice_index: Index into the list of installed voices (0 = first/default).
    """
    print(f"[TTS] Speaking: {text!r}")
    engine = _get_engine(rate=rate, volume=volume, voice_index=voice_index)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def save(text: str, output_path: str, rate: int = 175, volume: float = 1.0, voice_index: int = 0) -> str:
    """
    Synthesize text and save to a WAV file (no audio playback).

    Args:
        text:        The text to synthesize.
        output_path: Destination file path (should end in .wav).
        rate:        Speech rate in words per minute.
        volume:      Volume from 0.0 to 1.0.
        voice_index: Index into the list of installed voices.

    Returns:
        Absolute path to the saved file.
    """
    output_path = os.path.abspath(output_path)
    print(f"[TTS] Saving speech to: {output_path}")
    engine = _get_engine(rate=rate, volume=volume, voice_index=voice_index)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()
    print(f"[TTS] File saved.")
    return output_path


def list_voices() -> None:
    """Print all installed voices with their index, id, and name."""
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    print(f"[TTS] {len(voices)} installed voice(s):")
    for i, v in enumerate(voices):
        print(f"  [{i}] {v.name}  —  {v.id}")
    engine.stop()


# ---------------------------------------------------------------------------
# Voice-cloning TTS via Coqui XTTS-v2
# ---------------------------------------------------------------------------

def speak_cloned(
    text: str,
    reference_wav: str,
    language: str = "en",
    save_path: str | None = None,
) -> str:
    """
    Synthesize `text` in the voice captured in `reference_wav` using XTTS-v2.

    The model clones the speaker's voice — including their speaking style and
    emotional tone — from just a few seconds of reference audio.

    Args:
        text:          The text to synthesize.
        reference_wav: Path to the WAV that contains the voice to clone.
                       Should be at least 3 seconds of clean speech.
        language:      BCP-47 language code (default 'en').
        save_path:     Optional path to save the output WAV. If omitted a
                       temporary file is used and deleted after playback.

    Returns:
        Path to the generated WAV file (or the temp file path if no save_path).
    """
    import sounddevice as sd
    import scipy.io.wavfile as wav_io

    global _coqui_tts
    if _coqui_tts is None:
        # Accept the Coqui license agreement non-interactively
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS as CoquiTTS
        print("[TTS] Loading XTTS-v2 model (first run downloads ~1.9 GB)...")
        _coqui_tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("[TTS] XTTS-v2 model loaded.")

    delete_after = save_path is None
    out_path = save_path or tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    print(f"[TTS] Generating voice-cloned speech (this may take a moment on CPU)...")
    _coqui_tts.tts_to_file(
        text=text,
        speaker_wav=reference_wav,
        language=language,
        file_path=out_path,
    )

    # Play back the generated audio
    rate, data = wav_io.read(out_path)
    sd.play(data, rate)
    sd.wait()
    print("[TTS] Playback complete.")

    if delete_after and os.path.isfile(out_path):
        os.remove(out_path)

    return out_path


# ---------------------------------------------------------------------------
# Rule-based prosody renderer
# ---------------------------------------------------------------------------

def speak_prosody(
    segments,
    save_path: str | None = None,
    voice_index: int = 1,
    target_sr: int = 22_050,
) -> str:
    """
    Render a list of prosody `Segment`s (from prosody.analyze) to expressive
    speech and play it (and optionally save it).

    pyttsx3 only exposes *rate* and *volume*, and nothing for pitch or pauses.
    So we synthesise each segment independently, then shape it ourselves:

      • rate    → pyttsx3 engine rate per segment
      • pitch   → librosa.effects.pitch_shift (semitones) on the segment audio
      • volume  → amplitude gain applied to the segment audio
      • pause   → inserted as real silence between segments

    Segments are trimmed of leading/trailing silence so the inserted pauses are
    accurate, then concatenated into one waveform.

    Args:
        segments:    Iterable of objects with .text, .rate, .pitch, .volume,
                     .pause_after (duck-typed; prosody.Segment).
        save_path:   Optional WAV output path.
        voice_index: pyttsx3 voice index.
        target_sr:   Working sample rate for resampling/concatenation.

    Returns:
        The save path (if given) or an empty string.
    """
    import numpy as np
    import sounddevice as sd
    import scipy.io.wavfile as wav_io
    import librosa

    segments = list(segments)

    engine = pyttsx3.init()
    engine.setProperty("volume", 1.0)  # we apply volume ourselves, in post
    voices = engine.getProperty("voices")
    if voices and voice_index < len(voices):
        engine.setProperty("voice", voices[voice_index].id)

    # Phase 1: queue every speakable segment, then call runAndWait() exactly
    # ONCE. pyttsx3 blocks if runAndWait() is called repeatedly, so we batch.
    # setProperty('rate') is queued in order too, so each file gets its own rate.
    seg_files: dict[int, str] = {}
    for i, seg in enumerate(segments):
        if any(ch.isalnum() for ch in seg.text):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            engine.setProperty("rate", int(seg.rate))
            engine.save_to_file(seg.text.strip(), tmp.name)
            seg_files[i] = tmp.name
    engine.runAndWait()
    engine.stop()

    # Phase 2: load each segment's audio, apply pitch + volume, and assemble in
    # order with real silence for the pauses.
    pieces: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        path = seg_files.get(i)
        if path:
            try:
                y, _ = librosa.load(path, sr=target_sr, mono=True)
            except Exception:
                y = np.zeros(0, dtype="float32")
            finally:
                if os.path.isfile(path):
                    os.remove(path)
            if y.size:
                y, _ = librosa.effects.trim(y, top_db=30)
            if y.size and abs(seg.pitch) >= 0.1:
                y = librosa.effects.pitch_shift(y, sr=target_sr, n_steps=float(seg.pitch))
            if y.size:
                pieces.append((y * float(seg.volume)).astype("float32"))

        if seg.pause_after > 0:
            pieces.append(np.zeros(int(target_sr * seg.pause_after), dtype="float32"))

    if not pieces:
        print("[Prosody] Nothing to speak.")
        return ""

    audio = np.concatenate(pieces).astype("float32")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.97  # normalise, leave a little headroom
    pcm = (audio * 32767).astype("int16")

    if save_path:
        save_path = os.path.abspath(save_path)
        wav_io.write(save_path, target_sr, pcm)
        print(f"[Prosody] Saved speech to: {save_path}")

    print("[Prosody] Playing prosody-controlled speech...")
    sd.play(pcm, target_sr)
    sd.wait()
    print("[Prosody] Playback complete.")
    return save_path or ""
