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


def _get_coqui():
    """
    Lazily load the XTTS-v2 model and move it to GPU when one is available.

    Returns the cached CoquiTTS instance. The first call downloads the model
    (~1.9 GB) and, on a CUDA machine (e.g. Colab GPU runtime), places it on the
    GPU so voice cloning runs in near real-time instead of minutes per segment.
    """
    global _coqui_tts
    if _coqui_tts is None:
        # Accept the Coqui license agreement non-interactively
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        import torch
        from TTS.api import TTS as CoquiTTS

        print("[TTS] Loading XTTS-v2 model (first run downloads ~1.9 GB)...")
        _coqui_tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _coqui_tts.to(device)
        except Exception:
            pass  # older API: model is already on the default device
        print(f"[TTS] XTTS-v2 model loaded on {device}.")
    return _coqui_tts


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

    _get_coqui()

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
# Neural prosody renderer — Microsoft Edge TTS (online, free, natural voice)
# ---------------------------------------------------------------------------

def speak_prosody_edge(
    segments,
    save_path: str | None = None,
    voice: str = "en-GB-SoniaNeural",
    target_sr: int = 22_050,
    play: bool = True,
    use_va: bool = True,
) -> str:
    """
    Render prosody segments using Microsoft Edge Neural TTS with emotion-driven
    delivery: fear → lower/darker tone + slower pace; dramatic → slow + low;
    joy → bright + quicker; dialogue lines get a pre-pause for anticipation.

    Requires an internet connection (Microsoft Azure neural voices via edge-tts).

    Args:
        segments:   Iterable of AnnotatedSegment (from prosody_model.predict).
        save_path:  Optional WAV path to save the result.
        voice:      Edge-TTS voice (default: en-GB-SoniaNeural — British
                    storyteller style matching Storynory narration).
        target_sr:  Output sample rate.
    """
    import asyncio
    import re
    import numpy as np
    import sounddevice as sd
    import scipy.io.wavfile as wav_io
    import librosa
    import edge_tts

    segments = list(segments)

    # Edge-TTS default WPM for this voice family
    EDGE_BASELINE_WPM = 160

    # How many Hz per semitone of prosody-model pitch prediction.
    # Higher = more expressive pitch variation.
    HZ_PER_SEMITONE = 22

    # Dialogue detection: check if stripped text starts with any quote character.
    # Uses ord() comparisons to avoid any quote literals in source.
    _DIALOGUE_CODEPOINTS = {0x22, 0x27, 0x201C, 0x201D, 0x2018, 0x2019}

    def _is_dialogue(text: str) -> bool:
        t = text.lstrip()
        return bool(t) and ord(t[0]) in _DIALOGUE_CODEPOINTS

    def _emotion_adjustments_legacy(emotion: str, conf: float) -> tuple[int, int]:
        """Label-lookup table: fixed rate/pitch per discrete emotion class."""
        if emotion in ("scared", "fearful"):
            return (-int(conf * 18), -int(conf * 30))   # slow + dark/ominous
        if emotion == "dramatic":
            return (-int(conf * 12), -int(conf * 18))   # slow + slightly lower
        if emotion == "sad":
            return (-int(conf * 15), -int(conf * 22))   # slow + low
        if emotion == "joyful":
            return (int(conf * 8),  int(conf * 20))     # quicker + bright
        if emotion == "surprised":
            return (int(conf * 5),  int(conf * 28))     # bright + high
        if emotion == "angry":
            return (int(conf * 6),  int(conf * 12))     # slightly faster + raised
        return (0, 0)

    def _emotion_adjustments_va(valence: float, arousal: float, conf: float) -> tuple[int, int]:
        """
        Continuous V-A formula replacing the discrete label lookup.

        Valence drives pitch (negative = darker, positive = brighter).
        Valence direction × arousal magnitude drives rate (negative-valence
        emotions become slower, positive-valence emotions become faster,
        with arousal scaling how strong the effect is).
        Confidence scales the overall magnitude so low-confidence paragraphs
        stay closer to neutral delivery.

        Multipliers (100, 45) are large enough to produce clearly audible
        differences; the Edge-TTS clamps in the caller (±60 Hz, −48/+22%)
        act as the safety ceiling so nothing sounds unnatural.
        """
        pitch_delta = int(valence * conf * 100)                  # ≈ ±60 Hz at peak (clamped)
        rate_delta  = int(valence * abs(arousal) * conf * 45)    # ≈ ±25% at peak (clamped)
        return rate_delta, pitch_delta

    def _emotion_adjustments(emotion: str, conf: float, valence: float, arousal: float) -> tuple[int, int]:
        """Route to V-A formula or legacy lookup depending on use_va flag."""
        if use_va:
            return _emotion_adjustments_va(valence, arousal, conf)
        return _emotion_adjustments_legacy(emotion, conf)

    # Build per-segment TTS parameters
    synth_items: list[tuple] = []   # (idx, seg, rate_str | None, pitch_str | None)

    for idx, seg in enumerate(segments):
        if not any(ch.isalnum() for ch in seg.text):
            synth_items.append((idx, seg, None, None))
            continue

        emotion  = getattr(seg, "emotion", "neutral")
        conf     = getattr(seg, "emotion_confidence", 0.5)
        valence  = getattr(seg, "valence", 0.0)
        arousal  = getattr(seg, "arousal", 0.0)

        # Base rate from prosody model WPM → edge-tts percentage
        rate_pct = int((seg.rate / EDGE_BASELINE_WPM - 1.0) * 100)

        # Emotion overlay
        rate_d, pitch_d = _emotion_adjustments(emotion, conf, valence, arousal)
        rate_pct += rate_d

        # Dialogue: slow down for deliberate, character-voice delivery
        if _is_dialogue(seg.text):
            rate_pct -= 8

        rate_pct = max(-48, min(22, rate_pct))
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

        # Base pitch from prosody-model semitone prediction
        pitch_hz = int(float(seg.pitch) * HZ_PER_SEMITONE) + pitch_d
        pitch_hz = max(-60, min(60, pitch_hz))
        pitch_str = f"+{pitch_hz}Hz" if pitch_hz >= 0 else f"{pitch_hz}Hz"

        synth_items.append((idx, seg, rate_str, pitch_str))

    # Allocate a temp file for every text segment
    tmp_by_idx: dict[int, str] = {}
    for idx, seg, rate_str, _ in synth_items:
        if rate_str is not None:
            t = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            t.close()
            tmp_by_idx[idx] = t.name

    async def _synth_all() -> None:
        tasks = [
            edge_tts.Communicate(
                seg.text.strip(), voice, rate=rate_str, pitch=pitch_str
            ).save(tmp_by_idx[idx])
            for idx, seg, rate_str, pitch_str in synth_items
            if rate_str is not None
        ]
        await asyncio.gather(*tasks)

    n_text = sum(1 for _, _, r, _ in synth_items if r is not None)
    print(f"[TTS] Synthesising {n_text} segments with Edge Neural TTS ({voice})...")
    try:
        asyncio.run(_synth_all())
    except Exception as e:
        print(f"[TTS] Edge-TTS failed ({e}) — falling back to pyttsx3.")
        for p in tmp_by_idx.values():
            if os.path.isfile(p):
                os.remove(p)
        return speak_prosody(segments, save_path=save_path)

    # Assemble with emotion-aware pauses
    pieces: list[np.ndarray] = []
    prev_had_audio = False

    for idx, seg, rate_str, _ in synth_items:
        emotion     = getattr(seg, "emotion", "neutral")
        conf        = getattr(seg, "emotion_confidence", 0.5)
        valence     = getattr(seg, "valence", 0.0)
        arousal     = getattr(seg, "arousal", 0.0)
        is_dialogue = _is_dialogue(seg.text)

        # Emotional intensity: 0–1 scale used for V-A pause scaling.
        va_intensity = (abs(valence) + abs(arousal)) / 2.0 * conf

        # Pre-pause: insert extra silence before dialogue for anticipation,
        # and before tense/dramatic lines for dramatic effect.
        if prev_had_audio:
            pre_pause = 0.0
            if is_dialogue:
                pre_pause = 0.30
            elif use_va and valence < -0.3 and arousal > 0.3 and conf > 0.55:
                pre_pause = 0.10 + va_intensity * 0.20   # 0.10–0.30 s, intensity-scaled
            elif not use_va and emotion in ("scared", "fearful", "dramatic") and conf > 0.55:
                pre_pause = 0.20
            if pre_pause > 0:
                pieces.append(np.zeros(int(target_sr * pre_pause), dtype="float32"))

        tmp_path = tmp_by_idx.get(idx)
        if tmp_path:
            try:
                y, _ = librosa.load(tmp_path, sr=target_sr, mono=True)
                if y.size:
                    y, _ = librosa.effects.trim(y, top_db=28)
                if y.size:
                    pieces.append((y * float(seg.volume)).astype("float32"))
                    prev_had_audio = True
            except Exception as e:
                print(f"[TTS] Warning loading segment {idx}: {e}")
            finally:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)

        # Post-pause: scale model's pause predictions; emotional moments get more room.
        raw_pause = max(0.0, seg.pause_after)
        if use_va:
            # Scale pause by V-A intensity: highly emotional = longer breathing room
            pause_scale = 1.20 + va_intensity * 0.60   # 1.20 – 1.80×
            raw_pause = max(raw_pause * pause_scale, 0.10)
        else:
            if emotion in ("scared", "fearful", "dramatic") and conf > 0.5:
                raw_pause = max(raw_pause * 1.6, raw_pause + 0.15)
            else:
                raw_pause = max(raw_pause * 1.35, 0.10)
        raw_pause = min(raw_pause, 3.0)
        if raw_pause > 0:
            pieces.append(np.zeros(int(target_sr * raw_pause), dtype="float32"))

    if not pieces:
        print("[TTS] Nothing to speak.")
        return ""

    audio = np.concatenate(pieces).astype("float32")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.97
    pcm = (audio * 32767).astype("int16")

    if save_path:
        save_path = os.path.abspath(save_path)
        wav_io.write(save_path, target_sr, pcm)
        print(f"[TTS] Saved speech to: {save_path}")

    if play:
        print("[TTS] Playing neural speech...")
        sd.play(pcm, target_sr)
        sd.wait()
        print("[TTS] Playback complete.")
    return save_path or ""


# ---------------------------------------------------------------------------
# Voice-cloned prosody renderer — XTTS-v2 in YOUR voice + prosody model
# ---------------------------------------------------------------------------

def speak_prosody_cloned(
    segments,
    reference_wav: str,
    language: str = "en",
    save_path: str | None = None,
    target_sr: int = 24_000,
    play: bool = True,
    use_va: bool = True,
) -> str:
    """
    Render prosody segments in a CLONED voice using Coqui XTTS-v2.

    This is the voice-cloning counterpart to ``speak_prosody_edge``: it keeps the
    trained prosody model's per-segment emotion, pitch, volume and pause
    decisions, but synthesises each segment in the voice captured in
    ``reference_wav`` (e.g. a short clean recording of your own voice) instead of
    a fixed Microsoft neural voice.

    XTTS-v2 has no SSML, so prosody is applied differently than in the Edge
    renderer:
      • rate   → XTTS ``speed`` parameter (0.5–2.0), derived from the segment's
                 predicted WPM plus the emotion overlay.
      • pitch  → librosa semitone shift on the generated segment audio.
      • volume → amplitude gain on the segment audio.
      • pause  → real silence inserted between segments (V-A intensity-scaled).
    XTTS also naturally carries the emotional tone of the reference clip, so the
    result sounds like you, reading expressively.

    Runs on GPU automatically when CUDA is available (strongly recommended —
    use a Colab GPU runtime). On CPU it works but is slow (seconds per segment).

    Args:
        segments:      Iterable of AnnotatedSegment (from prosody_model.predict).
        reference_wav: Path to a clean WAV of the voice to clone (~6–30 s).
        language:      BCP-47 language code (default 'en').
        save_path:     Optional WAV path to save the result.
        target_sr:     Output sample rate (XTTS natively produces 24 kHz).
        play:          Play through the speakers (disable on headless/Colab).
        use_va:        Use the continuous valence-arousal emotion model.

    Returns:
        The save path (if given) or an empty string.
    """
    import numpy as np
    import librosa
    import scipy.io.wavfile as wav_io

    if not reference_wav or not os.path.isfile(reference_wav):
        raise FileNotFoundError(
            f"Reference voice WAV not found: {reference_wav!r}. "
            "Provide a clean ~6–30 s recording of the voice to clone."
        )

    segments = list(segments)
    model = _get_coqui()

    # Dialogue detection (same codepoint set as the Edge renderer).
    _DIALOGUE_CODEPOINTS = {0x22, 0x27, 0x201C, 0x201D, 0x2018, 0x2019}

    def _is_dialogue(text: str) -> bool:
        t = text.lstrip()
        return bool(t) and ord(t[0]) in _DIALOGUE_CODEPOINTS

    def _va_overlay(valence: float, arousal: float, conf: float) -> tuple[float, float]:
        """Return (speed_delta, pitch_semitone_delta) from valence-arousal."""
        # Positive valence → slightly faster & brighter; negative → slower & darker.
        speed_delta = valence * abs(arousal) * conf * 0.25   # ≈ ±0.25 at peak
        pitch_delta = valence * conf * 2.0                   # ≈ ±2 semitones at peak
        return speed_delta, pitch_delta

    def _legacy_overlay(emotion: str, conf: float) -> tuple[float, float]:
        table = {
            "scared":   (-0.18, -2.4), "fearful": (-0.18, -2.4),
            "dramatic": (-0.12, -1.4), "sad":     (-0.15, -1.8),
            "joyful":   (0.10, 1.6),   "surprised": (0.06, 2.2),
            "angry":    (0.06, 1.0),
        }
        s, p = table.get(emotion, (0.0, 0.0))
        return s * conf, p * conf

    BASELINE_WPM = 150  # XTTS reads at roughly this pace at speed = 1.0

    pieces: list[np.ndarray] = []
    prev_had_audio = False
    n_text = sum(1 for s in segments if any(ch.isalnum() for ch in s.text))
    print(f"[TTS] Synthesising {n_text} segments with XTTS-v2 (cloned voice)...")

    done = 0
    for seg in segments:
        has_text = any(ch.isalnum() for ch in seg.text)

        emotion  = getattr(seg, "emotion", "neutral")
        conf     = getattr(seg, "emotion_confidence", 0.5)
        valence  = getattr(seg, "valence", 0.0)
        arousal  = getattr(seg, "arousal", 0.0)
        va_intensity = (abs(valence) + abs(arousal)) / 2.0 * conf
        is_dialogue  = _is_dialogue(seg.text)

        # Pre-pause before dialogue / tense lines (mirrors the Edge renderer).
        if prev_had_audio:
            pre_pause = 0.0
            if is_dialogue:
                pre_pause = 0.30
            elif use_va and valence < -0.3 and arousal > 0.3 and conf > 0.55:
                pre_pause = 0.10 + va_intensity * 0.20
            elif not use_va and emotion in ("scared", "fearful", "dramatic") and conf > 0.55:
                pre_pause = 0.20
            if pre_pause > 0:
                pieces.append(np.zeros(int(target_sr * pre_pause), dtype="float32"))

        if has_text:
            # Rate → XTTS speed multiplier
            speed = seg.rate / BASELINE_WPM
            s_delta, p_delta = (
                _va_overlay(valence, arousal, conf) if use_va
                else _legacy_overlay(emotion, conf)
            )
            speed += s_delta
            if is_dialogue:
                speed -= 0.05  # slower, deliberate character delivery
            speed = max(0.6, min(1.6, speed))

            try:
                wav = model.tts(
                    text=seg.text.strip(),
                    speaker_wav=reference_wav,
                    language=language,
                    speed=speed,
                )
                y = np.asarray(wav, dtype="float32")
                # XTTS outputs 24 kHz; resample if the caller wants something else.
                if target_sr != 24_000 and y.size:
                    y = librosa.resample(y, orig_sr=24_000, target_sr=target_sr)
                if y.size:
                    y, _ = librosa.effects.trim(y, top_db=28)

                # Pitch: model semitones + emotion overlay (semitones).
                pitch_semitones = float(seg.pitch) + p_delta
                if y.size and abs(pitch_semitones) >= 0.1:
                    y = librosa.effects.pitch_shift(
                        y, sr=target_sr, n_steps=pitch_semitones
                    )
                if y.size:
                    pieces.append((y * float(seg.volume)).astype("float32"))
                    prev_had_audio = True
            except Exception as e:
                print(f"[TTS] Warning synthesising segment {done}: {e}")

            done += 1
            if done % 5 == 0 or done == n_text:
                print(f"[TTS]   {done}/{n_text} segments done")

        # Post-pause (V-A intensity-scaled, same policy as the Edge renderer).
        raw_pause = max(0.0, seg.pause_after)
        if use_va:
            raw_pause = max(raw_pause * (1.20 + va_intensity * 0.60), 0.10)
        elif emotion in ("scared", "fearful", "dramatic") and conf > 0.5:
            raw_pause = max(raw_pause * 1.6, raw_pause + 0.15)
        else:
            raw_pause = max(raw_pause * 1.35, 0.10)
        raw_pause = min(raw_pause, 3.0)
        if raw_pause > 0:
            pieces.append(np.zeros(int(target_sr * raw_pause), dtype="float32"))

    if not pieces:
        print("[TTS] Nothing to speak.")
        return ""

    audio = np.concatenate(pieces).astype("float32")
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.97
    pcm = (audio * 32767).astype("int16")

    if save_path:
        save_path = os.path.abspath(save_path)
        wav_io.write(save_path, target_sr, pcm)
        print(f"[TTS] Saved cloned-voice speech to: {save_path}")

    if play:
        try:
            import sounddevice as sd
            print("[TTS] Playing cloned-voice speech...")
            sd.play(pcm, target_sr)
            sd.wait()
            print("[TTS] Playback complete.")
        except Exception as e:
            print(f"[TTS] Skipping playback (no audio device: {e}).")
    return save_path or ""


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
