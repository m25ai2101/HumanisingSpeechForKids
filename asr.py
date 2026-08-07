"""
ASR module using OpenAI Whisper for local speech-to-text transcription.
No API key required. Runs fully offline after model download.

WAV files are loaded directly via scipy (no ffmpeg required).
Non-WAV formats (MP3, FLAC, etc.) require ffmpeg to be installed and on PATH.
"""
# python data_pipeline.py --hf-dataset openslr/librispeech_asr --hf-config clean --hf-split train.100 --max-samples 50 --output data/
# python train.py --data data/train.jsonl --val data/val.jsonl
import os

import numpy as np
import scipy.io.wavfile as wav_io
import scipy.signal
import whisper

WHISPER_SAMPLE_RATE = 16_000

_model = None
_model_size = None


def load_model(size: str = "base") -> whisper.Whisper:
    """Load the Whisper model (cached after first load)."""
    global _model, _model_size
    if _model is None or _model_size != size:
        print(f"[ASR] Loading Whisper model '{size}'...")
        _model = whisper.load_model(size)
        _model_size = size
        print(f"[ASR] Model '{size}' loaded.")
    return _model


def _load_wav(audio_path: str) -> np.ndarray:
    """
    Load a WAV file and return a float32 mono array at 16 kHz.
    Does not require ffmpeg.
    """
    rate, data = wav_io.read(audio_path)

    # Convert integer PCM to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # Stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample to 16 kHz if needed
    if rate != WHISPER_SAMPLE_RATE:
        num_samples = int(len(data) * WHISPER_SAMPLE_RATE / rate)
        data = scipy.signal.resample(data, num_samples)

    return data


def transcribe_with_timestamps(
    audio_path: str, model_size: str = "base", language: str = None
) -> tuple[str, list[dict]]:
    """
    Transcribe audio and return word-level timestamps from Whisper.

    Returns (text, words) where words is a list of:
        {"word": str, "start": float, "end": float}  — start/end in seconds.

    These timestamps let downstream code slice the original audio per word
    and extract acoustic features (pitch, energy, duration) directly —
    without any text-based analysis.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_model(model_size)
    print(f"[ASR] Transcribing with word timestamps: {audio_path}")

    options: dict = {"word_timestamps": True}
    if language:
        options["language"] = language

    audio_input = _load_wav(audio_path) if audio_path.lower().endswith(".wav") else audio_path
    result = model.transcribe(audio_input, **options)

    text = result["text"].strip()
    words: list[dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            word_text = w["word"].strip()
            if word_text:
                words.append({"word": word_text, "start": w["start"], "end": w["end"]})

    print(f"[ASR] Transcription: {text!r}  ({len(words)} word timestamps)")
    return text, words


def transcribe(audio_path: str, model_size: str = "base", language: str = None) -> str:
    """
    Transcribe an audio file to text using Whisper.

    WAV files are decoded with scipy (no ffmpeg needed).
    Other formats require ffmpeg installed and on PATH.

    Args:
        audio_path:  Path to the audio file.
        model_size:  Whisper model size — 'tiny', 'base', 'small', 'medium', 'large'.
        language:    ISO-639-1 language code to force (e.g. 'en'). None = auto-detect.

    Returns:
        Transcribed text string.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_model(model_size)
    print(f"[ASR] Transcribing: {audio_path}")

    options = {}
    if language:
        options["language"] = language

    # Use scipy for WAV to avoid the ffmpeg dependency
    if audio_path.lower().endswith(".wav"):
        audio_input = _load_wav(audio_path)
    else:
        # Non-WAV formats fall back to Whisper's ffmpeg-based loader
        audio_input = audio_path

    result = model.transcribe(audio_input, **options)
    text = result["text"].strip()
    print(f"[ASR] Transcription: {text!r}")
    return text
