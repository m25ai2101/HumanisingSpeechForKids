"""
ASR → TTS Pipeline
==================
Transcribes speech to text (Whisper) then speaks the result aloud (pyttsx3).

Usage
-----
# Text-only input with trained prosody model (primary new flow):
    python pipeline.py --text "Once upon a time..." --model-prosody

# Transcribe an audio file and speak the result:
    python pipeline.py --file path/to/audio.wav

# Record from microphone (press Enter to start, press Enter again to stop):
    python pipeline.py --mic

# Transcribe a file, speak the result, and also save the TTS output:
    python pipeline.py --file path/to/audio.wav --save output.wav

# Use a larger Whisper model for higher accuracy:
    python pipeline.py --file audio.wav --model small

# List available TTS voices:
    python pipeline.py --list-voices

Options
-------
--text TEXT       Raw story text to synthesise (no audio input needed).
--file PATH       Path to an audio file to transcribe.
--mic             Record from microphone (press Enter to start, Enter again to stop).
--save PATH       Save TTS output to this WAV file instead of (also) playing it.
--model SIZE      Whisper model: tiny | base | small | medium | large (default: base).
--language CODE   Force a language for Whisper, e.g. 'en' (default: auto-detect).
--rate WPM        TTS speech rate in words per minute (default: 175).
--voice INDEX     TTS voice index (default: 0). Use --list-voices to see options.
--list-voices     Print available TTS voices and exit.
--model-prosody   Use trained BERT prosody model (requires --checkpoint or default path).
--checkpoint PATH Path to trained ProsodyPredictor checkpoint (default: checkpoints/best.pt).
"""

import argparse
import os
import sys
import tempfile
import threading

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

import asr
import emotion
import fiction
import fusion
import humanize
import prosody
import prosody_model
import text_emotion
import tts


SAMPLE_RATE = 16_000  # Hz — Whisper expects 16 kHz


def record_microphone() -> str:
    """
    Record audio from the default microphone.
    Press Enter to start recording, press Enter again to stop.
    Returns the path to a temporary WAV file containing the recording.
    """
    input("[MIC] Press Enter to START recording...")
    print("[MIC] Recording... Press Enter to STOP.")

    chunks = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    stop_event = threading.Event()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input()  # block until user presses Enter

    print("[MIC] Recording complete.")

    audio = np.concatenate(chunks, axis=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name


def run_pipeline(
    audio_path: str | None,
    model_size: str,
    language: str,
    tts_rate: int,
    voice_index: int,
    save_path: str | None,
    use_clone: bool = False,
    use_prosody: bool = False,
    use_model_prosody: bool = False,
    checkpoint_path: str = "checkpoints/best.pt",
    input_text: str | None = None,
    cleanup_audio: bool = False,
) -> str:
    """Run ASR → Emotion → Humanize → TTS on `audio_path`, or text → TTS."""
    try:
        # ------------------------------------------------------------------
        # Model-prosody path (trained BERT predictor)
        # ------------------------------------------------------------------
        # Text-only: text → text_emotion → prosody_model → speak_prosody
        # Audio input: audio → ASR → text_emotion + speech_emotion → fusion
        #              → prosody_model → speak_prosody
        if use_model_prosody:
            prosody_model.load(
                checkpoint_path if os.path.exists(checkpoint_path) else None
            )

            if input_text:
                text = input_text.strip()
                print(f"\n[Pipeline] Input text:\n  {text}\n")
                text_vec  = text_emotion.detect(text)
                fused_vec = fusion.fuse(text_vec, speech_vec=None)
            else:
                text, word_timestamps = asr.transcribe_with_timestamps(
                    audio_path, model_size=model_size, language=language
                )
                if not text:
                    print("[Pipeline] No speech detected.")
                    return ""
                print(f"\n[Pipeline] Transcribed text:\n  {text}\n")
                text_vec   = text_emotion.detect(text)
                speech_vec = emotion.detect_vector(audio_path)
                fused_vec  = fusion.fuse(text_vec, speech_vec)

            # Fiction gate — non-fiction gets no audio output.
            content_type, confidence = fiction.detect(text)
            if content_type == "non_fiction":
                print(
                    f"[Pipeline] Non-fiction detected (confidence: {confidence:.0%}) "
                    "— skipping audio output."
                )
                return text

            print(f"[Pipeline] Fiction confirmed (confidence: {confidence:.0%}) — generating expressive audio.")
            segments = prosody_model.predict(text, fused_vec, content_type=content_type)
            tts.speak_prosody(segments, save_path=save_path, voice_index=voice_index)
            return text

        # ------------------------------------------------------------------
        # Acoustic prosody path — extract per-word features directly from audio.
        # Whisper word timestamps slice the audio per word; pitch, energy and
        # duration are measured from the raw waveform for each slice.
        # This preserves stretched words, emphasis, and pitch changes that
        # text-based analysis cannot see.
        # Whisper word timestamps slice the audio per word; pitch, energy and
        # duration are measured from the raw waveform for each slice.
        # This preserves stretched words, emphasis, and pitch changes that
        # text-based analysis cannot see.
        if use_prosody:
            text, word_timestamps = asr.transcribe_with_timestamps(
                audio_path, model_size=model_size, language=language
            )
            if not text:
                print("[Pipeline] No speech detected.")
                return ""
            print(f"\n[Pipeline] Transcribed text:\n  {text}\n")

            # Fiction gate: only fiction stories proceed to emotion + prosody.
            content_type, _ = fiction.detect(text)
            if content_type == "non_fiction":
                print("[Pipeline] Non-fiction detected — skipping emotion and prosody.")
                return text

            detected_emotion, _ = emotion.detect(audio_path)

            if word_timestamps:
                segments = prosody.analyze_acoustic(
                    audio_path, word_timestamps, content_type=content_type
                )
            else:
                # Fallback: no timestamps available — use text rules
                segments = prosody.analyze(
                    text, emotion_hint=detected_emotion, content_type=content_type
                )

            tts.speak_prosody(segments, save_path=save_path, voice_index=voice_index)
            return text

        # Step 1: ASR — audio to text
        text = asr.transcribe(audio_path, model_size=model_size, language=language)

        if not text:
            print("[Pipeline] No speech detected.")
            return ""

        print(f"\n[Pipeline] Transcribed text:\n  {text}\n")

        # Fiction gate: only fiction stories proceed to emotion + TTS.
        content_type, _ = fiction.detect(text)
        if content_type == "non_fiction":
            print("[Pipeline] Non-fiction detected — skipping emotion and TTS.")
            return text

        # Step 2: Emotion detection — extract prosodic features from the recording
        detected_emotion, _ = emotion.detect(audio_path)

        # Step 3: Humanize — stylise text + derive prosody params (rate, volume)
        result = humanize.enhance(text, detected_emotion, content_type=content_type)

        print(f"\n[Pipeline] Humanized text:\n  {result['text']}\n")

        # Step 4: TTS — choose between normal TTS and voice cloning

        if use_clone:
            tts.speak_cloned(
                text=result["text"],
                reference_wav=audio_path,
                language=language or "en",
                save_path=save_path,
            )
        else:
            # Use standard pyttsx3 TTS
            if save_path:
                tts.save(
                    result["text"], save_path,
                    rate=result["rate"], volume=result["volume"],
                    voice_index=1,
                )
                tts.speak(
                    result["text"],
                    rate=result["rate"], volume=result["volume"],
                    voice_index=1,
                )
            else:
                tts.speak(
                    result["text"],
                    rate=result["rate"], volume=result["volume"],
                    voice_index=1,
                )

        return text

    finally:
        if cleanup_audio and os.path.isfile(audio_path):
            os.remove(audio_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASR → TTS pipeline: transcribe speech then speak the result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", metavar="TEXT", help="Raw story text to synthesise (no audio needed).")
    source.add_argument("--text-file", metavar="PATH", help="Path to a .txt file to synthesise (no audio needed).")
    source.add_argument("--file", metavar="PATH", help="Path to an audio file.")
    source.add_argument("--mic", action="store_true", help="Record from microphone.")

    parser.add_argument(
        "--save", metavar="PATH",
        help="Also save TTS output to this WAV file.",
    )
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--language", default=None,
        metavar="CODE", help="Force language for Whisper, e.g. 'en'.",
    )
    parser.add_argument(
        "--rate", type=int, default=175,
        metavar="WPM", help="TTS words-per-minute rate (default: 175).",
    )
    parser.add_argument(
        "--voice", type=int, default=0,
        metavar="INDEX", help="TTS voice index (default: 0).",
    )
    parser.add_argument(
        "--list-voices", action="store_true",
        help="Print available TTS voices and exit.",
    )
    parser.add_argument(
        "--clone", action="store_true", default=True,
        help="Use XTTS voice cloning (default: enabled).",
    )
    parser.add_argument(
        "--no-clone", dest="clone", action="store_false",
        help="Use pyttsx3 instead of voice cloning.",
    )
    parser.add_argument(
        "--prosody", action="store_true",
        help="Use rule-based prosody control (per-segment rate/pitch/pauses "
             "via pyttsx3). Overrides --clone.",
    )
    parser.add_argument(
        "--model-prosody", action="store_true", dest="model_prosody",
        help="Use trained BERT prosody model for emotionally-conditioned speech. "
             "Overrides --prosody and --clone. Works with --text, --file, or --mic.",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best.pt",
        metavar="PATH", help="Trained ProsodyPredictor checkpoint (default: checkpoints/best.pt).",
    )

    args = parser.parse_args()

    if args.list_voices:
        tts.list_voices()
        sys.exit(0)

    if not args.text and not args.text_file and not args.file and not args.mic:
        parser.print_help()
        sys.exit(1)

    # Resolve --text-file into a text string
    input_text = args.text
    if args.text_file:
        text_file_path = args.text_file
        if not os.path.isfile(text_file_path):
            print(f"[Pipeline] Error: text file not found: {text_file_path}")
            sys.exit(1)
        with open(text_file_path, encoding="utf-8") as f:
            input_text = f.read()

    cleanup    = False
    audio_path = args.file

    if args.mic:
        audio_path = record_microphone()
        cleanup = True  # delete temp file after processing

    run_pipeline(
        audio_path=audio_path,
        model_size=args.model,
        language=args.language,
        tts_rate=args.rate,
        voice_index=args.voice,
        save_path=args.save,
        use_clone=args.clone,
        use_prosody=args.prosody,
        use_model_prosody=args.model_prosody,
        checkpoint_path=args.checkpoint,
        input_text=input_text,
        cleanup_audio=cleanup,
    )


if __name__ == "__main__":
    main()
