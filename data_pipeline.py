"""
Dataset Preparation Pipeline

Converts audio + text pairs into training records (jsonl) that the prosody
prediction model (prosody_model.py) consumes.

Supports two data sources:

  1. Local directory of audio + text file pairs — RECOMMENDED for kids stories:
       python data_pipeline.py --input <data_dir> --output data/

     Suggested free sources for kids audio + text:
       - LibriVox children's section (librivox.org/subject/children)
         Public-domain audiobooks read aloud; download MP3s + copy the
         Project Gutenberg text as .txt sidecars.
       - Storynory (storynory.com) — free kids audio stories with scripts.

     Expected local layout (either format works):

     Format A — flat directory:
       data_dir/
         story_001.wav   (or .mp3)
         story_001.txt   (transcript, one sentence per file)
         story_002.wav
         story_002.txt

     Format B — nested by story title:
       data_dir/
         Alice/
           scene_01.wav
           scene_01.txt
         Cinderella/
           ...

     If .txt files are absent, Whisper is used to transcribe the audio.

  2. HuggingFace dataset (bring your own dataset ID):
       python data_pipeline.py --hf-dataset <DATASET_ID> \
                               --hf-split train \
                               --output data/

     The dataset must have an "audio" column (with array + sampling_rate)
     and a "text" column. Find audio+text datasets at huggingface.co/datasets
     (filter by task: automatic-speech-recognition).

Output:
    data/train.jsonl
    data/val.jsonl
    data/test.jsonl

Each line of a jsonl file is one utterance:
{
  "text":        "Once upon a time there was a little girl...",
  "audio_path":  "data_dir/story_001.wav",
  "emotion":     "joyful",
  "emotion_vec": {"excited": 0.05, "joyful": 0.70, ...},
  "words": [
    {
      "word": "Once", "start": 0.12, "end": 0.38,
      "f0_hz": 195.3, "f0_semitone_shift": 1.1,
      "rms_energy": 0.038, "duration_ratio": 1.0,
      "pause_after_sec": 0.05, "rate_wpm": 152, "pitch_shift": 1.1, "volume": 0.82
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Audio file discovery
# ---------------------------------------------------------------------------

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _find_pairs(data_dir: str) -> list[tuple[Path, Path | None]]:
    """
    Walk data_dir and return (audio_path, text_path_or_None) pairs.
    text_path is None when no .txt sidecar exists — Whisper will transcribe.
    """
    root = Path(data_dir)
    pairs: list[tuple[Path, Path | None]] = []
    for audio in sorted(root.rglob("*")):
        if audio.suffix.lower() not in _AUDIO_EXTS:
            continue
        txt = audio.with_suffix(".txt")
        pairs.append((audio, txt if txt.exists() else None))
    return pairs


# ---------------------------------------------------------------------------
# Single utterance → training record
# ---------------------------------------------------------------------------

def _process_pair(
    audio_path: Path,
    text_path: Path | None,
    content_type: str = "fiction",
    verbose: bool = False,
) -> dict | None:
    """
    Convert one (audio, text) pair into a training record dict.
    Returns None if processing fails (corrupt audio, empty transcript, etc.).
    """
    import asr
    import text_emotion
    import emotion as speech_emotion_module
    import fusion
    import prosody_extractor

    # ---- 1. Transcript + word timestamps --------------------------------
    try:
        if text_path is not None:
            text = text_path.read_text(encoding="utf-8").strip()
            # Still use Whisper for word timestamps even when text is provided
            result = asr.transcribe_with_timestamps(str(audio_path))
            words  = result.get("words", [])
            # If Whisper text diverges badly, prefer the provided text
            if not text:
                text = result.get("text", "").strip()
        else:
            result = asr.transcribe_with_timestamps(str(audio_path))
            text   = result.get("text", "").strip()
            words  = result.get("words", [])
    except Exception as e:
        print(f"[Pipeline] SKIP {audio_path.name} — ASR failed: {e}")
        return None

    if not text or not words:
        print(f"[Pipeline] SKIP {audio_path.name} — empty transcript")
        return None

    # ---- 2. Text emotion ------------------------------------------------
    try:
        text_vec = text_emotion.detect(text)
    except Exception as e:
        print(f"[Pipeline] text emotion failed for {audio_path.name}: {e}")
        from emotion_schema import uniform_vector
        text_vec = uniform_vector()

    # ---- 3. Speech emotion ----------------------------------------------
    try:
        speech_vec = speech_emotion_module.detect_vector(str(audio_path))
    except Exception as e:
        print(f"[Pipeline] speech emotion failed for {audio_path.name}: {e}")
        speech_vec = None

    # ---- 4. Fuse --------------------------------------------------------
    fused_vec = fusion.fuse(text_vec, speech_vec)
    dominant, confidence = fusion.dominant_label(fused_vec)

    # ---- 5. Per-word prosody features -----------------------------------
    try:
        word_features = prosody_extractor.extract(
            str(audio_path), words,
            content_type=content_type,
            verbose=verbose,
        )
    except Exception as e:
        print(f"[Pipeline] prosody extraction failed for {audio_path.name}: {e}")
        return None

    if not word_features:
        print(f"[Pipeline] SKIP {audio_path.name} — no prosody features extracted")
        return None

    return {
        "text":        text,
        "audio_path":  str(audio_path),
        "emotion":     dominant,
        "emotion_confidence": confidence,
        "emotion_vec": fused_vec,
        "words":       word_features,
    }


# ---------------------------------------------------------------------------
# HuggingFace dataset processing
# ---------------------------------------------------------------------------

def _process_hf_row(
    row: dict,
    idx: int,
    content_type: str = "fiction",
    verbose: bool = False,
) -> dict | None:
    """
    Convert one HuggingFace kids stories row into a training record.

    The row is expected to have:
        row["audio"]["array"]          — numpy float32 waveform
        row["audio"]["sampling_rate"]  — int (typically 16000)
        row["text"]                    — exact transcript string

    The audio array is saved to a temp WAV file so existing modules that
    expect a file path (emotion.py, prosody_extractor.py, asr.py) work
    without modification. The temp file is deleted after processing.
    """
    import tempfile
    import numpy as np
    import scipy.io.wavfile as wavfile
    import asr
    import text_emotion
    import emotion as speech_emotion_module
    import fusion
    import prosody_extractor

    audio_raw = row.get("audio")
    text      = (row.get("text") or "").strip()

    # Decode audio: non-streaming returns a dict; streaming returns an
    # AudioDecoder lazy object whose internal format varies by datasets version.
    import io as _io
    import soundfile as _sf

    array, sr = None, 16_000

    if isinstance(audio_raw, dict):
        array = audio_raw.get("array")
        sr    = audio_raw.get("sampling_rate", 16_000)
    elif audio_raw is not None:
        # Method 1: subscript access — AudioDecoder supports [] but not .get()
        try:
            array = audio_raw["array"]
            try:
                sr = audio_raw["sampling_rate"]
            except (KeyError, TypeError):
                sr = 16_000
        except (TypeError, KeyError):
            pass

        # Method 2: bytes / path attributes → soundfile
        if array is None:
            _bytes = getattr(audio_raw, "bytes", None)
            _path  = getattr(audio_raw, "path", None)
            try:
                if _bytes:
                    _data, sr = _sf.read(_io.BytesIO(_bytes))
                    array = np.array(_data, dtype=np.float32)
                elif _path:
                    _data, sr = _sf.read(_path)
                    array = np.array(_data, dtype=np.float32)
            except Exception:
                pass

        # Method 3: datasets Audio.decode_example() — works in datasets 3.x
        if array is None:
            try:
                from datasets.features.audio import Audio as _HFAudio
                _decoded = _HFAudio(sampling_rate=16_000).decode_example(audio_raw)
                array = np.array(_decoded["array"], dtype=np.float32)
                sr    = _decoded.get("sampling_rate", 16_000)
            except Exception:
                pass

        if array is None:
            _attrs = [a for a in dir(audio_raw) if not a.startswith("_")]
            print(f"[Pipeline] SKIP row {idx} — cannot decode AudioDecoder "
                  f"(type={type(audio_raw).__name__}, public_attrs={_attrs})")
            return None

    if array is None or not text:
        print(f"[Pipeline] SKIP row {idx} — missing audio or text")
        return None

    # Save numpy array → temp WAV (int16 PCM)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        # Normalise to int16 range
        arr_int16 = (np.array(array, dtype=np.float32) * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(tmp_path, sr, arr_int16)

        # ---- 1. Word-level timestamps (Whisper) -------------------------
        # We already have the text from HF; Whisper is used only for alignment.
        try:
            _, words = asr.transcribe_with_timestamps(tmp_path)
            if not words:
                print(f"[Pipeline] row {idx} — no word timestamps, skipping")
                return None
        except Exception as e:
            print(f"[Pipeline] SKIP row {idx} — ASR failed: {e}")
            return None

        # ---- 2. Text emotion --------------------------------------------
        try:
            text_vec = text_emotion.detect(text)
        except Exception as e:
            print(f"[Pipeline] row {idx} text emotion failed: {e}")
            from emotion_schema import uniform_vector
            text_vec = uniform_vector()

        # ---- 3. Speech emotion ------------------------------------------
        try:
            speech_vec = speech_emotion_module.detect_vector(tmp_path)
        except Exception as e:
            print(f"[Pipeline] row {idx} speech emotion failed: {e}")
            speech_vec = None

        # ---- 4. Fuse ----------------------------------------------------
        fused_vec = fusion.fuse(text_vec, speech_vec)
        dominant, confidence = fusion.dominant_label(fused_vec)

        # ---- 5. Per-word prosody features --------------------------------
        try:
            word_features = prosody_extractor.extract(
                tmp_path, words,
                content_type=content_type,
                verbose=verbose,
            )
        except Exception as e:
            print(f"[Pipeline] row {idx} prosody extraction failed: {e}")
            return None

        if not word_features:
            print(f"[Pipeline] SKIP row {idx} — no prosody features")
            return None

        return {
            "text":               text,
            "audio_path":         f"hf_row_{idx}",   # no persistent path for HF data
            "speaker_id":         row.get("speaker_id"),
            "emotion":            dominant,
            "emotion_confidence": confidence,
            "emotion_vec":        fused_vec,
            "words":              word_features,
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_from_hf(
    dataset_name: str = "openslr/librispeech_asr",
    config: str = None,
    split: str = "train",
    output_dir: str = "data",
    train_val_test: tuple[float, float, float] = (0.8, 0.1, 0.1),
    content_type: str = "fiction",
    max_samples: int | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """
    Load a HuggingFace speech dataset and build training jsonl files.

    Args:
        dataset_name:  HuggingFace dataset id. Must have "audio" and "text" columns.
        config:        Dataset config/subset name, or None for the default config.
        split:         Dataset split, e.g. "train", "validation", "test".
        output_dir:    Directory to write train/val/test jsonl files.
        train_val_test: Fraction of records to put in each split.
        content_type:  "fiction" (recommended for story audio).
        max_samples:   Cap the number of rows processed (useful for quick tests).
        seed:          Random seed for shuffling.
        verbose:       Print per-word prosody feature tables.
    """
    from datasets import load_dataset

    print(f"[Pipeline] Loading HuggingFace dataset '{dataset_name}' "
          f"config='{config or 'default'}' split='{split}' (streaming — no full download) ...")
    load_kwargs = dict(split=split, streaming=True)
    if config is not None:
        load_kwargs["name"] = config
    ds = load_dataset(dataset_name, **load_kwargs)

    limit_str = str(max_samples) if max_samples else "all"
    print(f"[Pipeline] Processing up to {limit_str} rows")

    records: list[dict] = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        print(f"[Pipeline] {i + 1}/{limit_str}: {(row.get('text') or '')[:60]!r}")
        rec = _process_hf_row(row, idx=i, content_type=content_type, verbose=verbose)
        if rec is not None:
            records.append(rec)

    _finish_and_write(records, output_dir, train_val_test, seed)


# ---------------------------------------------------------------------------
# Train / val / test split and jsonl export
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[Pipeline] Wrote {len(records)} records → {path}")


def _finish_and_write(
    records: list[dict],
    output_dir: str,
    split: tuple[float, float, float],
    seed: int,
) -> None:
    """Shared: shuffle → split → write jsonl → print emotion distribution."""
    from collections import Counter

    if not records:
        print("[Pipeline] No valid records produced. Exiting.")
        return

    n = len(records)
    print(f"[Pipeline] {n} valid records total")

    random.seed(seed)
    random.shuffle(records)

    n_train = int(n * split[0])
    n_val   = int(n * split[1])
    train   = records[:n_train]
    val     = records[n_train:n_train + n_val]
    test    = records[n_train + n_val:]

    out = Path(output_dir)
    _write_jsonl(train, out / "train.jsonl")
    _write_jsonl(val,   out / "val.jsonl")
    _write_jsonl(test,  out / "test.jsonl")

    dist = Counter(r["emotion"] for r in records)
    print("\n[Pipeline] Emotion distribution:")
    for label, count in dist.most_common():
        print(f"  {label:<12} {count:4d}  ({count / n:.0%})")


def build(
    data_dir: str,
    output_dir: str = "data",
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    content_type: str = "fiction",
    max_samples: int | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """Full pipeline for local audio+text pairs: discover → process → write jsonl."""
    pairs = _find_pairs(data_dir)
    if not pairs:
        print(f"[Pipeline] No audio files found under {data_dir}")
        return

    if max_samples:
        pairs = pairs[:max_samples]

    print(f"[Pipeline] Found {len(pairs)} audio files in {data_dir}")

    records: list[dict] = []
    for i, (audio, txt) in enumerate(pairs):
        print(f"[Pipeline] Processing {i + 1}/{len(pairs)}: {audio.name}")
        rec = _process_pair(audio, txt, content_type=content_type, verbose=verbose)
        if rec is not None:
            records.append(rec)

    _finish_and_write(records, output_dir, split, seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build training dataset from local audio+text pairs or a HuggingFace speech dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local kids story audio files (recommended — see module docstring for free sources):
  python data_pipeline.py --input kids_stories_data/ --output data/

  # HuggingFace dataset (you must supply a valid dataset ID with audio+text columns):
  python data_pipeline.py --hf-dataset <DATASET_ID> --hf-split train --output data/

  # Quick test with 200 samples:
  python data_pipeline.py --input kids_stories_data/ --max-samples 200 --output data/
        """,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-dataset", metavar="DATASET_ID",
                        help="HuggingFace dataset id, e.g. 'openslr/librispeech_asr'")
    source.add_argument("--input", metavar="DIR",
                        help="Local directory of audio+text file pairs")

    # HuggingFace-specific options
    parser.add_argument("--hf-config", default=None,
                        help="HuggingFace dataset config/subset (default: None — uses dataset default)")
    parser.add_argument("--hf-split",  default="train",
                        help="HuggingFace dataset split (default: train). "
                             "Options: train, validation, test")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max rows to process (useful for quick tests, e.g. 500)")

    # Shared options
    parser.add_argument("--output", default="data",
                        help="Output directory for train/val/test jsonl files (default: data/)")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test split ratios (default: 0.8 0.1 0.1)")
    parser.add_argument("--content-type", default="fiction",
                        choices=["fiction", "non_fiction"],
                        help="Story type — fiction enables full pitch range (default: fiction)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-word prosody feature tables during processing")

    args = parser.parse_args()

    if args.hf_dataset:
        build_from_hf(
            dataset_name=args.hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            output_dir=args.output,
            train_val_test=tuple(args.split),
            content_type=args.content_type,
            max_samples=args.max_samples,
            seed=args.seed,
            verbose=args.verbose,
        )
    else:
        build(
            data_dir=args.input,
            output_dir=args.output,
            split=tuple(args.split),
            content_type=args.content_type,
            max_samples=args.max_samples,
            seed=args.seed,
            verbose=args.verbose,
        )
