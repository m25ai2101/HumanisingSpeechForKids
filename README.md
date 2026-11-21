# HumanizingSpeech

An end-to-end pipeline that transforms children's story text into expressive, emotionally aware narrated audio. The system detects emotion at the paragraph level, predicts prosody parameters using a trained BERT model, and renders natural-sounding speech via Microsoft Edge Neural TTS.

## Pipeline Overview

```
Text / Audio ──► ASR (Whisper) ──► Fiction Gate ──► Emotion Detection ──► Emotion Fusion ──► Prosody Prediction (BERT) ──► Neural TTS (Edge) ──► Expressive Audio
```

### Modules

| Module | Description |
|---|---|
| `asr.py` | Speech-to-text transcription using OpenAI Whisper (runs locally, no API key) |
| `fiction.py` | Fiction vs non-fiction classifier using POS-ratio logistic regression (Qureshi et al. 2019, Kazmi et al. 2022) |
| `text_emotion.py` | Text-based emotion detection using DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`) |
| `emotion.py` | Acoustic emotion detection using Wav2Vec2 fine-tuned on IEMOCAP (`superb/wav2vec2-base-superb-er`) |
| `fusion.py` | Multimodal fusion of text and speech emotion vectors with configurable weights |
| `prosody_model.py` | BERT-based prosody predictor — predicts pitch, duration, volume, and pause per token conditioned on emotion |
| `prosody.py` | Rule-based prosody analysis (acoustic and text-based fallback) |
| `humanize.py` | Text humanization layer — discourse detection, prosody profiles, and style presets |
| `tts.py` | TTS rendering with multiple backends: pyttsx3 (offline), Edge Neural TTS (online, expressive) |
| `pipeline.py` | End-to-end orchestrator connecting all modules |
| `app.py` | Streamlit web UI for interactive story narration |

### Training and Data

| Module | Description |
|---|---|
| `data_pipeline.py` | Builds training data from LibriSpeech/LibriTTS (HuggingFace datasets) |
| `train.py` | Trains the ProsodyPredictor model |
| `scrape_storynory.py` | Scrapes children's stories from Storynory for training data |
| `evaluate.py` | Quantitative evaluation — MAE, RMSE, Pearson r against ground-truth prosody |
| `evaluate_audio.py` | Audio-level evaluation of generated speech |
| `mos_eval.py` | Mean Opinion Score evaluation framework |

## Installation

**Python 3.10–3.11 recommended.**

```bash
pip install -r requirements.txt
```

### System Dependencies

- **FFmpeg** (required by torchcodec):
  - Windows: `winget install --id Gyan.FFmpeg.Shared` (add its `bin/` to PATH)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

- **espeak** (required by some TTS backends on Linux):
  - `sudo apt install espeak-ng`

## Usage

### Command Line

```bash
# Synthesise a story from text (primary flow)
python pipeline.py --text "Once upon a time..." --model-prosody --save output.wav

# Read from a text file
python pipeline.py --text-file story.txt --model-prosody --save output.wav

# Transcribe audio and generate expressive speech
python pipeline.py --file recording.wav --model-prosody --save output.wav

# Record from microphone
python pipeline.py --mic --model-prosody

# Compare standard vs V-A enhanced prosody
python pipeline.py --text-file story.txt --model-prosody --compare --save output.wav
```

### Streamlit Web App

```bash
streamlit run app.py
```

Upload a `.txt` story file to:
1. Classify fiction vs non-fiction (with confidence score and explanation)
2. Detect emotion per paragraph (with badges and confidence)
3. Generate expressive audio and compare against basic TTS
4. Choose from multiple narrator voices (British, American, Australian, Irish)

## Key Features

- **Paragraph-level emotion detection**: each paragraph gets its own emotion label (joyful, scared, dramatic, sad, etc.) for varied, expressive narration
- **Fiction gate**: POS-ratio classifier filters non-fiction content, with an override option
- **Trained BERT prosody model**: predicts pitch, duration, volume, and pause per token, conditioned on the fused emotion vector
- **Valence-Arousal (V-A) continuous emotion model**: replaces discrete emotion labels with a continuous V-A space for smoother prosody transitions
- **Edge Neural TTS**: Microsoft's neural voices with SSML prosody control for natural-sounding output
- **Multimodal emotion fusion**: combines text-based and speech-based emotion signals when audio input is available
- **Comparison mode**: generates both standard and V-A enhanced audio side-by-side with a prosody diff table

## Training the Prosody Model

```bash
# Build training data from LibriSpeech
python data_pipeline.py --hf-dataset openslr/librispeech_asr --hf-config clean --hf-split train.100 --max-samples 50 --output data/

# Train
python train.py --data data/train.jsonl --val data/val.jsonl

# Evaluate
python evaluate.py
```

The trained checkpoint is saved to `checkpoints/best.pt`.

## Evaluation

The `evaluate.py` script compares predicted prosody against ground-truth audio measurements on a held-out test set. It reports MAE, RMSE, and Pearson r for each prosody parameter (pitch, duration, volume, pause) against two baselines:
- **Mean predictor**: always predicts the dataset mean
- **Neutral predictor**: always predicts flat/neutral values (what a basic monotone TTS does)

A model that beats both baselines demonstrates learned emotion-conditioned prosody.
