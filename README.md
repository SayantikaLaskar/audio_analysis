# Audio Processing Pipeline (SED · STT · Diarization · Emotion) + API

This project provides a complete audio analysis pipeline and a simple web API:

- Sound Event Detection (SED) → Non‑speech audio events
- Speaker Diarization → Who spoke when
- Speech‑to‑Text (STT) → Transcription via OpenAI Whisper API
- Emotion Detection → Emotions inferred from text
- FastAPI service → Upload or reference audio and get structured JSON back

Results can be returned via API and/or saved to a local JSON file.

---

## 🚀 Features

- Converts input audio (MP3, WAV, FLAC, M4A) to a consistent format when needed
- Runs SED with the `MIT/ast-finetuned-audioset-10-10-0.4593` model
- Runs Speaker Diarization with `pyannote.audio` (requires HF token)
- Transcribes speech with OpenAI Whisper API
- Detects emotions for each transcript segment
- Exposes a `/transcribe` API endpoint and/or writes `final_output.json`

---

## 📂 Project Structure

```
├── sed_stt.py          # CLI pipeline (SED + Diarization + STT + Emotion)
├── api.py              # FastAPI server exposing /transcribe
├── requirements.txt    # Python dependencies
└── README.md           # This documentation
```

---

## ⚙️ Prerequisites

- Python 3.10+
- FFmpeg installed and available on PATH
  - Windows: download from `https://ffmpeg.org/download.html`, extract, add `bin/` to PATH
  - Linux: `sudo apt-get install ffmpeg`
- (Optional, enables diarization) Hugging Face account + token

---

## 🧪 Setup

1. Create and activate a virtual environment

```
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/macOS
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Environment variables (create a `.env` file in the project root)

```
# Required for diarization
HF_TOKEN=your_huggingface_read_token

# Required for speech-to-text
OPENAI_API_KEY=your_openai_api_key

# Optional: persist API results to MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster/db
MONGODB_DB=audio_db
MONGODB_COLLECTION=transcriptions
```

Notes:

- If `HF_TOKEN` is not set, diarization will be skipped (pipeline continues).
- If `OPENAI_API_KEY` is not set, speech transcription will fail.
- On Windows, if you encounter symlink warnings from Hugging Face, you can set:
  - `set HF_HUB_DISABLE_SYMLINKS_WARNING=1`
  - `set HF_HUB_DISABLE_SYMLINKS=1`

---

## ▶️ Usage (CLI pipeline)

The CLI pipeline is defined in `sed_stt.py`. It reads the file specified by `AUDIO_FILE` at the top of the script (default: `ENDE001.mp3`).

Steps:

1. Place your audio file in the project directory
2. Edit `AUDIO_FILE` in `sed_stt.py` to point to your file name
3. Run:

```
python sed_stt.py
```

Output: a `final_output.json` file will be created with transcript, speakers, emotions, and sound effects.

---

## 🌐 Usage (FastAPI server)

Start the server:

```
python api.py
```

Endpoints:

- GET /health → `{ "status": "ok" }`
- POST /transcribe → JSON body (project_id, task_id, user_id)
- POST /transcribe-upload → multipart file upload or form path input

JSON body for `POST /transcribe`:

```json
{
  "project_id": "68aab18274ec11e465e4fb91",
  "task_id": "68b58059c70e58f9921dbdfb",
  "user_id": "6857b9446800696c7aa3cdc1"
}
```

Examples (PowerShell):

```
curl -Method Post -Uri http://localhost:8000/transcribe -Body '{"project_id":"68aab18274ec11e465e4fb91","task_id":"68b58059c70e58f9921dbdfb","user_id":"6857b9446800696c7aa3cdc1"}' -ContentType 'application/json'
```

Using `curl` (Git Bash / Linux / macOS):

```
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "68aab18274ec11e465e4fb91",
    "task_id": "68b58059c70e58f9921dbdfb",
    "user_id": "6857b9446800696c7aa3cdc1"
  }'
```

File upload variant (`POST /transcribe-upload`):

```
curl -X POST http://localhost:8000/transcribe-upload \
  -F "file=@sample.mp3"
```

Response (example):

```json
{
  "file_id": "sample",
  "file_path": "C:/audio/sample.mp3",
  "detected_language": "en",
  "transcript": [
    {
      "speaker": "SPEAKER_1",
      "start_time": 0.0,
      "end_time": 5.2,
      "text": "Hello, how are you?",
      "emotion": "joy",
      "emotion_score": 0.92
    }
  ],
  "sound_effects": [{ "label": "Music", "score": 0.87 }],
  "metadata": { "annotator": "system_auto" }
}
```

If MongoDB is configured, the response will also include the inserted `_id` or a `db_error` if persistence fails.

---

## ⚙️ Models and runtime

- Device selection is automatic: CUDA if available, else CPU (for SED and emotion models)
- Whisper: OpenAI Whisper API (whisper-1 model)
- SED model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Diarization: `pyannote/speaker-diarization@2.1` (requires `HF_TOKEN`)

Tip: GPU significantly accelerates SED and emotion detection inference. Whisper API calls don't require local GPU.

---

## 📊 Output schema (summary)

```json
{
  "file_id": "<basename>",
  "file_path": "<original path>",
  "detected_language": "<lang>",
  "transcript": [
    {
      "speaker": "<label or Unknown>",
      "start_time": 0.0,
      "end_time": 1.23,
      "text": "...",
      "emotion": "joy|anger|...",
      "emotion_score": 0.0
    }
  ],
  "sound_effects": [{ "label": "...", "score": 0.0 }],
  "metadata": { "annotator": "system_auto" }
}
```

---

## 🛠️ Troubleshooting

- Windows symlink warnings from Hugging Face:
  - `set HF_HUB_DISABLE_SYMLINKS_WARNING=1`
  - `set HF_HUB_DISABLE_SYMLINKS=1`
- FFmpeg not found:
  - Ensure `ffmpeg` is on PATH; on Windows, verify `C:\ffmpeg\bin` or your install location is added
- Diarization failing:
  - Ensure `HF_TOKEN` is set and has read access; without it, diarization is skipped
- Speech transcription failing:
  - Ensure `OPENAI_API_KEY` is set with valid OpenAI API access
- Large model downloads/timeouts:
  - Pre-download models or ensure a stable connection for SED/emotion models

---
