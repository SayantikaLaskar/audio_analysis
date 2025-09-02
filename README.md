# Audio Processing Pipeline (SED · STT · Diarization · Emotion) + API

This project provides a complete audio analysis pipeline and a simple web API:

- Sound Event Detection (SED) → Non‑speech audio events
- Speaker Diarization → Who spoke when
- Speech‑to‑Text (STT) → Transcription via OpenAI Whisper
- Emotion Detection → Emotions inferred from text
- FastAPI service → Upload or reference audio and get structured JSON back

Results can be returned via API and/or saved to a local JSON file.

---

## 🚀 Features
- Converts input audio (MP3, WAV, FLAC, M4A) to a consistent format when needed
- Runs SED with the `MIT/ast-finetuned-audioset-10-10-0.4593` model
- Runs Speaker Diarization with `pyannote.audio` (requires HF token)
- Transcribes speech with OpenAI Whisper (medium model by default)
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

1) Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/macOS
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Environment variables (create a `.env` file in the project root)
```
# Required for diarization
HF_TOKEN=your_huggingface_read_token

# Optional: persist API results to MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster/db
MONGODB_DB=audio_db
MONGODB_COLLECTION=transcriptions
```

Notes:
- If `HF_TOKEN` is not set, diarization will be skipped (pipeline continues).
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
- POST /transcribe → multipart file upload or form path input

Request options for `POST /transcribe` (provide one of):
- `file`: multipart file upload (e.g., `audio/mpeg`, `audio/wav`)
- `audio_path`: path to an existing local audio file on the server

Examples (PowerShell):
```
curl -Method Post -Uri http://localhost:8000/transcribe -InFile .\sample.mp3 -ContentType 'audio/mpeg'
```

Using `curl` (Git Bash / Linux / macOS):
```
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample.mp3"
```

Using a local path (no upload):
```
curl -X POST http://localhost:8000/transcribe \
  -F "audio_path=C:\\audio\\sample.mp3"
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
  "sound_effects": [
    { "label": "Music", "score": 0.87 }
  ],
  "metadata": { "annotator": "system_auto" }
}
```

If MongoDB is configured, the response will also include the inserted `_id` or a `db_error` if persistence fails.

---

## ⚙️ Models and runtime
- Device selection is automatic: CUDA if available, else CPU
- Whisper model: `medium` (configured in `sed_stt.py`)
- SED model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Diarization: `pyannote/speaker-diarization@2.1` (requires `HF_TOKEN`)

Tip: GPU significantly accelerates Whisper and transformer inference.

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
- Large model downloads/timeouts:
  - Pre-download models or ensure a stable connection; consider smaller Whisper models if needed

---

## 📜 License
Choose a license and add it here (e.g., MIT, Apache-2.0).


