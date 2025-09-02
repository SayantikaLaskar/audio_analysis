# Audio Processing Pipeline (SED + STT + Diarization + Emotion)

This project is a complete **audio analysis pipeline** that combines:

- **Sound Event Detection (SED)** → Non-speech audio events  
- **Speaker Diarization** → Identifying who spoke when  
- **Speech-to-Text (STT)** → Transcribing speech using OpenAI Whisper  
- **Emotion Detection** → Inferring emotions from transcribed text  

Final results are saved into a **structured JSON file**.

---

## 🚀 Features
- Converts input audio (MP3, WAV, FLAC, M4A) into a consistent format
- Runs **Sound Event Detection** using Hugging Face AST model
- Runs **Speaker Diarization** with `pyannote.audio`
- Transcribes speech with **OpenAI Whisper**
- Detects **emotions** in text segments
- Outputs everything into `final_output.json`

---

## 📂 Project Structure
````
├── sed_stt.py          # Main pipeline script
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── ENDE001.mp3         # Example audio file (replace with your own)
````

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/audio-pipeline.git
cd audio-pipeline
````

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Install FFmpeg

* **Windows**: Download from [FFmpeg.org](https://ffmpeg.org/download.html), extract, and add `bin/` to PATH
* **Linux/macOS**:

  ```bash
  sudo apt-get install ffmpeg
  ```

---

## 🔑 Hugging Face Token

Speaker diarization requires a Hugging Face account & access token.

1. Sign up / log in at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens**
3. Generate a **read token**
4. Replace the placeholder in `sed_stt.py`:

   ```python
   use_auth_token="your_hf_token"
   ```

---

## ▶️ Usage

Place your audio file in the project directory and run:

```bash
python sed_stt.py
```

---

## 📊 Output

A `final_output.json` file will be generated:

```json
{
  "file_id": "ENDE001",
  "file_path": "ENDE001.mp3",
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
    {"label": "Music", "score": 0.87},
    {"label": "Applause", "score": 0.65}
  ],
  "metadata": {
    "annotator": "system_auto"
  }
}
```

---

## 🛠️ Troubleshooting

* **Windows symlink errors**: Run terminal as **Administrator** or set

  ```bash
  set HF_HUB_DISABLE_SYMLINKS=1
  ```
* **Model mismatch warnings**: Safe to ignore, but you can install matching versions of `torch` and `pyannote`.

