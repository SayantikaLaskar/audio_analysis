import os
import platform
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import librosa
from openai import OpenAI
from pyannote.audio import Inference
from transformers import pipeline as hf_pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import subprocess, json
from pathlib import Path
import shutil
from dotenv import load_dotenv
import torch


# ===================== CONFIG =====================
AUDIO_FILE = "ENDE001.mp3"
OUTPUT_JSON = "final_output.json"

# Auto-detect device with GPU support for AWS EC2, Mac, and Windows
def get_optimal_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = "mps"
    #     print("🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        print("💻 Using CPU (no GPU detected)")
    return device

DEVICE = get_optimal_device()

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ===================== HELPER =====================
def check_ffmpeg():
    return shutil.which("ffmpeg") is not None

def convert_to_wav(input_path, output_path="temp.wav"):
    if input_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        try:
            audio_data, sr = librosa.load(input_path, sr=16000, duration=1.0)
            return input_path
        except:
            pass
    if not check_ffmpeg():
        try:
            audio_data, sr = librosa.load(input_path, sr=16000)
            try:
                import soundfile as sf
                sf.write(output_path, audio_data, sr)
                return output_path
            except ImportError:
                return input_path
        except:
            return input_path
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", output_path
        ], check=True, capture_output=True, text=True)
        return output_path
    except:
        return input_path

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        return False
    return True

# ===================== === MODIFIED === Load waveform once =====================
def load_waveform(audio_path):
    """Load waveform once and reuse across models"""  # === MODIFIED ===
    waveform, sr = librosa.load(audio_path, sr=16000)
    return waveform, sr  # === MODIFIED ===

# ===================== 1. SOUND EVENT DETECTION =====================
def run_sed(waveform, sr, threshold=0.01):  # === MODIFIED ===
    """Sound Event Detection with error handling"""
    try:
        print("🔊 Loading SED model...")
        device_idx = 0 if DEVICE in ["cuda", "mps"] else -1
        
        # === MODIFIED: Use feature extractor to process waveform once ===
        extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        inputs = extractor(waveform, sampling_rate=sr, return_tensors="pt")
        if DEVICE == "cuda":
            model.to("cuda")
            inputs = {k: v.to("cuda") for k,v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # For simplicity, using HF pipeline for post-processing
        sed_pipeline = hf_pipeline(
            task="audio-classification",
            model="MIT/ast-finetuned-audioset-10-10-0.4593",
            device=device_idx
        )
        results = sed_pipeline(waveform)  # still waveform-based
        results = [e for e in results if "speech" not in e["label"].lower() and e["score"] >= threshold]
        return results
    except Exception as e:
        print(f"❌ SED failed: {e}")
        return []

# ===================== 2. SPEAKER DIARIZATION =====================
def run_diarization(waveform, sr):
    """Speaker activity segmentation using pyannote/segmentation"""
    try:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN not set")

        # Load segmentation model as Inference
        inference = Inference("pyannote/segmentation", use_auth_token=HF_TOKEN, device=DEVICE)

        # waveform must be (time,) or (1, time)
        waveform_tensor = torch.tensor(waveform).unsqueeze(0)  # shape [1, time]
        # Compute speech activity scores
        speech_scores = inference(waveform_tensor, sample_rate=sr)

        # Convert scores to speech segments
        # Thresholding example
        threshold = 0.5
        segments = []
        start = None
        for i, score in enumerate(speech_scores[0]):  # assuming batch=1
            t = i / sr
            if score >= threshold and start is None:
                start = t
            elif score < threshold and start is not None:
                segments.append({"start": start, "end": t, "speaker": "unknown"})
                start = None
        if start is not None:
            segments.append({"start": start, "end": len(waveform)/sr, "speaker": "unknown"})

        print(f"✅ Segmentation detected {len(segments)} segments")
        return segments
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return []

# ===================== 3. SPEECH TO TEXT =====================
# Remains unchanged; still uses audio file for Whisper API
def run_stt(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        detected_lang = transcription.language
        segments = []
        if hasattr(transcription, 'segments') and transcription.segments:
            for seg in transcription.segments:
                segments.append({
                    "start_time": seg.start,
                    "end_time": seg.end,
                    "text": seg.text,
                    "confidence": getattr(seg, 'avg_logprob', None)
                })
        else:
            segments.append({
                "start_time": 0.0,
                "end_time": 0.0,
                "text": transcription.text,
                "confidence": None
            })
        return detected_lang, segments
    except Exception as e:
        print(f"❌ STT failed: {e}")
        return "unknown", []

# ===================== 4. EMOTION DETECTION =====================
# Remains unchanged
def run_emotion_detection(texts):
    if not texts:
        return []
    try:
        device_idx = 0 if DEVICE in ["cuda", "mps"] else -1
        emo_pipeline = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=device_idx,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        results = emo_pipeline(texts)
        emotions = []
        for res in results:
            if isinstance(res, list):
                res = sorted(res, key=lambda x: x["score"], reverse=True)[0]
            emotions.append({"label": res["label"], "score": float(res["score"])})
        return emotions
    except Exception as e:
        print(f"❌ Emotion detection failed: {e}")
        return [{"label": "unknown", "score": 0.0} for _ in texts]

# ===================== 5. MERGE & SAVE JSON =====================
# Remains unchanged
def build_output(file_id, audio_path, sed_events, diarization, transcript, detected_lang, save_to_file=True):
    for t in transcript:
        t["speaker"] = "Unknown"
        for s in diarization:
            if (s["start"] <= t["start_time"] <= s["end"]) or (s["start"] <= t["end_time"] <= s["end"]):
                t["speaker"] = s["speaker"]
                break
    texts = [t["text"] for t in transcript]
    emotions = run_emotion_detection(texts)
    for i, t in enumerate(transcript):
        if i < len(emotions):
            t["emotion"] = emotions[i]["label"]
            t["emotion_score"] = emotions[i]["score"]
        else:
            t["emotion"] = "unknown"
            t["emotion_score"] = 0.0
    merged_transcript = [
        {
            "speaker": t["speaker"],
            "start_time": t["start_time"],
            "end_time": t["end_time"],
            "text": t["text"].strip(),
            "emotion": t["emotion"],
            "emotion_score": round(t["emotion_score"], 3)
        }
        for t in transcript
    ]
    sorted_sounds = sorted(sed_events, key=lambda x: x["score"], reverse=True)
    sound_effects = [
        {"label": s["label"], "score": round(s["score"], 3)}
        for s in sorted_sounds
    ]
    data = {
        "file_id": Path(file_id).stem,
        "file_path": audio_path,
        "detected_language": detected_lang,
        "transcript": merged_transcript,
        "sound_effects": sound_effects,
        "metadata": {
            "annotator": "system_auto"
        }
    }
    if save_to_file:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    return data

# ===================== MAIN =====================
# if __name__ == "__main__":
#     if not check_file_exists(AUDIO_FILE):
#         exit(1)
    
#     try:
#         print("\n📂 Preparing audio file...")
#         wav_path = convert_to_wav(AUDIO_FILE)
        
#         # === MODIFIED === Load waveform once ===
#         waveform, sr = load_waveform(wav_path)  # === MODIFIED ===

#         print("\n🔊 Running Sound Event Detection...")
#         sed_events = run_sed(waveform, sr)  # === MODIFIED ===

#         if DEVICE in ["cuda", "mps"]:
#             torch.cuda.empty_cache() if DEVICE == "cuda" else None

#         print("\n👥 Running Speaker Diarization...")
#         diarization = run_diarization(waveform, sr)  # === MODIFIED ===

#         if DEVICE in ["cuda", "mps"]:
#             torch.cuda.empty_cache() if DEVICE == "cuda" else None

#         print("\n🗣️  Running Speech-to-Text...")
#         detected_lang, transcript = run_stt(wav_path)

#         print("\n🔄 Building final output with emotions and timestamps...")
#         build_output(AUDIO_FILE, AUDIO_FILE, sed_events, diarization, transcript, detected_lang, save_to_file=True)

#         if wav_path != AUDIO_FILE and os.path.exists(wav_path):
#             os.remove(wav_path)

#         if DEVICE in ["cuda", "mps"]:
#             torch.cuda.empty_cache() if DEVICE == "cuda" else None

#         print("\n🎉 Processing completed successfully!")
        
#     except Exception as e:
#         print(f"\n❌ Fatal error: {e}")
#         if DEVICE in ["cuda", "mps"]:
#             torch.cuda.empty_cache() if DEVICE == "cuda" else None
#         exit(1)