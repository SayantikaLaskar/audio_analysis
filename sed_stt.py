import os
import platform
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import librosa
from transformers import pipeline as hf_pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import subprocess, json
from pathlib import Path
import shutil
from dotenv import load_dotenv
import torch
import numpy as np
from sklearn.cluster import KMeans
from faster_whisper import WhisperModel


# ===================== CONFIG =====================
AUDIO_FILE = "ENDE001.mp3"
OUTPUT_JSON = "final_output.json"

# Force CPU-only execution regardless of GPU availability
DEVICE = "cpu"
print("💻 Forcing CPU-only execution (GPU disabled by request)")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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
        device_idx = -1
        
        # === MODIFIED: Use feature extractor to process waveform once ===
        extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        inputs = extractor(waveform, sampling_rate=sr, return_tensors="pt")
        # CPU only
        
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
    """Fast CPU-only diarization via simple VAD + MFCC k-means clustering.

    Produces segments with speaker labels S1/S2. Falls back to single speaker S1.
    """
    try:
        # 1) Simple VAD using short-time energy
        frame_length = int(0.03 * sr)  # 30ms
        hop_length = int(0.015 * sr)   # 15ms
        if frame_length <= 0:
            frame_length = 512
        if hop_length <= 0:
            hop_length = 256
        frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length)
        energy = np.mean(frames**2, axis=0)
        thresh = np.percentile(energy, 75)
        voiced = energy > max(thresh, 1e-8)

        # 2) Build contiguous voiced segments
        segments = []
        start_idx = None
        for i, v in enumerate(voiced):
            if v and start_idx is None:
                start_idx = i
            elif not v and start_idx is not None:
                s = start_idx * hop_length / sr
                e = (i * hop_length + frame_length) / sr
                if e - s > 0.2:
                    segments.append({"start": s, "end": e})
                start_idx = None
        if start_idx is not None:
            s = start_idx * hop_length / sr
            e = (len(voiced) * hop_length + frame_length) / sr
            if e - s > 0.2:
                segments.append({"start": s, "end": e})

        if not segments:
            return []

        # 3) Extract MFCC embeddings per segment
        embeddings = []
        for seg in segments:
            s = int(seg["start"] * sr)
            e = int(seg["end"] * sr)
            clip = waveform[s:e]
            if len(clip) < int(0.2 * sr):
                # pad short clips
                pad = int(0.2 * sr) - len(clip)
                clip = np.pad(clip, (0, pad))
            mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
            emb = np.mean(mfcc, axis=1)
            embeddings.append(emb)
        embeddings = np.stack(embeddings, axis=0)

        # 4) KMeans into up to 2 speakers if enough segments, else 1
        num_segments = embeddings.shape[0]
        n_clusters = 2 if num_segments >= 2 else 1
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = km.fit_predict(embeddings)

        diarized = []
        for seg, lab in zip(segments, labels):
            speaker = f"S{int(lab)+1}"
            diarized.append({"start": seg["start"], "end": seg["end"], "speaker": speaker})

        print(f"✅ Diarization produced {len(diarized)} segments across {n_clusters} speaker(s) on CPU")
        return diarized
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return []

# ===================== 3. SPEECH TO TEXT =====================
# Remains unchanged; still uses audio file for Whisper API
def run_stt(audio_path):
    """Speech-to-text using local faster-whisper with the tiny model on CPU."""
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments_iter, info = model.transcribe(audio_path, vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
        detected_lang = getattr(info, "language", "unknown") or "unknown"
        segments = []
        for seg in segments_iter:
            segments.append({
                "start_time": float(seg.start),
                "end_time": float(seg.end),
                "text": seg.text,
                "confidence": float(seg.avg_logprob) if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None else None
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
        t["speaker"] = "S1"
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