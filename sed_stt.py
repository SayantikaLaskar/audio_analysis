
import os
import platform
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import librosa
from openai import OpenAI
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
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
    """Detect the best available device for computation"""
    if torch.cuda.is_available():
        # NVIDIA GPU available (AWS EC2 with GPU instances, Windows, Linux)
        device = "cuda"
        print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU (Mac M1/M2/M3)
        device = "mps"
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        # Fallback to CPU
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
    """Check if ffmpeg is available"""
    return shutil.which("ffmpeg") is not None

def convert_to_wav(input_path, output_path="temp.wav"):
    """Convert audio file to wav for processing with multiple fallback options"""
    
    # First, try to use the original file directly if it's already supported
    if input_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        print(f"🔄 Audio file {input_path} should be directly compatible. Trying to use as-is...")
        try:
            # Test if librosa can load it directly
            audio_data, sr = librosa.load(input_path, sr=16000, duration=1.0)  # Test with 1 second
            print(f"✅ File {input_path} is directly readable by librosa")
            return input_path  # Use original file
        except Exception as e:
            print(f"⚠️  Direct loading failed: {e}. Attempting conversion...")
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        print("⚠️  FFmpeg not found. Trying to load and convert with librosa + soundfile...")
        try:
            # Try to load the file directly with librosa
            audio_data, sr = librosa.load(input_path, sr=16000)
            
            # Try to save using soundfile
            try:
                import soundfile as sf
                sf.write(output_path, audio_data, sr)
                print(f"✅ Successfully converted {input_path} to {output_path} using librosa + soundfile")
                return output_path
            except ImportError:
                print("❌ soundfile not installed. Please install it: pip install soundfile")
                print("🔄 Attempting to use original file directly...")
                return input_path
                
        except Exception as e:
            print(f"❌ Failed to convert with librosa: {e}")
            print("🔄 Attempting to use original file directly...")
            return input_path
    
    try:
        # Use ffmpeg if available
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", output_path
        ], check=True, capture_output=True, text=True)
        print(f"✅ Successfully converted {input_path} to {output_path} using ffmpeg")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        print("🔄 Attempting to use original file directly...")
        return input_path
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
        print("🔄 Attempting to use original file directly...")
        return input_path

def check_file_exists(file_path):
    """Check if the input file exists"""
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        print(f"📁 Current directory: {os.getcwd()}")
        print("📋 Available files:", [f for f in os.listdir(".") if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))])
        return False
    return True

# ===================== 1. SOUND EVENT DETECTION =====================
def run_sed(audio_path, threshold=0.01):
    """Sound Event Detection with error handling"""
    try:
        print("🔊 Loading SED model...")
        
        # Convert device string to device index for transformers pipeline
        device_idx = 0 if DEVICE in ["cuda", "mps"] else -1
        
        sed_pipeline = hf_pipeline(
            task="audio-classification",
            model="MIT/ast-finetuned-audioset-10-10-0.4593",
            device=device_idx,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        
        print("🎵 Loading audio for SED...")
        audio_data, sr = librosa.load(audio_path, sr=16000)
        
        print("🔍 Running sound event detection...")
        results = sed_pipeline(audio_data)
        
        # keep only non-speech sounds above threshold
        results = [
            e for e in results
            if "speech" not in e["label"].lower() and e["score"] >= threshold
        ]
        print(f"✅ SED completed. Found {len(results)} sound events")
        return results
    except Exception as e:
        print(f"❌ SED failed: {e}")
        print("🔄 Continuing without sound event detection...")
        return []

# ===================== 2. SPEAKER DIARIZATION =====================
def run_diarization(audio_path):
    """Speaker diarization with error handling"""
    try:
        print("👥 Loading diarization model...")
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN not set in environment (.env)")
        
        # Set device for pyannote pipeline
        device = torch.device(DEVICE)
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=HF_TOKEN
        )
        
        # Move pipeline to the appropriate device
        if DEVICE != "cpu":
            pipeline = pipeline.to(device)
        
        print("🎯 Running speaker diarization...")
        diarization = pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker
            })
        
        print(f"✅ Diarization completed. Found {len(segments)} speaker segments")
        return segments
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        print("📋 Note: You need a valid Hugging Face token for pyannote models")
        print("🔄 Continuing without speaker diarization...")
        return []

# ===================== 3. SPEECH TO TEXT =====================
def run_stt(audio_path):
    """Speech to text using OpenAI Whisper API with error handling"""
    try:
        print("🗣️  Using OpenAI Whisper API...")
        
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set in environment (.env)")
        
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
            print("📝 Running speech transcription via API...")
            # Use the Whisper API with timestamps
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        detected_lang = transcription.language
        segments = []
        
        # Parse segments from API response
        if hasattr(transcription, 'segments') and transcription.segments:
            for seg in transcription.segments:
                segments.append({
                    "start_time": seg.start,
                    "end_time": seg.end,
                    "text": seg.text,
                    "confidence": getattr(seg, 'avg_logprob', None)
                })
        else:
            # Fallback: create a single segment with the full text
            segments.append({
                "start_time": 0.0,
                "end_time": 0.0,  # We don't have duration info
                "text": transcription.text,
                "confidence": None
            })
        
        print(f"✅ STT completed. Language: {detected_lang}, Segments: {len(segments)}")
        return detected_lang, segments
    except Exception as e:
        print(f"❌ STT failed: {e}")
        print("📋 Note: You need a valid OpenAI API key for Whisper API")
        return "unknown", []

# ===================== 4. EMOTION DETECTION =====================
def run_emotion_detection(texts):
    """Emotion detection with error handling"""
    if not texts:
        return []
    
    try:
        print("😊 Loading emotion detection model...")
        
        # Convert device string to device index for transformers pipeline
        device_idx = 0 if DEVICE in ["cuda", "mps"] else -1
        
        emo_pipeline = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=device_idx,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        
        print("🎭 Running emotion detection...")
        results = emo_pipeline(texts)
        
        # keep only top emotion
        emotions = []
        for res in results:
            if isinstance(res, list):
                res = sorted(res, key=lambda x: x["score"], reverse=True)[0]
            emotions.append({"label": res["label"], "score": float(res["score"])})
        
        print(f"✅ Emotion detection completed for {len(emotions)} segments")
        return emotions
    except Exception as e:
        print(f"❌ Emotion detection failed: {e}")
        print("🔄 Continuing without emotion detection...")
        return [{"label": "unknown", "score": 0.0} for _ in texts]

# ===================== 5. MERGE & SAVE JSON =====================
def build_output(file_id, audio_path, sed_events, diarization, transcript, detected_lang, save_to_file=True):
    """Build final output dict. Optionally persist to OUTPUT_JSON when save_to_file=True."""
    # ---- attach speakers to transcript ----
    for t in transcript:
        t["speaker"] = "Unknown"
        for s in diarization:
            if (s["start"] <= t["start_time"] <= s["end"]) or (s["start"] <= t["end_time"] <= s["end"]):
                t["speaker"] = s["speaker"]
                break

    # ---- add emotion detection ----
    texts = [t["text"] for t in transcript]
    emotions = run_emotion_detection(texts)
    
    # Handle case where emotion detection returns fewer results
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

    # ---- sort and clean sound effects ----
    sorted_sounds = sorted(sed_events, key=lambda x: x["score"], reverse=True)
    sound_effects = [
        {"label": s["label"], "score": round(s["score"], 3)}
        for s in sorted_sounds
    ]

    # ---- final JSON ----
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
        print(f"\n✅ Final JSON saved to {OUTPUT_JSON}")

    return data

# ===================== MAIN =====================
if __name__ == "__main__":
    print(f"🚀 Starting audio processing pipeline...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🎵 Input file: {AUDIO_FILE}")
    print(f"💻 Using device: {DEVICE}")
    print(f"🖥️  Platform: {platform.system()} {platform.machine()}")
    
    # Set optimizations for different platforms
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        print("⚡ CUDA optimizations enabled")
    elif DEVICE == "mps":
        print("🍎 MPS optimizations enabled")
        # Set memory fraction for Apple Silicon
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # Check if input file exists
    if not check_file_exists(AUDIO_FILE):
        exit(1)
    
    try:
        # Convert audio file (or use original if possible)
        print("\n📂 Preparing audio file...")
        wav_path = convert_to_wav(AUDIO_FILE)

        print("\n🔊 Running Sound Event Detection...")
        sed_events = run_sed(wav_path)
        
        # Clear GPU cache if using GPU
        if DEVICE in ["cuda", "mps"]:
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
            print("🗑️  GPU cache cleared")

        print("\n👥 Running Speaker Diarization...")
        diarization = run_diarization(wav_path)
        
        # Clear GPU cache again
        if DEVICE in ["cuda", "mps"]:
            torch.cuda.empty_cache() if DEVICE == "cuda" else None

        print("\n🗣️  Running Speech-to-Text...")
        detected_lang, transcript = run_stt(wav_path)

        print("\n🔄 Building final output with emotions and timestamps...")
        build_output(AUDIO_FILE, AUDIO_FILE, sed_events, diarization, transcript, detected_lang, save_to_file=True)
        
        # Clean up temporary file (only if we created one)
        if wav_path != AUDIO_FILE and os.path.exists(wav_path):
            os.remove(wav_path)
            print(f"🧹 Cleaned up temporary file: {wav_path}")
            
        # Final GPU cleanup
        if DEVICE in ["cuda", "mps"]:
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
            print("🗑️  Final GPU cleanup completed")
            
        print("\n🎉 Processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("📋 Please check your dependencies and file paths")
        # Clean up GPU memory on error
        if DEVICE in ["cuda", "mps"]:
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
        exit(1)