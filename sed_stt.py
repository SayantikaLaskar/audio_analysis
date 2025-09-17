import os
import platform
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
import librosa
import subprocess, json
from pathlib import Path
import shutil
from dotenv import load_dotenv
import assemblyai as aai

# ===================== CONFIG =====================
AUDIO_FILE = "/Users/abhangsudhirpawar/Downloads/harvard.wav"
OUTPUT_JSON = "final_output.json"

# Load environment variables
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Set AssemblyAI API key
aai.settings.api_key = ASSEMBLYAI_API_KEY

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

# ===================== ASSEMBLYAI TRANSCRIPTION WITH ALL FEATURES =====================
def run_assemblyai_transcription(audio_path):
    """Run AssemblyAI transcription with speaker diarization, sentiment analysis, and automatic language detection"""
    try:
        print("🎙️ Starting AssemblyAI transcription with all features...")
        
        # Configure transcription with all features
        config = aai.TranscriptionConfig(
            speaker_labels=True,  # Enable speaker diarization
            sentiment_analysis=True,  # Enable sentiment analysis
            language_detection=True,  # Enable automatic language detection
            language_confidence_threshold=0.4,  # Optional confidence threshold
            punctuate=True,
            format_text=True
        )
        
        # Create transcriber and transcribe
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"❌ Transcription failed: {transcript.error}")
            return "unknown", [], [], 0.0  # Return 4 values consistently
        
        # Extract detected language from json_response
        detected_lang = "unknown"
        language_confidence = 0.0
        
        try:
            # Access language_code from json_response
            detected_lang = transcript.json_response.get("language_code", "en")
            language_confidence = transcript.json_response.get("language_confidence", 0.0)
            print(f"🔍 Detected language: {detected_lang} (confidence: {language_confidence:.3f})")
        except Exception as e:
            print(f"⚠️ Could not extract language info: {e}")
            detected_lang = "en"
            language_confidence = 0.0
        
        # Process transcript segments (rest of your code remains the same)
        segments = []
        diarization = []
        
        if hasattr(transcript, 'utterances') and transcript.utterances:
            for utterance in transcript.utterances:
                segment_data = {
                    "start_time": utterance.start / 1000.0,
                    "end_time": utterance.end / 1000.0,
                    "text": utterance.text,
                    "speaker": utterance.speaker,
                    "confidence": getattr(utterance, 'confidence', 0.0)
                }
                
                if hasattr(utterance, 'sentiment') and utterance.sentiment:
                    segment_data['sentiment'] = utterance.sentiment
                    segment_data['sentiment_confidence'] = getattr(utterance, 'sentiment_confidence', 0.5)
                
                segments.append(segment_data)
                
                diarization.append({
                    "start": utterance.start / 1000.0,
                    "end": utterance.end / 1000.0,
                    "speaker": utterance.speaker
                })
        
        elif hasattr(transcript, 'sentences') and transcript.sentences:
            for sentence in transcript.sentences:
                segment_data = {
                    "start_time": sentence.start / 1000.0,
                    "end_time": sentence.end / 1000.0,
                    "text": sentence.text,
                    "speaker": "S1",
                    "confidence": getattr(sentence, 'confidence', 0.0)
                }
                
                if hasattr(sentence, 'sentiment') and sentence.sentiment:
                    segment_data['sentiment'] = sentence.sentiment
                    segment_data['sentiment_confidence'] = getattr(sentence, 'sentiment_confidence', 0.5)
                
                segments.append(segment_data)
        
        else:
            segments.append({
                "start_time": 0.0,
                "end_time": 0.0,
                "text": transcript.text,
                "speaker": "S1",
                "confidence": getattr(transcript, 'confidence', 0.0)
            })
        
        print(f"✅ AssemblyAI transcription completed. Language: {detected_lang}, Segments: {len(segments)}")
        return detected_lang, segments, diarization, language_confidence  # Always return 4 values
        
    except Exception as e:
        print(f"❌ AssemblyAI transcription failed: {e}")
        return "unknown", [], [], 0.0  # Always return 4 values


# ===================== EMOTION DETECTION (now using AssemblyAI sentiment) =====================
def extract_emotions_from_segments(segments):
    """Extract emotion data from AssemblyAI sentiment analysis results"""
    emotions = []
    
    for segment in segments:
        if 'sentiment' in segment and segment['sentiment']:
            # Map AssemblyAI sentiment to emotion-like labels
            sentiment_to_emotion = {
                'POSITIVE': 'joy',
                'NEGATIVE': 'sadness', 
                'NEUTRAL': 'neutral'
            }
            
            emotion_label = sentiment_to_emotion.get(segment['sentiment'], 'neutral')
            confidence = segment.get('sentiment_confidence', 0.5)
            
            emotions.append({
                "label": emotion_label,
                "score": confidence
            })
        else:
            emotions.append({
                "label": "neutral",
                "score": 0.5
            })
    
    return emotions

# ===================== MERGE & SAVE JSON =====================
def build_output(file_id, audio_path, segments, diarization, detected_lang, language_confidence=0.0):
    # Extract emotions from sentiment analysis
    emotions = extract_emotions_from_segments(segments)
    
    # Build merged transcript with all information
    merged_transcript = []
    for i, segment in enumerate(segments):
        emotion_data = emotions[i] if i < len(emotions) else {"label": "neutral", "score": 0.5}
        
        merged_transcript.append({
            "speaker": segment.get("speaker", "S1"),
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "text": segment["text"].strip(),
            "emotion": emotion_data["label"],
            "emotion_score": round(emotion_data["score"], 3),
            "confidence": segment.get("confidence", 0.0)
        })
    
    # Since we removed sound effect detection, this will be empty
    sound_effects = []
    
    data = {
        "file_id": Path(file_id).stem,
        "file_path": audio_path,
        "detected_language": detected_lang,
        "language_confidence": round(language_confidence, 4),
        "transcript": merged_transcript,
        "sound_effects": sound_effects,
        "metadata": {
            "annotator": "assemblyai_auto",
            "features_used": ["speech_to_text", "speaker_diarization", "sentiment_analysis", "language_detection"]
        }
    }
    
    return data


# ===================== MAIN (commented out for import use) =====================
# if __name__ == "__main__":
#     if not check_file_exists(AUDIO_FILE):
#         print("❌ Audio file not found!")
#         exit(1)
    
#     try:
#         print("\n📂 Preparing audio file...")
#         wav_path = convert_to_wav(AUDIO_FILE)
#         print(f"✅ Audio file prepared: {wav_path}")
        
#         print("\n🎙️ Running AssemblyAI transcription with all features...")
#         # Updated to expect 4 return values
#         detected_lang, segments, diarization, language_confidence = run_assemblyai_transcription(wav_path)
        
#         print(f"\n🔍 Transcription Results:")
#         print(f"   📍 Detected Language: {detected_lang}")
#         print(f"   🎯 Language Confidence: {language_confidence:.3f}")
#         print(f"   🎤 Total Segments: {len(segments)}")
#         print(f"   👥 Diarization Entries: {len(diarization)}")
        
#         print("\n🔄 Building final output with emotions and timestamps...")
#         final_data = build_output(
#             AUDIO_FILE, 
#             AUDIO_FILE, 
#             segments, 
#             diarization, 
#             detected_lang,
#             language_confidence
#         )
        
#         # Print transcript preview
#         transcripts = final_data.get("transcript", [])
#         print(f"\n📋 Transcript Preview ({len(transcripts)} segments):")
#         for i, segment in enumerate(transcripts[:3]):  # Show first 3 segments
#             print(f"   {i+1}. [{segment['speaker']}] {segment['text'][:100]}...")
#             print(f"      Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")
#             print(f"      Emotion: {segment['emotion']} ({segment['emotion_score']:.3f})")
#             print()
        
#         if len(transcripts) > 3:
#             print(f"   ... and {len(transcripts) - 3} more segments")
        
#         # Save to JSON file
#         with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
#             json.dump(final_data, f, indent=2, ensure_ascii=False)
        
#         print(f"\n💾 Results saved to: {OUTPUT_JSON}")
#         print(f"📊 File size: {os.path.getsize(OUTPUT_JSON) / 1024:.1f} KB")
        
#         # Cleanup
#         if wav_path != AUDIO_FILE and os.path.exists(wav_path):
#             os.remove(wav_path)
#             print(f"🗑️ Cleaned up temporary file: {wav_path}")
        
#         print("\n🎉 Processing completed successfully!")
#         print("\n" + "="*60)
#         print("📈 SUMMARY:")
#         print(f"   🔤 Language: {detected_lang} ({language_confidence:.1%} confidence)")
#         print(f"   🎯 Segments: {len(segments)}")
#         print(f"   👥 Speakers: {len(set(s.get('speaker', 'S1') for s in segments))}")
#         print(f"   📄 Output: {OUTPUT_JSON}")
#         print("="*60)
        
#     except Exception as e:
#         print(f"\n❌ Fatal error: {e}")
#         print(f"📋 Error type: {type(e).__name__}")
#         import traceback
#         print(f"🔍 Traceback:")
#         traceback.print_exc()
#         exit(1)
