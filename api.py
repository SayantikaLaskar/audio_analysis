import os
import tempfile
import requests
import re
from pydantic import BaseModel
try:
    from bson import ObjectId as BsonObjectId 
except Exception:
    BsonObjectId = None

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId

app = FastAPI(title="Audio Processing API", version="1.0.0")

# Load env and setup Mongo
load_dotenv()
# MongoDB connection
MONGODB_URL = os.getenv('MONGODB_URI')
if not MONGODB_URL:
    raise ValueError("MONGO_URI environment variable is not set")

client = MongoClient(MONGODB_URL)
db = client["AkaiDb0"]
datapoints_collection = db["audioDatapoints"]

class TranscribeRequest(BaseModel):
    project_id: str
    task_id: str
    user_id: str


@app.post("/transcribe")
async def transcribe_json(request: TranscribeRequest):
    """Transcribe and analyze audio using a JSON body.
    """
    try:
        task_id = ObjectId(request.task_id)
        project_id = ObjectId(request.project_id)
        user_id = ObjectId(request.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid task_id or project_id")

    # Lazy import heavy pipeline to avoid import failures when server boots
    try:
        from sed_stt import (
            convert_to_wav,
            run_sed,
            run_diarization,
            run_stt,
            build_output,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import pipeline: {e}")
    
    try:
        # Updated query to match your data structure
        query = {
            "task_id": task_id, 
            "project_id": project_id,
            "processingStatus": "created"
        }
        dp = datapoints_collection.find_one(query)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Datapoint query failed: {e}")

    if not dp:
        raise HTTPException(status_code=404, detail="No datapoint found for given task_id and project_id with status 'created'")
    audio_url = dp.get("mediaUrl")
    if not audio_url:
        raise HTTPException(status_code=404, detail="Datapoint has no media URL field")

    # Download audio to a temporary file (supports public HTTP and private S3 via boto3)
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tf:
            temp_file = tf.name
            r = requests.get(audio_url, timeout=30)
            if r.status_code == 200:
                tf.write(r.content)
            else:
                # Try S3 signed download if URL looks like S3 and boto3 credentials exist
                is_s3 = bool(re.match(r"^https?://[\w.-]*s3[\w.-]*/", audio_url)) or audio_url.startswith("s3://")
                if is_s3:
                    try:
                        import boto3
                        from urllib.parse import urlparse

                        def parse_s3(url: str):
                            if url.startswith("s3://"):
                                # s3://bucket/key
                                parts = url[5:].split("/", 1)
                                return parts[0], parts[1]
                            # https://bucket.s3.amazonaws.com/key or virtual-hosted-style
                            parsed = urlparse(url)
                            host = parsed.netloc
                            path = parsed.path.lstrip("/")
                            if ".s3" in host:
                                bucket = host.split(".s3")[0]
                            else:
                                # path-style: s3.amazonaws.com/bucket/key
                                first, rest = path.split("/", 1)
                                bucket, path = first, rest
                            return bucket, path

                        bucket, key = parse_s3(audio_url)
                        s3 = boto3.client("s3")
                        s3.download_file(bucket, key, temp_file)
                    except Exception as e:
                        raise HTTPException(status_code=502, detail=f"Failed to download audio via S3: {e}")
                else:
                    raise HTTPException(status_code=502, detail=f"Failed to download audio (status {r.status_code})")
    except requests.RequestException as e:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=f"Failed to download audio: {e}")
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=f"Failed to download audio: {e}")

    try:
        # Convert or use original
        wav_path = convert_to_wav(temp_file)

        # Run pipeline
        sed_events = run_sed(wav_path)
        diarization = run_diarization(wav_path)
        detected_lang, transcript = run_stt(wav_path)

        # Build output dict (do not write to local file)
        data = build_output(
            str(dp.get("_id")) if dp.get("_id") else "datapoint",
            audio_url,
            sed_events,
            diarization,
            transcript,
            detected_lang,
            save_to_file=False,
        )

        # Persist results back to audioDatapoints with pluralized fields
        transcripts = data.get("transcript", [])
        sound_effects = data.get("sound_effects", [])

        try:
            datapoints_collection.update_one(
                {"_id": dp["_id"]},
                {
                    "$set": {
                        "transcript": transcripts,
                        "soundEffect": sound_effects,
                        "processingStatus": "pre-label",
                        "updatedAt": datetime.utcnow(),
                    },
                },
            )
        except Exception as e:
            # Do not fail the request if DB write fails; return processing result instead
            print(f"Failed to write transcripts/soundEffects to MongoDB: {e}")

        # API response with pluralized keys and IDs
        response = {
            "project_id": request.project_id,
            "task_id": request.task_id,
            "user_id": request.user_id,
            "datapoint_id": str(dp.get("_id")) if dp.get("_id") else None,
            "audio_url": audio_url,
            "detected_language": data.get("detected_language"),
            "transcripts": transcripts,
            "soundEffects": sound_effects,
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary file
        try:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            # Also cleanup wav file if it's different from temp_file
            if 'wav_path' in locals() and wav_path != temp_file and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)