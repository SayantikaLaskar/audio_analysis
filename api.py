import os
import tempfile
import requests
import re
from typing import Optional
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

app = FastAPI(title="Audio Processing API", version="1.0.0")

# Load env and setup Mongo
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "AkaiDb0")  # Updated to match your DB name
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "transcriptions")
mongo_client: Optional[MongoClient] = None
mongo_collection = None
audio_datapoints_collection = None

if MONGODB_URI:
    try:
        mongo_client = MongoClient(MONGODB_URI)
        mongo_collection = mongo_client[MONGODB_DB][MONGODB_COLLECTION]
        # Audio datapoints collection where media URL is stored
        audio_datapoints_collection = mongo_client[MONGODB_DB]["audioDatapoints"]
    except Exception:
        mongo_client = None
        mongo_collection = None
        audio_datapoints_collection = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


class TranscribeRequest(BaseModel):
    project_id: str
    task_id: str
    user_id: str


@app.post("/transcribe")
async def transcribe_json(request: TranscribeRequest):
    """Transcribe and analyze audio using a JSON body.

    Body parameters:
    - project_id: string
    - task_id: string  
    - user_id: string
    """

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

    # Resolve audio URL from audioDatapoints collection
    if audio_datapoints_collection is None:
        raise HTTPException(status_code=500, detail="Audio datapoints collection not available")

    # Build filters that match either string IDs or ObjectId-stored IDs
    def build_id_candidates(value: str):
        # Trim whitespace/newlines from incoming IDs
        cleaned = (value or "").strip()
        candidates = [cleaned]
        if BsonObjectId is not None:
            try:
                candidates.append(BsonObjectId(cleaned))
            except Exception:
                pass
        return candidates

    try:
        # Updated query to match your data structure
        query = {
            "task_id": {"$in": build_id_candidates(request.task_id)},
            "project_id": {"$in": build_id_candidates(request.project_id)},
            "mediaUrl": {"$exists": True}
        }
        dp_list = list(audio_datapoints_collection.find(query))
        
        # Debug info
        print(f"Query: {query}")
        print(f"Task ID candidates: {build_id_candidates(request.task_id)}")
        print(f"Project ID candidates: {build_id_candidates(request.project_id)}")
        print(f"Found {len(dp_list)} documents")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Datapoint query failed: {e}")

    if not dp_list:
        # Try a broader search for debugging
        try:
            task_only = list(audio_datapoints_collection.find({"task_id": {"$in": build_id_candidates(request.task_id)}}).limit(5))
            project_only = list(audio_datapoints_collection.find({"project_id": {"$in": build_id_candidates(request.project_id)}}).limit(5))
            print(f"Task-only matches: {len(task_only)}")
            print(f"Project-only matches: {len(project_only)}")
        except Exception as e:
            print(f"Debug query failed: {e}")
            
        raise HTTPException(
            status_code=404, 
            detail=f"No datapoint found for task_id={request.task_id}, project_id={request.project_id}. Check /debug/audioDatapoints endpoint for available data."
        )

    # Get the first matching document
    dp = dp_list[0]
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
            if audio_datapoints_collection is not None and dp.get("_id") is not None:
                audio_datapoints_collection.update_one(
                    {"_id": dp["_id"]},
                    {
                        "$set": {
                            "transcripts": transcripts,
                            "soundEffects": sound_effects,
                            "processingStatus": "completed",
                            "updatedAt": datetime.utcnow(),
                        },
                        "$setOnInsert": {
                            "project_id": dp.get("project_id", request.project_id),
                            "task_id": dp.get("task_id", request.task_id),
                            "createdAt": datetime.utcnow(),
                        },
                    },
                    upsert=False,
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


@app.get("/debug/audioDatapoints")
async def debug_audio_datapoints(project_id: str, task_id: str):
    """Debug helper: check visibility of datapoints by IDs and collection/DB wiring."""
    if audio_datapoints_collection is None:
        raise HTTPException(status_code=500, detail="Audio datapoints collection not available")

    def build_id_candidates_local(value: str):
        cleaned = (value or "").strip()
        candidates = [cleaned]
        if BsonObjectId is not None:
            try:
                candidates.append(BsonObjectId(cleaned))
            except Exception:
                pass
        return candidates

    db_query = {
        "task_id": {"$in": build_id_candidates_local(task_id)},
        "project_id": {"$in": build_id_candidates_local(project_id)}
    }
    docs = list(audio_datapoints_collection.find(db_query).limit(10))
    # Build a JSON-serializable view of the query
    safe_query = {
        "task_id": {"$in": [str(v) for v in build_id_candidates_local(task_id)]},
        "project_id": {"$in": [str(v) for v in build_id_candidates_local(project_id)]},
    }
    return {
        "db": MONGODB_DB,
        "collection": "audioDatapoints",
        "query": safe_query,
        "count": len(docs),
        "sample": [
            {
                "_id": str(d.get("_id")),
                "has_mediaUrl": bool(d.get("mediaUrl")),
                "task_id_type": type(d.get("task_id")).__name__,
                "project_id_type": type(d.get("project_id")).__name__,
            }
            for d in docs
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)