import os
import shutil
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymongo import MongoClient

app = FastAPI(title="Audio Processing API", version="1.0.0")

# Load env and setup Mongo
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "audio_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "transcriptions")
mongo_client: Optional[MongoClient] = None
mongo_collection = None

if MONGODB_URI:
    try:
        mongo_client = MongoClient(MONGODB_URI)
        mongo_collection = mongo_client[MONGODB_DB][MONGODB_COLLECTION]
    except Exception:
        mongo_client = None
        mongo_collection = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: Optional[UploadFile] = File(default=None),
    audio_path: Optional[str] = Form(default=None),
):
    """Transcribe and analyze audio.

    One of `file` (multipart upload) or `audio_path` (existing local path) must be provided.
    Returns merged transcript with speakers, emotions, detected language and sound effects.
    """

    if not file and not audio_path:
        raise HTTPException(status_code=400, detail="Provide either file upload or audio_path")

    # Lazy import heavy pipeline to avoid import failures when server boots
    try:
        from sed_stt import (
            convert_to_wav,
            check_file_exists,
            run_sed,
            run_diarization,
            run_stt,
            build_output,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import pipeline: {e}")

    # Persist upload if provided
    temp_input_path: Optional[str] = None
    try:
        if file:
            filename = os.path.basename(file.filename) if file.filename else "uploaded_audio"
            temp_input_path = os.path.abspath(filename)
            with open(temp_input_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            source_path = temp_input_path
        else:
            # Validate local path
            if not check_file_exists(audio_path):
                raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
            source_path = audio_path

        # Convert or use original
        wav_path = convert_to_wav(source_path)

        # Run pipeline
        sed_events = run_sed(wav_path)
        diarization = run_diarization(wav_path)
        detected_lang, transcript = run_stt(wav_path)

        # Build output dict (do not write to local file)
        data = build_output(
            source_path,
            source_path,
            sed_events,
            diarization,
            transcript,
            detected_lang,
            save_to_file=False,
        )

        # Persist to MongoDB if configured
        if mongo_collection is not None:
            try:
                insert_result = mongo_collection.insert_one(data)
                data["_id"] = str(insert_result.inserted_id)
            except Exception as e:
                # Do not fail the request if DB insert fails; include warning
                data["db_error"] = str(e)

        return JSONResponse(content=data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary upload if created
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


