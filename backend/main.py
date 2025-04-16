from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv
import shutil
from pathlib import Path
import uuid

load_dotenv()

app = FastAPI(title="Speech-to-Text API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure temp directory exists
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Generate unique filename
    temp_file = TEMP_DIR / f"{uuid.uuid4()}.webm"
    
    try:
        # Save uploaded file
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ensure file is closed
        await file.close()
        
        # Transcribe using OpenAI
        with temp_file.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )
            
        return {"transcription": transcription.text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass  # Best effort cleanup

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Cleanup on shutdown
@app.on_event("shutdown")
async def cleanup():
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
    except Exception:
        pass  # Best effort cleanup