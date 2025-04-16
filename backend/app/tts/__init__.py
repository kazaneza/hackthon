from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
import logging
import io
from ..config import OPENAI_API_KEY, TTS_OPTIONS

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
client = OpenAI(api_key=OPENAI_API_KEY)

@router.get("/speak/{text}")
async def text_to_speech(text: str):
    try:
        logger.info(f"Generating speech for text: {text}")
        # Generate speech from text
        speech = client.audio.speech.create(
            input=text,
            **TTS_OPTIONS
        )
        
        # Create an in-memory bytes buffer
        audio_data = io.BytesIO()
        for chunk in speech.iter_bytes(chunk_size=8192):
            audio_data.write(chunk)
        audio_data.seek(0)
        
        logger.info("Speech generated successfully")
        # Return the audio as a streaming response
        return StreamingResponse(
            audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3"
            }
        )
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")