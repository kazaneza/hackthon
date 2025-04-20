from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
from openai import OpenAI
from ..config import OPENAI_API_KEY, WHISPER_OPTIONS
from ..nlp import NLPProcessor
import logging
from uuid import uuid4
from fastapi import Header

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
client = OpenAI(api_key=OPENAI_API_KEY)
nlp_processor = NLPProcessor()

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    x_session_id: str = Header(None)
):
    # Generate session ID if not provided
    session_id = x_session_id or str(uuid4())
    
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    logger.info(f"Received file: {file.filename} ({file.content_type})")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            try:
                # Save uploaded file
                content = await file.read()
                if not content:
                    raise HTTPException(status_code=400, detail="Empty file received")
                
                logger.info(f"File size: {len(content)} bytes")
                temp_file.write(content)
                temp_file.flush()
                
                # Transcribe with OpenAI
                with open(temp_file.name, 'rb') as audio_file:
                    try:
                        transcription = client.audio.transcriptions.create(
                            file=audio_file,
                            **WHISPER_OPTIONS
                        )
                        
                        result = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
                        logger.info(f"Transcription successful: {result}")
                        
                        # Process transcription with GPT using session ID
                        gpt_response = await nlp_processor.process_text(result, session_id)
                        logger.info(f"GPT response: {gpt_response}")
                        
                        return JSONResponse(
                            content={
                                "transcription": result,
                                "response": gpt_response,
                                "session_id": session_id
                            },
                            headers={"X-Session-ID": session_id}
                        )
                    except Exception as e:
                        logger.error(f"OpenAI processing error: {str(e)}")
                        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
            finally:
                try:
                    temp_file.close()
                    Path(temp_file.name).unlink()
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {str(e)}")
    finally:
        await file.close()

@router.post("/transcribe/text")
async def process_text(
    text: str = Body(..., embed=True),
    x_session_id: str = Header(None)
):
    """Process text directly without audio transcription"""
    session_id = x_session_id or str(uuid4())
    
    try:
        gpt_response = await nlp_processor.process_text(text, session_id)
        return JSONResponse(
            content={
                "response": gpt_response,
                "session_id": session_id
            },
            headers={"X-Session-ID": session_id}
        )
    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )