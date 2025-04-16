from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
import wave
import tempfile
from pathlib import Path
import numpy as np
from openai import OpenAI
from ..config import OPENAI_API_KEY, RATE, CHANNELS, WHISPER_OPTIONS
from .audio import AudioBuffer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
client = OpenAI(api_key=OPENAI_API_KEY)

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    logger.info(f"Received file: {file.filename} ({file.content_type})")
    
    try:
        # Create temporary WAV file
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
                        
                        # The transcription is now returned directly as a string
                        result = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
                        logger.info(f"Transcription successful: {result}")
                        
                        return JSONResponse(content={"transcription": result})
                    except Exception as e:
                        logger.error(f"OpenAI transcription error: {str(e)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Transcription failed: {str(e)}"
                        )
            except Exception as e:
                logger.error(f"File processing error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"File processing failed: {str(e)}"
                )
            finally:
                try:
                    # Make sure the file handle is closed before attempting to delete
                    temp_file.close()
                    Path(temp_file.name).unlink()
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        await file.close()

@router.websocket("/transcribe/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = AudioBuffer([])
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Add samples and check if we should process
            if audio_buffer.add_samples(audio_chunk):
                audio_data = audio_buffer.get_audio_data()
                
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    with wave.open(temp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(2)  # 16-bit audio
                        wav_file.setframerate(RATE)
                        wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                    
                    # Transcribe with OpenAI
                    with open(temp_file.name, 'rb') as audio_file:
                        transcription = client.audio.transcriptions.create(
                            file=audio_file,
                            **WHISPER_OPTIONS
                        )
                        
                        # Handle both string and object responses
                        result = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
                    
                    # Clean up temp file
                    temp_file.close()
                    Path(temp_file.name).unlink()
                    
                    # Send transcription back to client
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result
                    })
            
            # Send audio level for visualization
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            await websocket.send_json({
                "type": "audio_level",
                "level": float(rms)
            })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass