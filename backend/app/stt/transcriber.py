import wave
import tempfile
from pathlib import Path
import numpy as np
from openai import OpenAI
from ..config import OPENAI_API_KEY, RATE, CHANNELS, WHISPER_OPTIONS

client = OpenAI(api_key=OPENAI_API_KEY)

async def transcribe_audio(audio_data: np.ndarray) -> str:
    try:
        # Skip if audio is too quiet
        if np.max(np.abs(audio_data)) < 0.01:
            return ""
            
        # Convert float32 audio to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Transcribe with OpenAI
            with open(temp_file.name, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    **WHISPER_OPTIONS
                )
            
            # Clean up temp file
            Path(temp_file.name).unlink()
            
            return transcription.text.strip()
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""