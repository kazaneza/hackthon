import wave
import tempfile
from pathlib import Path
import numpy as np
from openai import OpenAI
import re
from ..config import (
    OPENAI_API_KEY, RATE, CHANNELS, WHISPER_OPTIONS,
    ACCOUNT_NUMBER_PATTERNS, NUMBER_WORD_MAP
)

client = OpenAI(api_key=OPENAI_API_KEY)

def normalize_numbers(text: str) -> str:
    """Convert spoken numbers to digits and standardize account number format"""
    # Convert spoken numbers to digits
    for word, digit in NUMBER_WORD_MAP.items():
        text = re.sub(r'\b' + word + r'\b', digit, text.lower())
    
    # Find and format account numbers
    for pattern in ACCOUNT_NUMBER_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            number = match.group()
            # Remove any existing dashes or spaces
            clean_number = re.sub(r'[-\s]', '', number)
            if len(clean_number) == 12:  # Valid account number length
                # Format as XXXX-XXX-XXXX
                formatted = f"{clean_number[:4]}-{clean_number[4:7]}-{clean_number[7:]}"
                text = text.replace(number, formatted)
    
    return text

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
            
            # Process and normalize the transcribed text
            text = transcription.text.strip()
            normalized_text = normalize_numbers(text)
            
            return normalized_text
            
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""