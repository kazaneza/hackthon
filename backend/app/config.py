import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Whisper Configuration
WHISPER_OPTIONS = {
    "model": "gpt-4o-transcribe",
    "language": "en",
    "temperature": 0.0,
    "response_format": "text"
}

# TTS Configuration
TTS_OPTIONS = {
    "model": "tts-1",
    "voice": "coral",
    "speed": 1.0
}