import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Whisper Configuration
WHISPER_OPTIONS = {
    "model": "whisper-1",
    "language": "en",
    "temperature": 0.0,
    "response_format": "text"
}

# TTS Configuration
TTS_OPTIONS = {
    "model": "tts-1",
    "voice": "alloy",
    "speed": 1.0
}

# Audio Configuration
RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.03
SILENCE_DURATION = 1.0
CHUNK_DURATION = 3.0