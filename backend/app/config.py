import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Audio Processing Configuration
RATE = 16000  # Changed to 16kHz - Whisper's preferred sample rate
CHANNELS = 1
CHUNK_DURATION = 3  # seconds
SILENCE_THRESHOLD = 0.01  # Adjusted for normalized audio
SILENCE_DURATION = 0.5  # seconds

# Whisper Configuration
WHISPER_OPTIONS = {
    "model": "whisper-1",
    "language": "en",     # Explicitly set language
    "temperature": 0.0,   # Reduced for more accurate transcription
    "response_format": "text"
}