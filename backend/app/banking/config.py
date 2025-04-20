import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database Configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", 
    'Driver={SQL Server};'
    'Server=your_server;'
    'Database=your_database;'
    'Trusted_Connection=yes;'
)

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