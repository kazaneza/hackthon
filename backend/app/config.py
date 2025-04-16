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