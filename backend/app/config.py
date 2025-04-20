import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database Configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

# Whisper Configuration
WHISPER_OPTIONS = {
    "model": "whisper-1",
    "language": "en",
    "temperature": 0.0,
    "response_format": "text",
    "prompt": "This is a banking conversation. Numbers and account details are important. The text may contain account numbers in format XXXX-XXX-XXXX or XXXXXXXXXXXX."
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

# Number Processing
ACCOUNT_NUMBER_PATTERNS = [
    r'\b\d{4}[-\s]?\d{3}[-\s]?\d{4}\b',  # Format: XXXX-XXX-XXXX
    r'\b\d{12}\b',                        # Format: XXXXXXXXXXXX
    r'one|two|three|four|five|six|seven|eight|nine|zero',  # Spoken numbers
    r'\b\d+\b'                            # Any number sequence
]

NUMBER_WORD_MAP = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}