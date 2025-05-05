import wave, tempfile, re
from pathlib import Path
from typing import Final

import numpy as np
from openai import OpenAI

from ..config import (
    OPENAI_API_KEY, RATE, CHANNELS, WHISPER_OPTIONS,
    ACCOUNT_NUMBER_PATTERNS, NUMBER_WORD_MAP,
)
from .utils import words_to_digits                   # NEW

client: Final = OpenAI(api_key=OPENAI_API_KEY)

def normalize_numbers(text: str) -> str:
    """
    1. convert spoken words to digits         (“one two” -> “12”)
    2. format any 12‑digit account as XXXX‑XXX‑XXXX
    """
    text = words_to_digits(text.lower())

    for pattern in ACCOUNT_NUMBER_PATTERNS:
        for m in re.finditer(pattern, text):
            raw = re.sub(r"[-\s]", "", m.group(0))
            if len(raw) == 12:
                pretty = f"{raw[:4]}-{raw[4:7]}-{raw[7:]}"
                text = text.replace(m.group(0), pretty)
    return text

async def transcribe_audio(audio: np.ndarray) -> str:
    try:
        if audio.size == 0 or np.max(np.abs(audio)) < 0.01:
            return ""

        int16 = (audio * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, "wb") as wav:
                wav.setnchannels(CHANNELS)
                wav.setsampwidth(2)
                wav.setframerate(RATE)
                wav.writeframes(int16.tobytes())

            with open(tmp.name, "rb") as f:
                result = client.audio.transcriptions.create(file=f, **WHISPER_OPTIONS)

        Path(tmp.name).unlink(missing_ok=True)
        return normalize_numbers(result.text.strip())

    except Exception as exc:
        print(f"[transcriber] error: {exc}")
        return ""
