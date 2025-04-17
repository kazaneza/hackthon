from openai import OpenAI
from fastapi import HTTPException
import logging
from ..config import OPENAI_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

# GPT Configuration
GPT_OPTIONS = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 150
}

# System message for chat context
SYSTEM_MESSAGE = """You are a helpful AI assistant. Analyze the user's transcribed speech and provide relevant, 
concise responses. Keep your responses friendly and natural."""

class NLPProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    async def process_text(self, text: str) -> str:
        """Process text with GPT and return the response."""
        try:
            logger.info(f"Processing text with GPT: {text}")
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": text}
                ],
                **GPT_OPTIONS
            )
            response = completion.choices[0].message.content.strip()
            logger.info(f"GPT response received: {response}")
            return response
        except Exception as e:
            logger.error(f"GPT processing error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"GPT processing failed: {str(e)}"
            )