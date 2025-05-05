from fastapi import APIRouter, UploadFile, File, Header, Body, HTTPException
from fastapi.responses import JSONResponse
from uuid import uuid4
from pathlib import Path
import tempfile, logging

from openai import OpenAI

from ..config import OPENAI_API_KEY, WHISPER_OPTIONS
from ..nlp.app import NLPProcessor
from ..transcriber.utils import words_to_digits         # NEW

router = APIRouter()
log    = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)
nlp    = NLPProcessor()

def _resp(payload: dict, sid: str):
    headers = {
        "X-Session-ID": sid,
        "Access-Control-Expose-Headers": "X-Session-ID",
    }
    return JSONResponse(content=payload, headers=headers)

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    x_session_id: str | None = Header(None)
):
    sid = x_session_id or str(uuid4())
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "File must be audio/*")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty file")
        tmp.write(data); tmp.flush()

        try:
            res = client.audio.transcriptions.create(file=open(tmp.name, "rb"), **WHISPER_OPTIONS)
            text = words_to_digits(res.text.strip())          # ← normalise
            out  = await nlp.process_text(text, sid)
            return _resp({"transcription": text, **out, "session_id": sid}, sid)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
            await file.close()

@router.post("/transcribe/text")
async def transcribe_text(
    text: str = Body(..., embed=True),
    x_session_id: str | None = Header(None)
):
    sid = x_session_id or str(uuid4())
    out = await nlp.process_text(words_to_digits(text), sid)
    return _resp({**out, "session_id": sid}, sid)
