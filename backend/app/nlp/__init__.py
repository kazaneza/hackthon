# nlp_context.py
"""Drop‑in replacement for your previous NLP module.

Features
--------
* Keeps the whole conversation (messages, account number, last intent)
  in memory **and** on disk (JSON file) – no DB required.
* Automatically forgets any session that has been inactive for more
  than ``EXPIRY_SECONDS`` (default: 60 s) to free memory.
* Thread‑/process‑safe enough for a single‑worker Uvicorn dev server.
  If you later run multiple workers, switch to Redis or another shared
  store.

Usage
-----
Instantiate **once** at startup and pass the same ``session_id``
(header / query param) on every request from the browser:

>>> nlp = NLPProcessor()
>>> reply = await nlp.process_text(text, session_id)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import HTTPException
from openai import OpenAI

from ..config import OPENAI_API_KEY, DB_CONNECTION_STRING
from ..banking import BankingOperations  # your existing class

# ---------------------------------------------------------------------------
# Globals & constants
# ---------------------------------------------------------------------------

STORE_FILE = "conversation_store.json"  # local JSON file on disk
EXPIRY_SECONDS = 60                      # forget after 1 minute
ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

GPT_OPTIONS = {
    "model": "gpt-4-turbo-preview",
    "temperature": 0.7,
    "max_tokens": 150,
}

SYSTEM_MESSAGE = """You are Alice, a helpful AI banking assistant for Bank of Kigali. Follow these guidelines:

1. ALWAYS maintain context of the conversation
2. If a user asks about balance or transactions:
   - If no account number provided, ask for it politely
   - If account number provided, respond with the actual balance
   - If they provided an account number earlier, use it without asking again
3. NEVER ask \"what would you like to do with this account\" after someone provides their number
4. Keep responses concise but friendly
5. When showing balance:
   - Format amounts as \"RWF X,XXX\"
   - Include last 3 transactions if available
   - Mention transaction dates
"""

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Message(**d)


@dataclass
class Conversation:
    messages: List[Message] = field(default_factory=list)
    account_number: Optional[str] = None
    last_intent: Optional[str] = None
    last_interaction: datetime = field(default_factory=lambda: datetime.utcnow().replace(tzinfo=timezone.utc))

    # --- helpers -----------------------------------------------------------
    def add_message(self, role: str, content: str):
        self.messages.append(Message(role, content))
        self.last_interaction = datetime.utcnow().replace(tzinfo=timezone.utc)

    def is_expired(self) -> bool:
        return (datetime.utcnow().replace(tzinfo=timezone.utc) - self.last_interaction) >= timedelta(seconds=EXPIRY_SECONDS)

    # --- (de)serialisation -------------------------------------------------
    def to_json(self):
        return {
            "messages": [m.to_dict() for m in self.messages],
            "account_number": self.account_number,
            "last_intent": self.last_intent,
            "last_interaction": self.last_interaction.strftime(ISO_FMT),
        }

    @staticmethod
    def from_json(data):
        conv = Conversation()
        conv.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        conv.account_number = data.get("account_number")
        conv.last_intent = data.get("last_intent")
        ts = data.get("last_interaction")
        if ts:
            conv.last_interaction = datetime.strptime(ts, ISO_FMT).replace(tzinfo=timezone.utc)
        return conv


# ---------------------------------------------------------------------------
# Simple file‑based conversation store
# ---------------------------------------------------------------------------

class FileConversationStore:
    """JSON file <-> Conversation objects."""

    def __init__(self, path: str = STORE_FILE):
        self.path = path
        self._conversations: Dict[str, Conversation] = {}
        self._load()

    # -- public API ---------------------------------------------------------

    def get(self, session_id: str) -> Optional[Conversation]:
        self._cleanup()
        return self._conversations.get(session_id)

    def save(self, session_id: str, convo: Conversation):
        self._conversations[session_id] = convo
        self._persist()

    # -- internal helpers ---------------------------------------------------

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, conv_json in data.items():
                self._conversations[sid] = Conversation.from_json(conv_json)
        except Exception as e:
            logger.warning("Failed to load conversation store – starting fresh: %s", e)
            self._conversations = {}

    def _persist(self):
        data = {sid: conv.to_json() for sid, conv in self._conversations.items()}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)  # atomic on POSIX

    def _cleanup(self):
        expired = [sid for sid, conv in self._conversations.items() if conv.is_expired()]
        if expired:
            for sid in expired:
                del self._conversations[sid]
            self._persist()


# ---------------------------------------------------------------------------
# NLP Processor – same public interface, but with file‑backed memory
# ---------------------------------------------------------------------------

class NLPProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.store = FileConversationStore()
        self.banking = BankingOperations(DB_CONNECTION_STRING)

    # ---------------------------------------------------------------------
    # Regex helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def extract_account_number(text: str) -> Optional[str]:
        patterns = [r"\b\d{12}\b", r"\b\d{4}[-\s]?\d{3}[-\s]?\d{4}\b"]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return re.sub(r"[-\s]", "", m.group(0))
        return None

    @staticmethod
    def detect_intent(text: str) -> Optional[str]:
        if re.search(r"balance|statement|transactions|mini.?statement|how much.*have", text, re.I):
            return "balance_inquiry"
        return None

    @staticmethod
    def format_transactions(txs: List[dict]) -> str:
        if not txs:
            return ""
        lines = ["\n\nRecent transactions:"]
        for tx in txs:
            amt = tx["amount"]
            sign = "+" if amt >= 0 else "-"
            lines.append(f"- {tx['date']}: {tx['description']} ({sign}RWF {abs(amt):,.0f})")
        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------------

    async def process_text(self, text: str, session_id: str) -> str:
        try:
            convo = self.store.get(session_id) or Conversation()
            convo.add_message("user", text)

            intent = self.detect_intent(text)
            convo.last_intent = intent  # remember even on error flows

            acct = self.extract_account_number(text)
            if acct:
                convo.account_number = acct

            # --------------------------------------------------------------
            # Handle balance inquiries locally so we don't pay GPT tokens
            # --------------------------------------------------------------
            if intent == "balance_inquiry" or convo.last_intent == "balance_inquiry":
                if convo.account_number:
                    try:
                        txs = self.banking.get_recent_transactions(convo.account_number)
                        balance = txs[0].balance if txs else 0
                        reply = f"Your current balance for account {convo.account_number} is RWF {balance:,.0f}" + \
                                self.format_transactions([tx.to_dict() for tx in txs[:3]])
                    except Exception as e:
                        logger.error("Banking operation error: %s", e)
                        reply = ("I apologize, but I'm having trouble accessing your account "
                                 "information right now. Please try again in a moment.")
                else:
                    reply = "I'll help you check your balance. Could you please provide your account number?"

                # Save and return
                convo.add_message("assistant", reply)
                self.store.save(session_id, convo)
                return reply

            # --------------------------------------------------------------
            # Otherwise fall back to GPT
            # --------------------------------------------------------------
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + [m.to_dict() for m in convo.messages]
            completion = self.client.chat.completions.create(messages=messages, **GPT_OPTIONS)
            reply = completion.choices[0].message.content.strip()

            convo.add_message("assistant", reply)
            self.store.save(session_id, convo)
            return reply

        except Exception as e:
            logger.exception("NLP processing failure")
            raise HTTPException(status_code=500, detail=f"NLP failure: {e}")
