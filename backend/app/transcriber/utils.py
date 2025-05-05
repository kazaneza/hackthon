# app/transcriber/utils.py
"""
Shared helpers for number normalisation.

• words_to_digits("one two three")      -> "123"
• collapse_run("1234567890123456", 14)  -> "12345678901234"
"""
import re

NUMBER_WORD_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

_WORD_RE = re.compile(r"\b(" + "|".join(NUMBER_WORD_MAP) + r")\b", re.I)

def words_to_digits(text: str) -> str:
    return _WORD_RE.sub(lambda m: NUMBER_WORD_MAP[m.group(1).lower()], text)

def collapse_run(digits: str, keep: int = 14) -> str:
    return digits[:keep]
