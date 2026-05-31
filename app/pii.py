from __future__ import annotations

import hashlib
import re


def mask_ssn(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) >= 4:
        return f"****{digits[-4:]}"
    return "****"


def mask_policy(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) > 3:
        return f'{value[:3]}{"*" * (len(value) - 3)}'
    return "***"


def mask_transcript(transcript: str | None) -> str | None:
    if not transcript:
        return None
    masked = re.sub(r"\b\d{4}\b", "****", transcript)
    masked = re.sub(r"[A-Z]{2,4}\d{3,4}", lambda m: mask_policy(m.group()) or "***", masked)
    return masked


def hash_policy(policy: str | None) -> str | None:
    if not policy:
        return None
    return hashlib.sha256(policy.encode()).hexdigest()[:16]
