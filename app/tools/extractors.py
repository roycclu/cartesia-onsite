from __future__ import annotations

import re


NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

SSN_PATTERN = re.compile(r"\b(\d{4})\b")
POLICY_PATTERN = re.compile(r"\b([A-Z]{2,4}\d?)[-\s]?([0-9OoIi]{3,4})\b", re.IGNORECASE)
SSN_CONTEXT_PATTERN = re.compile(
    r"(?:ssn|social security number|last four(?: digits)?)\D*(\d{4})",
    re.IGNORECASE,
)


def normalize_policy_number(value: str) -> str:
    cleaned = value.lower().replace("-", " ").replace(".", " ")
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    normalized: list[str] = []
    for token in tokens:
        if token in NUMBER_WORDS:
            normalized.append(NUMBER_WORDS[token])
            continue
        if token == "policy":
            continue
        if token == "number":
            continue
        if token == "pol":
            normalized.append("POL")
            continue
        if all(part in NUMBER_WORDS for part in token.split()):
            normalized.extend(NUMBER_WORDS[part] for part in token.split())
            continue
        normalized.append(token.upper())
    collapsed = "".join(normalized)
    if not collapsed:
        return ""
    digits = "".join(char for char in collapsed if char.isdigit())
    if collapsed.startswith("POL"):
        return f"POL{digits}" if digits else "POL"
    if digits:
        return f"POL{digits}"
    return collapsed


def extract_fields(transcript: str) -> dict[str, str | None]:
    lower = transcript.lower()
    split_at = len(transcript)
    for kw in ("social", "ssn", "last four"):
        idx = lower.find(kw)
        if idx != -1:
            split_at = min(split_at, idx)
    policy_section = transcript[:split_at]
    policy_section_clean = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])", "", policy_section)

    policy_number = None
    policy_match = POLICY_PATTERN.search(policy_section_clean)
    if policy_match:
        prefix = policy_match.group(1).upper()
        digits = policy_match.group(2).upper().replace("O", "0").replace("I", "1")
        policy_number = normalize_policy_number(f"{prefix}{digits}")

    ssn_match = SSN_CONTEXT_PATTERN.search(transcript)
    if ssn_match:
        ssn_last4 = ssn_match.group(1)
    elif any(kw in lower for kw in ("ssn", "last four", "social")):
        fallback_matches = SSN_PATTERN.findall(transcript)
        ssn_last4 = fallback_matches[-1] if fallback_matches else None
    else:
        ssn_last4 = None

    return {"policy_number": policy_number, "ssn_last4": ssn_last4}


def predict_intent_fast(partial: str) -> str | None:
    lower = partial.lower()
    if any(word in lower for word in ("claim", "claims", "status")):
        return "get_claim_status"
    if any(word in lower for word in ("policy", "coverage", "deductible")):
        return "get_policy_info"
    return None
