from __future__ import annotations

import os

import boto3
from dotenv import load_dotenv


DEFAULT_AWS_REGION = "us-east-2"
DEFAULT_SSM_PARAMETER_PREFIX = "/voice-agent-demo/"

REQUIRED_KEYS = (
    "CARTESIA_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "PUBLIC_BASE_URL",
)

OPTIONAL_DEFAULTS = {
    "CARTESIA_VOICE_ID": "a0e99841-438c-4a64-b679-ae501e7d6091",
    "CARTESIA_VERSION": "2026-03-01",
    "OPENAI_MODEL": "gpt-4o-mini",
    "AWS_REGION": DEFAULT_AWS_REGION,
    "SSM_PARAMETER_PREFIX": DEFAULT_SSM_PARAMETER_PREFIX,
    "LLM_TIMEOUT_SECONDS": "8",
}

SSM_MANAGED_KEYS = set(REQUIRED_KEYS) | set(OPTIONAL_DEFAULTS)

CARTESIA_API_KEY: str | None = None
TWILIO_ACCOUNT_SID: str | None = None
TWILIO_AUTH_TOKEN: str | None = None
OPENAI_API_KEY: str | None = None
DATABASE_URL: str | None = None
PUBLIC_BASE_URL: str | None = None
CARTESIA_VOICE_ID: str = OPTIONAL_DEFAULTS["CARTESIA_VOICE_ID"]
CARTESIA_VERSION: str = OPTIONAL_DEFAULTS["CARTESIA_VERSION"]
OPENAI_MODEL: str = OPTIONAL_DEFAULTS["OPENAI_MODEL"]
AWS_REGION: str = OPTIONAL_DEFAULTS["AWS_REGION"]
SSM_PARAMETER_PREFIX: str = OPTIONAL_DEFAULTS["SSM_PARAMETER_PREFIX"]
LLM_TIMEOUT_SECONDS: int = int(OPTIONAL_DEFAULTS["LLM_TIMEOUT_SECONDS"])


def require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. Check your .env file or SSM Parameter Store."
        )
    return value


def optional(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def load_runtime_env() -> None:
    if os.getenv("AWS_EXECUTION_ENV"):
        _load_from_ssm()
    else:
        load_dotenv(override=False)
    _bind_settings()


def _bind_settings() -> None:
    global CARTESIA_API_KEY
    global TWILIO_ACCOUNT_SID
    global TWILIO_AUTH_TOKEN
    global OPENAI_API_KEY
    global DATABASE_URL
    global PUBLIC_BASE_URL
    global CARTESIA_VOICE_ID
    global CARTESIA_VERSION
    global OPENAI_MODEL
    global AWS_REGION
    global SSM_PARAMETER_PREFIX
    global LLM_TIMEOUT_SECONDS

    CARTESIA_API_KEY = require("CARTESIA_API_KEY")
    TWILIO_ACCOUNT_SID = require("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = require("TWILIO_AUTH_TOKEN")
    OPENAI_API_KEY = require("OPENAI_API_KEY")
    DATABASE_URL = require("DATABASE_URL")
    PUBLIC_BASE_URL = require("PUBLIC_BASE_URL")

    CARTESIA_VOICE_ID = optional("CARTESIA_VOICE_ID", OPTIONAL_DEFAULTS["CARTESIA_VOICE_ID"]) or OPTIONAL_DEFAULTS[
        "CARTESIA_VOICE_ID"
    ]
    CARTESIA_VERSION = optional("CARTESIA_VERSION", OPTIONAL_DEFAULTS["CARTESIA_VERSION"]) or OPTIONAL_DEFAULTS[
        "CARTESIA_VERSION"
    ]
    OPENAI_MODEL = optional("OPENAI_MODEL", OPTIONAL_DEFAULTS["OPENAI_MODEL"]) or OPTIONAL_DEFAULTS["OPENAI_MODEL"]
    AWS_REGION = optional("AWS_REGION", OPTIONAL_DEFAULTS["AWS_REGION"]) or OPTIONAL_DEFAULTS["AWS_REGION"]
    SSM_PARAMETER_PREFIX = _normalize_prefix(
        optional("SSM_PARAMETER_PREFIX", OPTIONAL_DEFAULTS["SSM_PARAMETER_PREFIX"])
        or OPTIONAL_DEFAULTS["SSM_PARAMETER_PREFIX"]
    )
    LLM_TIMEOUT_SECONDS = int(
        optional("LLM_TIMEOUT_SECONDS", OPTIONAL_DEFAULTS["LLM_TIMEOUT_SECONDS"])
        or OPTIONAL_DEFAULTS["LLM_TIMEOUT_SECONDS"]
    )


def _load_from_ssm() -> None:
    region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    prefix = _normalize_prefix(os.getenv("SSM_PARAMETER_PREFIX", DEFAULT_SSM_PARAMETER_PREFIX))
    client = boto3.client("ssm", region_name=region)

    loaded_any = False
    next_token: str | None = None
    while True:
        request: dict[str, object] = {
            "Path": prefix,
            "Recursive": False,
            "WithDecryption": True,
        }
        if next_token:
            request["NextToken"] = next_token
        response = client.get_parameters_by_path(**request)
        for parameter in response.get("Parameters", []):
            name = parameter["Name"].rsplit("/", 1)[-1]
            if name in SSM_MANAGED_KEYS:
                os.environ[name] = parameter["Value"]
                loaded_any = True
        next_token = response.get("NextToken")
        if not next_token:
            break

    _load_explicit_parameters(client, loaded_any)


def _load_explicit_parameters(client: object, loaded_any: bool) -> None:
    missing = [key for key in SSM_MANAGED_KEYS if key not in os.environ]
    if not missing and loaded_any:
        return

    prefix = _normalize_prefix(os.getenv("SSM_PARAMETER_PREFIX", DEFAULT_SSM_PARAMETER_PREFIX))
    names = [f"{prefix}{key}" for key in missing]
    if not names:
        return

    for offset in range(0, len(names), 10):
        response = client.get_parameters(Names=names[offset : offset + 10], WithDecryption=True)
        for parameter in response.get("Parameters", []):
            name = parameter["Name"].rsplit("/", 1)[-1]
            if name in SSM_MANAGED_KEYS:
                os.environ[name] = parameter["Value"]


def _normalize_prefix(prefix: str) -> str:
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    if not prefix.endswith("/"):
        prefix = f"{prefix}/"
    return prefix
