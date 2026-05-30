from __future__ import annotations

import os

import boto3
from dotenv import load_dotenv


DEFAULTS = {
    "AWS_REGION": "us-east-1",
    "CARTESIA_VERSION": "2026-03-01",
    "CARTESIA_VOICE_ID": "a0e99841-438c-4a64-b679-ae501e7d6091",
    "DATABASE_URL": "postgresql://postgres:postgres@postgres:5432/voice_agent",
    "HUMAN_HANDOFF_NUMBER": "+15555550199",
    "LLM_TIMEOUT_SECONDS": "8",
    "OPENAI_MODEL": "gpt-4o-mini",
    "SSM_PARAMETER_PREFIX": "/voice-agent-demo/",
}

SSM_MANAGED_KEYS = {
    "AWS_REGION",
    "CARTESIA_API_KEY",
    "CARTESIA_VOICE_ID",
    "CARTESIA_VERSION",
    "DATABASE_URL",
    "HUMAN_HANDOFF_NUMBER",
    "LLM_TIMEOUT_SECONDS",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "PUBLIC_BASE_URL",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
}


def load_runtime_env() -> None:
    if os.getenv("AWS_EXECUTION_ENV"):
        _load_from_ssm()
    else:
        load_dotenv(override=False)
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)


def _load_from_ssm() -> None:
    region = os.getenv("AWS_REGION", DEFAULTS["AWS_REGION"])
    prefix = _normalize_prefix(os.getenv("SSM_PARAMETER_PREFIX", DEFAULTS["SSM_PARAMETER_PREFIX"]))
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

    names = [f"{_normalize_prefix(os.getenv('SSM_PARAMETER_PREFIX', DEFAULTS['SSM_PARAMETER_PREFIX']))}{key}" for key in missing]
    if not names:
        return

    response = client.get_parameters(Names=names, WithDecryption=True)
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
