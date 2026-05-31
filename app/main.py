from __future__ import annotations

import logging

from fastapi import FastAPI

from app import config
from app.api_routes import router as api_router
from app.app_state import set_orchestrator, set_transcriber, set_tts
from app.orchestration import InsuranceOrchestrator
from app.prompts import PROMPT_VERSION
from app.stt import CartesiaTranscriber
from app.telephony import router as telephony_router
from app.audio import CartesiaTTS
from mock_data.db import close_db, init_db


logging.basicConfig(level=logging.INFO)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)

logger = logging.getLogger("voice_agent")

app = FastAPI(title="Insurance Voice Agent Demo", version="0.1.0")
app.include_router(api_router)
app.include_router(telephony_router)


@app.on_event("startup")
async def startup() -> None:
    config.load_runtime_env()
    logger.info("CONFIG_OK model=%s region=%s prompt_version=%s", config.OPENAI_MODEL, config.AWS_REGION, PROMPT_VERSION)
    await init_db()
    set_orchestrator(InsuranceOrchestrator())
    set_transcriber(CartesiaTranscriber())
    set_tts(CartesiaTTS())


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_db()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
