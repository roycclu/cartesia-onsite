# Insurance Voice Agent Demo

This prototype is a FastAPI-based voice AI demo for an insurance call center. It accepts streaming call input, detects turn endings with VAD, transcribes speech, routes each turn through a LangGraph workflow, queries a seeded SQLite insurance database, and returns spoken responses. It also logs compliance events and defaults to human handoff on failures.

It now supports two ingress paths:

- Direct demo/testing via HTTP and WebSocket endpoints in this app
- Twilio Programmable Voice via TwiML webhook plus Twilio Media Streams

## What is mocked vs. production

Mocked in this repo:

- The insurance data store uses local SQLite with seeded fake records.
- Identity verification uses a mock `verification` table with policy number plus SSN last 4.
- If `CARTESIA_API_KEY` is missing, STT and TTS fall back to mock behavior so the demo still runs.
- LLM behavior falls back to deterministic rules when `OPENAI_API_KEY` is missing.

Production-facing pieces in this prototype:

- FastAPI app structure for inbound call/session handling.
- WebSocket envelope compatible with Cartesia's current Calls API pattern.
- Cartesia Ink STT WebSocket integration path.
- Cartesia Sonic TTS WebSocket integration path.
- LangGraph orchestration with intent classification, tool execution, and response generation nodes.
- Append-only compliance logging.

## Architecture

Core files:

- [main.py](/home/roy/cartesia_onsite/main.py): FastAPI app, session lifecycle, WebSocket/audio flow, mock text-turn endpoint.
- [orchestration.py](/home/roy/cartesia_onsite/orchestration.py): LangGraph state machine and fallback LLM helpers.
- [tools.py](/home/roy/cartesia_onsite/tools.py): Read-only insurance tools, identity verification, and handoff logging.
- [db.py](/home/roy/cartesia_onsite/db.py): SQLite schema, seeding, and query helpers.
- [compliance.py](/home/roy/cartesia_onsite/compliance.py): Append-only compliance log helpers.
- [vad.py](/home/roy/cartesia_onsite/vad.py): Silero-first VAD wrapper with an RMS fallback when `silero_vad` is unavailable.

Call flow:

1. Client starts a call session with `POST /calls/start` or directly opens `ws://.../ws/cartesia/{session_id}`.
2. Audio chunks arrive as WebSocket `media_input` events or raw bytes.
3. VAD marks end-of-turn after silence accumulation.
4. Audio is transcribed with Cartesia Ink STT.
5. LangGraph runs `intent_classification -> tool_execution -> response_generation`.
6. Tool calls hit the SQLite mock DB only after identity verification passes.
7. Response text is synthesized through Cartesia Sonic TTS and streamed back as `media_output`.
8. Compliance events are appended through the full path.

## Edge Cases Covered

- Two silence timeouts trigger a prompt, then handoff.
- Two low-confidence ASR results trigger handoff.
- Two human requests trigger immediate handoff.
- Out-of-scope or write requests trigger handoff.
- LLM errors trigger fallback messaging plus handoff.
- Unhandled exceptions trigger default handoff.

## How to Run

### 1. Install dependencies

The environment used for this prototype already had the core packages, but a clean environment should install:

```bash
pip install fastapi uvicorn langgraph openai websockets numpy torch onnxruntime
```

Optional for true Silero VAD support:

```bash
pip install silero-vad
```

### 2. Set environment variables

For full external integrations:

```bash
export CARTESIA_API_KEY=...
export CARTESIA_VOICE_ID=...
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export HUMAN_HANDOFF_NUMBER=+15555550199
```

If you skip the API keys, the app still runs in mock mode.

### 3. Start the server

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload
```

### 4. Try the text demo path

```bash
curl -X POST http://127.0.0.1:8000/demo/text-turn \
  -H 'Content-Type: application/json' \
  -d '{"transcript":"My policy number is POL-1001 and the last four of my SSN are 4821"}'
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/demo/text-turn \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<returned_session_id>","transcript":"What is my claim status?"}'
```

### 5. Try the WebSocket path

The WebSocket endpoint is:

```text
ws://127.0.0.1:8000/ws/cartesia/{session_id}
```

It accepts a Cartesia-like message envelope:

```json
{"event":"start","stream_id":"demo-stream","config":{"input_format":"mulaw_8000"}}
```

and then:

```json
{"event":"media_input","stream_id":"demo-stream","media":{"payload":"<base64 audio>"}}
```

### 6. Try the Twilio phone path

Set a public base URL so the webhook can generate the correct `wss://` stream target:

```bash
export PUBLIC_BASE_URL=https://your-public-domain.example.com
```

If you are testing locally, expose the app publicly first, for example with `ngrok`:

```bash
ngrok http 8000
export PUBLIC_BASE_URL=https://your-ngrok-subdomain.ngrok-free.app
```

Configure your Twilio phone number's voice webhook to:

```text
POST https://your-public-domain.example.com/twilio/voice
```

On inbound call, Twilio will request TwiML from `/twilio/voice`, then open a bidirectional Media Stream to:

```text
wss://your-public-domain.example.com/ws/twilio-media
```

Important notes for Twilio:

- Twilio Media Streams uses `audio/x-mulaw` at 8 kHz. The app converts inbound `mulaw` to PCM for VAD/STT and converts TTS PCM back to `mulaw` for playback.
- Live spoken responses require `CARTESIA_API_KEY` because the mock TTS path only emits silence for telephony testing.
- For a real phone demo, set your Twilio number webhook after the app is reachable over public HTTPS/WSS.

## Tradeoffs

- The telephony bridge is demo-focused. It mirrors Cartesia's current WebSocket call event shape rather than implementing a full production phone provider bridge.
- The VAD layer prefers Silero but falls back to RMS energy detection so the prototype stays runnable in restricted environments.
- The orchestration graph is intentionally narrow: three nodes only, matching the spec and keeping behavior easy to inspect.
- The compliance logger uses SQLite rather than PostgreSQL to reduce setup overhead.

## What Changes for a Real VPC Deployment

- Replace SQLite with managed PostgreSQL and move compliance/handoff logs into separate audited schemas.
- Run a dedicated media gateway for telephony provider bridging and jitter buffering.
- Store call audio and transcripts in durable object storage.
- Add auth around all operational endpoints.
- Use a real secrets manager for API keys.
- Split async workers for STT/TTS and handoff actions.
- Add proper observability: traces, structured logs, metrics, alarms, and dead-letter handling.
- Enforce network egress rules and private connectivity for model providers where available.

## Cartesia Docs Used

- Calls/WebSocket API: https://docs.cartesia.ai/line/integrations/websocket-api
- STT WebSocket API: https://docs.cartesia.ai/api-reference/stt/stt
- TTS WebSocket API: https://docs.cartesia.ai/api-reference/tts/websocket

## Word Export

`pandoc` was not installed in this environment. Run `brew install pandoc` then `pandoc SPEC.md -o SPEC.docx` to generate a Word version if you want a pandoc-based export.
