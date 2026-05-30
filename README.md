# Insurance Voice Agent Demo

This prototype is a FastAPI-based voice AI demo for an insurance call center. It accepts streaming call input, transcribes speech with Cartesia Ink-2 turn detection, routes each completed turn through a LangGraph workflow, queries a seeded PostgreSQL insurance database, and returns spoken responses. It also logs compliance events and defaults to human handoff on failures.

It now supports two ingress paths:

- Direct demo/testing via HTTP and WebSocket endpoints in this app
- Twilio Programmable Voice via TwiML webhook plus Twilio Media Streams

## What is mocked vs. production

Mocked in this repo:

- The insurance data store uses PostgreSQL with seeded fake records.
- Identity verification uses a mock `verification` table with policy number plus SSN last 4.
- If `CARTESIA_API_KEY` is missing, STT and TTS fall back to mock behavior so the demo still runs.
- LLM behavior falls back to deterministic rules when `OPENAI_API_KEY` is missing.

Production-facing pieces in this prototype:

- FastAPI app structure for inbound call/session handling.
- WebSocket envelope compatible with Cartesia's current Calls API pattern.
- Cartesia Ink-2 turn-based STT WebSocket integration path.
- Cartesia Sonic TTS WebSocket integration path.
- LangGraph orchestration with intent classification, tool execution, and response generation nodes.
- Append-only compliance logging.

## Architecture

Core files:

- [main.py](/home/roy/cartesia_onsite/main.py): FastAPI app, session lifecycle, WebSocket/audio flow, mock text-turn endpoint.
- [orchestration.py](/home/roy/cartesia_onsite/orchestration.py): LangGraph state machine and fallback LLM helpers.
- [tools.py](/home/roy/cartesia_onsite/tools.py): Read-only insurance tools, identity verification, and handoff logging.
- [db.py](/home/roy/cartesia_onsite/db.py): PostgreSQL schema, seeding, and query helpers.
- [compliance.py](/home/roy/cartesia_onsite/compliance.py): Append-only compliance log helpers.

Call flow:

1. Client starts a call session with `POST /calls/start` or directly opens `ws://.../ws/cartesia/{session_id}`.
2. Audio chunks arrive as WebSocket `media_input` events or raw bytes.
3. Twilio telephony streams raw `mulaw` audio directly to Cartesia Ink-2, which handles VAD and turn detection internally.
4. Completed user turns are transcribed with Cartesia Ink-2 STT.
5. LangGraph runs `intent_classification -> tool_execution -> response_generation`.
6. Tool calls hit the PostgreSQL mock DB only after identity verification passes.
7. Response text is synthesized through Cartesia Sonic TTS and streamed back as `media_output`.
8. Compliance events are appended through the full path.

## Edge Cases Covered

- Two human requests trigger immediate handoff.
- Out-of-scope or write requests trigger handoff.
- LLM errors trigger fallback messaging plus handoff.
- Unhandled exceptions trigger default handoff.

## How to Run

### 1. Install dependencies

The environment used for this prototype already had the core packages, but a clean environment should install:

```bash
pip install fastapi uvicorn langgraph openai websockets
```

### 2. Set environment variables

For full external integrations:

```bash
export CARTESIA_API_KEY=...
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
export DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/voice_agent
export AWS_REGION=us-east-1
export CARTESIA_VOICE_ID=...
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export HUMAN_HANDOFF_NUMBER=+15555550199
```

If you skip the API keys, the app still runs in mock mode.

Local startup loads `.env` automatically when `AWS_EXECUTION_ENV` is not set.

On AWS, startup skips `.env` and loads configuration from SSM Parameter Store instead. The app expects parameters under `SSM_PARAMETER_PREFIX` and maps names directly, for example:

```text
/voice-agent-demo/CARTESIA_API_KEY
/voice-agent-demo/TWILIO_ACCOUNT_SID
/voice-agent-demo/TWILIO_AUTH_TOKEN
/voice-agent-demo/DATABASE_URL
```

### 3. Start the server

```bash
docker compose up --build
```

Or:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
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

- Twilio Media Streams uses `audio/x-mulaw` at 8 kHz. The app forwards inbound `mulaw` directly to Cartesia Ink-2 STT and converts TTS PCM back to `mulaw` for playback.
- Live spoken responses require `CARTESIA_API_KEY` because the mock TTS path only emits silence for telephony testing.
- For a real phone demo, set your Twilio number webhook after the app is reachable over public HTTPS/WSS.

## AWS Deployment

- `deploy/buildspec.yml` builds the image in CodeBuild, writes the current commit hash into `VERSION`, and pushes both `latest` and the commit-tagged image to ECR.
- `deploy/taskdef.json` defines a Fargate task sized at 1 vCPU and 2 GB RAM for ECS and passes `AWS_REGION` plus `SSM_PARAMETER_PREFIX` so the app can load runtime secrets from SSM Parameter Store.
- `deploy/deploy.sh` builds locally, tags the image with `git rev-parse HEAD`, pushes to `voice-agent-demo` in `us-east-1`, registers a new task definition revision, and updates the ECS service.

## Tradeoffs

- The telephony bridge is demo-focused. It mirrors Cartesia's current WebSocket call event shape rather than implementing a full production phone provider bridge.
- Turn detection relies on Cartesia Ink-2 rather than a local VAD layer, which simplifies the telephony pipeline but couples turn timing to the STT provider.
- The orchestration graph is intentionally narrow: three nodes only, matching the spec and keeping behavior easy to inspect.
- The compliance logger is append-only and enforced at the PostgreSQL table level.

## What Changes for a Real VPC Deployment

- Move the demo PostgreSQL schema into managed RDS/PostgreSQL and split compliance/handoff logs into separate audited schemas.
- Run a dedicated media gateway for telephony provider bridging and jitter buffering.
- Store call audio and transcripts in durable object storage.
- Add auth around all operational endpoints.
- Use a real secrets manager for API keys.
- Split async workers for STT/TTS and handoff actions.
- Add proper observability: traces, structured logs, metrics, alarms, and dead-letter handling.
- Enforce network egress rules and private connectivity for model providers where available.

## Cartesia Docs Used

- Calls/WebSocket API: https://docs.cartesia.ai/line/integrations/websocket-api
- STT Turns WebSocket API: https://docs.cartesia.ai/api-reference/stt/turns/websocket
- TTS WebSocket API: https://docs.cartesia.ai/api-reference/tts/websocket

## Word Export

`pandoc` was not installed in this environment. Run `brew install pandoc` then `pandoc SPEC.md -o SPEC.docx` to generate a Word version if you want a pandoc-based export.
