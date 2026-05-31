# Acme Insurance CX Agent — Prototype

**What this is:** A working voice agent prototype for an insurance call center pilot. Handles claim status, policy questions, and human handoff over a real phone call.

**Try it:** Call **(628) 203-1893** · Identify with policy `POL1001` and SSN last 4 `4821`

## Approach

This prototype uses Twilio for telephony, Cartesia Ink-2 for streaming STT with built-in VAD and turn detection, LangGraph for orchestration, Cartesia Sonic for streaming TTS, and PostgreSQL for append-only compliance logging. Audio and control flow stream end-to-end. Ink-2 turn detection removes the need for a local VAD layer, which keeps the telephony pipeline simpler and avoids frame-size edge cases.

The deployment path is Docker-first. Images are built locally or in CodeBuild, pushed to AWS ECR, and referenced by ECS Fargate task-definition scaffolding. Terraform defines the core infrastructure primitives. Secrets come from `.env` locally and SSM Parameter Store in production. Every image is tagged with the git commit hash for rollback and traceability.

## Architecture Notes

PII masking applied at log time. SSNs masked to last 4 digits. Policy numbers masked in transcripts. Calls table stores policy hash not plaintext for audit correlation without PII exposure.

## Structure

```text
app/
  call_state.py            # Per-call state dataclass with lifecycle methods
  call_state_manager.py    # In-memory registry of active calls
  orchestration.py         # LangGraph — intent → tool → response
  prompts.py               # All prompts, centralized and versioned
  tools.py                 # Read-only tools: claim status, policy lookup, handoff

mock_data/                 # Prototype-only simulation of Acme's internal systems
                           # Replaced entirely by Acme's APIs in production

infrastructure/            # Terraform — ECR, ECS cluster, IAM roles
deploy/                    # deploy.sh, ECS task definition, CodeBuild spec
```

## Assumptions

- Audio format assumed 8kHz 16-bit LPCM mulaw from telephony — confirmed with Twilio, to be validated with Genesys in week 1.
- Identity verification scoped to policy number + last 4 SSN only — no DOB or name required for pilot.
- All tool calls are read-only — no write operations in pilot scope.
- Genesys SIP integration mocked by Twilio for prototype — production integration is a week 1 telephony team dependency.
- Mock DB simulates Acme's claims and policy systems — in production replaced by read-only API access to Acme's existing systems.
- Compliance log retention assumed 7 years per standard insurance regulation.
- Single tenant for prototype — multi-tenant isolation required for production.

## Tradeoffs

**Cartesia Ink-2 built-in VAD over local Silero VAD**  
Simplifies pipeline, eliminates frame-size errors on telephony audio.  
*Tradeoff: turn detection timing coupled to STT provider.*

**Explicit tool calls over RAG**  
Deterministic and fully auditable — no hallucination risk on regulated financial data.  
*Tradeoff: every new query type requires a new tool.*

**PostgreSQL append-only over NoSQL for compliance logs**  
Auditor-friendly, relational, queryable post-hoc by regulators.  
*Tradeoff: less scalable than NoSQL at 1500 concurrent — week 5 optimization.*

**Regex field extraction over LLM extraction for identity**  
Reliable policy number and SSN parsing independent of LLM behavior.  
*Tradeoff: brittle on unusual ASR output — normalization layer handles O/0 and format variants.*

## What Changes for Production

- Cartesia public API → self-hosted models inside Acme's AWS VPC
- Twilio → real Genesys SIP integration
- `mock_data/` → read-only API access to Acme's existing systems
- Local PostgreSQL → RDS with immutable compliance logging
- Single EC2 + ngrok → ECS Fargate with ALB, ACM certificate, auto-scaling
- CloudWatch only → LangSmith for full LLM trace observability
- Mutable ECR tags → immutable tags with git commit hash

## Running Locally

```bash
cp .env.example .env  # add your API keys
docker compose up --build
# app runs on http://localhost:8000
```

For Twilio phone testing:

```bash
ngrok http 8000
# set PUBLIC_BASE_URL to ngrok URL
# set Twilio webhook to POST /twilio/voice
```
