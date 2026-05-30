# Insurance Voice Agent Deployment Specification

## 1. System Overview

This system is a pilot voice AI platform for an insurance call center handling inbound policy and claim inquiries through a FastAPI application, Cartesia speech services, LangGraph orchestration, and a compliant audit trail. The pilot scope is limited to read-only servicing workflows, identity verification, and supervised human handoff, with a production trajectory toward hardened telephony integration, VPC-contained services, audited data retention, and enterprise observability suitable for regulated insurance operations.

## 2. Architecture Summary

### Components

- `PSTN / Carrier / CCaaS`: Delivers inbound customer calls into the enterprise telephony edge.
- `Telephony Gateway / Genesys Bridge`: Terminates SIP or provider media, normalizes call metadata, and forwards audio to the voice agent boundary.
- `FastAPI Voice Agent Service`: Primary application service handling session lifecycle, media ingress, orchestration, and outbound responses.
- `Silero VAD Layer`: Detects end-of-turn based on caller silence to control ASR commit timing.
- `Cartesia Ink STT`: Performs streaming speech-to-text on 8 kHz linear PCM call audio.
- `LangGraph Orchestrator`: Manages turn state through intent classification, tool execution, and response generation nodes.
- `Insurance Tooling Layer`: Enforces identity verification and exposes read-only policy and claim lookup functions plus handoff signaling.
- `Cartesia Sonic TTS`: Produces streaming outbound audio from generated response text.
- `SQLite Prototype Store`: Holds seeded mock insurance records, verification records, handoff queue entries, and compliance events.
- `Compliance Logging Layer`: Appends transcript, intent, tool, response, and handoff artifacts for auditability.

### System Boundaries

- Inside AWS VPC:
  - FastAPI voice agent service
  - Orchestration service runtime
  - Tool layer
  - Compliance logger
  - Production relational database replacement for SQLite
  - CloudWatch, Secrets Manager, IAM-based runtime controls
- On-prem or customer-managed telephony domain:
  - PBX, SIP trunk, or Genesys media control components
  - Any pre-authentication service that can enrich call metadata before transfer
- Crossing the boundary:
  - Audio media stream
  - Call start metadata
  - Human handoff control signal
  - Operational telemetry and health checks where approved

### Data Flow Narrative

1. A PSTN caller enters the customer telephony stack and is routed to the insurance voice AI queue.
2. The telephony gateway or WebSocket bridge establishes a session with the FastAPI voice agent service and sends call metadata including caller ID and queue origin.
3. Caller audio is streamed as 8 kHz 16-bit LPCM chunks to the application boundary.
4. The VAD layer accumulates audio and emits an end-of-turn signal after the configured silence threshold.
5. The completed utterance is streamed to Cartesia Ink STT for transcription.
6. The transcript and session state enter the LangGraph workflow for intent classification, tool execution, and response generation.
7. If the user is not yet verified, the orchestration layer requires policy number plus SSN last 4 before any claim or policy retrieval.
8. Read-only tools query the system of record replica for claims or policy information, or log a handoff request when escalation is required.
9. Response text is sent to Cartesia Sonic TTS and returned as streaming audio payloads to the telephony bridge.
10. The telephony bridge plays the response to the caller or executes a handoff to Genesys when the service emits a transfer event.
11. All critical events are appended to the compliance log for post-call audit and investigation.

## 3. DB Schema

Retention policy assumption for all regulated customer interaction data is seven years from call completion unless customer policy or jurisdiction imposes a stricter requirement.

### `claims`

| Field | Type | Constraints | Description |
| --- | --- | --- | --- |
| `claim_id` | `TEXT` | Primary key, not null | Unique claim identifier exposed to the service agent layer. |
| `policy_number` | `TEXT` | Not null, foreign-key logical reference to `policies.policy_number` | Policy associated with the claim. |
| `status` | `TEXT` | Not null | Current claim state such as `Under review`, `Approved`, or `Closed`. |
| `last_updated` | `TEXT` | Not null | ISO-8601 timestamp of last claim update in the source workflow. |
| `adjuster_name` | `TEXT` | Not null | Assigned adjuster display name. |

### `policies`

| Field | Type | Constraints | Description |
| --- | --- | --- | --- |
| `policy_number` | `TEXT` | Primary key, not null | Unique policy identifier. |
| `holder_name` | `TEXT` | Not null | Policy holder full name. |
| `coverage_type` | `TEXT` | Not null | Product or coverage family. |
| `coverage_limit` | `INTEGER` | Not null | Numeric policy limit in whole currency units. |
| `deductible` | `INTEGER` | Not null | Deductible amount in whole currency units. |
| `effective_date` | `TEXT` | Not null | ISO date on which the policy became effective. |

### `verification`

| Field | Type | Constraints | Description |
| --- | --- | --- | --- |
| `policy_number` | `TEXT` | Primary key, not null | Policy identifier used during caller verification. |
| `ssn_last4` | `TEXT` | Not null, length 4 expected | Mock verification factor used before data disclosure. |
| `holder_name` | `TEXT` | Not null | Reference name used for audit and future match expansion. |

### `handoff_queue`

| Field | Type | Constraints | Description |
| --- | --- | --- | --- |
| `id` | `INTEGER` | Primary key, auto-increment | Internal queue row identifier. |
| `session_id` | `TEXT` | Not null | Unique voice session identifier. |
| `reason_code` | `TEXT` | Not null | Normalized escalation reason such as `human_requested_twice` or `llm_error`. |
| `transcript_summary` | `TEXT` | Not null | Brief summary or last-turn text sufficient for routing context. |
| `timestamp` | `TEXT` | Not null | ISO-8601 insertion time. |

### `compliance_log`

| Field | Type | Constraints | Description |
| --- | --- | --- | --- |
| `id` | `INTEGER` | Primary key, auto-increment | Internal sequence identifier preserving event order. |
| `session_id` | `TEXT` | Not null | Unique voice session identifier. |
| `event_type` | `TEXT` | Not null | Event class such as `user_transcript`, `intent_classification`, `tool_call`, `llm_response`, or `handoff`. |
| `content` | `TEXT` | Not null | Serialized event payload captured at write time. |
| `timestamp` | `TEXT` | Not null | ISO-8601 append time. |

### Append-Only Tables

- `compliance_log` is append-only to preserve non-repudiation and audit lineage.
- `handoff_queue` should be operationally append-only in production, with downstream workflow systems creating derived status tables rather than mutating the source escalation record.

## 4. API Contracts

### Endpoint: `GET /health`

- Purpose: Liveness and readiness check for the application container.
- Input: None.
- Output:
  - `status: string`
- Auth mechanism:
  - Prototype: none
  - Production: internal load balancer allowlist or signed service auth
- SLA target: 99.9% monthly availability, `<100 ms` p95 within VPC
- Error / fallback:
  - `503` if dependency health gates fail in production

### Endpoint: `POST /calls/start`

- Purpose: Create or register a call session before media streaming starts.
- Input fields:
  - `session_id: string | optional`
  - `caller_id: string | optional`
  - `queue_origin: string | optional`
  - `account_number: string | optional`
  - `metadata: object | optional`
- Validation:
  - `session_id` must be unique if supplied
  - `caller_id` should be E.164 when present
  - `account_number` accepted only as opaque routing metadata, not proof of identity
- Output fields:
  - `session_id: string`
- Auth mechanism:
  - Prototype: none
  - Production: mTLS or signed service token from telephony gateway
- SLA target: `<250 ms` p95
- Error / fallback:
  - `400` invalid payload
  - `409` duplicate session ID
  - On failure, telephony stack should route directly to human queue

### Endpoint: `POST /demo/text-turn`

- Purpose: Non-telephony test endpoint for deterministic demo turns.
- Input fields:
  - `session_id: string | optional`
  - `transcript: string | required, 1..5000 chars`
- Output fields:
  - `session_id: string`
  - `response_text: string`
  - `verified: boolean`
  - `should_handoff: boolean`
  - `handoff_reason: string | null`
- Auth mechanism:
  - Prototype: none
  - Production: disabled outside non-prod environments
- SLA target: `<2 s` p95 in demo mode, excluding external model latency
- Error / fallback:
  - `400` validation failure
  - `500` internal error, caller should be redirected to human handling path

### Endpoint: `GET /sessions/{session_id}/compliance-log`

- Purpose: Retrieve ordered compliance events for audit or investigation.
- Input fields:
  - `session_id: string` path parameter
- Output fields:
  - Array of objects containing `session_id`, `event_type`, `content`, `timestamp`
- Auth mechanism:
  - Prototype: none
  - Production: role-restricted auditor or break-glass support role only
- SLA target: `<500 ms` p95 for current-day sessions
- Error / fallback:
  - `404` session not found if production lookup is partitioned by tenant
  - `500` storage query failure

### Endpoint: `WS /ws/cartesia/{session_id}`

- Purpose: Streaming media boundary between telephony bridge and agent runtime.
- Inbound message types:
  - `start`
    - `event: string = "start"`
    - `stream_id: string | optional`
    - `config.input_format: string`, expected `pcm_8000` equivalent for production bridge
    - `metadata.caller_id: string | optional`
    - `metadata.queue_origin: string | optional`
    - `metadata.account_number: string | optional`
  - `media_input`
    - `event: string = "media_input"`
    - `stream_id: string`
    - `media.payload: base64 string`, raw 8 kHz 16-bit PCM bytes
  - `custom`
    - `event: string = "custom"`
    - `stream_id: string`
    - `metadata: object`
  - `dtmf`
    - `event: string = "dtmf"`
    - `stream_id: string`
    - `dtmf: string`, one of `0-9`, `*`, `#`
- Outbound message types:
  - `ack`
    - `event: string = "ack"`
    - `stream_id: string`
    - `config: object`
  - `media_output`
    - `event: string = "media_output"`
    - `stream_id: string`
    - `media.payload: base64 string`
    - `text: string`
  - `transfer_call`
    - `event: string = "transfer_call"`
    - `stream_id: string`
    - `transfer.target_phone_number: string`
    - `reason: string`
    - `text: string`
- Auth mechanism:
  - Prototype: none
  - Production: mTLS plus JWT or signed service assertion
- SLA target:
  - Accept to `ack`: `<250 ms` p95
  - End-of-turn to first response audio byte: target `<1500 ms` p95 in pilot
- Error / fallback:
  - Unsupported event: close with protocol error and route caller to fallback queue
  - Internal exception: emit `transfer_call` and terminate session

### Tool Contract: `verify_identity(policy_number, ssn_last4)`

- Input:
  - `policy_number: string`, expected pattern `POL\d{4}` in prototype
  - `ssn_last4: string`, exactly 4 digits
- Output:
  - `verified: boolean`
  - `policy_number: string`
  - `holder_name: string | null`
- Auth mechanism: internal call only from orchestrator
- SLA target: `<100 ms` p95 against local data store
- Fallback:
  - Single failure prompts repeat
  - Repeated failure escalates to human handoff

### Tool Contract: `get_claim_status(policy_number)`

- Input:
  - `policy_number: string`
- Preconditions:
  - Caller must already be verified in the active session
- Output:
  - `claim_id: string`
  - `policy_number: string`
  - `status: string`
  - `last_updated: string`
  - `adjuster_name: string`
  - or `error: string`
- Auth mechanism: internal call only from orchestrator
- SLA target: `<100 ms` p95
- Fallback:
  - Missing record returns controlled error string
  - Any exception triggers handoff

### Tool Contract: `get_policy_info(policy_number)`

- Input:
  - `policy_number: string`
- Preconditions:
  - Caller must already be verified in the active session
- Output:
  - `policy_number: string`
  - `holder_name: string`
  - `coverage_type: string`
  - `coverage_limit: integer`
  - `deductible: integer`
  - `effective_date: string`
  - or `error: string`
- Auth mechanism: internal call only from orchestrator
- SLA target: `<100 ms` p95
- Fallback:
  - Missing record returns controlled error string
  - Any exception triggers handoff

### Tool Contract: `trigger_handoff(reason, session_id)`

- Input:
  - `reason: string`
  - `session_id: string`
- Output:
  - `session_id: string`
  - `reason_code: string`
  - `transcript_summary: string`
  - `timestamp: string`
- Auth mechanism: internal call only from orchestrator or fail-safe handlers
- SLA target: `<150 ms` p95
- Fallback:
  - If queue persistence fails, emit direct telephony transfer and raise production alert

## 5. Telephony Integration Spec

- Expected audio input format:
  - 8 kHz, 16-bit LPCM, mono
  - Transport payload encoded as base64 within WebSocket `media_input` envelopes
- VAD configuration:
  - Engine: Silero VAD
  - End-of-turn threshold: 8 consecutive silent frames in the prototype implementation
  - First silence timeout behavior: prompt caller after two silence windows
  - Second silence timeout behavior: route to human handoff
- SIP / WebSocket boundary assumptions:
  - Customer telephony platform owns SIP trunking, PSTN ingress, and media anchoring
  - Voice agent service accepts normalized WebSocket media and control messages, not raw SIP signaling
  - Any codec transcoding is completed before media reaches the application boundary
- Required call initiation metadata:
  - `caller_id`
  - `queue_origin`
  - `account_number` if pre-authenticated upstream
  - Optional future fields: ANI, DNIS, language preference, customer segment, interaction ID
- Human handoff signal:
  - Method: outbound WebSocket event `transfer_call`
  - Payload:
    - `event: "transfer_call"`
    - `stream_id: string`
    - `transfer.target_phone_number: string`
    - `reason: string`
    - `text: string`
  - Expected Genesys response:
    - `200`-equivalent acknowledgment at the bridge layer
    - Transfer execution status correlated to session ID
    - Failure path routes to default live queue and emits incident metric

## 6. Authentication & Security Spec

- Service boundary authentication:
  - Assumption: mTLS between on-prem or customer telephony bridge and AWS VPC ingress
  - Client certificates rotated by enterprise PKI
  - Only designated bridge identities may open media sessions
- Internal service authentication:
  - One IAM role per service workload
  - No standing credentials on disk
  - Short-lived role credentials only
- Secrets management:
  - AWS Secrets Manager for Cartesia, LLM, and integration credentials
  - Automatic rotation every 90 days or tighter per customer standard
  - Rotation coordinated with deployment health checks before cutover
- User identity verification:
  - The agent must collect policy number and SSN last 4 in conversation before exposing claim or policy data
  - Pre-authenticated metadata may reduce prompts but does not bypass server-side verification policy unless explicitly approved
- Data encryption:
  - In transit: TLS 1.2+ for all service calls; mTLS at network boundary
  - At rest: encrypted block storage and encrypted managed database volumes using customer-approved KMS keys

## 7. Observability & Compliance Spec

### Compliance Log Structure

The compliance log records, at minimum, each user transcript, ASR result, intent classification, tool invocation, generated response, handoff event, and system exception together with session ID and timestamp. Logging occurs synchronously within the turn path for audit completeness, and records are retained for seven years under the stated insurance retention assumption.

### CloudWatch Integration Assumptions

- Structured JSON application logs shipped to CloudWatch Logs
- Custom metrics published for:
  - ASR confidence failures
  - LLM timeouts
  - Handoff counts by reason
  - End-of-turn latency
  - First-byte TTS latency
- Dashboards segmented by environment, tenant, and queue origin

### Alert Conditions

- LLM timeout rate exceeds 2% over 5 minutes
- ASR low-confidence rate exceeds 8% over 15 minutes
- Handoff rate increases 50% above rolling 7-day baseline
- Health endpoint unavailable for 2 consecutive checks
- Compliance log write failures greater than 0 in any 5-minute window

### Audit Query Examples

- Full call transcript reconstruction:
  - `SELECT timestamp, event_type, content FROM compliance_log WHERE session_id = ? ORDER BY id ASC;`
- All handoffs for a day:
  - `SELECT session_id, reason_code, timestamp FROM handoff_queue WHERE timestamp >= ? AND timestamp < ?;`
- Verification failures:
  - `SELECT session_id, timestamp, content FROM compliance_log WHERE event_type = 'identity_verification' AND content LIKE '%"verified": false%';`

## 8. Prototype vs Production Delta

| Prototype (current) | Production (required) |
| --- | --- |
| SQLite file database on local disk | Managed PostgreSQL or Aurora with backups, retention controls, and tenant partitioning |
| Mock insurance records seeded at startup | System-of-record integration or replicated servicing datastore |
| Optional mock STT/TTS fallback when API keys are absent | Mandatory managed credentials, health checks, and controlled failover paths |
| No auth on HTTP or WebSocket endpoints | mTLS, private ingress, service identity, and operator RBAC |
| Demo WebSocket bridge shaped after Cartesia events | Customer-certified telephony gateway integrated with SIP, Genesys, and call routing policies |
| Single-process FastAPI runtime | Horizontally scaled stateless services plus external session and queue stores |
| Local append-only compliance log table | Centralized immutable audit storage with retention enforcement and legal hold support |
| Single-tenant assumptions | Explicit tenant isolation across data, metrics, secrets, and routing |
| Best-effort local observability | CloudWatch dashboards, alerts, traces, and incident response runbooks |
| Inline LLM fallback logic | Approved production model policy, timeout budgets, and prompt governance |

## 9. Open Items for Week 1 Sign-off

- Telephony: Confirm whether inbound media will arrive from Genesys, Twilio, or a customer SIP bridge, and who owns codec normalization to 8 kHz LPCM.
- Telephony: Confirm whether transfer execution should be blind transfer, warm transfer, or queue re-route, and what Genesys acknowledgment payload is expected.
- Telephony: Confirm the authoritative source for `caller_id`, `queue_origin`, and any pre-authenticated `account_number` metadata.
- Infra: Confirm AWS account structure, VPC ingress pattern, certificate authority for mTLS, and whether private connectivity to model providers is required.
- Infra: Confirm production database platform, backup policy, and customer-approved KMS key ownership model.
- Infra: Confirm log retention class, legal hold process, and whether compliance data must be replicated cross-region.
- Engineering: Confirm the production LLM provider, timeout budget, and prompt approval workflow.
- Engineering: Confirm the target system of record for policy and claim lookup, including API latency expectations and read replica strategy.
- Engineering: Confirm whether SSN last 4 is the approved verification factor for pilot and what fallback verification path is required when verification fails.
