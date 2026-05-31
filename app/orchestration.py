from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from app import config
from app.call_state import CallState
from app.compliance import log_event
from app.prompts import (
    END_CONVERSATION_PROMPT,
    GREETING_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    LLM_ERROR_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    PROMPT_VERSION,
    REPEATED_QUERY_INSTRUCTION,
    SYSTEM_PROMPT_VERIFIED,
    VERIFICATION_FAILED_HANDOFF_PROMPT,
    VERIFICATION_PROMPT,
    VERIFICATION_SUCCESS_PROMPT,
    VERIFICATION_SUCCESS_WITH_PENDING,
    WRITE_REQUEST_PROMPT,
)
from app.tools import get_claim_status, get_policy_info, normalize_policy_number, trigger_handoff, verify_identity

logger = logging.getLogger("voice_agent")

SSN_PATTERN = re.compile(r"\b(\d{4})\b")
POLICY_PATTERN = re.compile(r"\b([A-Z]{2,4}\d?)[-\s]?([0-9OoIi]{3,4})\b", re.IGNORECASE)
SSN_CONTEXT_PATTERN = re.compile(
    r"(?:ssn|social security number|last four(?: digits)?)\D*(\d{4})",
    re.IGNORECASE,
)

ALLOWED_INTENTS = {
    "get_claim_status",
    "get_policy_info",
    "handoff",
    "out_of_scope",
    "write_request",
    "end_conversation",
    "unknown",
}


Intent = Literal[
    "get_claim_status",
    "get_policy_info",
    "handoff",
    "out_of_scope",
    "write_request",
    "end_conversation",
    "unknown",
]


class GraphState(TypedDict, total=False):
    session_id: str
    transcript: str
    call_state: CallState
    intent: Intent
    tool_result: dict[str, Any] | None
    response_text: str
    should_handoff: bool
    handoff_reason: str | None
    llm_error: str | None
    repeated_query: bool
    sentence_handler: Callable[[str], Awaitable[None]]


class LLMHelper:
    def __init__(self) -> None:
        self.model = config.OPENAI_MODEL
        self.timeout_seconds = float(config.LLM_TIMEOUT_SECONDS)
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.max_tokens = 80

    def _validate_llm_params(self, params: dict[str, Any], caller: str) -> None:
        if "max_tokens" not in params and "max_output_tokens" not in params:
            logger.warning("LLM_MISSING_MAX_TOKENS caller=%s", caller)

    async def classify_intent(self, transcript: str) -> Intent:
        lowered = transcript.lower()
        if any(phrase in lowered for phrase in ("that's all", "that is all", "all set", "no that's it", "no that is it")):
            return "end_conversation"
        if any(phrase in lowered for phrase in ("representative", "human", "agent")):
            return "handoff"
        if any(phrase in lowered for phrase in ("update my", "change my", "file a claim", "cancel my policy", "pay my bill")):
            return "write_request"
        if any(phrase in lowered for phrase in ("weather", "sports", "restaurant", "flight")):
            return "out_of_scope"
        if "claim" in lowered or "status" in lowered:
            return "get_claim_status"
        if "policy" in lowered or "coverage" in lowered or "deductible" in lowered:
            return "get_policy_info"
        prompt = INTENT_CLASSIFICATION_PROMPT.format(transcript=transcript)
        try:
            params = {"model": self.model, "input": prompt, "max_output_tokens": self.max_tokens}
            self._validate_llm_params(params, "classify_intent")
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.responses.create(**params)
            text = (response.output_text or "unknown").strip().split()[0]
            return text if text in ALLOWED_INTENTS else "unknown"
        except Exception:
            return "unknown"

    async def generate_greeting(self) -> str:
        try:
            params = {"model": self.model, "input": GREETING_PROMPT, "max_output_tokens": self.max_tokens}
            self._validate_llm_params(params, "generate_greeting")
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.responses.create(**params)
            text = (response.output_text or "").strip()
            return text or self._fallback_greeting()
        except Exception:
            return self._fallback_greeting()

    async def generate_verification_prompt(self, attempts: int, recent_history: list[dict[str, str]] | None = None) -> str:
        prompt = VERIFICATION_PROMPT.format(
            attempts=attempts,
            recent_history=recent_history or [],
        )
        try:
            params = {"model": self.model, "input": prompt, "max_output_tokens": self.max_tokens}
            self._validate_llm_params(params, "generate_verification_prompt")
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.responses.create(**params)
            text = (response.output_text or "").strip()
            return text or self._fallback_verification_prompt(attempts)
        except Exception:
            return self._fallback_verification_prompt(attempts)

    async def generate_verification_success(self, holder_name: str | None, pending_intent: str | None = None) -> str:
        if pending_intent:
            prompt = VERIFICATION_SUCCESS_WITH_PENDING.format(
                holder_name=holder_name or "the caller",
                pending_intent=pending_intent.replace("_", " "),
            )
        else:
            prompt = VERIFICATION_SUCCESS_PROMPT.format(holder_name=holder_name or "the caller")
        try:
            params = {"model": self.model, "input": prompt, "max_output_tokens": self.max_tokens}
            self._validate_llm_params(params, "generate_verification_success")
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.responses.create(**params)
            text = (response.output_text or "").strip()
            return text or self._fallback_verification_success(holder_name, pending_intent)
        except Exception:
            return self._fallback_verification_success(holder_name, pending_intent)

    async def generate_response(self, state: GraphState) -> str:
        call_state = state["call_state"]
        repeated_instruction = f"{REPEATED_QUERY_INSTRUCTION} " if state.get("repeated_query") else ""
        prompt = SYSTEM_PROMPT_VERIFIED.format(
            holder_name=(call_state.holder_name or "there").split()[0],
            latest_tool_result=call_state.latest_tool_result or state.get("tool_result") or {},
            repeated_query_instruction=repeated_instruction,
            state=call_state.to_llm_state(),
        )
        logger.info("TURN [%s] PROMPT: %s", state["session_id"], prompt)
        try:
            params = {"model": self.model, "input": prompt, "max_output_tokens": self.max_tokens}
            self._validate_llm_params(params, "generate_response")
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.responses.create(**params)
            text = (response.output_text or "").strip() or self._fallback_response(state)
            logger.info("TURN [%s] LLM: %s", state["session_id"], text)
            return text
        except Exception as exc:
            state["llm_error"] = str(exc)
            text = self._fallback_response(state)
            logger.info("TURN [%s] LLM: %s", state["session_id"], text)
            return text

    async def stream_response(self, state: GraphState, sentence_handler: Callable[[str], Awaitable[None]]) -> str:
        call_state = state["call_state"]
        repeated_instruction = f"{REPEATED_QUERY_INSTRUCTION} " if state.get("repeated_query") else ""
        prompt = SYSTEM_PROMPT_VERIFIED.format(
            holder_name=(call_state.holder_name or "there").split()[0],
            latest_tool_result=call_state.latest_tool_result or state.get("tool_result") or {},
            repeated_query_instruction=repeated_instruction,
            state=call_state.to_llm_state(),
        )
        logger.info("TURN [%s] PROMPT: %s", state["session_id"], prompt)

        if call_state.should_handoff or state.get("intent") == "end_conversation" or state.get("repeated_query"):
            text = self._fallback_response(state)
            await _emit_sentences(text, sentence_handler)
            logger.info("TURN [%s] LLM: %s", state["session_id"], text)
            return text

        try:
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "max_tokens": self.max_tokens,
            }
            self._validate_llm_params(params, "stream_response")
            async with asyncio.timeout(self.timeout_seconds):
                stream = await self.client.chat.completions.create(**params)
                parts: list[str] = []
                sentence_buffer = ""
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    parts.append(delta)
                    sentence_buffer += delta
                    sentences, sentence_buffer = _split_complete_sentences(sentence_buffer)
                    for sentence in sentences:
                        await sentence_handler(sentence)
                if sentence_buffer.strip():
                    await sentence_handler(sentence_buffer.strip())
                    sentence_buffer = ""
            text = "".join(parts).strip() or self._fallback_response(state)
            logger.info("TURN [%s] LLM: %s", state["session_id"], text)
            return text
        except Exception as exc:
            state["llm_error"] = str(exc)
            text = self._fallback_response(state)
            await _emit_sentences(text, sentence_handler)
            logger.info("TURN [%s] LLM: %s", state["session_id"], text)
            return text

    def _fallback_response(self, state: GraphState) -> str:
        call_state = state["call_state"]
        result = state.get("tool_result") or {}
        intent = state.get("intent", "unknown")
        if call_state.should_handoff:
            return VERIFICATION_FAILED_HANDOFF_PROMPT if call_state.handoff_reason == "verification_failed" else OUT_OF_SCOPE_PROMPT
        if state.get("repeated_query"):
            if intent == "get_claim_status" and "claim_id" in result:
                return f"As I mentioned, your claim {result['claim_id']} is {result['status']}."
            if intent == "get_policy_info" and "coverage_type" in result:
                return f"As I mentioned, your policy is {result['coverage_type']} with a deductible of {result['deductible']} dollars."
        if intent == "end_conversation":
            return END_CONVERSATION_PROMPT
        if intent == "get_claim_status":
            if "error" in result:
                return result["error"]
            return (
                f"Your claim {result['claim_id']} is {result['status']}. "
                f"It was last updated on {result['last_updated']} by adjuster {result['adjuster_name']}."
            )
        if intent == "get_policy_info":
            if "error" in result:
                return result["error"]
            return (
                f"Your policy is {result['coverage_type']} with a coverage limit of {result['coverage_limit']} dollars "
                f"and a deductible of {result['deductible']} dollars."
            )
        if call_state.verified:
            return self._fallback_verification_success(None, None)
        return self._fallback_verification_prompt(call_state.verification_attempts)

    def _fallback_verification_prompt(self, attempts: int) -> str:
        if attempts <= 0:
            return "Welcome in. I just need your policy number and the last four of your Social Security number."
        if attempts == 1:
            return "I didn't catch that. Please share your policy number and the last four of your Social Security number."
        return "Sorry for the confusion. One more time: your policy number and last four of your Social Security number, or I'll transfer you."

    def _fallback_verification_success(self, holder_name: str | None, pending_intent: str | None) -> str:
        first_name = (holder_name or "there").split()[0]
        if pending_intent:
            return f"Great, I've got you verified {first_name} — I can help with your {pending_intent.replace('_', ' ')}."
        return f"Great, I've got you verified {first_name} — what can I help you with today?"

    def _fallback_greeting(self) -> str:
        return "Thanks for calling Acme Insurance, how can I help you today?"


class InsuranceOrchestrator:
    def __init__(self) -> None:
        self.llm = LLMHelper()
        graph = StateGraph(GraphState)
        graph.add_node("field_extraction", self._extract_fields_node)
        graph.add_node("verification", self._verification_node)
        graph.add_node("intent_classification", self._classify_intent)
        graph.add_node("tool_execution", self._execute_tools)
        graph.add_node("response_generation", self._generate_response)
        graph.set_entry_point("field_extraction")
        graph.add_edge("field_extraction", "verification")
        graph.add_conditional_edges(
            "verification",
            self._route_after_verification,
            {"end": END, "intent_classification": "intent_classification"},
        )
        graph.add_edge("intent_classification", "tool_execution")
        graph.add_edge("tool_execution", "response_generation")
        graph.add_edge("response_generation", END)
        self.graph = graph.compile()

    async def run_turn(self, state: GraphState) -> GraphState:
        result = await self.graph.ainvoke(state)
        await log_event(result["session_id"], "graph_result", {"prompt_version": PROMPT_VERSION, "state": result["call_state"].to_llm_state()})
        return result

    async def generate_greeting(self) -> str:
        return await self.llm.generate_greeting()

    async def _extract_fields_node(self, state: GraphState) -> GraphState:
        call_state = state["call_state"]
        extracted = extract_fields(state["transcript"])
        call_state.merge_extracted_fields(extracted)
        logger.info(
            "EXTRACT [%s] raw_transcript=%r extracted_policy=%r extracted_ssn=%r",
            state["session_id"],
            state["transcript"],
            extracted.get("policy_number"),
            extracted.get("ssn_last4"),
        )
        await log_event(
            state["session_id"],
            "field_extraction",
            {
                "transcript": state["transcript"],
                "policy_number": call_state.policy_number,
                "ssn_last4": call_state.ssn_last4,
            },
        )
        return state

    async def _verification_node(self, state: GraphState) -> GraphState:
        call_state = state["call_state"]
        if call_state.verified:
            return state

        policy_number = call_state.policy_number
        ssn_last4 = call_state.ssn_last4
        logger.info(
            "VERIFY_CHECK [%s] state_policy=%r state_ssn=%r both_present=%s",
            state["session_id"],
            policy_number,
            ssn_last4,
            bool(policy_number and ssn_last4),
        )

        if not policy_number or not ssn_last4:
            if call_state.verification_attempts >= 3:
                call_state.should_handoff = True
                call_state.handoff_reason = "verification_failed"
                state["should_handoff"] = True
                state["handoff_reason"] = "verification_failed"
                state["tool_result"] = await trigger_handoff(state["session_id"], "verification_failed", state["transcript"])
                state["response_text"] = VERIFICATION_FAILED_HANDOFF_PROMPT
                return state
            state["tool_result"] = {"verified": False}
            state["response_text"] = await self.llm.generate_verification_prompt(
                call_state.verification_attempts,
                call_state.history[-4:],
            )
            call_state.verification_attempts += 1
            return state

        logger.info("VERIFY_CALL [%s] calling verify_identity with policy=%r ssn=%r", state["session_id"], policy_number, ssn_last4)
        verification = await verify_identity(state["session_id"], policy_number, ssn_last4)
        state["tool_result"] = verification
        call_state.policy_number = verification.get("policy_number", call_state.policy_number)
        logger.info(
            "VERIFY_RESULT [%s] verified=%s holder=%r message=%r",
            state["session_id"],
            verification.get("verified"),
            verification.get("holder_name"),
            verification.get("message"),
        )

        if verification["verified"]:
            call_state.verified = True
            call_state.holder_name = verification.get("holder_name")
            call_state.latest_tool_result = verification
            call_state.verification_attempts = 0
            call_state.last_verification_failed = False
            holder_name = verification.get("holder_name")
            pending_intent = call_state.pending_intent
            if pending_intent:
                intent = pending_intent
                transcript = call_state.pending_intent_transcript or state["transcript"]
                call_state.pending_intent = None
                call_state.pending_intent_transcript = None
                state["intent"] = intent
                state["transcript"] = transcript
                logger.info("PENDING_INTENT_RESOLVED [%s] intent=%s", state["session_id"], intent)
                state = await self._execute_tools(state)
                return await self._generate_response(state)
            state["response_text"] = await self.llm.generate_verification_success(holder_name, None)
            return state

        if call_state.verification_attempts >= 3:
            call_state.should_handoff = True
            call_state.handoff_reason = "verification_failed"
            state["should_handoff"] = True
            state["handoff_reason"] = "verification_failed"
            state["tool_result"] = await trigger_handoff(state["session_id"], "verification_failed", state["transcript"])
            call_state.latest_tool_result = state["tool_result"]
            state["response_text"] = VERIFICATION_FAILED_HANDOFF_PROMPT
            return state

        call_state.mark_verification_failed()
        state["response_text"] = await self.llm.generate_verification_prompt(
            call_state.verification_attempts,
            call_state.history[-4:],
        )
        return state

    def _route_after_verification(self, state: GraphState) -> str:
        if state.get("response_text") or state["call_state"].should_handoff:
            return "end"
        return "intent_classification"

    async def _classify_intent(self, state: GraphState) -> GraphState:
        call_state = state["call_state"]
        intent = await self.llm.classify_intent(state["transcript"])
        state["intent"] = intent
        call_state.capture_pending_intent(intent, state["transcript"])
        if call_state.pending_intent == intent and call_state.pending_intent_transcript == state["transcript"]:
            logger.info("PENDING_INTENT captured: %s from: %s", intent, state["transcript"])
        await log_event(state["session_id"], "intent_classification", {"transcript": state["transcript"], "intent": intent})
        return state

    async def _execute_tools(self, state: GraphState) -> GraphState:
        call_state = state["call_state"]
        intent = state["intent"]
        transcript = state["transcript"]

        if intent == "end_conversation":
            call_state.should_close = True
            call_state.resolved = True
            state["tool_result"] = {"message": END_CONVERSATION_PROMPT}
            return state

        if intent == "handoff":
            call_state.should_handoff = True
            call_state.handoff_reason = "human_requested"
            state["should_handoff"] = True
            state["handoff_reason"] = "human_requested"
            state["tool_result"] = await trigger_handoff(state["session_id"], "human_requested", transcript)
            call_state.latest_tool_result = state["tool_result"]
            return state

        if intent in {"write_request", "out_of_scope"}:
            call_state.should_handoff = True
            call_state.handoff_reason = intent
            state["should_handoff"] = True
            state["handoff_reason"] = intent
            state["tool_result"] = await trigger_handoff(state["session_id"], intent, transcript)
            call_state.latest_tool_result = state["tool_result"]
            return state

        if call_state.already_answered(intent):
            state["repeated_query"] = True
            state["tool_result"] = call_state.answered_queries[intent]
            call_state.latest_tool_result = state["tool_result"]
            return state

        prefetched = call_state.prefetched_data.get(intent)
        if prefetched is not None:
            logger.info("PREFETCH [%s] intent=%s hit=true", state["session_id"], intent)
            call_state.record_answered_query(intent, prefetched)
            state["tool_result"] = prefetched
            return state
        if intent in {"get_claim_status", "get_policy_info"}:
            logger.info("PREFETCH [%s] intent=%s hit=false", state["session_id"], intent)

        if intent == "get_claim_status":
            result = await get_claim_status(state["session_id"], call_state.policy_number or "")
            call_state.record_answered_query(intent, result)
            state["tool_result"] = result
        elif intent == "get_policy_info":
            result = await get_policy_info(state["session_id"], call_state.policy_number or "")
            call_state.record_answered_query(intent, result)
            state["tool_result"] = result
        else:
            state["tool_result"] = {"message": "verified"}
        return state

    async def _generate_response(self, state: GraphState) -> GraphState:
        logger.info(
            "LLM_INPUT [%s] verified=%s proceeding_to_intent=%s",
            state["session_id"],
            state["call_state"].verified,
            state["call_state"].verified,
        )
        sentence_handler = state.get("sentence_handler")
        if sentence_handler is not None:
            response = await self.llm.stream_response(state, sentence_handler)
        else:
            response = await self.llm.generate_response(state)
        state["response_text"] = response
        if state.get("llm_error"):
            state["call_state"].should_handoff = True
            state["call_state"].handoff_reason = "llm_error"
            state["tool_result"] = await trigger_handoff(state["session_id"], "llm_error", state["transcript"])
            state["response_text"] = LLM_ERROR_PROMPT
        await log_event(
            state["session_id"],
            "llm_response",
            {"text": state["response_text"], "handoff": state["call_state"].should_handoff},
        )
        return state


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


def _split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    matches = list(re.finditer(r"[^.!?]*[.!?]", buffer))
    if not matches:
        return [], buffer
    sentences = [match.group(0).strip() for match in matches]
    remainder = buffer[matches[-1].end() :]
    return [sentence for sentence in sentences if sentence], remainder


async def _emit_sentences(text: str, sentence_handler: Callable[[str], Awaitable[None]]) -> None:
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if sentence:
            await sentence_handler(sentence)
