from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from compliance import log_event
from tools import get_claim_status, get_policy_info, trigger_handoff, verify_identity


POLICY_PATTERN = re.compile(r"\bPOL-\d{4}\b", re.IGNORECASE)
SSN_PATTERN = re.compile(r"\b(\d{4})\b")
SSN_CONTEXT_PATTERN = re.compile(
    r"(?:ssn|social security number|last four(?: digits)?)(?:\D+)(\d{4})",
    re.IGNORECASE,
)
ALLOWED_INTENTS = {
    "verify_identity",
    "get_claim_status",
    "get_policy_info",
    "handoff",
    "out_of_scope",
    "write_request",
    "unknown",
}


Intent = Literal[
    "verify_identity",
    "get_claim_status",
    "get_policy_info",
    "handoff",
    "out_of_scope",
    "write_request",
    "unknown",
]


class GraphState(TypedDict, total=False):
    session_id: str
    transcript: str
    history: list[dict[str, str]]
    verified: bool
    policy_number: str | None
    ssn_last4: str | None
    intent: Intent
    tool_result: dict[str, Any] | None
    response_text: str
    should_handoff: bool
    handoff_reason: str | None
    llm_error: str | None


class LLMHelper:
    def __init__(self) -> None:
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

    async def classify_intent(self, transcript: str) -> Intent:
        lowered = transcript.lower()
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
        if POLICY_PATTERN.search(transcript) or ("ssn" in lowered and SSN_PATTERN.search(transcript)):
            return "verify_identity"
        if self.client is None:
            return "unknown"
        prompt = (
            "Classify the insurance call center request into one of: "
            "verify_identity, get_claim_status, get_policy_info, handoff, out_of_scope, write_request, unknown. "
            f"Transcript: {transcript}"
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.responses.create(model=self.model, input=prompt)
            text = (response.output_text or "unknown").strip().split()[0]
            return text if text in ALLOWED_INTENTS else "unknown"
        except Exception:
            return "unknown"

    async def generate_response(self, state: GraphState) -> str:
        if self.client is None:
            return self._fallback_response(state)
        prompt = (
            "You are a concise insurance call center voice agent. "
            "Use the state below and produce a short spoken response.\n"
            f"State: {state}"
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.responses.create(model=self.model, input=prompt)
            return (response.output_text or "").strip() or self._fallback_response(state)
        except Exception as exc:
            state["llm_error"] = str(exc)
            return self._fallback_response(state)

    def _fallback_response(self, state: GraphState) -> str:
        if state.get("should_handoff"):
            return "I’m transferring you to a human representative for further help."
        result = state.get("tool_result") or {}
        if state["intent"] == "verify_identity":
            return (
                "Thanks, your identity is verified. How can I help with your claim or policy today?"
                if result.get("verified")
                else "I couldn't verify your identity. Please repeat your policy number and the last four digits of your Social Security number."
            )
        if state["intent"] == "get_claim_status":
            if "error" in result:
                return result["error"]
            if "claim_id" not in result:
                return "Thanks, your identity is verified. How can I help with your claim or policy today?"
            return (
                f"Your claim {result['claim_id']} is {result['status']}. "
                f"It was last updated on {result['last_updated']} by adjuster {result['adjuster_name']}."
            )
        if state["intent"] == "get_policy_info":
            if "error" in result:
                return result["error"]
            if "coverage_type" not in result:
                return "Thanks, your identity is verified. How can I help with your claim or policy today?"
            return (
                f"Your policy is {result['coverage_type']} with a coverage limit of {result['coverage_limit']} dollars "
                f"and a deductible of {result['deductible']} dollars."
            )
        if state["intent"] == "write_request":
            return "I can’t make account changes in this demo, so I’m connecting you with a human representative."
        if state["intent"] == "out_of_scope":
            return "That request is outside this insurance support demo, so I’m transferring you to a human representative."
        return "Please share your policy number and the last four digits of your Social Security number so I can verify your identity."


class InsuranceOrchestrator:
    def __init__(self) -> None:
        self.llm = LLMHelper()
        graph = StateGraph(GraphState)
        graph.add_node("intent_classification", self._classify_intent)
        graph.add_node("tool_execution", self._execute_tools)
        graph.add_node("response_generation", self._generate_response)
        graph.set_entry_point("intent_classification")
        graph.add_edge("intent_classification", "tool_execution")
        graph.add_edge("tool_execution", "response_generation")
        graph.add_edge("response_generation", END)
        self.graph = graph.compile()

    async def run_turn(self, state: GraphState) -> GraphState:
        result = await self.graph.ainvoke(state)
        await log_event(result["session_id"], "graph_result", result)
        return result

    async def _classify_intent(self, state: GraphState) -> GraphState:
        transcript = state["transcript"]
        state["intent"] = await self.llm.classify_intent(transcript)
        state["policy_number"] = state.get("policy_number") or _extract_policy_number(transcript)
        state["ssn_last4"] = state.get("ssn_last4") or _extract_ssn_last4(transcript)
        await log_event(state["session_id"], "intent_classification", {"transcript": transcript, "intent": state["intent"]})
        return state

    async def _execute_tools(self, state: GraphState) -> GraphState:
        intent = state["intent"]
        transcript = state["transcript"]
        policy_number = state.get("policy_number")

        if intent == "handoff":
            state["should_handoff"] = True
            state["handoff_reason"] = "human_requested"
            state["tool_result"] = await trigger_handoff(state["session_id"], "human_requested", transcript)
            return state

        if intent in {"write_request", "out_of_scope"}:
            state["should_handoff"] = True
            state["handoff_reason"] = intent
            state["tool_result"] = await trigger_handoff(state["session_id"], intent, transcript)
            return state

        if not state.get("verified"):
            if policy_number and state.get("ssn_last4"):
                verification = await verify_identity(state["session_id"], policy_number, state["ssn_last4"] or "")
                state["tool_result"] = verification
                state["verified"] = verification["verified"]
                if not verification["verified"]:
                    return state
                if intent == "verify_identity" or _is_verification_only(transcript):
                    state["intent"] = "verify_identity"
                    return state
            else:
                state["tool_result"] = {
                    "verified": False,
                    "message": "Identity verification required before claim or policy access.",
                }
                state["intent"] = "verify_identity"
                return state

        if intent == "get_claim_status":
            state["tool_result"] = await get_claim_status(state["session_id"], policy_number or "")
        elif intent == "get_policy_info":
            state["tool_result"] = await get_policy_info(state["session_id"], policy_number or "")
        elif intent == "verify_identity":
            state["tool_result"] = {"verified": True, "policy_number": policy_number}
        else:
            state["tool_result"] = {"message": "No tool executed."}
        return state

    async def _generate_response(self, state: GraphState) -> GraphState:
        response = await self.llm.generate_response(state)
        state["response_text"] = response
        if state.get("llm_error"):
            state["should_handoff"] = True
            state["handoff_reason"] = "llm_error"
            state["tool_result"] = await trigger_handoff(state["session_id"], "llm_error", state["transcript"])
            state["response_text"] = "I’m having trouble completing that request. I’ll connect you with a human representative."
        await log_event(state["session_id"], "llm_response", {"text": state["response_text"], "handoff": state.get("should_handoff", False)})
        return state


def _extract_policy_number(text: str) -> str | None:
    match = POLICY_PATTERN.search(text)
    return match.group(0).upper() if match else None


def _extract_ssn_last4(text: str) -> str | None:
    match = SSN_CONTEXT_PATTERN.search(text)
    if match:
        return match.group(1)
    if "ssn" not in text.lower() and "last four" not in text.lower():
        return None
    matches = SSN_PATTERN.findall(text)
    if not matches:
        return None
    match_value = matches[-1]
    if match_value.upper().startswith("POL"):
        return None
    return match_value


def _is_verification_only(text: str) -> bool:
    lowered = text.lower()
    request_terms = ("claim", "status", "coverage", "deductible", "policy info", "policy information")
    return not any(term in lowered for term in request_terms)
