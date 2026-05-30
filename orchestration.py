from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from compliance import log_event
from tools import get_claim_status, get_policy_info, normalize_policy_number, trigger_handoff, verify_identity

logger = logging.getLogger("voice_agent")

SSN_PATTERN = re.compile(r"\b(\d{4})\b")
POLICY_PATTERN = re.compile(r"\bPOL[-\s]?(\d{4})\b", re.IGNORECASE)
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
    "unknown",
}


Intent = Literal[
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
    verification_attempts: int


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
        if self.client is None:
            return "unknown"
        prompt = (
            "Classify the insurance call center request into one of: "
            "get_claim_status, get_policy_info, handoff, out_of_scope, write_request, unknown. "
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
        prompt = (
            "You are a concise insurance call center voice agent. "
            "The caller has already been verified. "
            "Answer briefly and directly based on the current state. "
            "Do not ask for verification details again. "
            "Use the state below and produce a short spoken response.\n"
            f"State: {state}"
        )
        logger.info("TURN [%s] PROMPT: %s", state["session_id"], prompt)
        if self.client is None:
            response = self._fallback_response(state)
            logger.info("TURN [%s] LLM: %s", state["session_id"], response)
            return response
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.responses.create(model=self.model, input=prompt)
            text = (response.output_text or "").strip() or self._fallback_response(state)
            logger.info("TURN [%s] LLM: %s", state["session_id"], text)
            return text
        except Exception as exc:
            state["llm_error"] = str(exc)
            text = self._fallback_response(state)
            logger.info("TURN [%s] LLM: %s", state["session_id"], text)
            return text

    def _fallback_response(self, state: GraphState) -> str:
        if state.get("should_handoff"):
            return "I’m transferring you to a human representative for further help."
        result = state.get("tool_result") or {}
        if state["intent"] == "get_claim_status":
            if "error" in result:
                return result["error"]
            return (
                f"Your claim {result['claim_id']} is {result['status']}. "
                f"It was last updated on {result['last_updated']} by adjuster {result['adjuster_name']}."
            )
        if state["intent"] == "get_policy_info":
            if "error" in result:
                return result["error"]
            return (
                f"Your policy is {result['coverage_type']} with a coverage limit of {result['coverage_limit']} dollars "
                f"and a deductible of {result['deductible']} dollars."
            )
        if state["intent"] == "write_request":
            return "I can’t make account changes in this demo, so I’m connecting you with a human representative."
        if state["intent"] == "out_of_scope":
            return "That request is outside this insurance support demo, so I’m transferring you to a human representative."
        return "Identity verified. How can I help you today? I can check your claim status or answer policy questions."


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
            {
                "end": END,
                "intent_classification": "intent_classification",
            },
        )
        graph.add_edge("intent_classification", "tool_execution")
        graph.add_edge("tool_execution", "response_generation")
        graph.add_edge("response_generation", END)
        self.graph = graph.compile()

    async def run_turn(self, state: GraphState) -> GraphState:
        result = await self.graph.ainvoke(state)
        await log_event(result["session_id"], "graph_result", result)
        return result

    async def _extract_fields_node(self, state: GraphState) -> GraphState:
        extracted = extract_fields(state["transcript"])
        if extracted["policy_number"]:
            state["policy_number"] = extracted["policy_number"]
        if extracted["ssn_last4"]:
            state["ssn_last4"] = extracted["ssn_last4"]
        await log_event(
            state["session_id"],
            "field_extraction",
            {
                "transcript": state["transcript"],
                "policy_number": state.get("policy_number"),
                "ssn_last4": state.get("ssn_last4"),
            },
        )
        return state

    async def _verification_node(self, state: GraphState) -> GraphState:
        if state.get("verified"):
            return state

        transcript = state["transcript"]
        policy_number = state.get("policy_number")
        ssn_last4 = state.get("ssn_last4")

        if not policy_number or not ssn_last4:
            state["tool_result"] = {
                "verified": False,
                "message": "To verify your identity I need your policy number and the last 4 digits of your Social Security number. Please provide both.",
            }
            state["response_text"] = "To verify your identity I need your policy number and the last 4 digits of your Social Security number. Please provide both."
            return state

        verification = await verify_identity(state["session_id"], policy_number, ssn_last4)
        state["tool_result"] = verification
        state["policy_number"] = verification.get("policy_number", policy_number)

        if verification["verified"]:
            state["verified"] = True
            state["verification_attempts"] = 0
            state["response_text"] = "Identity verified. How can I help you today?"
            return state

        state["verification_attempts"] = state.get("verification_attempts", 0) + 1
        if state["verification_attempts"] >= 3:
            state["should_handoff"] = True
            state["handoff_reason"] = "verification_failed"
            state["tool_result"] = await trigger_handoff(state["session_id"], "verification_failed", transcript)
            state["response_text"] = "I’m transferring you to a human representative."
            return state

        state["response_text"] = "I need your policy number and last 4 digits of your SSN."
        return state

    def _route_after_verification(self, state: GraphState) -> str:
        if not state.get("verified") or state.get("response_text") or state.get("should_handoff"):
            return "end"
        return "intent_classification"

    async def _classify_intent(self, state: GraphState) -> GraphState:
        state["intent"] = await self.llm.classify_intent(state["transcript"])
        await log_event(
            state["session_id"],
            "intent_classification",
            {"transcript": state["transcript"], "intent": state["intent"]},
        )
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

        if intent == "get_claim_status":
            state["tool_result"] = await get_claim_status(state["session_id"], policy_number or "")
        elif intent == "get_policy_info":
            state["tool_result"] = await get_policy_info(state["session_id"], policy_number or "")
        else:
            state["tool_result"] = {"message": "Identity verified. How can I help you today?"}
        return state

    async def _generate_response(self, state: GraphState) -> GraphState:
        response = await self.llm.generate_response(state)
        state["response_text"] = response
        if state.get("llm_error"):
            state["should_handoff"] = True
            state["handoff_reason"] = "llm_error"
            state["tool_result"] = await trigger_handoff(state["session_id"], "llm_error", state["transcript"])
            state["response_text"] = "I’m having trouble completing that request. I’ll connect you with a human representative."
        await log_event(
            state["session_id"],
            "llm_response",
            {"text": state["response_text"], "handoff": state.get("should_handoff", False)},
        )
        return state


def extract_fields(transcript: str) -> dict[str, str | None]:
    policy_match = POLICY_PATTERN.search(transcript)
    policy_number = normalize_policy_number(policy_match.group(0)) if policy_match else None

    ssn_match = SSN_CONTEXT_PATTERN.search(transcript)
    if ssn_match:
        ssn_last4 = ssn_match.group(1)
    else:
        fallback_matches = SSN_PATTERN.findall(transcript)
        ssn_last4 = fallback_matches[-1] if fallback_matches else None

    return {
        "policy_number": policy_number,
        "ssn_last4": ssn_last4,
    }
