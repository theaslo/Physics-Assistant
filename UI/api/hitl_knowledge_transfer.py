"""
Human-in-the-loop (HITL) Knowledge Transfer Gate for pilot agents.

Pilot scope:
- forces_agent
- kinematics_agent
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from strands import Agent
    from strands.models.ollama import OllamaModel
    from strands.tools.mcp import MCPClient
    from mcp.client.streamable_http import streamablehttp_client
    STRANDS_AVAILABLE = True
except Exception:
    STRANDS_AVAILABLE = False
    Agent = Any  # type: ignore
    OllamaModel = Any  # type: ignore
    MCPClient = Any  # type: ignore
    streamablehttp_client = None  # type: ignore


logger = logging.getLogger(__name__)


class KnowledgeTransferGate:
    """Coordinates teaching-opportunity detection and MCQ gating."""

    def __init__(
        self,
        database_api_url: str,
        llm_host: Optional[str] = None,
        model_id: Optional[str] = None,
        mcp_host: Optional[str] = None,
        mcp_port: Optional[int] = None,
    ) -> None:
        self.enabled = os.getenv("HITL_GATE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.threshold = float(os.getenv("HITL_GATE_THRESHOLD", "0.70"))
        self.llm_host = llm_host or os.getenv("HITL_LLM_HOST", "http://ds.stat.uconn.edu:11434")
        self.model_id = model_id or os.getenv("HITL_LLM_MODEL_ID", "qwen3:8b-q8_0")
        self.database_api_url = (database_api_url or "http://database-api:8001").rstrip("/")
        self.mcp_host = mcp_host or os.getenv("MCP_KNOWLEDGE_TRANSFER_HOST", "mcp-knowledge-transfer")
        self.mcp_port = int(mcp_port or int(os.getenv("MCP_KNOWLEDGE_TRANSFER_PORT", "10112")))

        pilot_agents_raw = os.getenv("HITL_PILOT_AGENTS", "forces_agent,kinematics_agent")
        self.pilot_agents = {a.strip() for a in pilot_agents_raw.split(",") if a.strip()}

        self._pending_checks: Dict[str, Dict[str, Any]] = {}
        self._mcp_client: Optional[MCPClient] = None
        self._mcp_agent: Optional[Agent] = None
        self._mcp_lock = asyncio.Lock()

    def is_enabled_for(self, agent_id: str) -> bool:
        return self.enabled and agent_id in self.pilot_agents

    def _append_trace_stage(
        self,
        stages: List[Dict[str, Any]],
        stage: str,
        started_at: float,
        **extra: Any,
    ) -> None:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        entry: Dict[str, Any] = {"stage": stage, "duration_ms": duration_ms}
        entry.update(extra)
        stages.append(entry)

    def _finalize_trace(self, operation: str, stages: List[Dict[str, Any]], started_at: float) -> Dict[str, Any]:
        total_ms = int((time.perf_counter() - started_at) * 1000)
        slowest = max(stages, key=lambda x: int(x.get("duration_ms", 0)), default={"stage": "none", "duration_ms": 0})
        return {
            "component": "hitl_gate",
            "operation": operation,
            "total_ms": total_ms,
            "slowest_stage": slowest.get("stage", "none"),
            "slowest_duration_ms": int(slowest.get("duration_ms", 0)),
            "stages": stages,
        }

    async def maybe_create_check(
        self,
        agent_id: str,
        problem: str,
        user_id: str,
        session_id: Optional[str] = None,
        class_identifier: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible wrapper without exposing trace."""
        check, _ = await self.maybe_create_check_with_trace(
            agent_id=agent_id,
            problem=problem,
            user_id=user_id,
            session_id=session_id,
            class_identifier=class_identifier,
        )
        return check

    async def maybe_create_check_with_trace(
        self,
        agent_id: str,
        problem: str,
        user_id: str,
        session_id: Optional[str] = None,
        class_identifier: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Return frontend-safe knowledge check payload if gating is required."""
        trace_started = time.perf_counter()
        trace_stages: List[Dict[str, Any]] = []
        if not self.is_enabled_for(agent_id):
            self._append_trace_stage(trace_stages, "gate_disabled_or_not_pilot", trace_started, agent_id=agent_id)
            return None, self._finalize_trace("maybe_create_check", trace_stages, trace_started)

        stage_started = time.perf_counter()
        concept_tag, rule_conf, reason_tags = self._deterministic_assessment(agent_id, problem)
        self._append_trace_stage(
            trace_stages,
            "deterministic_assessment",
            stage_started,
            concept_tag=concept_tag,
            rule_confidence=round(rule_conf, 4),
        )

        llm_conf = None
        llm_concept = None
        llm_tags: List[str] = []
        skip_llm_reason = self._skip_llm_assessment_reason(problem)
        if (concept_tag is None or rule_conf < self.threshold) and not skip_llm_reason:
            stage_started = time.perf_counter()
            llm_conf, llm_concept, llm_tags = self._llm_assessment(agent_id, problem)
            self._append_trace_stage(
                trace_stages,
                "llm_assessment",
                stage_started,
                llm_confidence=(round(llm_conf, 4) if llm_conf is not None else None),
                llm_concept=llm_concept,
            )
        else:
            stage_started = time.perf_counter()
            self._append_trace_stage(
                trace_stages,
                "llm_assessment_skipped",
                stage_started,
                reason=skip_llm_reason or "deterministic_confidence_met_threshold",
            )
        if llm_concept and concept_tag is None:
            concept_tag = llm_concept
        if llm_tags:
            reason_tags = list(dict.fromkeys([*reason_tags, *llm_tags]))

        stage_started = time.perf_counter()
        combined_conf = self._combine_confidence(rule_conf, llm_conf)
        self._append_trace_stage(
            trace_stages,
            "confidence_combination",
            stage_started,
            combined_confidence=round(combined_conf, 4),
            threshold=self.threshold,
        )
        if concept_tag is None or combined_conf < self.threshold:
            stage_started = time.perf_counter()
            self._append_trace_stage(
                trace_stages,
                "below_threshold_or_no_concept",
                stage_started,
                concept_tag=concept_tag,
                combined_confidence=round(combined_conf, 4),
            )
            return None, self._finalize_trace("maybe_create_check", trace_stages, trace_started)

        stage_started = time.perf_counter()
        kt_payload = await self._fetch_question_from_database(agent_id, concept_tag)
        self._append_trace_stage(
            trace_stages,
            "database_question_fetch",
            stage_started,
            database_payload_received=bool(kt_payload),
        )
        if not kt_payload:
            stage_started = time.perf_counter()
            kt_payload = await self._fetch_question_from_mcp(agent_id, concept_tag, problem)
            self._append_trace_stage(
                trace_stages,
                "mcp_question_fetch",
                stage_started,
                mcp_payload_received=bool(kt_payload),
            )
        if not kt_payload:
            logger.error(
                "HITL_GATE_FALLBACK: no knowledge-transfer question available; continuing without gate. "
                "agent=%s concept=%s user=%s score=%.3f threshold=%.3f",
                agent_id,
                concept_tag,
                user_id,
                combined_conf,
                self.threshold,
            )
            return None, self._finalize_trace("maybe_create_check", trace_stages, trace_started)

        check_id = uuid.uuid4().hex
        now_ts = int(time.time())
        correct_option_id = str(kt_payload.get("correct_option_id", "")).strip()
        if not correct_option_id:
            logger.error(
                "HITL_GATE_FALLBACK: invalid MCP payload (missing correct option); continuing without gate. "
                "agent=%s concept=%s user=%s",
                agent_id,
                concept_tag,
                user_id,
            )
            stage_started = time.perf_counter()
            self._append_trace_stage(
                trace_stages,
                "invalid_mcp_payload",
                stage_started,
                reason="missing_correct_option_id",
            )
            return None, self._finalize_trace("maybe_create_check", trace_stages, trace_started)

        options = kt_payload.get("options", [])
        if not isinstance(options, list) or not options:
            stage_started = time.perf_counter()
            self._append_trace_stage(
                trace_stages,
                "invalid_mcp_payload",
                stage_started,
                reason="missing_or_empty_options",
            )
            return None, self._finalize_trace("maybe_create_check", trace_stages, trace_started)

        self._pending_checks[check_id] = {
            "check_id": check_id,
            "created_at": now_ts,
            "agent_id": agent_id,
            "problem": problem,
            "concept_tag": str(kt_payload.get("concept_tag", concept_tag)),
            "question_id": kt_payload.get("question_id"),
            "question": str(kt_payload.get("question", "")),
            "options": options,
            "correct_option_id": correct_option_id,
            "correct_feedback": str(kt_payload.get("correct_feedback", "Correct.")),
            "incorrect_feedback": str(kt_payload.get("incorrect_feedback", "Let's correct the concept before solving.")),
            "distractor_feedback": kt_payload.get("distractor_feedback", {}) or {},
            "combined_confidence": combined_conf,
            "threshold": self.threshold,
            "rule_confidence": rule_conf,
            "llm_confidence": llm_conf,
            "reason_tags": reason_tags,
            "user_id": user_id,
            "session_id": session_id,
            "class_identifier": class_identifier,
        }

        response_payload = {
            "status": "question_required",
            "check_id": check_id,
            "agent_id": agent_id,
            "concept_tag": str(kt_payload.get("concept_tag", concept_tag)),
            "question_id": kt_payload.get("question_id"),
            "question": str(kt_payload.get("question", "")),
            "options": [{"id": str(opt.get("id", "")), "text": str(opt.get("text", ""))} for opt in options],
            "confidence": round(combined_conf, 4),
            "threshold": self.threshold,
            "reason_tags": reason_tags,
        }
        stage_started = time.perf_counter()
        self._append_trace_stage(trace_stages, "pending_check_stored", stage_started, check_id=check_id)
        return response_payload, self._finalize_trace("maybe_create_check", trace_stages, trace_started)

    def process_answer(
        self,
        check_id: str,
        selected_option_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        class_identifier: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible wrapper without exposing trace."""
        result, _ = self.process_answer_with_trace(
            check_id=check_id,
            selected_option_id=selected_option_id,
            user_id=user_id,
            session_id=session_id,
            class_identifier=class_identifier,
        )
        return result

    def process_answer_with_trace(
        self,
        check_id: str,
        selected_option_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        class_identifier: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Process one-shot answer and return guidance + original problem."""
        trace_started = time.perf_counter()
        trace_stages: List[Dict[str, Any]] = []
        stage_started = time.perf_counter()
        check = self._pending_checks.pop(check_id, None)
        self._append_trace_stage(trace_stages, "pending_check_lookup", stage_started, check_found=bool(check))
        if not check:
            return None, self._finalize_trace("process_answer", trace_stages, trace_started)

        selected = (selected_option_id or "").strip()
        correct = str(check["correct_option_id"]).strip()
        was_correct = selected == correct

        distractor_feedback = check.get("distractor_feedback", {}) or {}
        selected_feedback = str(distractor_feedback.get(selected, "")).strip()

        if was_correct:
            status = "answer_processed"
            remediation = None
            guidance = (
                "Knowledge Check: Correct.\n"
                f"{check['correct_feedback']}\n"
                "Proceeding to the full solution."
            )
        else:
            status = "remediation_required"
            pieces = [
                "Knowledge Check: Not quite.",
            ]
            if selected_feedback:
                pieces.append(selected_feedback)
            pieces.append(str(check["incorrect_feedback"]))
            pieces.append("")
            pieces.append("Before we continue: do you understand why this answer is not correct?")
            pieces.append('If you would like help, reply "next step" and I will guide you one step at a time.')
            guidance = "\n".join(pieces)
            remediation = {
                "prompt": "Do you understand why this answer is not correct?",
                "next_step_prompt": 'Reply "next step" if you want the next step of the solution.',
                "can_continue": False,
            }

        stage_started = time.perf_counter()
        self._log_attempt(
            check=check,
            selected_option_id=selected,
            was_correct=was_correct,
            user_id=user_id,
            session_id=session_id,
            class_identifier=class_identifier,
        )
        self._append_trace_stage(trace_stages, "attempt_logged", stage_started, was_correct=was_correct)

        response_payload = {
            "status": status,
            "check_id": check_id,
            "agent_id": check["agent_id"],
            "original_problem": check["problem"],
            "concept_tag": check["concept_tag"],
            "was_correct": was_correct,
            "guidance": guidance,
            "confidence": check["combined_confidence"],
            "threshold": check["threshold"],
            "reason_tags": check["reason_tags"],
        }
        if remediation:
            response_payload["remediation"] = remediation
        stage_started = time.perf_counter()
        self._append_trace_stage(trace_stages, "guidance_prepared", stage_started, was_correct=was_correct)
        return response_payload, self._finalize_trace("process_answer", trace_stages, trace_started)

    async def cleanup(self) -> None:
        if self._mcp_client:
            try:
                self._mcp_client.stop(None, None, None)
            except Exception as e:
                logger.warning("Knowledge-transfer MCP cleanup warning: %s", e)
        self._mcp_client = None
        self._mcp_agent = None
        self._pending_checks.clear()

    def _combine_confidence(self, rule_conf: float, llm_conf: Optional[float]) -> float:
        if llm_conf is None:
            return max(0.0, min(1.0, rule_conf))
        combined = 0.45 * rule_conf + 0.55 * llm_conf
        return max(0.0, min(1.0, combined))

    def _skip_llm_assessment_reason(self, problem: str) -> Optional[str]:
        lower = problem.lower()
        if (
            "free body" in lower
            or "free-body" in lower
            or "freebody" in lower
            or "fbd" in lower
        ):
            return "explicit_free_body_diagram_request"
        return None

    def _deterministic_assessment(self, agent_id: str, problem: str) -> Tuple[Optional[str], float, List[str]]:
        lower = problem.lower()

        if agent_id == "forces_agent":
            checks = [
                (
                    "newton_first_law",
                    [
                        r"newton'?s?\s*first",
                        r"\binertia\b",
                        r"stays?\s+at\s+rest",
                        r"stays?\s+in\s+motion",
                        r"constant\s+velocity",
                    ],
                    0.80,
                ),
                ("newton_second_law", [r"newton'?s?\s*second", r"\bf\s*=\s*m\s*a\b", r"\bnet force\b"], 0.82),
                (
                    "newton_third_law",
                    [
                        r"newton'?s?\s*third",
                        r"action[- ]reaction",
                        r"equal\s+and\s+opposite",
                    ],
                    0.80,
                ),
                (
                    "hookes_law",
                    [
                        r"hooke",
                        r"\bspring\b",
                        r"\bk\s*=",
                        r"\bf\s*=\s*-?\s*kx\b",
                        r"restoring\s+force",
                    ],
                    0.84,
                ),
                ("incline_components", [r"\bincline\b", r"\binclined\b", r"\bramp\b"], 0.78),
            ]
        elif agent_id == "kinematics_agent":
            checks = [
                ("projectile_components", [r"\bprojectile\b", r"\blaunch\b", r"angle"], 0.83),
                ("projectile_peak", [r"maximum height", r"\bpeak\b"], 0.82),
                ("range_formula", [r"\brange\b", r"maximum x distance"], 0.82),
                ("equation_selection", [r"which (equation|formula)", r"kinematic equation"], 0.80),
                ("sign_convention", [r"sign convention", r"\+y", r"-g"], 0.72),
            ]
        else:
            return None, 0.0, []

        matched: List[Tuple[str, float, str]] = []
        for concept, patterns, base_score in checks:
            for pattern in patterns:
                if re.search(pattern, lower):
                    matched.append((concept, base_score, pattern))
                    break

        if not matched:
            return None, 0.0, []

        concept, score, _ = matched[0]
        reason_tags = [m[0] for m in matched]

        complexity_bonus = 0.0
        if re.search(r"\b(calculate|derive|find|solve|determine)\b", lower):
            complexity_bonus += 0.06
        if re.search(r"\b(max(imum)?|range|components?|equation)\b", lower):
            complexity_bonus += 0.04

        return concept, max(0.0, min(1.0, score + complexity_bonus)), reason_tags

    def _llm_assessment(self, agent_id: str, problem: str) -> Tuple[Optional[float], Optional[str], List[str]]:
        """Guided LLM assessment returning confidence + concept tag."""
        prompt = (
            "You are classifying whether a physics tutoring question should trigger a pre-solution concept check. "
            "Return STRICT JSON with keys: confidence (0..1 float), concept_tag (string), reason_tags (array of strings). "
            "No markdown.\n\n"
            f"Agent: {agent_id}\n"
            f"Question: {problem}\n\n"
            "Allowed concept tags for forces_agent: newton_first_law,newton_second_law,newton_third_law,hookes_law,incline_components\n"
            "Allowed concept tags for kinematics_agent: projectile_components,projectile_peak,range_formula,equation_selection,sign_convention\n"
        )
        try:
            resp = requests.post(
                f"{self.llm_host.rstrip('/')}/api/generate",
                json={
                    "model": self.model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=14,
            )
            if resp.status_code != 200:
                return None, None, []
            data = resp.json()
            raw_text = str(data.get("response", "")).strip()
            parsed = self._extract_json_object(raw_text)
            if not parsed:
                return None, None, []

            conf = parsed.get("confidence")
            concept_tag = parsed.get("concept_tag")
            reason_tags = parsed.get("reason_tags", [])
            conf_value = float(conf)
            conf_value = max(0.0, min(1.0, conf_value))
            if not isinstance(reason_tags, list):
                reason_tags = []
            clean_tags = [str(t) for t in reason_tags if str(t).strip()]
            return conf_value, str(concept_tag).strip() if concept_tag else None, clean_tags
        except Exception:
            return None, None, []

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    async def _fetch_question_from_mcp(self, agent_id: str, concept_tag: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        if not STRANDS_AVAILABLE:
            return None

        await self._ensure_mcp_agent()
        if not self._mcp_agent:
            return None

        payload = {
            "agent_id": agent_id,
            "concept_tag": concept_tag,
            "user_prompt": user_prompt,
        }

        mcp_prompt = (
            "You must call the MCP tool get_guided_knowledge_question first using exactly this JSON payload:\n"
            f"{json.dumps(payload)}\n\n"
            "Return only the tool output text unchanged."
        )

        async with self._mcp_lock:
            try:
                result = await asyncio.wait_for(asyncio.to_thread(self._mcp_agent, mcp_prompt), timeout=45)
            except Exception as e:
                logger.error("HITL_GATE_FALLBACK: knowledge-transfer MCP invoke failed: %s", e)
                return None

        payload = self._extract_kt_payload_from_result(result)
        if payload:
            return payload

        text = self._extract_agent_text(result)
        return self._extract_kt_json(text)

    async def _fetch_question_from_database(self, agent_id: str, concept_tag: str) -> Optional[Dict[str, Any]]:
        """Direct DB fallback keeps HITL visible if MCP orchestration is unavailable."""
        agent_type = agent_id
        if agent_type.startswith("physics_"):
            agent_type = agent_type[len("physics_"):]
        if agent_type.endswith("_agent"):
            agent_type = agent_type[: -len("_agent")]
        if agent_type == "math":
            agent_type = "math_helper"

        def _request() -> Optional[Dict[str, Any]]:
            response = requests.get(
                f"{self.database_api_url}/knowledge-transfer/questions/pick",
                params={"agent_type": agent_type, "concept_tag": concept_tag},
                timeout=8,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None

        try:
            return await asyncio.to_thread(_request)
        except Exception as e:
            logger.error(
                "HITL_GATE_FALLBACK: database question fetch failed: %s agent=%s concept=%s",
                e,
                agent_id,
                concept_tag,
            )
            return None

    async def _ensure_mcp_agent(self) -> None:
        if self._mcp_agent is not None and self._mcp_client is not None:
            return

        mcp_url = f"http://{self.mcp_host}:{self.mcp_port}/mcp"
        try:
            self._mcp_client = MCPClient(
                lambda url=mcp_url: streamablehttp_client(url=url),
                startup_timeout=20,
            )
            self._mcp_client.start()
            tools = self._mcp_client.list_tools_sync()

            model = OllamaModel(
                host=self.llm_host,
                model_id=self.model_id,
                temperature=0.0,
            )
            self._mcp_agent = Agent(
                model=model,
                tools=tools,
                system_prompt=(
                    "Always invoke MCP tools when asked. "
                    "For knowledge-check retrieval, call the provided tool and echo raw tool output."
                ),
                name="knowledge_transfer_gate_agent",
                description="Internal HITL knowledge transfer orchestrator",
            )
        except Exception as e:
            logger.error("HITL_GATE_FALLBACK: unable to initialize mcp-knowledge-transfer: %s", e)
            self._mcp_agent = None
            self._mcp_client = None

    def _extract_agent_text(self, result: Any) -> str:
        output = ""
        if result and getattr(result, "message", None) and result.message.get("content"):
            for block in result.message["content"]:
                if isinstance(block, dict) and "text" in block:
                    output += str(block["text"])
        return output

    def _extract_kt_payload_from_result(self, result: Any) -> Optional[Dict[str, Any]]:
        """Prefer raw tool output over model paraphrases when decoding KT payloads."""
        if result and getattr(result, "message", None) and result.message.get("content"):
            for block in result.message["content"]:
                if not isinstance(block, dict):
                    continue

                tool_result = block.get("toolResult")
                if isinstance(tool_result, dict):
                    content = tool_result.get("content")
                    if isinstance(content, str):
                        payload = self._extract_kt_json(content)
                        if payload:
                            return payload
                    if isinstance(content, list):
                        for content_block in content:
                            if isinstance(content_block, dict) and isinstance(content_block.get("text"), str):
                                payload = self._extract_kt_json(content_block["text"])
                                if payload:
                                    return payload

                if isinstance(block.get("text"), str):
                    payload = self._extract_kt_json(block["text"])
                    if payload:
                        return payload

        return None

    def _extract_kt_json(self, text: str) -> Optional[Dict[str, Any]]:
        start = text.find("KT_JSON_START")
        end = text.find("KT_JSON_END", start + 1) if start >= 0 else -1
        if start < 0 or end < 0:
            return None
        payload_text = text[start + len("KT_JSON_START"):end].strip()
        try:
            parsed = json.loads(payload_text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _log_attempt(
        self,
        check: Dict[str, Any],
        selected_option_id: str,
        was_correct: bool,
        user_id: str,
        session_id: Optional[str],
        class_identifier: Optional[str],
    ) -> None:
        payload = {
            "user_identifier": user_id,
            "session_identifier": session_id or check.get("session_id"),
            "class_identifier": class_identifier or check.get("class_identifier"),
            "agent_type": str(check.get("agent_id", "")).replace("_agent", ""),
            "concept_tag": check.get("concept_tag"),
            "question_id": check.get("question_id"),
            "check_id": check.get("check_id"),
            "confidence_score": check.get("combined_confidence"),
            "threshold_score": check.get("threshold"),
            "was_correct": was_correct,
            "selected_option_id": selected_option_id,
            "metadata": {
                "reason_tags": check.get("reason_tags", []),
                "rule_confidence": check.get("rule_confidence"),
                "llm_confidence": check.get("llm_confidence"),
            },
        }

        try:
            requests.post(
                f"{self.database_api_url}/knowledge-transfer/attempts",
                json=payload,
                timeout=5,
            )
        except Exception as e:
            logger.warning("Failed to log HITL attempt: %s", e)

        try:
            requests.post(
                f"{self.database_api_url}/analytics/realtime/event",
                json={
                    "event_type": "knowledge_transfer_attempt",
                    "user_id": user_id,
                    "agent_type": check.get("agent_id"),
                    "timestamp": int(time.time()),
                    "data": {
                        "concept_tag": check.get("concept_tag"),
                        "was_correct": was_correct,
                        "confidence": check.get("combined_confidence"),
                        "threshold": check.get("threshold"),
                    },
                },
                timeout=4,
            )
        except Exception as e:
            logger.warning("Failed to emit HITL realtime event: %s", e)
