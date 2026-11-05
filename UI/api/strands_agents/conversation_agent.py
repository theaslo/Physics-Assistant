"""Shared MCP-grounded conversation and reflection workflow."""

import asyncio
import json
import math
import re
from typing import Any, Dict, Optional
from force_quiz_reflection import infer_newton_reflection_focus
from tutoring_profiles import MECHANICS_AGENTS, PRINCIPLES_TOOLS, is_guided_reflection
from .base_physics_agent import StrandsPhysicsAgent


MATH_FORMATTING_GUIDANCE = r"""Student-facing formatting: use readable Markdown paragraphs with $...$ for inline math and $$ on separate lines for displayed equations. Never use backticks or code fences for equations.
For a worked numerical solution, show the formula and substitution as displayed equations with one logical step per block. Do not concatenate several equations in one block; use separate blocks or an aligned environment with explicit row breaks. State the final result in a short prose sentence with units and direction if applicable. Keep conceptual follow-ups and hints brief; formatting must not reveal an unrequested solution.
Inside JSON strings, double every LaTeX backslash (for example, \\text{kg} or \\times). Do not encode LaTeX commands as tab, carriage-return, or form-feed escapes.
"""


QUIZ_REFLECTION_GUIDANCE = """You are a forces tutor for both ordinary questions and optional quiz reflection.
Default to direct answers. Use quiz modes ONLY when the student explicitly asks for quiz/test reflection or hints in the current turn or relevant history. An ordinary physics question or prior worked example does not activate quiz mode.
Quiz reflection workflow, only when requested:
If the student explicitly switches to an unrelated problem or requests a full solution, answer that new request directly; earlier quiz/hint instructions do not force a new quiz on it.
When the student requests quiz or test reflection, in this turn or the relevant history, follow their actual quiz rather than starting a general lesson.
If no specific quiz question has been provided, ask them to paste one quiz question together with their original answer and reasoning. Make this one intake request.
A review introduction, including "my answer and reasoning are below", is not itself a quiz question. Do not assume an unrelated earlier example is the quiz they want to review.
Do not start with a generic Newton's-law definition question or choose a misconception before seeing the student's quiz question and reasoning.
If the quiz question is already in the current message or relevant history, use it; do not ask the student to paste it again. Ask only for missing information needed to understand their reasoning.
Do not require the correct answer or an answer key before helping.
Once their question, answer, and reasoning are available, briefly acknowledge their idea without endorsing it and ask one targeted question about that specific situation. Do not give the correct choice, final result, or full explanation in this first reply.
Give only one small hint at a time. Do not reveal the solution and then append "Does that make sense?" or a check question.
While guiding unfinished reasoning, keep each reply to one brief acknowledgement or hint followed by one targeted question. Ask the student to work out the next inference; do not state it for them. Once they have explicitly answered the original quiz question AND justified it, confirm it instead of inventing further steps. A correct intermediate observation alone does not complete the original question.
On later turns, give feedback on the student's actual response before asking the next question. Confirm or correct only the step they attempted, not the final quiz answer. Reveal the final answer only after they derive it or explicitly request a full solution. Keep the same quiz context unless the student changes it.
If they request a new practice question, test the SAME misconception and physical principle in a different, physically consistent situation and withhold the answer until they attempt it. Do not substitute an unrelated forces topic.
Use precise force language: zero net force or zero horizontal force does not mean no forces act. Weight and a supporting normal force may still act vertically. Friction for a crate at rest on a stationary floor is static, not kinetic.
These reflection instructions take priority over general instructions to answer directly or show a complete solution.
"""


REFLECTION_STEP_PROMPT = """Identify the CURRENT student's task, not an earlier request in the history. Return JSON with action, problem_message_index, problem_statement.
Choose action in this order:
1. direct: the current student supplies a NEW problem to solve (especially if they say it is not the quiz), asks for the full solution, or asks for an explanation. A supplied problem is NOT a request for you to generate practice.
2. practice: the student asks YOU to INVENT a different practice exercise. They have not supplied the new exercise to solve. Reference the reviewed problem, never invent its replacement in this routing step.
3. intake: no actual quiz has been supplied, or the student explicitly starts another review and has not supplied its question.
4. hint: the current message supplies the actual quiz question and their original mistaken reasoning for review.
5. feedback: the student is responding to the tutor's question about the existing problem. This includes short words, numbers, equations, uncertainty, and corrected reasoning.
problem_message_index: the integer message index containing the ORIGINAL active problem, NOT the tutor's intermediate hint. A generated practice question becomes the active problem. null if absent.
problem_statement: copy ONLY the active problem's setup and ORIGINAL requested outcome, excluding the student's original answer and reasoning. Do NOT copy the intermediate tutor hint here. Do not solve physics. A short answer is still feedback if the tutor asked a physics question. An earlier review introduction is not the current message.
"""

REFLECTION_FEEDBACK_PROMPT = """Give feedback on the CURRENT student reply to the last tutor question, using the ORIGINAL problem and verified MCP evidence.
Return JSON with assessment (correct, incorrect, partial, unclear, or unassessed), outcome_evidence, reason_evidence, complete (boolean), and question.
The original student's answer can be wrong. Do not treat it as a premise or grade it instead of their latest correction.
A short word, number, direction, or equation can correctly answer an intermediate question. Grade that attempted step, not an unrequested later calculation.
If the CURRENT reply only expresses uncertainty or asks for a hint, use assessment=unclear; there is no new incorrect claim to grade. Do not grade the student's earlier wrong choice again. Ask one smaller prerequisite question without revealing the solution.
assessment is correct when the LAST TUTOR QUESTION was answered correctly, even when complete is false. Do NOT use partial just because the ORIGINAL quiz remains unfinished. Use partial only if the LAST TUTOR QUESTION itself requested several things and the reply supplied only some of them.
Complete means the student has explicitly supplied BOTH the ORIGINAL requested result/prediction AND a correct reason. It never means just that the latest tutor sub-question is answered.
outcome_evidence and reason_evidence must be exact quotes from student_corrected_work, not from the original mistaken submission or tutor explanations. Use empty strings for missing evidence.
For a motion prediction, stating a net force or acceleration alone is NOT the requested motion prediction. For a requested numerical result, an equation alone is NOT the result.
If complete is true, supply both evidence quotes and leave question empty; do not invent a further task. Otherwise give ONE next-step question ending in '?', without revealing the missing inference or final answer.
For a correct intermediate reply, acknowledge progress by asking the NEXT inference; do not repeat a question already answered or ask them to defend an old wrong belief.
For an incorrect reply, ask a smaller prerequisite question to help them reconsider. Never reveal the full solution unless explicitly requested.
For a new quiz submission, assessment is unassessed; ask a concrete prerequisite about this situation, not a generic law recital or a repeat of the student's supplied reasoning.
Check the premise of your question against physics. Zero NET force does not mean that no forces act. Do not invent forces or change the active problem.
"""


class ClarificationRequest(str):
    """A verified request for missing givens, rather than a numerical result."""


class MissingGivens(Exception):
    def __init__(self, question, tools_used):
        super().__init__(question)
        self.tools_used = list(dict.fromkeys(tools_used))


class ConversationPhysicsAgent(StrandsPhysicsAgent):
    tutoring_guidance = ""
    check_in_guidance = """
For a multiple-choice check-in, ask which physical principle or system assumption
applies to the given situation. Do not offer equivalent general and special
forms of an equation as competing options. Exactly one option must be correct
under the stated conditions; do not supply the final numerical result. Keep
MCP, tool names, tool selection and internal verification out of student text.
"""
    worked_solution_required = False
    verify_numerical_completeness = False

    async def prepare_contextual_check(self, question, context):
        """Let the model choose a relevant conceptual check, never a generic question bank."""
        if (context or {}).get("full_solution_requested"):
            return None
        # A direct explanation should not become a quiz merely because it names
        # a different physical setup from the previous turn.
        if (re.search(r"(?:^|[.!?]\s+)(?:why|is|are|does|do|can|would|could|should|explain)\b", question, re.I)
                and not re.search(r"\b(?:calculate|find|determine|compute|how much|how many)\b", question, re.I)):
            return None
        history = self._conversation_messages(context)
        if history:
            intent = await self._reflection_model(
                json.dumps({"history": history, "current_student_message": question}),
                "Classify conversational intent only; do not solve physics or write a quiz. "
                "Return JSON with new_problem (boolean). TRUE only when the current student supplies a distinct "
                "new physical setup requiring a calculation or multi-step solution, not when referring to an already discussed setup. "
                "A direct conceptual question asking for an explanation, definition, or yes/no clarification is FALSE; "
                "answer it conversationally rather than converting the question into a quiz. "
                "FALSE for asking why, asking about conservation/assumptions in THAT problem, changing a condition "
                "of the same problem, returning to an earlier problem, answering the tutor, requesting hints, "
                "or asking you to generate a practice exercise. A different question about the SAME situation "
                "is a continuation, not a new problem. Use the history to resolve references.", tokens=180,
            )
            if type(intent.get("new_problem")) is not bool:
                raise RuntimeError("The tutor could not identify this conversation step. Please try again.")
            if not intent["new_problem"]:
                return None
        result = await asyncio.wait_for(
            asyncio.to_thread(self._call_mcp_tool_direct, self.principles_tool, {}),
            timeout=self.fast_mcp_explain_timeout_seconds,
        )
        evidence = self._extract_text_from_mcp_tool_result(result)
        if not evidence or self._is_mcp_error_text(evidence) or result.get("status") == "error":
            raise RuntimeError("The tutor could not verify the physics for this check-in. Please try again.")
        check = await self._reflection_model(
            json.dumps({"current_question": question, "history": history}),
            "Decide if the CURRENT student request needs a brief conceptual check before solving. "
            "Return JSON with needs_check (boolean). Set false for a clarification, a response to the last tutor "
            "question (including one word, number, or equation), quiz reflection/review, requests for hints, "
            "a practice-question request, a greeting, or an explicit request for the full solution. "
            "Do not repeat a check already answered for this problem. A genuinely NEW quantitative problem "
            "can need a check even when earlier problems are in history. "
            "Use a check when choosing a physical principle, equation, system, or conservation assumption "
            "would help solve a NEW calculation or multi-step application problem. "
            "For a direct conceptual question, definition, or yes/no explanation, set false: answer it conversationally, "
            "do not turn that same question into a quiz. "
            "The check must refer to THIS setup and its givens, not a generic definition. "
            "If true, also return confidence (0 to 1), concept_tag, question, options (3 or 4 objects with id and text), "
            "and correct_option_id. Exactly one option must be correct under the stated conditions. "
            "Ask about the method or physical reasoning, NOT arithmetic or the final numerical answer. "
            "Do not reveal the correct option in the question. Options must be distinct and plausible. "
            "If necessary assumptions are absent, set false so the tutor can ask for them instead. "
            + self.tutoring_guidance + self.check_in_guidance
            + "\nDo not invent input values. Ground the question and answer key in these verified MCP principles:\n" + evidence,
            tokens=1100,
        )
        if type(check.get("needs_check")) is not bool:
            raise RuntimeError("The tutor could not prepare the check-in. Please try again.")
        if not check["needs_check"]:
            return None
        check = self._validate_contextual_check(check)
        return await self._review_contextual_check(question, check, evidence)

    def _validate_contextual_check(self, check):
        options = check.get("options")
        if (check.get("needs_check") is not True or not isinstance(options, list) or not 3 <= len(options) <= 4
                or not all(isinstance(o, dict) and isinstance(o.get("id"), str) and o["id"].strip()
                           and isinstance(o.get("text"), str) and o["text"].strip() for o in options)
                or len({o["id"] for o in options}) != len(options)
                or len({o["text"].strip().casefold() for o in options}) != len(options)
                or check.get("correct_option_id") not in {o["id"] for o in options}
                or not isinstance(check.get("question"), str) or not check["question"].strip()
                or not isinstance(check.get("concept_tag"), str) or not check["concept_tag"].strip()
                or type(check.get("confidence")) not in (int, float) or not 0 <= check["confidence"] <= 1):
            raise RuntimeError("The tutor could not validate the check-in. Please try again.")
        check["options"] = [{"id": o["id"], "text": o["text"]} for o in options]
        if self._mentions_internal_tools(check["question"] + "\n" + "\n".join(o["text"] for o in options)):
            raise RuntimeError("The tutor could not prepare a student-facing check-in. Please try again.")
        return check

    async def _review_contextual_check(self, question, check, evidence):
        for attempt in range(2):
            review = await self._reflection_model(
                json.dumps({"original_problem": question, "check": check}),
                "Independently review this multiple-choice physics check against the original problem and verified principles. "
                "Return JSON with correct_option_ids (array of ALL physically correct option IDs) and issue (string). "
                "Evaluate EVERY option, not just the supplied answer key. General and special forms of the same "
                "conservation equation can both be valid. No option may require unstated assumptions or invented inputs. "
                "Every option must answer the same precise quantity or decision asked in the question. "
                "Reject mixed referents (for example work BY versus work ON the system in different choices), "
                "or an abstract definition that fails to refer to the original physical setup. "
                "Set correct_option_ids=[] if the question is ambiguous.\n" + self.tutoring_guidance
                + "\nVerified principles:\n" + evidence,
                tokens=400,
            )
            if review.get("correct_option_ids") == [check["correct_option_id"]]:
                return check
            if attempt == 0:
                check = await self._reflection_model(
                    json.dumps({"original_problem": question, "rejected_check": check, "review": review}),
                    "Rewrite the check to remove ambiguity and have exactly ONE correct option. "
                    "If several listed laws are valid, the QUESTION is too broad: replace it with a NEW narrow question "
                    "about one sign, process condition, or system assumption. Do not reuse the rejected question or options. "
                    "All distractors must be physically false under the stated conditions, not merely less specific. "
                    "Ask about the physical principle or system assumption for the original setup, not equations or arithmetic. "
                    "Return JSON with needs_check=true, confidence (0 to 1), concept_tag, question, "
                    "options (3 or 4 objects with id and text), and correct_option_id. "
                    "Do not reveal the final answer.\n" + self.check_in_guidance + self.tutoring_guidance
                    + "\nVerified principles:\n" + evidence,
                    tokens=900,
                )
                check = self._validate_contextual_check(check)
        raise RuntimeError("The tutor could not prepare an unambiguous check-in. Please try again.")

    @property
    def principles_tool(self):
        return PRINCIPLES_TOOLS[self.agent_id]

    def _feedback_instructions(self):
        if self.agent_id == "forces_agent":
            return REFLECTION_FEEDBACK_PROMPT + MATH_FORMATTING_GUIDANCE
        return REFLECTION_FEEDBACK_PROMPT.replace(
            "Zero NET force does not mean that no forces act. Do not invent forces or change the active problem.",
            "Respect the system boundaries and conservation assumptions. Do not invent inputs, add unrelated tasks, or change the active problem.",
        ) + self.tutoring_guidance + MATH_FORMATTING_GUIDANCE

    async def _solve_conversation_turn(
        self, problem: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            return await self._run_conversation_turn(problem, context)
        except MissingGivens as clarification:
            return {"solution": str(clarification), "tools_used": clarification.tools_used, "diagram": None}

    async def _run_conversation_turn(
        self, problem: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Let the model select the active problem without keyword-based answer fallbacks."""
        question = self._current_followup_text(problem, context)
        history = self._conversation_context_text(problem, context, include_current=False)
        if not history and "Previous conversation:" in problem:
            history = problem.split("Previous conversation:", 1)[1].split("Current follow-up question:", 1)[0]

        # Principles are already executed below; they must not masquerade as a numerical tool.
        catalog = [tool for tool in self._fast_mcp_tool_catalog() if tool["name"] != self.principles_tool]
        tools_used = []
        tool_outputs = []
        completed_requests = []
        diagrams = []

        async def call_tool(name: str, arguments: Dict[str, Any]) -> None:
            requested = {"tool": name, "arguments": arguments}
            result = await asyncio.wait_for(
                asyncio.to_thread(self._call_mcp_tool_direct, name, arguments),
                timeout=self.fast_mcp_explain_timeout_seconds,
            )
            output = self._extract_text_from_mcp_tool_result(result)
            if (self.verify_numerical_completeness and name != self.principles_tool
                    and output and (self._is_mcp_error_text(output) or result.get("status") == "error")):
                repair = await self._reflection_model(
                    json.dumps({"current_question": question, "history": history,
                                "rejected_call": {"tool_name": name, "arguments": arguments},
                                "validation_error": output,
                                "completed_calculations": tool_outputs[1:]}),
                    "Repair the argument contract for this rejected calculation, using only the original givens "
                    "and completed calculations. Return JSON with tool_calls, containing ONE corrected call to the SAME tool. "
                    "Do not invent inputs or alter given physical quantities. For a genuinely unknown JSON field, "
                    "omit it or use null; never insert a placeholder zero or guessed result. Actual given zero heat, "
                    "work or temperature changes must remain zero. If genuinely required givens are missing, "
                    "return tool_calls=[] and clarification_question as ONE question asking for them. An unknown "
                    "that can be calculated from the supplied data is not a missing given. If supplied values are "
                    "invalid and no clarification is possible, return tool_calls=[]. "
                    "Do not answer the student or treat a failed call as verification.\nCatalog:\n"
                    + json.dumps([tool for tool in catalog if tool["name"] == name]),
                    tokens=700,
                )
                corrected_calls = self._extract_fast_mcp_tool_calls(repair, catalog)
                clarification = repair.get("clarification_question")
                if (repair.get("tool_calls") == [] and isinstance(clarification, str) and "?" in clarification
                        and not self._mentions_internal_tools(clarification)
                        and not self._has_new_numbers(clarification, question + "\n" + history)):
                    raise MissingGivens(self._one_question(clarification), tools_used)
                if (len(corrected_calls) == 1 and corrected_calls[0]["tool_name"] == name
                        and corrected_calls[0]["arguments"] != arguments
                        and not self._calculation_input_error(corrected_calls[0])):
                    arguments = corrected_calls[0]["arguments"]
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self._call_mcp_tool_direct, name, arguments),
                        timeout=self.fast_mcp_explain_timeout_seconds,
                    )
                    output = self._extract_text_from_mcp_tool_result(result)
            if not output or self._is_mcp_error_text(output) or result.get("status") == "error":
                raise RuntimeError("The tutor could not verify this calculation. Please check the given quantities and try again.")
            tools_used.append(name)
            completed_requests.append(requested)
            tool_outputs.append({"tool": name, "arguments": arguments, "result": self._strip_embedded_diagram_json(output)})
            diagram = self._extract_diagram_from_text(output)
            if diagram:
                diagrams.append(diagram)

        # Conceptual questions need physics grounding without invented numerical inputs.
        await call_tool(self.principles_tool, {})
        reflection_turn = bool((context or {}).get("guided_tutoring")) or is_guided_reflection(self.agent_id, question + "\n" + history)
        if reflection_turn and not (context or {}).get("full_solution_requested"):
            reflection = await self._solve_reflection_turn(
                question, self._conversation_messages(context), catalog, tool_outputs, call_tool,
            )
            if reflection is not None:
                answer, stage = reflection
                return {
                    "solution": answer, "tools_used": list(dict.fromkeys(tools_used)), "diagram": None,
                    "quiz_reflection_mode": {
                        "enabled": True, "stage": stage, "mcp_tool_called": True,
                        "concept_tag": (infer_newton_reflection_focus(question + "\n" + history)
                                        if self.agent_id == "forces_agent" else self.agent_id.removesuffix("_agent")),
                    },
                }
        prompt = (
            "You are a conversational physics tutor. Interpret the CURRENT QUESTION using the history only when relevant.\n"
            "The history may contain several unrelated problems. Decide which situation the student means now. "
            "A new object or situation starts a new problem, even in a short or misspelled question. "
            "Do not transfer masses, spring constants, angles, coefficients, or motion directions from a different problem. "
            "For a genuine follow-up, preserve the relevant setup; apply any changed conditions the student gives. "
            "Explicit references to an earlier problem can return to that problem. If ambiguous, ask one specific question.\n"
            "For ordinary questions outside quiz reflection, answer the current question directly and briefly; "
            "do not repeat a full previous solution. "
            "Say yes or no only when the student proposes a claim, and ensure it agrees with your explanation. "
            "For a direction question, do not require numbers that cannot affect the direction. "
            "If motion direction is unknown, explain the conditional directions and ask which way it is moving. "
            "Distinguish zero NET force from no forces acting: weight and a supporting normal force can cancel vertically. "
            "When discussing frictionless horizontal motion, specify zero horizontal force, not absence of all forces. "
            "For quiz reflection or hints, respect the student's requested pace and ask at most one question.\n"
            "A current explicit request for the full solution overrides an earlier hint-only preference; give the requested explanation.\n"
            "Use MCP principles for conceptual answers and MCP calculation tools for numerical results. "
            "Explain a conceptual follow-up using those principles rather than replaying the earlier calculation. "
            "Keep the system boundary consistent: redistributing momentum among its objects does not increase the system's total mass. "
            "Never invent inputs to satisfy a tool requirement. Tool results below apply ONLY to their stated arguments. "
            "Do not mention tools or implementation details in the student answer.\n"
            "Return JSON with these fields: "
            '{"answer":"student-facing answer, or empty if calculations are needed",'
            '"tool_calls":[{"tool_name":"catalog name","arguments":{}}],"show_diagram":false}. '
            "A request for missing details or a tutoring question is a valid answer. "
            "When tool_calls is empty, answer must contain a nonempty reply to the student. "
            "If you need a calculation, request up to three tools and wait for their results before answering. "
            "Request dependent calculations in separate rounds. If no calculation is needed, answer using the principles. "
            "The Verified MCP results are completed tool calls, not proposed calls. "
            "If they already contain the calculation needed for the current question, use that result in answer "
            "and return tool_calls=[]. Do not request the same tool with the same arguments again. "
            "For quiz intake with no quiz question yet, put the request for the student's quiz details in answer "
            "and return tool_calls=[] and show_diagram=false. No numerical calculation is needed for intake. "
            "Set show_diagram=true only if the student requests a diagram and a relevant tool produced one.\n\n"
            + self.tutoring_guidance
            + ("\nFor a numerical solution, also return worked_solution, an object with nonempty string fields "
               "givens, principle, equation, substitution, result, and check. Use the executed MCP calculations. "
               "equation and substitution must be raw LaTeX equations without dollar delimiters. "
               "result must include units and direction when applicable. check must verify the result or explain its meaning. "
               "Use physics language only: do not mention MCP, tools, tool names, verified results, or internal checks. "
               "The check should discuss physical units, direction, assumptions, or conservation. "
               "Complete the requested calculation and any supplied verification; do not ask permission to finish it. "
               "A short summary in answer alone is insufficient for a worked numerical solution.\n"
               if self.worked_solution_required else "")
            + "\n"
            f"Tool catalog: {json.dumps(catalog, ensure_ascii=True)}\n\n"
        )
        for _ in range(3):
            turn_prompt = (
                prompt
                + f"\n\nConversation history (oldest first):\n{history}\n\n"
                + f"CURRENT QUESTION:\n{question}\n\n"
                + "\nVerified MCP results (already executed):\n" + json.dumps(tool_outputs, ensure_ascii=True)
                + "Respond only to this current turn. If the student requested quiz reflection or hints "
                "in this turn or the relevant history, follow that pace instead of giving a full solution."
                " If a tool result above answers the current question, give the answer with tool_calls=[] "
                "rather than requesting that same calculation again."
            )
            try:
                response = await asyncio.to_thread(
                    self._request_ollama_generate,
                    turn_prompt,
                    self.fast_mcp_explain_timeout_seconds,
                    900,
                    True,
                    self.model_id,
                    think=False if self.model_id.startswith("qwen3") else None,
                    system=("You are a conversational physics tutor. Answer the current question directly, using only relevant history and verified MCP results. "
                            "Keep the same system boundary before and after an event. Distinguish redistribution among its objects from a change in a conserved total. "
                            "Your explanation must agree with this MCP reference, including its qualifications, rather than a generic remembered explanation.\n"
                            + MATH_FORMATTING_GUIDANCE
                            + "\nVerified conceptual MCP reference:\n" + tool_outputs[0]["result"]),
                )
            except Exception as exc:
                raise RuntimeError(
                    "The tutor could not reach the language model to interpret your current question. Please try again."
                ) from exc
            parsed = self._parse_llm_json_object(response)
            if not parsed:
                raise RuntimeError("The tutor could not interpret this question. Please try again.")
            calls = self._extract_fast_mcp_tool_calls(parsed, catalog)
            if calls:
                fresh_calls = [call for call in calls if not any(
                    self._same_calculation(output, call) for output in tool_outputs + completed_requests
                )]
                if fresh_calls:
                    for call in fresh_calls:
                        if not any(self._same_calculation(output, call) for output in tool_outputs + completed_requests):
                            await call_tool(call["tool_name"], call["arguments"])
                    continue
                # A repeated request is not another physics step. Ask for the
                # completed explanation using the existing calculation evidence.
                worked = await self._reflection_model(
                    json.dumps({"current_question": question, "history": history, "verified_results": tool_outputs}),
                    "The requested calculations have already been executed. Do not request them again. "
                    "Write the complete worked solution using only those results. Return JSON with nonempty "
                    "string fields givens, principle, equation, substitution, result, and check. "
                    "Cover every requested part. Insert numerical givens in substitution. Use raw LaTeX "
                    "without dollar delimiters for equation and substitution. Keep tool names and internal "
                    "verification out of the explanation.\n" + MATH_FORMATTING_GUIDANCE,
                    tokens=1200,
                )
                parsed = {"answer": worked.get("result", ""), "worked_solution": worked, "tool_calls": [],
                          "show_diagram": parsed.get("show_diagram")}
            answer = parsed.get("answer")
            if (parsed.get("tool_calls") or not isinstance(answer, str)
                    or (not answer.strip() and not isinstance(parsed.get("worked_solution"), dict))):
                raise RuntimeError("The tutor could not verify an answer to this question. Please try again.")
            if self.verify_numerical_completeness and len(tool_outputs) == 1 and "?" in answer:
                clarification = await self._reflection_model(
                    json.dumps({"current_question": question, "history": history, "draft_question": answer}),
                    "Check only whether the ORIGINAL task genuinely lacks necessary information. Do not plan tools "
                    "or calculate. Return JSON with needs_clarification (boolean) and question (string). "
                    "Set true when a numerical answer needs a quantity that was not given and cannot be derived, "
                    "such as moment of inertia when only angular velocity is known. Ask ONE targeted question "
                    "for that missing quantity, ending in '?'. Set false for a conceptual question, or when the "
                    "requested unknown can be calculated from the supplied data. Do not require mass when it "
                    "cancels from a speed calculation. Do not invent input values.\nVerified physics:\n"
                    + tool_outputs[0]["result"],
                    tokens=500,
                )
                if clarification.get("needs_clarification") is True or clarification.get("task_type") == "clarification":
                    followup = clarification.get("question") or clarification.get("corrected_answer") or answer
                    if (isinstance(followup, str) and "?" in followup and not self._mentions_internal_tools(followup)
                            and not self._has_new_numbers(followup, question + "\n" + history)):
                        return {"solution": self._one_question(followup), "tools_used": list(dict.fromkeys(tools_used)), "diagram": None}
            corrected_answer = None
            if len(tool_outputs) == 1:
                # A prose answer is not proof that a numerical check was done.
                # Independently check tool selection before accepting an answer
                # grounded only in the general principles tool.
                corrected_answer = await self._verify_reflection_calculations(
                    {"problem": history, "question_to_answer": question, "draft_answer": answer},
                    catalog, tool_outputs, call_tool,
                )
                if len(tool_outputs) > 1:
                    continue
                if corrected_answer:
                    answer = corrected_answer
            elif self.verify_numerical_completeness:
                completed_before = len(tool_outputs)
                corrected_answer = await self._verify_reflection_calculations(
                    {"problem": history, "question_to_answer": question,
                     "draft_answer": parsed.get("worked_solution") or answer},
                    catalog, tool_outputs, call_tool,
                )
                if len(tool_outputs) > completed_before:
                    continue
                if corrected_answer:
                    answer = corrected_answer
                    # Rebuild all worked sections together when the numerical
                    # draft omitted a requested part; don't retain partial work.
                    parsed["worked_solution"] = None
            if isinstance(corrected_answer, ClarificationRequest):
                return {"solution": str(corrected_answer), "tools_used": list(dict.fromkeys(tools_used)), "diagram": None}
            if self.worked_solution_required and len(tool_outputs) > 1:
                worked = parsed.get("worked_solution")
                if isinstance(worked, dict):
                    worked = self._ground_worked_solution(worked, tool_outputs)
                fields = ("givens", "principle", "equation", "substitution", "result", "check")
                def valid_worked(value):
                    if not isinstance(value, dict) or not all(isinstance(value.get(key), str) and value[key].strip() for key in fields):
                        return False
                    if any("$" in value[key] for key in ("equation", "substitution")):
                        return False
                    substitution = re.sub(r"\\(?:d?frac|tfrac)\s*\{?1\}?\s*\{?2\}?", "", value["substitution"])
                    substitution = re.sub(r"\^(?:\{[^}]*\}|[-+]?\d+)", "", substitution)
                    substitution = re.sub(r"(?<![\d.])(?:0?\.5|1\s*/\s*2)(?![\d.])", "", substitution)
                    if not re.search(r"\d", substitution):
                        return False
                    content = "\n".join(value[key] for key in fields)
                    return (not re.search(r"\b(?:MCP|numerical tools?|calculation tools?|tool outputs?|tool results?|verified results?)\b", content, re.I)
                            and not any(tool["name"] in content for tool in catalog)
                            and self.principles_tool not in content)
                if not valid_worked(worked):
                    worked = await self._reflection_model(
                        json.dumps({"current_question": question, "history": history, "verified_results": tool_outputs}),
                        "Write the COMPLETE worked solution to the current numerical question using only verified MCP results. "
                        "Return JSON with nonempty string fields givens, principle, equation, substitution, result, and check. "
                        "equation and substitution must be raw LaTeX equations without dollar delimiters. "
                        "result must include units and direction when relevant. Complete all requested verification. "
                        "Substitution MUST insert the actual numerical givens and computed values, not repeat symbolic formulas. "
                        "Do not include dollar delimiters inside equation or substitution. "
                        "Use physics language only. Never mention MCP, tools, tool names, verified results, or internal implementation. "
                        "check must discuss physical units, direction, assumptions, or conservation. "
                        "Do not ask permission to finish or introduce unverified calculations.\n" + MATH_FORMATTING_GUIDANCE,
                        tokens=1200,
                    )
                    if isinstance(worked, dict):
                        worked = self._ground_worked_solution(worked, tool_outputs)
                if not valid_worked(worked):
                    raise RuntimeError("The tutor could not finish the worked solution. Please try again.")
                sections = []
                for key, title in zip(fields, ("Given", "Principle", "Equation", "Substitution", "Result", "Check")):
                    value = worked[key].strip()
                    if key in {"equation", "substitution"}:
                        value = "$$\n" + value.strip("$").strip() + "\n$$"
                    sections.append(f"**{title}**\n\n{value}")
                answer = "\n\n".join(sections)
            response_data = {
                "solution": self._strip_embedded_diagram_json(answer).strip(),
                "tools_used": list(dict.fromkeys(tools_used)),
                "diagram": diagrams[-1] if diagrams and parsed.get("show_diagram") is True else None,
            }
            error = self._conversation_answer_error(response_data["solution"], numerical_verified=len(tool_outputs) > 1)
            if error:
                repaired = await self._reflection_model(
                    json.dumps({"current_question": question, "history": history,
                                "draft_answer": response_data["solution"], "error": error}),
                    "Rewrite the draft to correct the identified physics error. Address only the current question. "
                    "Return JSON with answer, a concise complete explanation consistent with the verified principles. "
                    "Replace the incorrect claim; do not repeat it or append a contradictory statement. "
                    "Do not introduce new numerical results.\nVerified physics:\n" + tool_outputs[0]["result"],
                )
                corrected = repaired.get("answer")
                if (not isinstance(corrected, str) or not corrected.strip()
                        or self._conversation_answer_error(corrected, numerical_verified=len(tool_outputs) > 1)):
                    raise RuntimeError("The tutor could not verify this explanation. Please try again.")
                response_data["solution"] = self._strip_embedded_diagram_json(corrected).strip()
            return response_data
        raise RuntimeError("This calculation needs more steps than the tutor could complete. Please try a smaller step.")

    def _conversation_answer_error(self, answer, *, numerical_verified=False):
        if self._mentions_internal_tools(answer):
            return "Use physics language only. Remove MCP, tool names and internal verification details from the student-facing explanation."
        return None

    def _mentions_internal_tools(self, text):
        if re.search(r"\bMCP\b|\btool_(?:calls|outputs|used)\b|\bnumerical tools?\b|\bverified results?\b", text, re.I):
            return True
        names = [self.principles_tool] + [tool["name"] for tool in getattr(self, "mcp_tool_specs", [])]
        return any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text) for name in names)

    def _ground_worked_solution(self, worked, tool_outputs):
        return worked

    @staticmethod
    def _same_calculation(output, call):
        def normalized(value):
            if isinstance(value, str):
                try:
                    return normalized(json.loads(value))
                except (ValueError, TypeError):
                    return value
            if isinstance(value, dict):
                return {key: normalized(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalized(item) for item in value]
            return value
        return (output["tool"] == call["tool_name"]
                and normalized(output["arguments"]) == normalized(call["arguments"]))

    @staticmethod
    def _has_student_evidence(quotes, work):
        """Require student text, tolerating whitespace and sentence capitalization."""
        normalized_work = [" ".join(item.split()) for item in work]
        for quote in quotes:
            if not isinstance(quote, str) or not quote.strip():
                return False
            quote = " ".join(quote.split())
            if any(quote in item for item in normalized_work):
                continue
            # Capitalizing a copied sentence does not change its evidence. Keep
            # case significant for single-letter variables and unit symbols.
            if not re.match(r"[A-Za-z]{3,}\b", quote):
                return False
            pattern = "[" + quote[0].lower() + quote[0].upper() + "]" + re.escape(quote[1:])
            if not any(re.search(pattern, item) for item in normalized_work):
                return False
        return True

    @staticmethod
    def _conversation_messages(context: Optional[Dict[str, Any]]) -> list[Dict[str, str]]:
        """Preserve roles and order; legacy summaries are fallbacks, never extra turns."""
        conversation = context.get("conversation_context") if isinstance(context, dict) else None
        if not isinstance(conversation, dict):
            return []
        raw_messages = conversation.get("recent_messages")
        messages = [
            {"role": message["role"], "content": message["content"].strip()}
            for message in (raw_messages if isinstance(raw_messages, list) else [])
            if isinstance(message, dict) and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str) and message["content"].strip()
        ]
        if messages:
            return messages
        for role, field in (("user", "previous_user_problem"), ("assistant", "previous_assistant_response")):
            content = conversation.get(field)
            if isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})
        return messages

    async def _reflection_model(self, prompt, system, *, messages=None, reasoning=False, tokens=700, repair_format=True):
        response = await asyncio.to_thread(
            self._request_ollama_generate, prompt,
            max(30, self.fast_mcp_explain_timeout_seconds) if reasoning else self.fast_mcp_explain_timeout_seconds,
            tokens, not reasoning, self.model_id,
            think=reasoning if self.model_id.startswith("qwen3") else None,
            system=system, messages=messages,
            model_options=({
                "temperature": 0.6 if reasoning else 0.7, "top_p": 0.95 if reasoning else 0.8,
                "top_k": 20, "min_p": 0, "seed": 42,
            } if self.model_id.startswith("qwen3") else None),
        )
        parsed = self._parse_llm_json_object(response)
        if parsed is None and not response.strip() and reasoning and repair_format:
            # A thinking budget can expire before final content is emitted.
            # Retry the same task once in structured, non-thinking mode.
            return await self._reflection_model(
                prompt, system, messages=messages, tokens=900, repair_format=False,
            )
        if parsed is None and response.strip() and repair_format:
            # Thinking models can return sound decisions in prose. Repair only the
            # serialization once; never treat unparsed prose as a tool plan/grade.
            return await self._reflection_model(
                json.dumps({"original_task": system, "original_input": prompt, "result_to_format": response}),
                "Reformat the supplied model result as the JSON object required by original_task. "
                "Preserve its decisions; do not redo the physics, add assumptions, or answer the student. "
                "Return only the requested JSON fields.",
                messages=messages, repair_format=False,
            )
        if not parsed:
            raise RuntimeError("The tutor could not interpret this conversation step. Please try again.")
        return parsed

    async def _solve_reflection_turn(self, question, messages, catalog, tool_outputs, call_tool):
        indexed = [{"role": message["role"], "content": f"Message {index}:\n{message['content']}"}
                   for index, message in enumerate(messages)]
        numerical_catalog = [tool for tool in catalog if tool["name"] not in {
            "check_equilibrium", "create_free_body_diagram", self.principles_tool,
        }]
        planning = (
            REFLECTION_STEP_PROMPT
            + "\nAlso return calculation_required (boolean) and tool_calls (a list). "
            "Select numerical MCP tools ONLY when the current turn requires arithmetic using supplied numerical givens. "
            "A word such as zero or equal, a symbolic equation, a direction, or a qualitative motion prediction is conceptual, "
            "not a calculation. For these, calculation_required=false and tool_calls=[]. "
            "Grade ONLY the attempted step: a formula with constants such as 0.5 or an exponent 2 is still symbolic. "
            "A qualitative proportionality question (doubling speed, halving inertia) also needs principles, not "
            "sample numerical masses or speeds. Do not use a numerical tool with empty data just to retrieve a formula. "
            "For quantitative feedback, calculate from ORIGINAL givens, never from the student's proposed result. "
            "Never invent a mass or other missing inputs. With no numerical givens, use the verified principles instead. "
            "For direct, intake, hint or practice, leave tool_calls empty. "
            'Each call is {"tool_name":"catalog name","arguments":{}}. '
            "Use completed results without repeating the same call.\nCatalog:\n"
            + json.dumps(numerical_catalog)
            + self.tutoring_guidance
        )
        for _ in range(4):
            frame = await self._reflection_model(
                f"Message {len(messages)} (the CURRENT student turn):\n{question}\nCompleted calculations:\n"
                + json.dumps([item for item in tool_outputs if item["tool"] != self.principles_tool]),
                planning + "\nA short answer to the previous tutor question is feedback, NOT direct. "
                "Choose direct only for an explicit current request to solve a new problem or give an explanation/full solution.",
                messages=indexed, reasoning=False, tokens=1000,
            )
            action = frame.get("action")
            if action not in {"intake", "hint", "feedback", "practice", "direct"}:
                raise RuntimeError("The tutor could not identify the current task. Please try again.")
            if action == "direct":
                # A formula submitted as an answer can look like a new calculation
                # when the router sees the whole tool catalog. Check the speech act
                # against the last tutor turn before leaving the hint-only workflow.
                intent = await self._reflection_model(
                    json.dumps({
                        "last_tutor_question": next((m["content"] for m in reversed(messages)
                                                     if m["role"] == "assistant"), ""),
                        "current_student_message": question,
                    }),
                    "Classify the speech act of current_student_message. Return JSON with request_to_solve (boolean). "
                    "TRUE for an instruction or question asking the tutor to calculate, find, solve, explain, or give a full solution. "
                    "An explicit NEW problem overrides the last tutor question, even when it includes equations or numbers. "
                    "FALSE for a proposed answer, equation, value, direction, or the student's own reasoning. "
                    "Examples: 'Calculate the unknown for this new setup' -> true; 'Please explain the answer' -> true; "
                    "'F=ma' -> false; 'My answer is 3 because I divided' -> false. "
                    "Identify the request, not whether the physics is right. Do not solve physics.", tokens=200,
                )
                if type(intent.get("request_to_solve")) is not bool:
                    raise RuntimeError("The tutor could not identify the current task. Please try again.")
                if intent["request_to_solve"]:
                    return None
                action = "feedback"
            problem = frame.get("problem_statement")
            index = frame.get("problem_message_index")
            if (action == "practice" and index is None and isinstance(problem, str) and problem.strip()
                    and " ".join(problem.split()) in " ".join(question.split())):
                intent = await self._reflection_model(
                    question,
                    'Does the student ask you to solve the supplied problem or invent a new exercise? '
                    'Return JSON with intent: "answer_supplied" or "invent_new".', tokens=200,
                )
                if intent.get("intent") == "answer_supplied":
                    return None
                if intent.get("intent") != "invent_new":
                    raise RuntimeError("The tutor could not identify the current task. Please try again.")
                index = len(messages)
            if action == "practice":
                intent = await self._reflection_model(
                    json.dumps({"current_student_message": question,
                                "last_tutor_message": messages[-1]["content"] if messages else ""}),
                    "Does the CURRENT student message ask you to invent a NEW practice question? "
                    "Return JSON with invent_new (boolean). A student submitting their attempt at a practice question "
                    "is NOT asking for another exercise. Do not reuse requests from earlier turns.", tokens=200,
                )
                if type(intent.get("invent_new")) is not bool:
                    raise RuntimeError("The tutor could not identify the current task. Please try again.")
                if not intent["invent_new"]:
                    action = "feedback"
            has_problem = isinstance(problem, str) and bool(problem.strip()) and type(index) is int and 0 <= index <= len(messages)
            if action in {"intake", "hint", "feedback"} and has_problem:
                action = "hint" if index == len(messages) else "feedback"
            if action == "intake":
                return "Please paste the quiz question, your original answer, and your reasoning.", "intake"
            if not has_problem:
                raise RuntimeError("The tutor could not identify the problem being discussed. Please try again.")
            source_problem = question if index == len(messages) else messages[index]["content"]
            if " ".join(problem.split()) not in " ".join(source_problem.split()):
                problem = source_problem
            if action != "feedback" or not frame.get("calculation_required"):
                break
            calls = self._extract_fast_mcp_tool_calls(frame, numerical_catalog)
            if not isinstance(frame.get("tool_calls"), list) or len(calls) != len(frame["tool_calls"]):
                raise RuntimeError("The tutor could not verify its calculation. Please try again.")
            if calls and self.agent_id in MECHANICS_AGENTS:
                grounding = await self._reflection_model(
                    json.dumps({"original_problem": source_problem, "current_reply": question,
                                "last_tutor_question": messages[-1]["content"] if messages else "",
                                "proposed_calculations": calls, "completed_calculations": tool_outputs[1:]}),
                    "Audit a proposed calculation plan for grading the CURRENT attempted step. "
                    "Return JSON with calculation_needed and inputs_grounded (booleans). "
                    "Choosing a symbolic equation, identifying a direction, or predicting a proportional change "
                    "needs no numerical evaluation: calculation_needed=false. Constants/exponents in a formula "
                    "are not numerical givens for mass, speed, or inertia. Do not solve a later, unasked step. "
                    "For an actual numerical attempt, calculation_needed=true; inputs_grounded=true ONLY when "
                    "every numerical input is supplied in the original problem or a completed calculation, "
                    "apart from standard gravity. Values chosen merely as examples, unit masses, or a student's "
                    "unverified final result are not valid input evidence. Do not solve physics in this audit.",
                    tokens=250,
                )
                if type(grounding.get("calculation_needed")) is not bool or type(grounding.get("inputs_grounded")) is not bool:
                    raise RuntimeError("The tutor could not validate the calculation inputs. Please try again.")
                if not grounding["calculation_needed"]:
                    break
                if not grounding["inputs_grounded"]:
                    planning += "\nThe preceding plan used unprovided quantities. Correct it using only the original numerical givens."
                    continue
            input_errors = [self._calculation_input_error(call) for call in calls]
            if any(input_errors):
                planning += "\nThe previous tool plan was invalid: " + "; ".join(error for error in input_errors if error)
                planning += ". Correct the plan without inventing inputs. Symbolic equation answers need principles, not numerical tools."
                continue
            if not calls:
                if len(tool_outputs) == 1:
                    raise RuntimeError("The tutor did not verify the requested calculation. Please try again.")
                break
            fresh = [call for call in calls if not any(
                item["tool"] == call["tool_name"] and item["arguments"] == call["arguments"]
                for item in tool_outputs)]
            if not fresh:
                break
            for call in fresh:
                await call_tool(call["tool_name"], call["arguments"])
        else:
            raise RuntimeError("The tutor could not finish checking this calculation. Please try a smaller step.")

        physics = "\n".join(output["result"] for output in tool_outputs)
        if action == "practice":
            exercise = await self._reflection_model(
                json.dumps({"original_problem": problem, "request": question}),
                "Create ONE new physics practice problem testing the SAME principle and misconception as the reviewed problem. "
                "Use a different object and setting, not just different numbers. Preserve the timing and relevant force assumptions. "
                "State all conditions needed for an unambiguous answer, including any neglected forces. "
                "Give a short setup and exactly ONE open prediction question ending in a question mark. "
                "Do not provide multiple-choice options. Do not reveal or suggest the answer. "
                "Return JSON with setup (declarative sentences describing the new situation) and question (the single open question).\nVerified physics:\n" + physics,
            )
            setup = exercise.get("setup")
            if not isinstance(setup, str) or not setup.strip():
                raise RuntimeError("The tutor could not prepare the practice question. Please try again.")
            return self._one_question(setup.strip() + "\n\n" + self._one_question(exercise.get("question"))), "practice"

        last_tutor = next((message["content"] for message in reversed(messages) if message["role"] == "assistant"), "")
        corrected_work = [m["content"] for m in messages[index + 1:] if m["role"] == "user"] + ([question] if action == "feedback" else [])
        feedback = await self._reflection_model(
            json.dumps({
                "original_problem": problem, "original_submission": source_problem, "last_tutor_question": last_tutor,
                "latest_student_reply": question, "action": action,
                "student_corrected_work": corrected_work,
                "recent_steps": messages[index + 1:][-6:],
            }),
            self._feedback_instructions() + "\nVerified physics:\n" + physics,
            reasoning=True, tokens=1200,
        )
        verdict = feedback.get("assessment")
        if action == "hint":
            verdict = "unassessed"
        if verdict not in {"correct", "incorrect", "partial", "unclear", "unassessed"}:
            raise RuntimeError("The tutor could not assess your reply. Please try again.")
        outcome, reason = feedback.get("outcome_evidence"), feedback.get("reason_evidence")
        has_evidence = self._has_student_evidence((outcome, reason), corrected_work)
        if verdict == "correct" and feedback.get("complete") is True and len(" ".join(corrected_work).split()) <= 2:
            next_step = await self._reflection_model(
                json.dumps({"problem": problem, "last_tutor_question": last_tutor, "correct_student_reply": question}),
                "The student correctly answered the last tutor question, but has not explained the original problem. "
                "Write ONE next-inference question. Return JSON with question only, ending in '?'. "
                "Build on the latest correct step. Do not re-grade it, repeat it, or ask them to defend an earlier wrong belief. "
                "Do not state an outcome the student has not yet supplied. Do not invent a new problem.\nVerified physics:\n" + physics,
                tokens=900,
            )
            feedback = {"complete": False, "question": next_step.get("question")}
        elif verdict == "correct" and (
                (self.agent_id in MECHANICS_AGENTS and feedback.get("complete") is True)
                or
                (feedback.get("complete") is True and not has_evidence)
                or (feedback.get("complete") is False and has_evidence and len(" ".join(corrected_work).split()) > 2)):
            # Resolve contradictory evidence/flags before repeating an answered question.
            proposed_question = feedback.get("question")
            feedback = await self._reflection_model(
                json.dumps({"original_problem": problem, "student_corrected_work": corrected_work}),
                "Check completion of the ORIGINAL problem, not an intermediate tutor question. "
                "First identify exactly what result or prediction original_problem asks for. "
                "Then check whether student_corrected_work EXPLICITLY states that result AND a physics reason. "
                "Do not infer an unstated result from a correct intermediate quantity or equation. "
                "Return JSON with original_requested_result (a short description), complete (boolean), "
                "outcome_evidence and reason_evidence (exact quotes from student_corrected_work). "
                "A quote may contain both the result and its reason. A force statement alone does not complete a motion prediction. "
                "A single word, value, or formula alone is not an explained solution. Do not borrow from the original submission. "
                "The submitted steps are correct so far. Build on them without asking for an answer already supplied. "
                "Do not ask the student to defend an earlier wrong belief; it is not a premise of this question. "
                "If either piece is absent, set complete=false and give one targeted next-inference question ending in '?'. "
                "Otherwise set complete=true and question=''. Do not generate a new exercise. "
                "This is an evidence check, not a request to solve the problem yourself.",
                reasoning=True, tokens=1200,
            )
            if feedback.get("complete") is False and not feedback.get("question"):
                # The evidence check may omit a question already supplied by the grader.
                feedback["question"] = proposed_question
            has_evidence = self._has_student_evidence(
                (feedback.get("outcome_evidence"), feedback.get("reason_evidence")), corrected_work)
        if (action == "feedback" and verdict == "correct" and feedback.get("complete") is True
                and has_evidence and len(" ".join(corrected_work).split()) > 2):
            return "Correct. You have answered this question and explained your reasoning.", "complete"
        if self.agent_id in MECHANICS_AGENTS and verdict == "unclear":
            previous_question = last_tutor.rsplit("\n\n", 1)[-1]
            def normalize(text):
                if not isinstance(text, str) or not text.strip():
                    return ""
                return " ".join(re.findall(r"\w+", self._one_question(text).casefold()))
            if normalize(feedback.get("question")) == normalize(previous_question):
                smaller = await self._reflection_model(
                    json.dumps({"original_problem": problem, "unanswered_question": previous_question,
                                "student_reply": question}),
                    "The student is lost and could not answer this tutor question. Ask ONE simpler prerequisite "
                    "about a stated given, transfer direction, or system condition. Do not repeat or paraphrase "
                    "the unanswered question. Do not supply the final result or invent quantities. Return JSON "
                    "with question, ending in '?'.\n" + self.tutoring_guidance + "\nVerified physics:\n" + physics,
                    tokens=500,
                )
                if normalize(smaller.get("question")) == normalize(previous_question):
                    raise RuntimeError("The tutor could not prepare a smaller hint. Please try again.")
                feedback["question"] = self._one_question(smaller.get("question"))
        if self.agent_id in MECHANICS_AGENTS and not str(feedback.get("question") or "").strip():
            next_step = await self._reflection_model(
                json.dumps({"original_problem": problem, "last_tutor_question": last_tutor,
                            "latest_student_reply": question, "assessment": verdict, "corrected_work": corrected_work}),
                "The latest step has been graded, but the ORIGINAL requested outcome has not yet been explained. "
                "Return JSON with question: ONE targeted next-inference question ending in '?'. "
                "Build on correct work instead of repeating the previous question. Do not reveal the final answer, "
                "invent numerical givens, or add a new exercise. For an incorrect step, ask a smaller prerequisite. "
                "Use only these verified principles and results:\n" + physics, tokens=350,
            )
            feedback["question"] = next_step.get("question")
        if self.agent_id in MECHANICS_AGENTS:
            # A hint must not add a numerical scenario that the student never supplied.
            source = "\n".join([source_problem, last_tutor, *corrected_work])
            candidate = str(feedback.get("question") or "")
            if self._has_new_numbers(candidate, source):
                repair = await self._reflection_model(
                    json.dumps({"original_problem": problem, "student_reply": question,
                                "last_tutor_question": last_tutor, "draft_question": candidate}),
                    "Rewrite draft_question to ask the SAME next inference WITHOUT introducing numerical values, "
                    "factors, sample masses, or changed assumptions. Keep the original situation. "
                    "Ask qualitatively when the original setup is qualitative. Do not reveal the answer. "
                    "Return JSON with question, one question ending in '?'.", tokens=300,
                )
                candidate = self._one_question(repair.get("question"))
                if self._has_new_numbers(candidate, source):
                    raise RuntimeError("The tutor could not keep the hint aligned with your problem. Please try again.")
                feedback["question"] = candidate
        labels = {
            "unassessed": "", "correct": "That step is correct.",
            "incorrect": "Not quite. Let's reconsider that step.",
            "partial": "Part of that reasoning is correct; let's check the remaining part.",
            "unclear": "Let's work through that step together.",
        }
        answer = "\n\n".join(part for part in (labels[verdict], self._one_question(feedback.get("question"))) if part)
        return answer, "hint" if action == "hint" else "feedback"

    def _calculation_input_error(self, call):
        """Validate numerical JSON contracts before a symbolic placeholder reaches MCP."""
        if self.agent_id in MECHANICS_AGENTS:
            def numbers(value):
                if isinstance(value, str):
                    try:
                        return numbers(json.loads(value))
                    except (ValueError, TypeError):
                        return []
                if isinstance(value, dict):
                    return [number for item in value.values() for number in numbers(item)]
                if isinstance(value, list):
                    return [number for item in value for number in numbers(item)]
                return [value] if type(value) in (float, int) else []
            values = numbers(call["arguments"])
            if not values or not all(math.isfinite(value) for value in values):
                return "Numerical tools require finite numerical givens; use principles for symbolic answers, not empty inputs."
            return None
        field = {"newton_second_law": "newton_data", "calculate_spring_force_tool": "spring_data"}.get(call["tool_name"])
        if not field:
            return None
        try:
            data = call["arguments"][field]
            data = json.loads(data) if isinstance(data, str) else data
            aliases = ({"force": ("force", "f", "F"), "mass": ("mass", "m"), "acceleration": ("acceleration", "a")}
                       if field == "newton_data" else {"spring_constant": ("spring_constant", "k"), "displacement": ("displacement", "x")})
            values = {}
            for name, keys in aliases.items():
                raw = next((data[key] for key in keys if key in data), None)
                if raw is not None:
                    if isinstance(raw, bool):
                        raise ValueError("boolean input")
                    values[name] = float(raw)
                    if not math.isfinite(values[name]):
                        raise ValueError("non-finite input")
            if len(values) < 2:
                raise ValueError("two numerical givens are required")
            if "mass" in values and values["mass"] <= 0:
                raise ValueError("mass must be positive")
            if field == "newton_data" and "mass" not in values and values.get("acceleration") == 0:
                raise ValueError("zero force/acceleration cannot determine mass")
        except (KeyError, TypeError, ValueError) as exc:
            return f"{call['tool_name']} requires actual finite numerical givens, not symbols or placeholders ({exc})"
        return None

    async def _verify_reflection_calculations(self, exchange, catalog, tool_outputs, call_tool):
        # Diagram/equilibrium presentation tools are not numerical graders. Qualitative
        # force balance is already covered by get_force_principles, without invented forces.
        catalog = [tool for tool in catalog if tool["name"] not in {
            "check_equilibrium", "create_free_body_diagram", self.principles_tool,
        }]
        correction = ""
        for _ in range(3):
            parsed = await self._reflection_model(
                json.dumps(exchange),
                "You are an internal physics validator. draft_answer is the TUTOR'S unverified draft, not a student's attempt. "
                "FIRST classify question_to_answer as task_type: numerical or conceptual. "
                "Finding or checking a numerical value, including a worked solution, is numerical even if the draft already contains correct arithmetic. "
                "Explaining why, discussing assumptions/directions, or choosing a symbolic equation is conceptual. "
                + ("Classify the ORIGINAL requested task, not the wording of draft_answer. If genuinely necessary "
                   "givens are absent from a numerical problem, task_type MUST be clarification: return tool_calls=[] "
                   "and corrected_answer as ONE targeted question asking for the missing information. "
                   "A draft that only explains that a quantity was not provided is incomplete: turn it into a question. "
                   "Do not fabricate numerical inputs or require a calculation before asking for missing givens. "
                   "An unknown that CAN be solved from the supplied data is not missing information; "
                   "mass is not needed for free-fall speed when it cancels from the equation. "
                   if self.verify_numerical_completeness else "")
                + "Return ONLY a JSON object with task_type, tool_calls (array), and corrected_answer (string, empty if no correction is needed). "
                "For task_type=numerical, you MUST select a calculation tool unless an applicable completed calculation is listed. "
                "Check EVERY quantity requested in question_to_answer. One completed calculation does not verify "
                "a different requested quantity: for example pressure-volume work does not itself calculate Delta U. "
                "Select the additional tools needed for any uncomputed part, using only givens and completed results. "
                "If all calculations are completed but the draft omits a requested result or physical explanation, "
                "put the COMPLETE answer covering every requested part in corrected_answer, with no new arithmetic. "
                "A draft result is NEVER a completed calculation. Do not repair a numerical draft in place of selecting a tool. "
                "Check the draft against the Verified conceptual MCP reference and for internal contradictions. "
                "When no calculation is needed but the draft contradicts that reference or itself, put a complete, concise corrected explanation in corrected_answer. "
                "Replace the incorrect claim; do not merely append a contradictory sentence. Address the current question without replaying the full solution. "
                "Do not say the student was wrong: you are correcting the tutor's draft. Do not add unverified numerical results. "
                "Select calculation tools needed for the CURRENT question_to_answer, not for an earlier task in problem. "
                "A why/how conceptual follow-up can use the already-executed principles tool even when the history contains numbers. "
                "Do not recompute an earlier solution merely to explain its physical meaning. "
                "You must call a numerical tool for a requested new numerical result or to grade numerical arithmetic, "
                "even if the arithmetic seems obvious or the student's answer looks correct. "
                "Use the quantities given in the original problem and question_to_answer, with the current question overriding earlier conditions. "
                "Never use a proposed result from draft_answer as an input; it is an unverified draft. "
                'Return JSON with tool_calls: [] for conceptual questions, directions, symbolic equations, or completed calculations listed below. '
                'Otherwise use tool_calls: [{"tool_name":"catalog name","arguments":{}}]. '
                "Each input quantity must be given in the problem or a completed calculation, apart from standard gravity. "
                "Never assume a unit mass or substitute placeholders for unknown values. "
                "A conceptual conclusion of zero net force or zero acceleration is NOT a numerical calculation. "
                "A motion prediction without given numerical quantities needs NO calculation tools. "
                "Only the completed calculations below count as verification, not a student's answer or general physics principles. "
                "Use completed results without requesting the same calculation again.\n"
                + "Verified conceptual MCP reference:\n" + tool_outputs[0]["result"] + "\n"
                + "Catalog:\n" + json.dumps(catalog, ensure_ascii=True)
                + "\nCompleted calculations:\n" + json.dumps(
                    [output for output in tool_outputs if output["tool"] != self.principles_tool], ensure_ascii=True)
                + correction,
                reasoning=False, tokens=900,
            )
            if not isinstance(parsed.get("tool_calls"), list):
                correction = '\nYour previous result omitted the tool_calls array. Return tool_calls: [] for a conceptual explanation; otherwise supply valid catalog calls. Put any corrected draft in corrected_answer, not prose outside JSON.'
                continue
            if parsed.get("task_type") == "clarification" and self.verify_numerical_completeness:
                question = parsed.get("corrected_answer")
                if (parsed["tool_calls"] or not isinstance(question, str) or "?" not in question
                        or self._mentions_internal_tools(question)):
                    correction = "\nA clarification must ask one student-facing question for missing givens, with tool_calls=[]."
                    continue
                return ClarificationRequest(self._one_question(question))
            if parsed.get("task_type") not in {"conceptual", "numerical"}:
                correction = "\nClassify the CURRENT request with task_type='conceptual' or 'numerical' and include tool_calls and corrected_answer."
                continue
            if not parsed["tool_calls"]:
                if parsed["task_type"] == "numerical" and len(tool_outputs) == 1:
                    correction = "\nThis is a numerical request and NO calculation has been executed. Select a numerical catalog tool using the original givens. The draft's arithmetic does not count as verification."
                    continue
                corrected = parsed.get("corrected_answer", "")
                if not isinstance(corrected, str):
                    correction = "\ncorrected_answer must be a string; use an empty string when no correction is needed."
                    continue
                return corrected.strip()
            calls = self._extract_fast_mcp_tool_calls(parsed, catalog)
            if len(calls) != len(parsed["tool_calls"]):
                correction = "\nYour previous plan did not match the catalog. Use only listed tool names and their argument contracts; for a conceptual explanation return an empty tool_calls array."
                continue
            errors = [self._calculation_input_error(call) for call in calls]
            if any(errors):
                correction = "\nCorrect the previous plan: " + "; ".join(error for error in errors if error)
                continue
            correction = ""
            fresh_calls = [
                call for call in calls
                if not any(self._same_calculation(output, call)
                           for output in tool_outputs)
            ]
            if not fresh_calls:
                return
            for call in fresh_calls:
                await call_tool(call["tool_name"], call["arguments"])
        raise RuntimeError("The tutor could not finish checking this calculation. Please try a smaller step.")

    @staticmethod
    def _has_new_numbers(question, source):
        pattern = r"(?<![A-Za-z_])\d+(?:\.\d+)?"
        return bool(set(re.findall(pattern, question)) - set(re.findall(pattern, source)))

    @staticmethod
    def _one_question(text: Any) -> str:
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("The tutor could not verify its next question. Please try again.")
        text = text.strip()
        return text.split("?", 1)[0] + "?" if "?" in text else text
