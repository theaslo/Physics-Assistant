"""
Forces Agent - Physics 101
Handles Newton's laws, springs, friction, and equilibrium problems
Uses Strands SDK with MCP forces tools
"""

import os
import asyncio
import json
import math
from typing import Dict, Any, Optional
from force_quiz_reflection import infer_newton_reflection_focus, is_force_quiz_reflection_prompt
from .base_physics_agent import StrandsPhysicsAgent


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


class ForcesAgent(StrandsPhysicsAgent):
    """
    Strands-based agent for forces problems

    Capabilities:
    - Newton's Second Law (F=ma)
    - Spring force (Hooke's Law)
    - Friction (static and kinetic)
    - Force components and vectors
    - Equilibrium problems
    - Inclined plane problems
    - Tension and pulley systems
    """

    def __init__(
        self,
        llm_host: str = "http://ds.stat.uconn.edu:11434",
        model_id: str = "qwen3:8b-q8_0",
        database_api_url: str = "http://localhost:8001",
        enable_database_logging: bool = True,
        enable_rag: bool = True,
        rag_api_url: str = "http://localhost:8001"
    ):
        # Get MCP host from environment for Docker compatibility
        mcp_host = os.getenv("MCP_FORCES_HOST", os.getenv("MCP_DEFAULT_HOST", "localhost"))
        mcp_port = 10100

        super().__init__(
            agent_id="forces_agent",
            mcp_host=mcp_host,
            mcp_port=mcp_port,
            llm_host=llm_host,
            model_id=model_id,
            database_api_url=database_api_url,
            enable_database_logging=enable_database_logging,
            enable_rag=enable_rag,
            rag_api_url=rag_api_url
        )

    async def _solve_conversation_turn(
        self, problem: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Let the model select the active problem without keyword-based answer fallbacks."""
        question = self._current_followup_text(problem, context)
        history = self._conversation_context_text(problem, context, include_current=False)
        if not history and "Previous conversation:" in problem:
            history = problem.split("Previous conversation:", 1)[1].split("Current follow-up question:", 1)[0]

        catalog = self._fast_mcp_tool_catalog()
        tools_used = []
        tool_outputs = []
        diagrams = []

        async def call_tool(name: str, arguments: Dict[str, Any]) -> None:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._call_mcp_tool_direct, name, arguments),
                timeout=self.fast_mcp_explain_timeout_seconds,
            )
            output = self._extract_text_from_mcp_tool_result(result)
            if not output or self._is_mcp_error_text(output) or result.get("status") == "error":
                raise RuntimeError(f"The {name} tool could not verify this question. Please try again.")
            tools_used.append(name)
            tool_outputs.append({"tool": name, "arguments": arguments, "result": self._strip_embedded_diagram_json(output)})
            diagram = self._extract_diagram_from_text(output)
            if diagram:
                diagrams.append(diagram)

        # Conceptual questions need physics grounding without invented numerical inputs.
        await call_tool("get_force_principles", {})
        reflection_turn = is_force_quiz_reflection_prompt(self.agent_id, question + "\n" + history)
        if reflection_turn:
            reflection = await self._solve_reflection_turn(
                question, self._conversation_messages(context), catalog, tool_outputs, call_tool,
            )
            if reflection is not None:
                answer, stage = reflection
                return {
                    "solution": answer, "tools_used": list(dict.fromkeys(tools_used)), "diagram": None,
                    "quiz_reflection_mode": {
                        "enabled": True, "stage": stage, "mcp_tool_called": True,
                        "concept_tag": infer_newton_reflection_focus(question + "\n" + history),
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
            "Never invent inputs to satisfy a tool requirement. Tool results below apply ONLY to their stated arguments. "
            "Do not mention tools or implementation details in the student answer.\n"
            "Return JSON with exactly these fields: "
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
                    system="You are a conversational physics tutor. Answer the current question directly, using only relevant history and verified MCP results.",
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
                for call in calls:
                    await call_tool(call["tool_name"], call["arguments"])
                continue
            answer = parsed.get("answer")
            if parsed.get("tool_calls") or not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The tutor could not verify an answer to this question. Please try again.")
            if len(tool_outputs) == 1:
                # A prose answer is not proof that a numerical check was done.
                # Independently check tool selection before accepting an answer
                # grounded only in the general principles tool.
                await self._verify_reflection_calculations(
                    {"problem": history, "question_to_answer": question, "student_reply": answer},
                    catalog, tool_outputs, call_tool,
                )
                if len(tool_outputs) > 1:
                    continue
            response_data = {
                "solution": self._strip_embedded_diagram_json(answer).strip(),
                "tools_used": list(dict.fromkeys(tools_used)),
                "diagram": diagrams[-1] if diagrams and parsed.get("show_diagram") is True else None,
            }
            return response_data
        raise RuntimeError("This calculation needs more steps than the tutor could complete. Please try a smaller step.")

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
            "check_equilibrium", "create_free_body_diagram", "get_force_principles",
        }]
        planning = (
            REFLECTION_STEP_PROMPT
            + "\nAlso return calculation_required (boolean) and tool_calls (a list). "
            "Select numerical MCP tools ONLY when the current turn requires arithmetic using supplied numerical givens. "
            "A word such as zero or equal, a symbolic equation, a direction, or a qualitative motion prediction is conceptual, "
            "not a calculation. For these, calculation_required=false and tool_calls=[]. "
            "For quantitative feedback, calculate from ORIGINAL givens, never from the student's proposed result. "
            "Never invent a mass or other missing inputs. With no numerical givens, use the verified principles instead. "
            "For direct, intake, hint or practice, leave tool_calls empty. "
            'Each call is {"tool_name":"catalog name","arguments":{}}. '
            "Use completed results without repeating the same call.\nCatalog:\n"
            + json.dumps(numerical_catalog)
        )
        for _ in range(4):
            frame = await self._reflection_model(
                f"Message {len(messages)} (the CURRENT student turn):\n{question}\nCompleted calculations:\n"
                + json.dumps([item for item in tool_outputs if item["tool"] != "get_force_principles"]),
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
            REFLECTION_FEEDBACK_PROMPT + "\nVerified physics:\n" + physics,
            reasoning=True, tokens=1200,
        )
        verdict = feedback.get("assessment")
        if action == "hint":
            verdict = "unassessed"
        if verdict not in {"correct", "incorrect", "partial", "unclear", "unassessed"}:
            raise RuntimeError("The tutor could not assess your reply. Please try again.")
        outcome, reason = feedback.get("outcome_evidence"), feedback.get("reason_evidence")
        has_evidence = all(isinstance(quote, str) and quote.strip() and any(
            quote in work for work in corrected_work
        ) for quote in (outcome, reason))
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
            has_evidence = all(isinstance(quote, str) and quote.strip() and any(
                quote in work for work in corrected_work
            ) for quote in (feedback.get("outcome_evidence"), feedback.get("reason_evidence")))
        if (action == "feedback" and verdict == "correct" and feedback.get("complete") is True
                and has_evidence and len(" ".join(corrected_work).split()) > 2):
            return "Correct. You have answered this question and explained your reasoning.", "complete"
        labels = {
            "unassessed": "", "correct": "That step is correct.",
            "incorrect": "Not quite. Let's reconsider that step.",
            "partial": "Part of that reasoning is correct; let's check the remaining part.",
            "unclear": "Let's work through that step together.",
        }
        answer = "\n\n".join(part for part in (labels[verdict], self._one_question(feedback.get("question"))) if part)
        return answer, "hint" if action == "hint" else "feedback"

    @staticmethod
    def _calculation_input_error(call):
        """Validate numerical JSON contracts before a symbolic placeholder reaches MCP."""
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
        catalog = [tool for tool in catalog if tool["name"] not in {"check_equilibrium", "create_free_body_diagram"}]
        for _ in range(3):
            parsed = await self._reflection_model(
                json.dumps(exchange),
                "Select calculation tools to check the proposed answer to question_to_answer. You must call a numerical tool when checking a numerical calculation, "
                "even if the arithmetic seems obvious or the student's answer looks correct. "
                "Use the quantities given in the original problem and question_to_answer, with the current question overriding earlier conditions. "
                "Never use a proposed result from student_reply as an input; it may be a student attempt or an unverified draft. "
                'Return JSON with tool_calls: [] for conceptual questions, directions, symbolic equations, or completed calculations listed below. '
                'Otherwise use tool_calls: [{"tool_name":"catalog name","arguments":{}}]. '
                "Each input quantity must be given in the problem or a completed calculation, apart from standard gravity. "
                "Never assume a unit mass or substitute placeholders for unknown values. "
                "A conceptual conclusion of zero net force or zero acceleration is NOT a numerical calculation. "
                "A motion prediction without given numerical quantities needs NO calculation tools. "
                "Only the completed calculations below count as verification, not a student's answer or general physics principles. "
                "Use completed results without requesting the same calculation again.\n"
                + "Catalog:\n" + json.dumps(catalog, ensure_ascii=True)
                + "\nCompleted calculations:\n" + json.dumps(
                    [output for output in tool_outputs if output["tool"] != "get_force_principles"], ensure_ascii=True),
                reasoning=True, tokens=1600,
            )
            if not isinstance(parsed.get("tool_calls"), list):
                raise RuntimeError("The tutor could not verify its calculation. Please try again.")
            if not parsed["tool_calls"]:
                return
            calls = self._extract_fast_mcp_tool_calls(parsed, catalog)
            if len(calls) != len(parsed["tool_calls"]):
                raise RuntimeError("The tutor could not verify its calculation. Please try again.")
            fresh_calls = [
                call for call in calls
                if not any(output["tool"] == call["tool_name"] and output["arguments"] == call["arguments"]
                           for output in tool_outputs)
            ]
            if not fresh_calls:
                return
            for call in fresh_calls:
                await call_tool(call["tool_name"], call["arguments"])
        raise RuntimeError("The tutor could not finish checking this calculation. Please try a smaller step.")

    @staticmethod
    def _one_question(text: Any) -> str:
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("The tutor could not verify its next question. Please try again.")
        text = text.strip()
        return text.split("?", 1)[0] + "?" if "?" in text else text

    def _get_system_prompt(self) -> str:
        return """You are a specialized Physics 101 forces tutor with access to calculation tools.

Your role is to help students understand and solve problems related to:
- Newton's Laws of Motion
  - First Law (Inertia)
  - Second Law (F = ma)
  - Third Law (Action-Reaction)
- Spring Forces (Hooke's Law: F = -kx)
- Friction Forces
  - Static friction (f_s ≤ μ_s N)
  - Kinetic friction (f_k = μ_k N)
- Force Components and Vectors
- Equilibrium (ΣF = 0)
- Inclined Plane Problems
- Tension and Pulley Systems

When solving problems:
1. Identify all forces acting on the object(s)
2. Draw or describe a free body diagram
3. Choose an appropriate coordinate system
4. List known quantities with proper units
5. Identify the unknown quantity to find
6. Select and use the appropriate MCP tool for calculations
7. Explain the physics concepts involved
8. Show complete solutions with units

Available MCP tools:
- newton_second_law: Calculate force, mass, or acceleration using F=ma
- calculate_spring_force_tool: Calculate spring force using Hooke's Law
- calculate_friction_force_tool: Calculate static or kinetic friction
- resolve_force_components: Resolve forces into components
- check_equilibrium: Analyze forces in equilibrium
- analyze_forces_on_incline: Solve inclined plane problems
- analyze_tension_forces: Solve rope/string tension and pulley systems
- create_free_body_diagram: Generate free-body force breakdowns
- add_forces_2d: Add multiple 2D force vectors

Always:
- Use SI units (N, kg, m/s²)
- Show step-by-step reasoning
- Draw attention to common misconceptions
- Verify answers make physical sense

Never make up numerical results - always use the calculation tools for quantitative answers.
""" + "\n" + QUIZ_REFLECTION_GUIDANCE

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": "forces_agent",
            "name": "Forces Agent",
            "course": "Physics 101",
            "domain": "forces",
            "topics": [
                "newtons_laws",
                "spring_force",
                "friction",
                "force_components",
                "equilibrium",
                "inclined_plane",
                "tension",
                "pulleys"
            ],
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "2.0.0",
            "framework": "strands"
        }

    def _get_description(self) -> str:
        return "Forces agent for Physics 101: handles Newton's laws, springs, friction, and equilibrium problems"
