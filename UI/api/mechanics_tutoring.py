"""Contextual check-ins and model-graded follow-ups for the extended mechanics agents."""

import asyncio


async def solve_mechanics_turn(agent, gate, request, conversation):
    context = {**(request.context or {}), "conversation_context": conversation or {}}
    user_id = request.user_id or "react_user"
    selection = context.get("knowledge_transfer_response")
    remediation = context.get("knowledge_transfer_remediation_followup")
    problem = request.problem
    hitl = None
    receipt = None
    if isinstance(selection, dict):
        if gate is None:
            raise ValueError("Check-ins are unavailable. Please resend your question.")
        check, hitl = await asyncio.to_thread(
            gate.take_contextual_answer, agent.agent_id, selection.get("check_id"),
            selection.get("selected_option_id"), user_id, request.session_id,
        )
        receipt = gate.get_contextual_answer(agent.agent_id, check["check_id"], user_id, request.session_id)
        problem = check["problem"]
        history = list(check.get("history", []))
        history.extend([
            {"role": "user", "content": problem},
            {"role": "assistant", "content": check["question"] + "\n" + "\n".join(
                f'{o["id"]}: {o["text"]}' for o in check["options"])},
        ])
        context["conversation_context"] = {"recent_messages": history}
        if hitl["was_correct"]:
            problem = "Please show the worked solution to this problem: " + problem
            context["full_solution_requested"] = True
        else:
            option = next(o for o in check["options"] if o["id"] == selection["selected_option_id"])
            problem = f'My choice is {option["id"]}: {option["text"]}'
            context["guided_tutoring"] = True
    elif isinstance(remediation, dict):
        if gate is None:
            raise ValueError("Check-ins are unavailable. Please resend your question.")
        receipt = gate.get_contextual_answer(
            agent.agent_id, remediation.get("check_id"), user_id, request.session_id,
        )
        check = receipt["check"]
        problem = str(remediation.get("student_message") or "").strip()
        if not problem:
            raise ValueError("Please type your attempted step or ask for a hint.")
        context["guided_tutoring"] = True
        # The browser can reload or truncate its transcript. Keep the actual
        # problem, selected answer, and successful tutoring turns server-side.
        context["conversation_context"] = {"recent_messages": list(receipt.get("history", []))}
        if remediation.get("mode") == "full_solution":
            problem = "Please give the full solution to: " + check["problem"]
            context["full_solution_requested"] = True
        hitl = {"status": "remediation_followup", "check_id": check["check_id"],
                "agent_id": agent.agent_id, "concept_tag": check["concept_tag"]}
    elif gate and gate.enabled and agent.agent_id in gate.contextual_agents:
        check = await agent.prepare_contextual_check(problem, context)
        if check and check["confidence"] >= gate.threshold:
            check["history"] = agent._conversation_messages(context)
            hitl = gate.register_contextual_check(
                agent.agent_id, problem, check, user_id, request.session_id, context.get("class_identifier"),
            )
            return {"success": True, "agent_id": agent.agent_id, "problem": request.problem,
                    "solution": "", "hitl": hitl, "tools_used": [agent.principles_tool],
                    "metadata": {"framework": "strands", "llm_response_used": True, "hitl": hitl}}

    result = await agent.solve_problem(problem=problem, context=context, user_id=request.user_id, session_id=request.session_id)
    result["problem"] = request.problem
    if hitl and result.get("success"):
        history = agent._conversation_messages(context)
        history.extend([{"role": "user", "content": problem},
                        {"role": "assistant", "content": result["solution"]}])
        receipt["history"] = history
        stage = result.get("metadata", {}).get("quiz_reflection_mode", {}).get("stage")
        if isinstance(remediation, dict):
            if context.get("full_solution_requested"):
                hitl["status"] = "full_solution_requested"
            elif stage == "complete" or not stage:
                hitl["status"] = "remediation_complete"
        if isinstance(selection, dict) and hitl["was_correct"]:
            result["solution"] = "Your check-in answer is correct.\n\n" + result["solution"]
        # Never return stored answer keys or legacy canned remediation guidance.
        hitl = {k: v for k, v in hitl.items() if k in {
            "status", "check_id", "agent_id", "concept_tag", "was_correct", "confidence", "threshold"}}
        result["hitl"] = hitl
        result.setdefault("metadata", {})["hitl"] = hitl
    return result
