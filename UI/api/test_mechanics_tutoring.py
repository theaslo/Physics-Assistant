import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from hitl_knowledge_transfer import KnowledgeTransferGate
from mechanics_tutoring import solve_mechanics_turn
from strands_agents.momentum_agent import MomentumAgent


AGENTS = (MomentumAgent,)
CHECK = {
    "needs_check": True, "confidence": 0.95, "concept_tag": "isolated_collision",
    "question": "For these sticking carts, which quantity is conserved during the collision?",
    "options": [{"id": "A", "text": "System momentum"}, {"id": "B", "text": "Kinetic energy"},
                {"id": "C", "text": "Each cart's velocity"}], "correct_option_id": "A",
}
WORKED = {
    "givens": "Mass is 2 kg and velocity is 4 m/s to the right.",
    "principle": "Momentum equals mass times velocity and points along the velocity.",
    "equation": "p = mv",
    "substitution": r"p = (2\,\text{kg})(4\,\text{m/s}) = 8\,\text{kg m/s}",
    "result": "The momentum is 8 kg m/s to the right.",
    "check": "Its direction matches the velocity, and its units are mass times velocity.",
}


def agent_for(cls):
    agent = cls(enable_database_logging=False, enable_rag=False)
    agent.initialized = True
    agent.mcp_client = Mock()
    agent.mcp_client.call_tool_sync.return_value = {"status": "success", "content": [{"text": "Verified physics."}]}
    return agent


def request(problem="Two carts stick. Find their final speed.", context=None, user="student-a"):
    return SimpleNamespace(problem=problem, context=context, user_id=user, session_id="session-a")


class MechanicsConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_conceptual_followup_repairs_wrong_verifier_schema_without_skipping_mcp(self):
        cases = [
            (MomentumAgent, "Two carts of 2 kg and 3 kg stick; the first initially moves at 4 m/s.",
             "Their final velocity is 1.6 m/s.", "why vf is smaller than vi?"),
        ]
        for cls, setup, solution, question in cases:
            with self.subTest(agent=cls.__name__):
                agent = agent_for(cls)
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"answer": "Explanation of the current question using the original setup.", "tool_calls": []}),
                    json.dumps({"answer": "No numerical calculation is needed."}),
                    json.dumps({"task_type": "conceptual", "tool_calls": []}),
                ])
                result = await agent.solve_problem(question, {"conversation_context": {"recent_messages": [
                    {"role": "user", "content": setup}, {"role": "assistant", "content": solution},
                ]}})
                self.assertTrue(result["success"], result)
                self.assertEqual(result["tools_used"], [agent.principles_tool])
                self.assertTrue(result["metadata"]["llm_response_used"])
                for call in agent._request_ollama_generate.call_args_list[1:]:
                    self.assertEqual(json.loads(call.args[0])["question_to_answer"], question)
                    self.assertTrue(call.args[3])  # Structured JSON, not an open-ended reasoning response.
                    self.assertFalse(call.kwargs["think"])
                    self.assertNotIn("Student-facing formatting", call.kwargs["system"])
                self.assertIn("omitted the tool_calls array", agent._request_ollama_generate.call_args.kwargs["system"])

    async def test_repeated_invalid_verification_never_returns_an_unverified_draft(self):
        agent = agent_for(MomentumAgent)
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "Unverified answer", "tool_calls": []}),
            *[json.dumps({"answer": "Not the requested schema"}) for _ in range(3)],
        ])
        result = await agent.solve_problem("Calculate the momentum of a 2 kg object at 4 m/s.")
        self.assertFalse(result["success"])
        self.assertNotIn("solution", result)
        self.assertEqual(agent._request_ollama_generate.call_count, 4)

    async def test_mcp_grounded_model_correction_replaces_the_draft_without_extra_model_call(self):
        for cls in AGENTS:
            with self.subTest(agent=cls.__name__):
                agent = agent_for(cls)
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"answer": "An inconsistent draft explanation.", "tool_calls": []}),
                    json.dumps({"task_type": "conceptual", "tool_calls": [], "corrected_answer": "A corrected explanation using the verified principle."}),
                ])
                result = await agent.solve_problem("Why does that quantity change?")
                self.assertTrue(result["success"], result)
                self.assertEqual(result["solution"], "A corrected explanation using the verified principle.")
                self.assertEqual(result["tools_used"], [agent.principles_tool])
                self.assertEqual(agent._request_ollama_generate.call_count, 2)
                self.assertIn("Verified conceptual MCP reference:\nVerified physics.",
                              agent._request_ollama_generate.call_args.kwargs["system"])
                exchange = json.loads(agent._request_ollama_generate.call_args.args[0])
                self.assertEqual(exchange["draft_answer"], "An inconsistent draft explanation.")

    async def test_numerical_request_cannot_substitute_a_rewrite_for_a_calculation(self):
        agent = agent_for(MomentumAgent)
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "Unverified arithmetic", "tool_calls": []}),
            *[json.dumps({"task_type": "numerical", "tool_calls": [], "corrected_answer": "8 kg m/s"})
              for _ in range(3)],
        ])
        result = await agent.solve_problem("Calculate the momentum of a 2 kg object at 4 m/s.")
        self.assertFalse(result["success"])
        self.assertNotIn("solution", result)
        self.assertIn("NO calculation has been executed", agent._request_ollama_generate.call_args.kwargs["system"])

    async def test_repaired_numerical_plan_still_executes_mcp_and_rewrites_draft(self):
        agent = agent_for(MomentumAgent)
        agent.mcp_tool_specs = [{"name": "calculate_momentum_1d", "description": "Calculate momentum",
                                 "inputSchema": {"json": {"type": "object"}}}]
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "Unverified answer", "tool_calls": []}),
            json.dumps({"task_type": "numerical", "tool_calls": [{"tool_name": "invented_tool", "arguments": {}}]}),
            json.dumps({"task_type": "numerical", "tool_calls": [{"tool_name": "calculate_momentum_1d",
                                        "arguments": {"momentum_data": {"mass": 2, "velocity": 4}}}]}),
            json.dumps({"task_type": "numerical", "tool_calls": []}),
            json.dumps({"answer": "The verified momentum is 8 kg m/s.", "tool_calls": [], "worked_solution": WORKED}),
        ])
        result = await agent.solve_problem("Calculate the momentum of a 2 kg object at 4 m/s.")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["tools_used"], [agent.principles_tool, "calculate_momentum_1d"])
        self.assertNotIn("Unverified", result["solution"])
        self.assertEqual(agent.mcp_client.call_tool_sync.call_count, 2)

    async def test_short_numerical_answer_is_completed_using_verified_results(self):
        agent = agent_for(MomentumAgent)
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "", "tool_calls": [{"tool_name": "calculate_momentum_1d",
                                                        "arguments": {"mass": 2, "velocity": 4}}]}),
            json.dumps({"answer": "8 kg m/s to the right.", "tool_calls": []}),
            json.dumps(WORKED),
        ])
        result = await agent.solve_problem("A 2 kg object moves at 4 m/s right. Calculate its momentum.")
        self.assertTrue(result["success"], result)
        for heading in ("Given", "Equation", "Substitution", "Result", "Check"):
            self.assertIn(f"**{heading}**", result["solution"])
        self.assertIn("p = mv", result["solution"])
        self.assertIn("8 kg m/s to the right", result["solution"])
        self.assertIn("verified_results", agent._request_ollama_generate.call_args.args[0])
        self.assertEqual(agent.mcp_client.call_tool_sync.call_count, 2)

    async def test_incomplete_worked_solution_repair_cannot_return_a_bare_answer(self):
        agent = agent_for(MomentumAgent)
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "", "tool_calls": [{"tool_name": "calculate_momentum_1d",
                                                        "arguments": {"mass": 2, "velocity": 4}}]}),
            json.dumps({"answer": "8 kg m/s.", "tool_calls": []}),
            json.dumps({"result": "8 kg m/s."}),
        ])
        result = await agent.solve_problem("Calculate momentum for 2 kg at 4 m/s.")
        self.assertFalse(result["success"])
        self.assertNotIn("solution", result)

    async def test_worked_solution_replaces_internal_tool_commentary_with_physics(self):
        agent = agent_for(MomentumAgent)
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "", "tool_calls": [{"tool_name": "calculate_momentum_1d",
                                                        "arguments": {"mass": 2, "velocity": 4}}]}),
            json.dumps({"answer": "8 kg m/s.", "tool_calls": [], "worked_solution": {
                **WORKED, "check": "The numerical tool calculate_momentum_1d verified this result."}}),
            json.dumps(WORKED),
        ])
        result = await agent.solve_problem("Calculate momentum for 2 kg at 4 m/s.")
        self.assertTrue(result["success"], result)
        self.assertNotIn("tool", result["solution"])
        self.assertNotIn("calculate_momentum_1d", result["solution"])
        self.assertIn("direction matches", result["solution"])
        self.assertIn("calculate_momentum_1d", result["tools_used"])

    def test_impulse_working_uses_both_executed_results_and_correct_theorem(self):
        agent = agent_for(MomentumAgent)
        worked = agent._ground_worked_solution({**WORKED, "principle": "J = dp/dt", "substitution": "An incomplete draft"}, [{
            "tool": "calculate_impulse_1d", "arguments": {"force": 120, "time": 0.35,
                "initial_momentum": 10, "final_momentum": 52},
            "result": "J = 120.000 × 0.350\nJ = 42.000 N⋅s\nJ = 52.000 - 10.000\nJ = 42.000 N⋅s",
        }])
        self.assertNotIn("dp/dt", worked["principle"])
        self.assertIn(r"J = \Delta p", worked["equation"])
        for value in ("120", "0.35", "52", "10", "42.000"):
            self.assertIn(value, worked["substitution"])
        self.assertNotIn("$", worked["substitution"])

    def test_instantaneous_momentum_does_not_invent_force_or_constant_velocity_assumptions(self):
        agent = agent_for(MomentumAgent)
        worked = agent._ground_worked_solution({**WORKED, "check": "Assume no external forces and constant velocity."}, [{
            "tool": "calculate_momentum_1d", "arguments": {"mass": 5.0, "velocity": 10.0},
            "result": "p = 5.000 × 10.000\np = 50.000 kg⋅m/s",
        }])
        self.assertNotIn("force", worked["check"])
        self.assertNotIn("constant velocity", worked["check"])
        self.assertIn("50.000", worked["substitution"])

    async def test_all_domains_use_model_and_own_mcp_principles_for_followups(self):
        for cls in AGENTS:
            with self.subTest(agent=cls.__name__):
                agent = agent_for(cls)
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"answer": "No. Those assumptions no longer apply.", "tool_calls": []}),
                    json.dumps({"task_type": "conceptual", "tool_calls": []}),
                ])
                history = [{"role": "user", "content": "An isolated system is being studied."},
                           {"role": "assistant", "content": "The total conserved quantity stays the same."}]
                result = await agent.solve_problem("What if the system is not isolated?", {"conversation_context": {"recent_messages": history}})
                self.assertTrue(result["success"], result)
                self.assertEqual(result["tools_used"], [agent.principles_tool])
                self.assertTrue(result["metadata"]["llm_response_used"])
                self.assertIn("What if the system", agent._request_ollama_generate.call_args_list[0].args[0])
                self.assertIn("An isolated system", agent._request_ollama_generate.call_args_list[0].args[0])
                system = agent._request_ollama_generate.call_args_list[0].kwargs["system"]
                self.assertIn("Verified conceptual MCP reference:\nVerified physics.", system)

    async def test_short_equation_is_graded_as_attempt_not_new_problem(self):
        for cls, equation in zip(AGENTS, ("p=mv", "K=0.5*m*v^2", "L=I*omega")):
            with self.subTest(agent=cls.__name__):
                agent = agent_for(cls)
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"action": "feedback", "problem_message_index": 0, "problem_statement": "Find the final result.",
                                "calculation_required": False, "tool_calls": []}),
                    json.dumps({"assessment": "correct", "complete": False, "outcome_evidence": "", "reason_evidence": "",
                                "question": "Which values belong in that equation?"}),
                ])
                result = await agent.solve_problem(equation, {"guided_tutoring": True, "conversation_context": {"recent_messages": [
                    {"role": "user", "content": "Find the final result."},
                    {"role": "assistant", "content": "Which equation would you use?"},
                ]}})
                self.assertTrue(result["success"], result)
                self.assertTrue(result["solution"].startswith("That step is correct."))
                self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")

    async def test_live_principles_tool_is_excluded_from_calculation_catalog(self):
        agent = agent_for(MomentumAgent)
        agent.mcp_tool_specs = [
            {"name": agent.principles_tool, "description": "concepts", "inputSchema": {"json": {}}},
            {"name": "calculate_momentum_1d", "description": "numerical momentum", "inputSchema": {"json": {}}},
        ]
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "Momentum is a vector.", "tool_calls": []}),
            json.dumps({"task_type": "conceptual", "tool_calls": []}),
        ])
        result = await agent.solve_problem("Is momentum a vector?")
        self.assertTrue(result["success"])
        for call in agent._request_ollama_generate.call_args_list:
            # The executed evidence may name the principles tool, but selectable catalogs must not.
            prompt = call.args[0]
            self.assertNotIn('"name": "get_momentum_principles"', prompt)

    async def test_model_or_mcp_failure_never_falls_back_to_canned_answer(self):
        for cls in AGENTS:
            agent = agent_for(cls)
            agent._request_ollama_generate = Mock(side_effect=RuntimeError("offline"))
            self.assertFalse((await agent.solve_problem("Explain the physical principle."))["success"])
            agent.mcp_client.call_tool_sync.return_value = {"status": "error", "content": [{"text": "offline"}]}
            agent._request_ollama_generate.reset_mock()
            self.assertFalse((await agent.solve_problem("Explain the physical principle."))["success"])
            agent._request_ollama_generate.assert_not_called()

    async def test_formula_completion_claim_requires_independent_evidence_check(self):
        agent = agent_for(MomentumAgent)
        equation = "m1*v1i + m2*v2i = (m1+m2)*vf"
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"action": "feedback", "problem_message_index": 0,
                        "problem_statement": "Two carts stick together. What is their final velocity?",
                        "calculation_required": False, "tool_calls": []}),
            json.dumps({"assessment": "correct", "complete": True, "outcome_evidence": equation,
                        "reason_evidence": equation, "question": ""}),
            json.dumps({"complete": False, "outcome_evidence": "", "reason_evidence": equation,
                        "question": ""}),
            json.dumps({"question": "If mass increases by a factor of 3, what happens to final velocity?"}),
            json.dumps({"question": "Which initial momenta enter that equation?"}),
        ])
        result = await agent.solve_problem(equation, {"guided_tutoring": True, "conversation_context": {"recent_messages": [
            {"role": "user", "content": "Two carts stick together. What is their final velocity?"},
            {"role": "assistant", "content": "Which conservation equation relates the two states?"},
        ]}})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")
        self.assertTrue(result["solution"].startswith("That step is correct."))
        self.assertNotIn("factor of 3", result["solution"])

    def test_empty_or_nonfinite_numerical_plans_are_rejected(self):
        agent = agent_for(MomentumAgent)
        for data in ({}, {"mass": "m", "initial_velocity": "v"}, {"mass": float("inf")}):
            self.assertIsNotNone(agent._calculation_input_error({"tool_name": "momentum_conservation_1d",
                                                                "arguments": {"collision_data": json.dumps(data)}}))
        self.assertIsNone(agent._calculation_input_error({"tool_name": "calculate_momentum_1d",
                                                          "arguments": {"mass": 2, "velocity": 3}}))

    async def test_contextual_check_is_model_generated_with_history_and_mcp(self):
        for cls in AGENTS:
            agent = agent_for(cls)
            agent._reflection_model = AsyncMock(side_effect=[{"new_problem": True}, copy.deepcopy(CHECK),
                                                            {"correct_option_ids": ["A"]}])
            result = await agent.prepare_contextual_check("New setup", {"conversation_context": {"recent_messages": [
                {"role": "user", "content": "Earlier setup"}]}})
            self.assertEqual(result["correct_option_id"], "A")
            self.assertIn("Earlier setup", agent._reflection_model.call_args_list[1].args[0])
            self.assertEqual(agent.mcp_client.call_tool_sync.call_args.kwargs["name"], agent.principles_tool)
            agent._reflection_model = AsyncMock(return_value={"needs_check": False})
            self.assertIsNone(await agent.prepare_contextual_check("Why?", {}))

    async def test_continuations_cannot_be_turned_into_another_multiple_choice_question(self):
        for cls in AGENTS:
            agent = agent_for(cls)
            agent._reflection_model = AsyncMock(return_value={"new_problem": False})
            self.assertIsNone(await agent.prepare_contextual_check("What about conservation of energy in that collision?", {
                "conversation_context": {"recent_messages": [{"role": "user", "content": "Two carts stick."}]}}))
            agent._reflection_model.assert_awaited_once()

    async def test_invalid_generated_options_fail_closed(self):
        for field, value in (("options", []), ("correct_option_id", "Z"), ("confidence", float("nan")),
                             ("needs_check", "true"), ("question", "")):
            agent = agent_for(MomentumAgent)
            agent._reflection_model = AsyncMock(return_value={**copy.deepcopy(CHECK), field: value})
            with self.assertRaises(RuntimeError):
                await agent.prepare_contextual_check("A collision", {})

    async def test_momentum_checks_and_feedback_use_domain_guidance(self):
        agent = agent_for(MomentumAgent)
        agent._reflection_model = AsyncMock(side_effect=[copy.deepcopy(CHECK), {"correct_option_ids": ["A"]}])
        await agent.prepare_contextual_check("A 0.2 kg ball reverses from 5 m/s right to 3 m/s left. Find the impulse.", {})
        check_instructions = agent._reflection_model.call_args_list[0].args[1]
        self.assertIn("retain the signs", check_instructions)
        self.assertIn("net external impulse", check_instructions)
        self.assertIn('"stuck"', agent._feedback_instructions())

    async def test_ambiguous_momentum_check_is_rewritten_and_independently_reviewed(self):
        agent = agent_for(MomentumAgent)
        agent._reflection_model = AsyncMock(side_effect=[
            copy.deepcopy(CHECK), {"correct_option_ids": ["A", "B"], "issue": "Two options are equivalent."},
            copy.deepcopy(CHECK), {"correct_option_ids": ["A"]},
        ])
        result = await agent.prepare_contextual_check("Two carts stick. Find their speed.", {})
        self.assertEqual(result["correct_option_id"], "A")
        self.assertIn("Two options are equivalent", agent._reflection_model.call_args_list[2].args[0])
        agent._reflection_model = AsyncMock(side_effect=[
            copy.deepcopy(CHECK), {"correct_option_ids": ["A", "B"]},
            copy.deepcopy(CHECK), {"correct_option_ids": ["B"]},
        ])
        with self.assertRaisesRegex(RuntimeError, "unambiguous"):
            await agent.prepare_contextual_check("Two carts stick. Find their speed.", {})

    async def test_conceptual_momentum_answer_cannot_relabel_velocity_as_momentum(self):
        agent = agent_for(MomentumAgent)
        bad = r"The ball's momentum changes from $5 \text{kg m/s}$ to $-3 \text{kg m/s}$."
        good = "The ball's momentum changes direction because the wall gives it an external impulse."
        for draft in (bad, bad.replace(r"\text", "\t" + "ext")):
            with self.subTest(draft=draft):
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"answer": draft, "tool_calls": []}),
                    json.dumps({"task_type": "conceptual", "tool_calls": []}),
                    json.dumps({"answer": good}),
                ])
                result = await agent.solve_problem("Is the ball's momentum conserved?")
                self.assertTrue(result["success"], result)
                self.assertEqual(result["solution"], good)

    def test_student_evidence_tolerates_sentence_capitalization_but_keeps_variables_distinct(self):
        agent = agent_for(MomentumAgent)
        work = ["The velocity is 2 m/s to the right because momentum is conserved."]
        self.assertTrue(agent._has_student_evidence(
            ["The velocity is 2 m/s to the right", "Momentum is conserved."], work))
        self.assertFalse(agent._has_student_evidence(["Energy is conserved."], work))
        self.assertFalse(agent._has_student_evidence(["P = 8"], ["p = 8"]))

    async def test_momentum_explanation_repairs_a_changing_system_mass_claim(self):
        agent = agent_for(MomentumAgent)
        bad = "In a perfectly inelastic collision, the total mass of the system increases after the collision."
        good = "Both carts are in the system throughout. The same total momentum is shared by both carts after they stick together."
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": bad, "tool_calls": []}),
            json.dumps({"task_type": "conceptual", "tool_calls": []}),
            json.dumps({"answer": good}),
        ])
        result = await agent.solve_problem("Why is the common speed smaller?")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["solution"], good)
        self.assertEqual(result["tools_used"], ["get_momentum_principles"])

    async def test_direct_conceptual_question_about_new_setup_skips_quiz_generation(self):
        agent = agent_for(MomentumAgent)
        agent._reflection_model = AsyncMock(return_value=copy.deepcopy(CHECK))
        self.assertIsNone(await agent.prepare_contextual_check(
            "Different problem: a ball rebounds from a wall. Is the ball's own momentum conserved?", {}))
        agent._reflection_model.assert_not_awaited()


class MechanicsGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.gate = KnowledgeTransferGate("http://unused")
        self.gate._log_attempt = Mock()
        self.agent = agent_for(MomentumAgent)
        self.agent.prepare_contextual_check = AsyncMock(return_value=copy.deepcopy(CHECK))
        self.agent.solve_problem = AsyncMock(return_value={"success": True, "solution": "One question?",
            "agent_id": "momentum_agent", "tools_used": [self.agent.principles_tool],
            "metadata": {"quiz_reflection_mode": {"stage": "feedback"}}})

    async def create(self):
        return await solve_mechanics_turn(self.agent, self.gate, request(), None)

    async def test_safe_payload_and_correct_and_wrong_second_legs(self):
        for selection in ("A", "B"):
            first = await self.create()
            self.assertNotIn("correct_option_id", json.dumps(first))
            check_id = first["hitl"]["check_id"]
            second = await solve_mechanics_turn(self.agent, self.gate, request(context={
                "knowledge_transfer_response": {"check_id": check_id, "selected_option_id": selection}}), None)
            self.assertEqual(second["hitl"]["was_correct"], selection == "A")
            self.assertEqual(second["hitl"]["status"], "answer_processed" if selection == "A" else "remediation_required")
            context = self.agent.solve_problem.call_args.kwargs["context"]
            self.assertEqual(context.get("full_solution_requested", False), selection == "A")
            self.assertEqual(context.get("guided_tutoring", False), selection == "B")
            self.assertIn("carts", context["conversation_context"]["recent_messages"][0]["content"])
            self.assertNotIn(check_id, self.gate._pending_checks)
            self.gate._log_attempt.assert_called()
            count = self.gate._log_attempt.call_count
            self.gate.take_contextual_answer("momentum_agent", check_id, selection, "student-a", "session-a")
            self.assertEqual(self.gate._log_attempt.call_count, count)
            with self.assertRaises(ValueError):
                self.gate.take_contextual_answer("momentum_agent", check_id, "C", "student-a", "session-a")

    async def test_ownership_options_expiry_and_one_shot(self):
        first = await self.create()
        cid = first["hitl"]["check_id"]
        for agent, option, user, session in (("energy_agent", "A", "student-a", "session-a"),
                                           ("momentum_agent", "Z", "student-a", "session-a"),
                                           ("momentum_agent", "A", "other", "session-a"),
                                           ("momentum_agent", "A", "student-a", "other")):
            with self.assertRaises(ValueError):
                self.gate.take_contextual_answer(agent, cid, option, user, session)
            self.assertIn(cid, self.gate._pending_checks)
        self.gate._pending_checks[cid]["created_at"] = 0
        with self.assertRaises(ValueError):
            self.gate.take_contextual_answer("momentum_agent", cid, "A", "student-a", "session-a")

    async def test_remediation_keeps_actual_reply_and_history_and_clears_on_completion(self):
        first = await self.create()
        cid = first["hitl"]["check_id"]
        await solve_mechanics_turn(self.agent, self.gate, request(context={
            "knowledge_transfer_response": {"check_id": cid, "selected_option_id": "B"}}), None)
        stored_history = copy.deepcopy(self.gate._contextual_answers[cid]["history"])
        history = {"recent_messages": [{"role": "assistant", "content": "Unrelated old problem"}]}
        followup = {"student_message": "p=mv", "original_problem": "A forged problem", "check_id": cid,
                    "concept_tag": "forged_concept"}
        result = await solve_mechanics_turn(self.agent, self.gate, request(context={
            "knowledge_transfer_remediation_followup": followup}), history)
        self.assertEqual(self.agent.solve_problem.call_args.kwargs["problem"], "p=mv")
        actual_history = self.agent.solve_problem.call_args.kwargs["context"]["conversation_context"]["recent_messages"]
        self.assertEqual(actual_history, stored_history)
        self.assertIn("carts", actual_history[0]["content"])
        self.assertIn("B: Kinetic energy", actual_history[-2]["content"])
        self.assertEqual(result["hitl"]["concept_tag"], CHECK["concept_tag"])
        self.assertEqual(result["hitl"]["status"], "remediation_followup")
        self.assertEqual(self.gate._contextual_answers[cid]["history"][-2]["content"], "p=mv")
        self.agent.solve_problem.return_value["metadata"]["quiz_reflection_mode"]["stage"] = "complete"
        result = await solve_mechanics_turn(self.agent, self.gate, request(context={
            "knowledge_transfer_remediation_followup": followup}), history)
        self.assertEqual(result["hitl"]["status"], "remediation_complete")

    async def test_full_solution_and_disabled_gate(self):
        first = await self.create()
        cid = first["hitl"]["check_id"]
        await solve_mechanics_turn(self.agent, self.gate, request(context={
            "knowledge_transfer_response": {"check_id": cid, "selected_option_id": "B"}}), None)
        result = await solve_mechanics_turn(self.agent, self.gate, request(context={
            "knowledge_transfer_remediation_followup": {"student_message": "I'm lost", "mode": "full_solution",
                "check_id": cid, "original_problem": "A forged problem"}}), None)
        self.assertIn("Two carts stick", self.agent.solve_problem.call_args.kwargs["problem"])
        self.assertNotIn("forged", self.agent.solve_problem.call_args.kwargs["problem"])
        self.assertEqual(result["hitl"]["status"], "full_solution_requested")
        self.gate.enabled = False
        self.agent.prepare_contextual_check.reset_mock()
        await self.create()
        self.agent.prepare_contextual_check.assert_not_called()

    async def test_remediation_rejects_wrong_owner_session_agent_and_expired_check(self):
        first = await self.create()
        cid = first["hitl"]["check_id"]
        await solve_mechanics_turn(self.agent, self.gate, request(context={
            "knowledge_transfer_response": {"check_id": cid, "selected_option_id": "B"}}), None)
        followup = {"knowledge_transfer_remediation_followup": {"check_id": cid, "student_message": "p=mv"}}
        self.agent.solve_problem.reset_mock()
        for user, session, agent in (("another-student", "session-a", "momentum_agent"),
                                     ("student-a", "another-session", "momentum_agent"),
                                     ("student-a", "session-a", "energy_agent")):
            with self.subTest(user=user, session=session, agent=agent):
                turn = request(context=followup, user=user)
                turn.session_id = session
                self.agent.agent_id = agent
                with self.assertRaises(ValueError):
                    await solve_mechanics_turn(self.agent, self.gate, turn, None)
        self.agent.agent_id = "momentum_agent"
        self.gate._contextual_answers[cid]["check"]["created_at"] = 0
        with self.assertRaises(ValueError):
            await solve_mechanics_turn(self.agent, self.gate, request(context=followup), None)
        self.agent.solve_problem.assert_not_awaited()

    async def test_failed_guided_turn_preserves_history_for_retry_without_logging_twice(self):
        first = await self.create()
        cid = first["hitl"]["check_id"]
        selection = {"knowledge_transfer_response": {"check_id": cid, "selected_option_id": "B"}}
        await solve_mechanics_turn(self.agent, self.gate, request(context=selection), None)
        receipt = self.gate._contextual_answers[cid]
        history = copy.deepcopy(receipt["history"])
        self.agent.solve_problem.return_value = {"success": False, "error": "Model unavailable"}
        followup = {"knowledge_transfer_remediation_followup": {"check_id": cid, "student_message": "I'm lost"}}
        failed = await solve_mechanics_turn(self.agent, self.gate, request(context=followup), None)
        self.assertFalse(failed["success"])
        self.assertEqual(receipt["history"], history)
        context = self.agent.solve_problem.call_args.kwargs["context"]
        self.assertTrue(context["guided_tutoring"])
        self.assertFalse(context.get("full_solution_requested", False))
        self.agent.solve_problem.return_value = {"success": True, "solution": "Which system are we studying?",
            "metadata": {"quiz_reflection_mode": {"stage": "feedback"}}}
        retried = await solve_mechanics_turn(self.agent, self.gate, request(context=followup), None)
        self.assertEqual(retried["hitl"]["status"], "remediation_followup")
        self.assertEqual(len(receipt["history"]), len(history) + 2)
        self.gate._log_attempt.assert_called_once()


if __name__ == "__main__":
    unittest.main()
