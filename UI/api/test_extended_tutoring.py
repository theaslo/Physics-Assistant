"""Cross-layer contracts for the three additional contextual tutors."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from hitl_knowledge_transfer import KnowledgeTransferGate
from mechanics_tutoring import solve_mechanics_turn
from strands_agents.energy_agent import EnergyAgent
from strands_agents.angular_motion_agent import AngularMotionAgent
from strands_agents.thermodynamics_agent import ThermodynamicsAgent
from physics_worked_solutions import ground_domain_working


CASES = (
    (EnergyAgent, "A 2 kg object moves at 3 m/s. Calculate its kinetic energy.",
     "calculate_kinetic_energy_tool", {"mass": 2, "velocity": 3}, "KE = 9.000 J", "9 J",
     "K = 0.5mv^2", "Nonnegative kinetic energy depends on speed squared."),
    (AngularMotionAgent, "A wheel starts at 2 rad/s with constant acceleration 3 rad/s^2 for 4 s. Find its final angular velocity.",
     "angular_kinematics", {"kinematics_data": json.dumps({"omega_0": 2, "alpha": 3, "time": 4})},
     "omega_0 = 2 rad/s\nalpha = 3 rad/s^2\nt = 4 s\nomega_f = omega_0 + alpha*t = 2 + 3*4 = 14 rad/s\ntheta_f = theta_0 + omega_0*t + 0.5*alpha*t^2 = 32 rad", "14 rad/s",
     "omega_f = omega_i + alpha t", "These equations require constant angular acceleration."),
    (ThermodynamicsAgent, "A gas absorbs 500 J of heat and does 200 J of work. Find the change in its internal energy.",
     "first_law_thermodynamics", {"energy_data": json.dumps({"heat": 500, "work_by_system": 200})},
     "Heat into the system: Q = 500 J\nWork by the system: W_by = 200 J\nΔU = 500 - (200) = 300 J", "300 J",
     "Delta U = Q - W_by", "Heat into the gas is positive and work by it removes energy."),
)


def check_for(problem):
    return {
        "needs_check": True, "confidence": 0.95, "concept_tag": "system_assumption",
        "question": "Which physical assumption applies to this situation?",
        "options": [{"id": "A", "text": "Use the stated physical conditions."},
                    {"id": "B", "text": "Assume an additional conserved quantity."},
                    {"id": "C", "text": "Ignore the energy or momentum transfer."}],
        "correct_option_id": "A",
    }


def agent_for(case):
    cls, problem, tool, args, output, answer, equation, principle = case
    agent = cls(enable_database_logging=False, enable_rag=False)
    agent.initialized = True
    agent.mcp_client = Mock()
    agent.mcp_tool_specs = [{"name": tool, "description": "Compute the requested quantity",
                             "inputSchema": {"json": {"type": "object"}}}]
    agent.mcp_client.call_tool_sync.side_effect = lambda **kw: {
        "status": "success", "content": [{"text": principle if kw["name"] == agent.principles_tool else output}]
    }
    return agent


def request(problem, context=None, user="synthetic-user", session="synthetic-session"):
    return SimpleNamespace(problem=problem, context=context, user_id=user, session_id=session)


class ExtendedTutorTests(unittest.IsolatedAsyncioTestCase):
    def test_skater_working_completes_the_energy_comparison_from_executed_results(self):
        worked = {"result": "The final angular velocity is 6 rad/s."}
        outputs = [{"tool": "angular_momentum_conservation", "arguments": {"momentum_data": {
            "initial": {"I": 4, "omega": 3}, "final": {"I": 2}}}, "result": (
                "ω₂ = (I₁ω₁)/I₂ = (4 × 3)/2 = 6.000 rad/s\n"
                "Initial kinetic energy: KE₁ = ½I₁ω₁² = ½ × 4 × 3² = 18.000 J\n"
                "Final kinetic energy: KE₂ = ½I₂ω₂² = ½ × 2 × 6² = 36.000 J\n"
                "Energy change: ΔKE = KE₂ - KE₁ = 18.000 J\n") }]
        result = ground_domain_working(worked, outputs)
        self.assertIn("36.000", result["substitution"])
        self.assertIn("18.000", result["substitution"])
        self.assertIn("Rotational kinetic energy is not conserved", result["result"])
        self.assertEqual(ground_domain_working(result, outputs), result)

    async def test_missing_inputs_rejected_by_a_calculator_become_a_clarification(self):
        agent = agent_for(CASES[2])
        agent.mcp_tool_specs = [{"name": "heat_transfer", "inputSchema": {"json": {"type": "object"}}}]
        agent.mcp_client.call_tool_sync.side_effect = lambda **kw: {
            "status": "success", "content": [{"text": "Q=mc Delta T." if kw["name"] == agent.principles_tool
                                                 else "Error: Provide three known values among heat, mass, specific_heat and temp_change"}],
        }
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "", "tool_calls": [{"tool_name": "heat_transfer", "arguments": {"heat_data": {"temp_change": 10}}}]}),
            json.dumps({"tool_calls": [], "clarification_question": "What are the object's mass and specific heat?"}),
        ])
        result = await agent.solve_problem("An object warms by 10 K without a phase change. Find the heat absorbed.")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["solution"], "What are the object's mass and specific heat?")
        self.assertEqual(result["tools_used"], [agent.principles_tool])
        self.assertNotIn("heat_transfer", result["solution"])

    async def test_a_missing_data_explanation_is_repaired_into_an_actual_question(self):
        agent = agent_for(CASES[0])
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "The mass was not provided.", "tool_calls": []}),
            json.dumps({"task_type": "clarification", "tool_calls": [], "corrected_answer": "The mass was not provided."}),
            json.dumps({"task_type": "clarification", "tool_calls": [], "corrected_answer": "What is the cart's mass?"}),
        ])
        result = await agent.solve_problem("A cart moves at 3 m/s. Find its kinetic energy.")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["solution"], "What is the cart's mass?")
        self.assertEqual(agent.mcp_client.call_tool_sync.call_count, 1)

    async def test_lost_student_gets_a_smaller_question_instead_of_the_same_question(self):
        agent = agent_for(CASES[1])
        problem = "A skater changes inertia under negligible external torque. Find the final angular speed."
        previous = "When there is negligible external torque, which quantity is conserved?"
        smaller = "Is the net external torque zero in this setup?"
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"action": "feedback", "problem_message_index": 0, "problem_statement": problem,
                        "calculation_required": False, "tool_calls": []}),
            json.dumps({"assessment": "unclear", "complete": False, "outcome_evidence": "", "reason_evidence": "", "question": previous + " (Hint: It is not mechanical energy.)"}),
            json.dumps({"question": smaller}),
        ])
        result = await agent.solve_problem("I'm lost.", {"guided_tutoring": True, "conversation_context": {
            "recent_messages": [{"role": "user", "content": problem},
                                {"role": "assistant", "content": "Not quite.\n\n" + previous}],
        }})
        self.assertTrue(result["success"], result)
        self.assertIn(smaller, result["solution"])
        self.assertNotIn(previous, result["solution"])
        self.assertEqual(result["solution"].count("?"), 1)

    async def test_repeated_completed_calculation_finishes_without_rerunning_it(self):
        for repair_first in (False, True):
            with self.subTest(repair_first=repair_first):
                agent = agent_for(CASES[2])
                name = "ideal_gas_law"
                agent.mcp_tool_specs = [{"name": name, "inputSchema": {"json": {"type": "object"}}}]
                known = {"volume": 0.01, "moles": 1, "temperature": 300}
                initial = {**known, "pressure": 0} if repair_first else known
                def call(**kwargs):
                    text = "PV=nRT."
                    if kwargs["name"] == name:
                        data = json.loads(kwargs["arguments"]["gas_data"])
                        text = "Error: pressure must be positive" if "pressure" in data else "P = 249420.00 Pa"
                        if "pressure" not in data:
                            text += '\nDIAGRAM_JSON_START\n' + json.dumps({
                                "type": "pv_diagram", "state": {"pressure_pa": 249420, "volume_m3": 0.01,
                                "moles": 1, "temperature_k": 300}, "pv_points": [], "isotherms": [],
                            }) + '\nDIAGRAM_JSON_END'
                    return {"status": "success", "content": [{"text": text}]}
                agent.mcp_client.call_tool_sync.side_effect = call
                first_call = {"tool_name": name, "arguments": {"gas_data": json.dumps(initial)}}
                repeated = initial if repair_first else {"temperature": 300, "moles": 1, "volume": 0.01}
                sequence = [json.dumps({"answer": "", "tool_calls": [first_call]})]
                if repair_first:
                    sequence.append(json.dumps({"tool_calls": [{"tool_name": name, "arguments": {"gas_data": known}}]}))
                sequence.extend([
                    json.dumps({"answer": "", "show_diagram": True, "tool_calls": [{"tool_name": name, "arguments": {"gas_data": repeated}}]}),
                    json.dumps({"givens": "1 mol at 300 K in 0.01 m^3", "principle": "Ideal gas law",
                                "equation": "PV=nRT", "substitution": "P=(1)(8.314)(300)/0.01=249420",
                                "result": "249420 Pa", "check": "Absolute pressure is positive."}),
                    json.dumps({"task_type": "numerical", "tool_calls": [], "corrected_answer": ""}),
                ])
                agent._request_ollama_generate = Mock(side_effect=sequence)
                result = await agent.solve_problem("An ideal gas has 1 mole at 300 K in 0.010 m^3. Find its pressure.")
                self.assertTrue(result["success"], result)
                self.assertIn("249420", result["solution"])
                self.assertEqual(agent.mcp_client.call_tool_sync.call_count, 3 if repair_first else 2)
                self.assertEqual(result["tools_used"].count(name), 1)
                self.assertEqual(result["diagram"]["type"], "pv_diagram")

    async def test_missing_givens_can_prompt_one_question_without_fabricated_calculations(self):
        for case, question in zip(CASES, ("What is the mass of the cart?", "What is its moment of inertia about this axis?", "What is the mass of the material?")):
            with self.subTest(agent=case[0].__name__):
                agent = agent_for(case)
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"answer": question, "tool_calls": []}),
                    json.dumps({"needs_clarification": True, "question": question}),
                ])
                result = await agent.solve_problem("Find the requested energy or momentum; a necessary physical quantity is not given.")
                self.assertTrue(result["success"], result)
                self.assertEqual(result["solution"], question)
                self.assertEqual(result["tools_used"], [agent.principles_tool])
                self.assertEqual(agent.mcp_client.call_tool_sync.call_count, 1)

    async def test_unknown_gas_pressure_placeholder_is_repaired_without_inventing_a_given(self):
        agent = agent_for(CASES[2])
        tool = "ideal_gas_law"
        agent.mcp_tool_specs = [{"name": tool, "inputSchema": {"json": {"type": "object"}}}]
        valid = {"volume": 0.01, "moles": 1, "temperature": 300}
        invalid = {**valid, "pressure": 0}
        def call(**kwargs):
            if kwargs["name"] == agent.principles_tool:
                text = "PV = nRT with positive absolute pressure and kelvin temperature."
            else:
                data = json.loads(kwargs["arguments"]["gas_data"])
                text = "Error: pressure must be positive" if "pressure" in data else "P = 249420.00 Pa"
            return {"status": "success", "content": [{"text": text}]}
        agent.mcp_client.call_tool_sync.side_effect = call
        worked = {"givens": "n=1 mol, T=300 K, V=0.01 m^3", "principle": "PV=nRT",
                  "equation": "P=nRT/V", "substitution": "(1)(8.314)(300)/0.01=249420",
                  "result": "249420 Pa", "check": "Absolute pressure is positive."}
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "", "tool_calls": [{"tool_name": tool, "arguments": {"gas_data": json.dumps(invalid)}}]}),
            json.dumps({"tool_calls": [{"tool_name": tool, "arguments": {"gas_data": valid}}]}),
            json.dumps({"answer": "249420 Pa", "tool_calls": [], "worked_solution": worked}),
            json.dumps({"task_type": "numerical", "tool_calls": [], "corrected_answer": ""}),
        ])
        result = await agent.solve_problem("An ideal gas has 1 mole at 300 K in volume 0.010 m^3. Find its pressure.")
        self.assertTrue(result["success"], result)
        self.assertIn("249420", result["solution"])
        self.assertNotIn("ideal_gas_law", result["solution"])
        executed = [call.kwargs["arguments"] for call in agent.mcp_client.call_tool_sync.call_args_list
                    if call.kwargs["name"] == tool]
        self.assertEqual(json.loads(executed[-1]["gas_data"]), valid)
        self.assertEqual(len(executed), 2)

    async def test_unrepairable_calculation_does_not_expose_internal_tool_names(self):
        agent = agent_for(CASES[2])
        agent.mcp_tool_specs = [{"name": "ideal_gas_law", "inputSchema": {"json": {"type": "object"}}}]
        agent.mcp_client.call_tool_sync.side_effect = lambda **kw: {
            "status": "success", "content": [{"text": "Principles" if kw["name"] == agent.principles_tool else "Error: invalid pressure"}],
        }
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "", "tool_calls": [{"tool_name": "ideal_gas_law", "arguments": {"gas_data": {"pressure": -1}}}]}),
            json.dumps({"tool_calls": []}),
        ])
        result = await agent.solve_problem("A gas is stated to have negative absolute pressure; calculate its state.")
        self.assertFalse(result["success"])
        self.assertNotIn("ideal_gas_law", result["error"])
        self.assertNotIn("MCP", result["error"])

    async def test_two_part_thermodynamics_problem_cannot_stop_after_work(self):
        agent = agent_for(CASES[2])
        tools = ("constant_pressure_work", "first_law_thermodynamics")
        agent.mcp_tool_specs = [{"name": name, "inputSchema": {"json": {"type": "object"}}} for name in tools]
        outputs = {
            agent.principles_tool: "Delta U = Q - W_by. Constant-pressure work is P Delta V.",
            tools[0]: "W_by = 100000 * (0.013 - 0.01) = 300 J",
            tools[1]: "Heat into the system: Q = 500 J\nWork by the system: W_by = 300 J\nΔU = 500 - (300) = 200 J",
        }
        agent.mcp_client.call_tool_sync.side_effect = lambda **kw: {
            "status": "success", "content": [{"text": outputs[kw["name"]]}],
        }
        first_call = {"tool_name": tools[0], "arguments": {"process_data": json.dumps({
            "pressure": 100000, "initial_volume": 0.01, "final_volume": 0.013})}}
        second_call = {"tool_name": tools[1], "arguments": {"energy_data": json.dumps({"heat": 500, "work_by_system": 300})}}
        partial = {"givens": "500 J absorbed at constant pressure", "principle": "Work at constant pressure",
                   "equation": "W=P(V_f-V_i)", "substitution": "100000(0.013-0.01)=300",
                   "result": "Work is 300 J", "check": "Expansion does positive work."}
        complete = {**partial, "result": "Work is 300 J and internal energy increases by 200 J."}
        agent._request_ollama_generate = Mock(side_effect=[
            json.dumps({"answer": "", "tool_calls": [first_call]}),
            json.dumps({"answer": "Work is 300 J", "tool_calls": [], "worked_solution": partial}),
            json.dumps({"task_type": "numerical", "tool_calls": [second_call], "corrected_answer": ""}),
            json.dumps({"task_type": "numerical", "tool_calls": [], "corrected_answer": complete["result"]}),
            json.dumps({"answer": complete["result"], "tool_calls": [], "worked_solution": complete}),
            json.dumps({"task_type": "numerical", "tool_calls": [], "corrected_answer": ""}),
        ])
        result = await agent.solve_problem("A gas expands at constant pressure 100000 Pa from 0.010 to 0.013 m^3 and absorbs 500 J. Find the work and change in internal energy.")
        self.assertTrue(result["success"], result)
        self.assertIn("300 J and internal energy increases by 200 J", result["solution"])
        self.assertTrue(set(tools).issubset(result["tools_used"]))
        self.assertEqual(agent._request_ollama_generate.call_count, 6)

    async def test_formula_only_substitution_is_repaired_before_finishing(self):
        for case, substitution in zip(CASES, (r"\frac12(2)(3)^2=9", "2+(3)(4)=14", "500-200=300")):
            with self.subTest(agent=case[0].__name__):
                agent = agent_for(case)
                tool = "composite_domain_calculation"
                agent.mcp_tool_specs = [{"name": tool, "inputSchema": {"json": {"type": "object"}}}]
                worked = {"givens": case[1], "principle": case[7], "equation": case[6],
                          "substitution": r"\frac{1}{2} m v_i^2", "result": case[5], "check": case[7]}
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"answer": "", "tool_calls": [{"tool_name": tool, "arguments": {"known": 3}}]}),
                    json.dumps({"answer": case[5], "tool_calls": [], "worked_solution": worked}),
                    json.dumps({"task_type": "numerical", "tool_calls": [], "corrected_answer": ""}),
                    json.dumps({**worked, "substitution": substitution}),
                ])
                result = await agent.solve_problem(case[1])
                self.assertTrue(result["success"], result)
                self.assertIn(substitution, result["solution"])
                self.assertIn("actual numerical givens", agent._request_ollama_generate.call_args.kwargs["system"])

    async def test_conceptual_answers_remove_internal_tool_details(self):
        for case in CASES:
            with self.subTest(agent=case[0].__name__):
                agent = agent_for(case)
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"answer": "According to MCP " + agent.principles_tool + ", use the stated conditions.", "tool_calls": []}),
                    json.dumps({"task_type": "conceptual", "tool_calls": []}),
                    json.dumps({"answer": "Use the stated physical conditions and energy-transfer directions."}),
                ])
                result = await agent.solve_problem("Explain the relevant physical assumptions.")
                self.assertTrue(result["success"], result)
                self.assertNotIn("MCP", result["solution"])
                self.assertNotIn(agent.principles_tool, result["solution"])
                self.assertEqual(result["tools_used"], [agent.principles_tool])

    async def test_explicit_solution_mode_skips_question_generation(self):
        for case in CASES:
            agent = agent_for(case)
            agent._reflection_model = AsyncMock()
            self.assertIsNone(await agent.prepare_contextual_check(case[1], {"full_solution_requested": True}))
            agent._reflection_model.assert_not_awaited()
            agent.mcp_client.call_tool_sync.assert_not_called()

    async def test_simple_calculations_can_skip_check_but_must_show_complete_working(self):
        for case in CASES:
            with self.subTest(agent=case[0].__name__):
                agent = agent_for(case)
                worked = {"givens": case[1], "principle": case[7], "equation": case[6],
                          "substitution": "Incomplete draft", "result": case[5], "check": case[7]}
                agent._request_ollama_generate = Mock(side_effect=[
                    json.dumps({"needs_check": False}),
                    json.dumps({"answer": "", "tool_calls": [{"tool_name": case[2], "arguments": case[3]}]}),
                    json.dumps({"answer": case[5], "tool_calls": []}),
                    json.dumps({"task_type": "numerical", "tool_calls": [], "corrected_answer": ""}),
                    json.dumps(worked),
                ])
                gate = KnowledgeTransferGate("http://unused")
                result = await solve_mechanics_turn(agent, gate, request(case[1]), None)
                self.assertTrue(result["success"], result)
                self.assertIsNone(result.get("hitl"))
                for heading in ("Given", "Principle", "Equation", "Substitution", "Result", "Check"):
                    self.assertIn(f"**{heading}**", result["solution"])
                self.assertIn(case[5], result["solution"])
                self.assertNotIn("Incomplete draft", result["solution"])
                self.assertNotRegex(result["solution"], r"\bMCP\b|numerical tool|verified results")
                self.assertNotIn(case[2], result["solution"])
                self.assertIn(case[2], result["tools_used"])

    async def test_each_new_tutor_independently_reviews_ambiguous_options(self):
        for case in CASES:
            with self.subTest(agent=case[0].__name__):
                agent = agent_for(case)
                check = check_for(case[1])
                agent._reflection_model = AsyncMock(side_effect=[
                    copy.deepcopy(check), {"correct_option_ids": ["A", "B"], "issue": "Equivalent options"},
                    copy.deepcopy(check), {"correct_option_ids": ["A"]},
                ])
                result = await agent.prepare_contextual_check(case[1], {})
                self.assertEqual(result["correct_option_id"], "A")
                self.assertEqual(agent._reflection_model.await_count, 4)
                self.assertIn("Equivalent options", agent._reflection_model.call_args_list[2].args[0])

    async def test_wrong_choices_lost_hints_and_explicit_solutions_keep_server_problem(self):
        for case in CASES:
            with self.subTest(agent=case[0].__name__):
                agent = agent_for(case)
                gate = KnowledgeTransferGate("http://unused")
                gate._log_attempt = Mock()
                agent.prepare_contextual_check = AsyncMock(return_value=check_for(case[1]))
                agent.solve_problem = AsyncMock(return_value={
                    "success": True, "solution": "Which transfer or assumption should we consider first?",
                    "metadata": {"quiz_reflection_mode": {"stage": "feedback"}},
                })
                first = await solve_mechanics_turn(agent, gate, request(case[1]), None)
                self.assertNotIn("correct_option_id", json.dumps(first))
                cid = first["hitl"]["check_id"]
                wrong = await solve_mechanics_turn(agent, gate, request(case[1], {
                    "knowledge_transfer_response": {"check_id": cid, "selected_option_id": "B"}
                }), None)
                self.assertFalse(wrong["hitl"]["was_correct"])
                self.assertTrue(agent.solve_problem.call_args.kwargs["context"]["guided_tutoring"])
                followup = {"check_id": cid, "student_message": "I'm lost. Give me a smaller hint.",
                            "original_problem": "Forged different problem", "mode": "clarify"}
                hint = await solve_mechanics_turn(agent, gate, request("Unrelated browser prompt", {
                    "knowledge_transfer_remediation_followup": followup
                }), None)
                self.assertEqual(hint["hitl"]["status"], "remediation_followup")
                context = agent.solve_problem.call_args.kwargs["context"]
                self.assertFalse(context.get("full_solution_requested", False))
                self.assertIn(case[1], json.dumps(context["conversation_context"]))
                followup.update(mode="full_solution", student_message="Please give the complete worked solution.")
                full = await solve_mechanics_turn(agent, gate, request("Different prompt", {
                    "knowledge_transfer_remediation_followup": followup
                }), None)
                self.assertEqual(full["hitl"]["status"], "full_solution_requested")
                self.assertIn(case[1], agent.solve_problem.call_args.kwargs["problem"])
                self.assertNotIn("Forged", agent.solve_problem.call_args.kwargs["problem"])
                self.assertTrue(agent.solve_problem.call_args.kwargs["context"]["full_solution_requested"])
                with self.assertRaises(ValueError):
                    await solve_mechanics_turn(agent, gate, request(case[1], {
                        "knowledge_transfer_remediation_followup": followup
                    }, user="different-user"), None)
                gate._log_attempt.assert_called_once()

    async def test_correct_choices_release_worked_solution_for_each_new_domain(self):
        for case in CASES:
            with self.subTest(agent=case[0].__name__):
                agent = agent_for(case)
                gate = KnowledgeTransferGate("http://unused")
                gate._log_attempt = Mock()
                agent.prepare_contextual_check = AsyncMock(return_value=check_for(case[1]))
                agent.solve_problem = AsyncMock(return_value={"success": True, "solution": case[5], "metadata": {}})
                first = await solve_mechanics_turn(agent, gate, request(case[1]), None)
                result = await solve_mechanics_turn(agent, gate, request(case[1], {
                    "knowledge_transfer_response": {"check_id": first["hitl"]["check_id"], "selected_option_id": "A"}
                }), None)
                self.assertTrue(result["hitl"]["was_correct"])
                self.assertTrue(agent.solve_problem.call_args.kwargs["context"]["full_solution_requested"])


if __name__ == "__main__":
    unittest.main()
