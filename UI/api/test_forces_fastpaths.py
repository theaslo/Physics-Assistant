import json
import unittest
from unittest.mock import Mock

from strands_agents.forces_agent import ForcesAgent


class ForcesFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_spring_force_uses_direct_mcp_path(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        calls = {}

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id, name, arguments):
                calls["tool_use_id"] = tool_use_id
                calls["name"] = name
                calls["arguments"] = arguments
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                "Spring Force Calculation (Hooke's Law):\n"
                                "F = -(50.00) x (0.20)\n"
                                "F = -10.00 N\n"
                                "Force magnitude: 10.00 N"
                            )
                        }
                    ],
                }

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def forbidden_llm_plan(*args, **kwargs):
            raise AssertionError("Direct spring path should not wait for the LLM router")

        async def forbidden_full_agent(*args, **kwargs):
            raise AssertionError("Direct spring path should not invoke the full agent")

        agent.initialize = fake_initialize
        agent._build_fast_mcp_plan_with_llm = forbidden_llm_plan
        agent._call_agent_async = forbidden_full_agent

        result = await agent.solve_problem(
            problem="A spring stretches 0.2 m with k=50 N/m. Use Hooke law to find the force.",
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["calculate_spring_force_tool"])
        self.assertEqual(calls["name"], "calculate_spring_force_tool")
        self.assertEqual(calls["arguments"]["spring_data"], '{"spring_constant": 50.0, "displacement": 0.2}')
        self.assertIn("10.00 N", result["solution"])
        self.assertIn("Hooke", result["solution"])
        self.assertTrue(result["metadata"]["fastpath_recovery"])

    async def test_incline_fast_path_uses_direct_mcp_and_reports_acceleration(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        calls = []

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id, name, arguments):
                calls.append((name, arguments))
                if name == "analyze_forces_on_incline":
                    return {
                        "status": "success",
                        "content": [
                            {
                                "text": (
                                    "Forces on Inclined Plane Analysis:\n"
                                    "Net force down incline = 11.78 N\n"
                                    "DIAGRAM_JSON_START\n"
                                    "{\"type\":\"inclined_plane_diagram\",\"title\":\"Inclined Plane Forces\","
                                    "\"mass_kg\":5.0,\"angle_deg\":30.0,\"coefficient_friction\":0.3,"
                                    "\"weight_n\":49.05,\"weight_parallel_n\":24.525,"
                                    "\"weight_perpendicular_n\":42.48,\"normal_n\":42.48,"
                                    "\"net_down_n\":11.78,\"has_friction\":true,\"friction_n\":12.74}"
                                    "\nDIAGRAM_JSON_END"
                                )
                            }
                        ],
                    }
                raise AssertionError(f"Unexpected tool call: {name}")

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def forbidden_llm_plan(*args, **kwargs):
            raise AssertionError("Direct incline path should not wait for the LLM router")

        async def forbidden_full_agent(*args, **kwargs):
            raise AssertionError("Direct incline path should not invoke the full agent")

        agent.initialize = fake_initialize
        agent._build_fast_mcp_plan_with_llm = forbidden_llm_plan
        agent._call_agent_async = forbidden_full_agent

        result = await agent.solve_problem(
            problem=(
                "A 5.0 kg box slides down a 30 degree incline with coefficient of kinetic friction 0.30. "
                "Draw the free-body diagram and find the acceleration down the ramp."
            ),
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual([call[0] for call in calls], ["analyze_forces_on_incline"])
        self.assertEqual(result["tools_used"], ["analyze_forces_on_incline"])
        self.assertNotIn("Error:", result["solution"])
        self.assertIn("Acceleration magnitude: 2.36 m/s", result["solution"])
        self.assertEqual(result["diagram"]["type"], "inclined_plane_diagram")

    async def test_level_surface_friction_example_uses_direct_mcp_without_llm(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        calls = []

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id, name, arguments):
                calls.append((name, arguments))
                if name == "resolve_force_components":
                    return {
                        "status": "success",
                        "content": [
                            {
                                "text": (
                                    "Force Component Resolution:\n"
                                    "Given Force:\n"
                                    "- Magnitude: 40.00 N\n"
                                    "- Angle: 25.0° (from positive x-axis)\n"
                                    "Fx = F × cos(θ) = 40.00 × cos(25.0°) = 36.25 N\n"
                                    "Fy = F × sin(θ) = 40.00 × sin(25.0°) = 16.90 N\n"
                                )
                            }
                        ],
                    }
                if name == "calculate_weight_force":
                    return {
                        "status": "success",
                        "content": [{"text": "Weight Force Calculation:\nW = 117.72 N\nWeight force magnitude: 117.72 N"}],
                    }
                if name == "calculate_friction_force_tool":
                    return {
                        "status": "success",
                        "content": [{"text": "Friction Force Calculation:\nf = 0.100 × 100.82\nf = 10.08 N\nMagnitude: 10.08 N"}],
                    }
                if name == "add_forces_1d":
                    return {"status": "success", "content": [{"text": "Net Force: 26.17 N"}]}
                if name == "newton_second_law":
                    return {"status": "success", "content": [{"text": "a = 2.18 m/s²"}]}
                raise AssertionError(f"Unexpected tool call: {name}")

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def forbidden_llm_plan(*args, **kwargs):
            raise AssertionError("Level-surface friction path should not wait for the LLM router")

        async def forbidden_full_agent(*args, **kwargs):
            raise AssertionError("Level-surface friction path should not invoke the full agent")

        agent.initialize = fake_initialize
        agent._build_fast_mcp_plan_with_llm = forbidden_llm_plan
        agent._call_agent_async = forbidden_full_agent

        result = await agent.solve_problem(
            problem=(
                "A 12 kg sled is pulled across level snow by a 40 N rope at 25° above the horizontal. "
                "If μk = 0.10, find the normal force and the sled’s acceleration."
            ),
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["tools_used"],
            [
                "resolve_force_components",
                "calculate_weight_force",
                "calculate_friction_force_tool",
                "add_forces_1d",
                "newton_second_law",
            ],
        )
        self.assertEqual([call[0] for call in calls], result["tools_used"])
        self.assertIn("N = W - F_y", result["solution"])
        self.assertIn("100.82 N", result["solution"])
        self.assertIn("10.08 N", result["solution"])
        self.assertIn("2.18 m/s", result["solution"])
        self.assertTrue(result["metadata"]["fastpath_recovery"])

    async def test_simple_tension_equilibrium_uses_direct_mcp_without_llm(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        calls = []

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id, name, arguments):
                calls.append((name, arguments))
                if name == "calculate_weight_force":
                    return {
                        "status": "success",
                        "content": [{"text": "Weight Force Calculation:\nWeight force magnitude: 49.05 N"}],
                    }
                raise AssertionError(f"Unexpected tool call: {name}")

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def forbidden_llm_plan(*args, **kwargs):
            raise AssertionError("Tension equilibrium path should not wait for the LLM router")

        async def forbidden_full_agent(*args, **kwargs):
            raise AssertionError("Tension equilibrium path should not invoke the full agent")

        agent.initialize = fake_initialize
        agent._build_fast_mcp_plan_with_llm = forbidden_llm_plan
        agent._call_agent_async = forbidden_full_agent

        result = await agent.solve_problem(
            problem="A 5.0 kg mass hangs at rest from a vertical rope. Find the tension.",
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["calculate_weight_force"])
        self.assertEqual(calls[0][0], "calculate_weight_force")
        self.assertIn("T = W = 49.05 N", result["solution"])
        self.assertTrue(result["metadata"]["fastpath_recovery"])

    async def test_newton_quiz_reflection_uses_mcp_without_generic_hitl(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        calls = []
        case = self

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id, name, arguments):
                calls.append((name, arguments))
                case.assertEqual(name, "check_equilibrium")
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                "Equilibrium Analysis:\n"
                                "Net Force = 6.00 N at 0.0 degrees\n"
                                "NOT IN EQUILIBRIUM"
                            )
                        }
                    ],
                }

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def forbidden_llm_router(*args, **kwargs):
            raise AssertionError("Reflection mode should use the direct MCP grounding path")

        async def forbidden_full_agent(*args, **kwargs):
            raise AssertionError("Reflection mode should not fall through to the full agent")

        agent.initialize = fake_initialize
        agent._build_fast_mcp_plan_with_llm = forbidden_llm_router
        agent._call_agent_async = forbidden_full_agent

        result = await agent.solve_problem(
            problem=(
                "I am analyzing a mistake from my Newton's 2nd law quiz. "
                "The quiz question asked about net force and acceleration. "
                "My answer used the biggest force as ma, but the correct answer says to use net force."
            ),
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["check_equilibrium"])
        self.assertEqual(calls[0][0], "check_equilibrium")
        self.assertIn("Quiz 4 Reflection with Physics AI Tutor", result["solution"])
        self.assertIn("First question:", result["solution"])
        self.assertNotIn("MCP force-balance check", result["solution"])
        self.assertNotIn("Concept focus:", result["solution"])
        self.assertIsNone(result["diagram"])


        self.assertTrue(result["metadata"]["quiz_reflection_mode"]["enabled"])
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["concept_tag"], "newton_second_law")

    async def test_hockey_puck_quiz_reflection_asks_targeted_first_question(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id, name, arguments):
                return {
                    "status": "success",
                    "content": [{"text": "Equilibrium Analysis:\nEQUILIBRIUM ACHIEVED!"}],
                }

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def forbidden_full_agent(*args, **kwargs):
            raise AssertionError("Reflection mode should not fall through to the full agent")

        agent.initialize = fake_initialize
        agent._call_agent_async = forbidden_full_agent

        result = await agent.solve_problem(
            problem=(
                "(5 pts) Suppose you are playing hockey on a new-age ice surface for which there is no friction "
                "between the ice and the hockey puck. You wind up and hit the puck as hard as you can. "
                "After the puck loses contact with your stick, the puck will A) start to slow down. "
                "B) not slow down or speed up. C) speed up a little, and then slow down. "
                "D) speed up a little, and then move at a constant speed. "
                "I said A because it should speed up since I hit it first."
            ),
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["check_equilibrium"])
        self.assertIn("Quiz 4 Reflection with Physics AI Tutor", result["solution"])
        self.assertIn("After the puck loses contact with the stick", result["solution"])
        self.assertNotIn("What should you do before calculating this force problem", result["solution"])
        self.assertNotIn("MCP force-balance check", result["solution"])
        self.assertIsNone(result["diagram"])


class ForcesConversationTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, responses, calculation_output="Verified calculation: F = -18 N"):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        agent.initialized = True
        agent.mcp_client = Mock()
        agent.mcp_client.call_tool_sync.side_effect = lambda **kwargs: {
            "status": "success",
            "content": [{"text": "Kinetic friction opposes relative sliding; spring force is -kx."
                         if kwargs["name"] == "get_force_principles" else calculation_output}],
        }
        agent._request_ollama_generate = Mock(side_effect=responses)
        agent._build_forces_direct_solution = Mock(side_effect=AssertionError("Must not replay an old problem"))
        return agent

    def context(self, question, messages=None):
        return {
            "contextual_followup": True,
            "student_followup": question,
            "conversation_context": {
                "recent_messages": messages or [
                    {"role": "user", "content": "A spring has k = 50 N/m and is stretched 0.20 m to the right."},
                    {"role": "assistant", "content": "The spring force is 10 N to the left."},
                    {"role": "user", "content": "What is the direction of spring force?"},
                    {"role": "assistant", "content": "Opposite the stretch, to the left."},
                ]
            },
        }

    async def test_topic_switch_is_interpreted_by_model_with_mcp_principles(self):
        question = "on an inclide sliding box, which was would be the friction?"
        answer = "Friction opposes sliding: up the ramp if sliding down, down if sliding up. Which way is the box sliding?"
        agent = self.make_agent([json.dumps({"answer": answer, "tool_calls": []})])
        result = await agent.solve_problem(question, context=self.context(question))

        self.assertTrue(result["success"], result)
        self.assertEqual(result["solution"], answer)
        self.assertEqual(result["tools_used"], ["get_force_principles"])
        self.assertTrue(result["metadata"]["llm_response_used"])
        prompt = agent._request_ollama_generate.call_args.args[0]
        self.assertIn("CURRENT QUESTION:\n" + question, prompt)
        self.assertIn("k = 50 N/m", prompt)
        self.assertIn("Kinetic friction opposes relative sliding", prompt)
        self.assertNotIn("Preserve the earlier problem context", prompt)
        agent._build_forces_direct_solution.assert_not_called()

    async def test_quantitative_topic_switch_uses_new_inputs_then_model_reads_mcp_result(self):
        question = "Now a different spring has k = 90 N/m and extension 0.20 m. What is its force?"
        plan = {"answer": "", "tool_calls": [{"tool_name": "calculate_spring_force_tool",
                "arguments": {"spring_data": {"spring_constant": 90, "displacement": 0.20}}}]}
        agent = self.make_agent([json.dumps(plan), json.dumps({"answer": "The force is -18 N.", "tool_calls": []})])
        result = await agent.solve_problem(question, context=self.context(question))

        self.assertTrue(result["success"], result)
        self.assertEqual(result["tools_used"], ["get_force_principles", "calculate_spring_force_tool"])
        calls = agent.mcp_client.call_tool_sync.call_args_list
        self.assertEqual(json.loads(calls[1].kwargs["arguments"]["spring_data"]),
                         {"spring_constant": 90, "displacement": 0.20})
        self.assertEqual(agent._request_ollama_generate.call_count, 2)
        self.assertIn("Verified calculation: F = -18 N", agent._request_ollama_generate.call_args.args[0])

    async def test_varied_followups_all_reach_model_with_current_question_separate(self):
        messages = [
            {"role": "user", "content": "A box slides down a 30 degree ramp with friction."},
            {"role": "assistant", "content": "Friction points up the ramp."},
            {"role": "user", "content": "A spring is stretched to the right."},
            {"role": "assistant", "content": "The spring pulls left."},
        ]
        for question in (
            "What if it is compressed instead?",
            "Back to the box: what if it slides UP the ramp?",
            "Does that mean the acceleration must point left too?",
            "A new elevator accelerates upward. How does the cable tension compare to its weight?",
            "How do action-reaction pairs differ from balanced forces?",
            "What if the ramp is frictionless?",
        ):
            with self.subTest(question=question):
                agent = self.make_agent([json.dumps({"answer": "Model response for this turn.", "tool_calls": []})])
                result = await agent.solve_problem(question, context=self.context(question, messages))
                self.assertTrue(result["success"], result)
                prompt = agent._request_ollama_generate.call_args.args[0]
                self.assertIn("CURRENT QUESTION:\n" + question, prompt)
                self.assertLess(prompt.index("box slides down"), prompt.index("spring is stretched"))
                agent._build_forces_direct_solution.assert_not_called()

    async def test_model_failure_never_replays_spring_or_raw_tool_output(self):
        for response in (TimeoutError("offline"), "", "{}", '{"tool_calls":[{"tool_name":"invented_tool"}]}'):
            with self.subTest(response=response):
                agent = self.make_agent([response])
                question = "What about a box on an incline?"
                result = await agent.solve_problem(question, context=self.context(question))
                self.assertFalse(result["success"])
                self.assertNotIn("solution", result)
                agent._build_forces_direct_solution.assert_not_called()

    async def test_mcp_failure_is_not_claimed_as_verified_answer(self):
        agent = self.make_agent([])
        agent.mcp_client.call_tool_sync.side_effect = None
        agent.mcp_client.call_tool_sync.return_value = {"status": "error", "content": [{"text": "Error: offline"}]}
        question = "What about the friction?"
        result = await agent.solve_problem(question, context=self.context(question))
        self.assertFalse(result["success"])
        agent._request_ollama_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
