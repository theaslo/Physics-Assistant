import json
import unittest
from unittest.mock import Mock, patch

from strands_agents.forces_agent import ForcesAgent, QUIZ_REFLECTION_GUIDANCE, REFLECTION_STEP_PROMPT


class ForcesFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_introduction_without_history_requests_quiz_details(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        agent.initialized = True
        agent.mcp_client = Mock()
        agent.mcp_client.call_tool_sync.return_value = {
            "status": "success",
            "content": [{"text": "Equilibrium Analysis: net force is zero."}],
        }
        agent._request_ollama_generate = Mock(return_value=json.dumps({
            "action": "intake",
            "question": "Paste that one quiz question with your original answer and your original reasoning.",
            "tool_calls": [],
        }))

        result = await agent.solve_problem(
            "I am reviewing a quiz about Newton's laws and forces. "
            "My answer and reasoning are below. Help me identify my misunderstanding. "
            "Please ask me one question at a time and give hints before revealing the answer."
        )

        self.assertTrue(result["success"], result)
        self.assertIn("Please paste the quiz question, your original answer, and your reasoning", result["solution"])
        self.assertNotIn("what horizontal forces", result["solution"])
        self.assertNotIn("net external force is zero", result["solution"])
        self.assertIsNone(result["diagram"])
        self.assertTrue(result["metadata"]["quiz_reflection_mode"]["enabled"])
        self.assertTrue(result["metadata"]["llm_response_used"])
        self.assertEqual(result["tools_used"], ["get_force_principles"])

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
                case.assertEqual(name, "get_force_principles")
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
        agent._request_ollama_generate = Mock(side_effect=[json.dumps({"action": "hint", "problem_message_index": 0,
            "problem_statement": "A quiz about net force and acceleration."}), json.dumps({
            "assessment": "unassessed", "completed_problem": False, "question": "Which forces act on the object?",
        })])

        result = await agent.solve_problem(
            problem=(
                "I am analyzing a mistake from my Newton's 2nd law quiz. "
                "The quiz question asked about net force and acceleration. "
                "My answer used the biggest force as ma, but the correct answer says to use net force."
            ),
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["get_force_principles"])
        self.assertEqual(calls[0][0], "get_force_principles")
        self.assertEqual(result["solution"], "Which forces act on the object?")
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
        agent._request_ollama_generate = Mock(side_effect=[json.dumps({"action": "hint", "problem_message_index": 0,
            "problem_statement": "After a hockey puck leaves the stick on frictionless ice, will it slow down?"}), json.dumps({
            "assessment": "unassessed", "completed_problem": False,
            "question": "After the puck loses contact with the stick, what horizontal forces act on it?",
        })])

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
        self.assertEqual(result["tools_used"], ["get_force_principles"])
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
        return {"contextual_followup": True, "student_followup": question, "conversation_context": {
            "recent_messages": messages or [
                {"role": "user", "content": "A spring has k = 50 N/m and is stretched 0.20 m to the right."},
                {"role": "assistant", "content": "The spring force is 10 N to the left."},
            ],
        }}

    def quiz_history(self):
        return [
            {"role": "user", "content": "Help me review my forces quiz with hints."},
            {"role": "assistant", "content": "Please provide the quiz and your reasoning."},
            {"role": "user", "content": "A puck leaves a stick on frictionless ice. Will it speed up? I said yes because I hit it."},
            {"role": "assistant", "content": "What is the net force after contact ends?"},
        ]

    def frame(self, action="feedback", index=2):
        return json.dumps({"action": action, "problem_message_index": index,
                           "problem_statement": "A puck leaves a stick on frictionless ice. Will it speed up?"})

    def feedback_responses(self, verdict="correct", complete=False):
        return [self.frame(), json.dumps({
            "assessment": verdict, "complete": complete,
            "outcome_evidence": "Velocity stays constant" if complete else "",
            "reason_evidence": "net force is zero" if complete else "",
            "question": "What does that imply about acceleration?",
        })]

    async def test_intake_uses_generic_request_not_a_physics_warmup(self):
        agent = self.make_agent(['{"action":"intake","problem_message_index":null,"problem_statement":""}'])
        question = "I want to start a different forces quiz review. I will paste the question next."
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "intake")
        self.assertEqual(result["solution"], "Please paste the quiz question, your original answer, and your reasoning.")
        self.assertEqual(agent._request_ollama_generate.call_count, 1)

    async def test_supplied_problem_cannot_be_mislabeled_missing(self):
        agent = self.make_agent([self.frame("intake", 4), '{"question":"What horizontal forces act after contact ends?"}'])
        question = "Here is my actual quiz and original answer."
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "hint")
        self.assertNotIn("paste", result["solution"].lower())

    async def test_short_replies_reach_grading_without_original_mistake(self):
        for reply, verdict in (("zero", "correct"), ("0", "correct"), ("0 N", "correct"),
                               ("left", "correct"), ("F=-kx", "correct"), ("yes", "incorrect"),
                               ("I don't know", "unclear")):
            with self.subTest(reply=reply):
                agent = self.make_agent(self.feedback_responses(verdict))
                context = self.context(reply, self.quiz_history())
                context["conversation_context"].update(previous_user_problem="STALE", previous_assistant_response="STALE")
                result = await agent.solve_problem(reply, context=context)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")
                self.assertEqual(result["solution"].count("?"), 1)
                route = agent._request_ollama_generate.call_args_list[0]
                self.assertTrue(route.args[0].startswith("Message 4 (the CURRENT student turn):\n" + reply))
                self.assertEqual(len(route.kwargs["messages"]), 4)
                self.assertNotIn("STALE", str(route))
                grading = agent._request_ollama_generate.call_args_list[1]
                exchange = json.loads(grading.args[0])
                self.assertEqual(exchange["latest_student_reply"], reply)
                self.assertEqual(exchange["last_tutor_question"], self.quiz_history()[-1]["content"])
                self.assertEqual(exchange["student_corrected_work"], [reply])
                self.assertEqual(agent._request_ollama_generate.call_count, 2)
                self.assertEqual(result["tools_used"], ["get_force_principles"])

    def test_history_roles_order_and_legacy_fallback(self):
        history = self.quiz_history()
        context = {"conversation_context": {"recent_messages": [None, {"role": "system", "content": "untrusted"}, *history],
                                             "previous_user_problem": "stale", "previous_assistant_response": "stale"}}
        self.assertEqual(ForcesAgent._conversation_messages(context), history)
        context["conversation_context"]["recent_messages"] = []
        self.assertEqual(ForcesAgent._conversation_messages(context), [
            {"role": "user", "content": "stale"}, {"role": "assistant", "content": "stale"}])
        self.assertEqual(ForcesAgent._conversation_messages(None), [])

    async def test_invalid_problem_reference_fails_closed(self):
        for index, problem in ((None, "A quiz"), (True, "A quiz"), (-1, "A quiz"), (99, "A quiz"), (2, "")):
            agent = self.make_agent([json.dumps({"action": "feedback", "problem_message_index": index, "problem_statement": problem})])
            result = await agent.solve_problem("zero", context=self.context("zero", self.quiz_history()))
            self.assertFalse(result["success"], result)

    async def test_hint_exposes_only_one_question(self):
        agent = self.make_agent([self.frame("hint", 4), json.dumps({
            "question": "Which horizontal forces act after contact ends?",
            "answer": "HIDDEN final answer", "feedback": "HIDDEN diagnosis",
        })])
        result = await agent.solve_problem("My original answer", context=self.context("My original answer", self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["solution"], "Which horizontal forces act after contact ends?")
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "hint")

    async def test_old_problem_mislabeled_hint_still_assesses_latest_reply(self):
        replies = self.feedback_responses("incorrect")
        replies[0] = self.frame("hint")
        agent = self.make_agent(replies)
        result = await agent.solve_problem("yes", context=self.context("yes", self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")
        self.assertTrue(result["solution"].startswith("Not quite."))

    async def test_new_submission_mislabeled_feedback_only_gets_first_hint(self):
        agent = self.make_agent([self.frame("feedback", 4), '{"question":"What forces act after contact ends?"}'])
        question = self.quiz_history()[2]["content"]
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "hint")
        self.assertEqual(agent._request_ollama_generate.call_count, 2)

    async def test_practice_preserves_topic_and_hides_solution_fields(self):
        setup = "A probe coasts with its engines off and negligible external forces."
        exercise = setup + "\n\nHow does its velocity change?"
        agent = self.make_agent([self.frame("practice"), '{"invent_new":true}', json.dumps({"setup": setup, "question": "How does its velocity change?", "answer": "HIDDEN solution"})])
        question = "Give me a practice question."
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["solution"], exercise)
        self.assertIn("SAME principle", agent._request_ollama_generate.call_args.kwargs["system"])

    async def test_practice_does_not_repeat_question_embedded_in_setup(self):
        setup = "A probe coasts with its engines off and negligible external forces. How does its velocity change?"
        agent = self.make_agent([self.frame("practice"), '{"invent_new":true}', json.dumps({
            "setup": setup, "question": "What happens to its speed? Does it slow down?",
        })])
        question = "Give me a practice question."
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["solution"], setup)
        self.assertEqual(result["solution"].count("?"), 1)

    async def test_practice_attempt_mislabeled_new_practice_is_graded(self):
        replies = [self.frame("practice"), '{"invent_new":false}', self.feedback_responses(complete=True)[1]]
        agent = self.make_agent(replies)
        reply = "Velocity stays constant because net force is zero."
        result = await agent.solve_problem(reply, context=self.context(reply, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "complete")
        self.assertEqual(json.loads(agent._request_ollama_generate.call_args_list[2].args[0])["action"], "feedback")

    async def test_invented_router_problem_is_replaced_by_actual_source(self):
        replies = self.feedback_responses()
        frame = json.loads(replies[0])
        frame["problem_statement"] = "An astronaut pushes a tool away."
        replies[0] = json.dumps(frame)
        agent = self.make_agent(replies)
        result = await agent.solve_problem("zero", context=self.context("zero", self.quiz_history()))
        self.assertTrue(result["success"], result)
        exchange = json.loads(agent._request_ollama_generate.call_args_list[1].args[0])
        self.assertEqual(exchange["original_problem"], self.quiz_history()[2]["content"])
        self.assertNotIn("astronaut", exchange["original_problem"])

    async def test_complete_work_is_confirmed_without_another_question(self):
        agent = self.make_agent(self.feedback_responses(complete=True))
        question = "Velocity stays constant because net force is zero."
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "complete")
        self.assertNotIn("?", result["solution"])
        self.assertEqual(agent._request_ollama_generate.call_count, 2)

    async def test_bare_answer_cannot_be_claimed_as_an_explained_solution(self):
        agent = self.make_agent(self.feedback_responses(complete=True) + ['{"question":"Why do the forces balance?"}'])
        result = await agent.solve_problem("equal", context=self.context("equal", self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")
        self.assertTrue(result["solution"].startswith("That step is correct."))
        self.assertEqual(result["solution"].count("?"), 1)

    async def test_completion_evidence_excludes_old_mistake_and_tutor_explanations(self):
        history = self.quiz_history() + [
            {"role": "user", "content": "There is no net force, so acceleration is zero."},
            {"role": "assistant", "content": "TUTOR_ONLY_EXPLANATION. What happens to velocity?"},
        ]
        replies = self.feedback_responses(complete=True)
        replies[1] = json.dumps({"assessment": "correct", "complete": True,
            "outcome_evidence": "Its velocity stays constant.", "reason_evidence": "There is no net force, so acceleration is zero.", "question": ""})
        agent = self.make_agent(replies)
        question = "Its velocity stays constant."
        result = await agent.solve_problem(question, context=self.context(question, history))
        self.assertTrue(result["success"], result)
        completion = agent._request_ollama_generate.call_args_list[1]
        work = "\n".join(json.loads(completion.args[0])["student_corrected_work"])
        self.assertIn("acceleration is zero", work)
        self.assertIn(question, work)
        self.assertNotIn("I said yes", work)
        self.assertNotIn("TUTOR_ONLY", work)
        self.assertTrue(completion.kwargs["think"])
        self.assertFalse(completion.args[3])

    async def test_incorrect_reply_cannot_trigger_completion(self):
        agent = self.make_agent(self.feedback_responses("incorrect"))
        result = await agent.solve_problem("yes", context=self.context("yes", self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertTrue(result["solution"].startswith("Not quite."))
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")

    async def test_numerical_grading_uses_mcp_before_assessing(self):
        plan = json.loads(self.frame())
        plan.update(calculation_required=True, tool_calls=[{"tool_name": "newton_second_law",
            "arguments": {"newton_data": {"force": 10, "mass": 5}}}])
        agent = self.make_agent([json.dumps(plan), self.frame(),
            '{"assessment":"incorrect","complete":false,"question":"How do you isolate acceleration in F=ma?"}',
        ], calculation_output="Verified acceleration: 2 m/s^2")
        result = await agent.solve_problem("6 m/s^2", context=self.context("6 m/s^2", self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["tools_used"], ["get_force_principles", "newton_second_law"])
        self.assertIn("Verified acceleration: 2 m/s^2", agent._request_ollama_generate.call_args_list[2].kwargs["system"])
        self.assertTrue(result["solution"].startswith("Not quite."))
        planner = agent._request_ollama_generate.call_args_list[0]
        self.assertFalse(planner.kwargs["think"])
        self.assertNotIn('"name": "check_equilibrium"', planner.kwargs["system"])
        self.assertNotIn('"name": "create_free_body_diagram"', planner.kwargs["system"])

    async def test_repeated_verified_tool_is_not_executed_again(self):
        call = json.loads(self.frame())
        call.update(calculation_required=True, tool_calls=[{"tool_name": "calculate_spring_force_tool",
                           "arguments": {"spring_data": {"spring_constant": 90, "displacement": 0.2}}}])
        agent = self.make_agent([json.dumps(call), json.dumps(call),
                                '{"assessment":"correct","complete":false,"question":"Which way does the force point?"}'])
        result = await agent.solve_problem("18 N", context=self.context("18 N", self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(len(agent.mcp_client.call_tool_sync.call_args_list), 2)

    async def test_symbolic_inputs_are_replanned_not_sent_to_numerical_mcp(self):
        plan = json.loads(self.frame())
        plan.update(calculation_required=True, tool_calls=[{"tool_name": "calculate_spring_force_tool",
            "arguments": {"spring_data": {"spring_constant": "k", "displacement": "2x"}}}])
        agent = self.make_agent([json.dumps(plan), *self.feedback_responses()])
        result = await agent.solve_problem("F=-kx", context=self.context("F=-kx", self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["tools_used"], ["get_force_principles"])
        self.assertEqual(agent.mcp_client.call_tool_sync.call_count, 1)

    def test_calculation_errors_cannot_be_treated_as_verified_physics(self):
        agent = self.make_agent([])
        for text in ("Error: bad input", "Error in calculation: cannot convert k", " Error calculating force"):
            self.assertTrue(agent._is_mcp_error_text(text))
        self.assertFalse(agent._is_mcp_error_text("Error Analysis: standard deviation is 2."))
        for values in ({"force": 0, "acceleration": 0}, {"force": 10}, {"force": 10, "mass": 0},
                       {"force": "NaN", "mass": 5}):
            self.assertIsNotNone(agent._calculation_input_error({"tool_name": "newton_second_law",
                "arguments": {"newton_data": json.dumps(values)}}))

    async def test_invalid_tool_or_model_output_is_not_shown_as_feedback(self):
        for responses in (
            ["{}"], [TimeoutError("offline")],
            [json.dumps({**json.loads(self.frame()), "calculation_required": True,
                         "tool_calls": [{"tool_name": "invented_tool", "arguments": {}}]})],
            [self.frame(), '{"assessment":"invented"}'],
            [self.frame(), '{"assessment":"correct","complete":"true"}'],
        ):
            agent = self.make_agent(responses)
            result = await agent.solve_problem("zero", context=self.context("zero", self.quiz_history()))
            self.assertFalse(result["success"], result)
            self.assertNotIn("solution", result)

    async def test_prose_model_result_gets_one_bounded_json_format_repair(self):
        prose = "This is a conceptual comparison. No calculation tools are required."
        agent = self.make_agent([prose, '{"tool_calls":[]}'])
        result = await agent._reflection_model("compare forces", "Return JSON with tool_calls", reasoning=True)
        self.assertEqual(result, {"tool_calls": []})
        repair = agent._request_ollama_generate.call_args_list[1]
        self.assertEqual(json.loads(repair.args[0])["result_to_format"], prose)
        self.assertTrue(repair.args[3])
        self.assertFalse(repair.kwargs["think"])
        agent = self.make_agent([prose, "Still not JSON"])
        with self.assertRaises(RuntimeError):
            await agent._reflection_model("compare forces", "Return JSON with tool_calls", reasoning=True)
        self.assertEqual(agent._request_ollama_generate.call_count, 2)

    async def test_empty_thinking_result_retries_same_task_once(self):
        agent = self.make_agent(["", '{"tool_calls":[]}'])
        result = await agent._reflection_model("check this step", "Return JSON with tool_calls", reasoning=True)
        self.assertEqual(result, {"tool_calls": []})
        retry = agent._request_ollama_generate.call_args_list[1]
        self.assertEqual(retry.args[0], "check this step")
        self.assertEqual(retry.kwargs["system"], "Return JSON with tool_calls")
        self.assertFalse(retry.kwargs["think"])
        self.assertTrue(retry.args[3])

    async def test_leaving_reflection_still_uses_mcp_for_new_calculations(self):
        question = "New problem: calculate the force for a spring with k=90 N/m stretched 0.20 m."
        agent = self.make_agent([
            '{"action":"direct","answer":"UNVERIFIED estimate"}',
            '{"request_to_solve":true}',
            json.dumps({"tool_calls": [{"tool_name": "calculate_spring_force_tool", "arguments": {
                "spring_data": {"spring_constant": 90, "displacement": 0.20}}}]}),
            '{"answer":"The force is 18 N opposite the stretch.","tool_calls":[]}',
        ])
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertNotIn("UNVERIFIED", result["solution"])
        self.assertEqual(result["tools_used"], ["get_force_principles", "calculate_spring_force_tool"])

    async def test_equation_mislabeled_direct_stays_in_reflection(self):
        replies = self.feedback_responses()
        replies[0] = self.frame("direct")
        replies.insert(1, '{"request_to_solve":false}')
        agent = self.make_agent(replies)
        result = await agent.solve_problem("F=-kx", context=self.context("F=-kx", self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")
        self.assertEqual(result["tools_used"], ["get_force_principles"])
        self.assertEqual(result["solution"].count("?"), 1)

    async def test_completion_cannot_borrow_evidence_from_tutor(self):
        replies = self.feedback_responses(complete=True)
        replies.append('{"complete":false,"question":"Why did you choose that answer?"}')
        agent = self.make_agent(replies)
        history = self.quiz_history() + [{"role": "assistant", "content": "Velocity stays constant because net force is zero."}]
        result = await agent.solve_problem("I think it is B", context=self.context("I think it is B", history))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")

    async def test_missing_completion_quote_gets_one_bounded_evidence_check(self):
        question = "Velocity stays constant because net force is zero."
        feedback = json.loads(self.feedback_responses(complete=True)[1])
        feedback["reason_evidence"] = ""
        agent = self.make_agent([self.frame(), json.dumps(feedback), json.dumps({
            "complete": True, "outcome_evidence": "Velocity stays constant",
            "reason_evidence": "net force is zero", "question": "",
        })])
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "complete")
        self.assertEqual(agent._request_ollama_generate.call_count, 3)

    async def test_incomplete_flag_with_both_evidence_quotes_is_checked(self):
        question = "Velocity stays constant because net force is zero."
        feedback = json.loads(self.feedback_responses(complete=True)[1])
        feedback["complete"] = False
        agent = self.make_agent([self.frame(), json.dumps(feedback), json.dumps({
            "complete": True, "outcome_evidence": "Velocity stays constant",
            "reason_evidence": "net force is zero", "question": "",
        })])
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "complete")
        self.assertNotIn("?", result["solution"])
        completion_input = json.loads(agent._request_ollama_generate.call_args.args[0])
        self.assertEqual(set(completion_input), {"original_problem", "student_corrected_work"})
        self.assertEqual(completion_input["student_corrected_work"], [question])

    async def test_incomplete_evidence_check_preserves_model_next_question(self):
        question = "The net force is zero because the forces balance."
        next_question = "What does this imply about the motion?"
        agent = self.make_agent([self.frame(), json.dumps({
            "assessment": "correct", "complete": False, "outcome_evidence": question,
            "reason_evidence": question, "question": next_question,
        }), json.dumps({"complete": False, "outcome_evidence": "", "reason_evidence": question})])
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")
        self.assertEqual(result["solution"], "That step is correct.\n\n" + next_question)

    async def test_supplied_problem_mislabeled_practice_is_disambiguated_by_model(self):
        question = "New problem: find the spring force for k=90 N/m and extension 0.20 m."
        agent = self.make_agent([
            json.dumps({"action": "practice", "problem_message_index": None, "problem_statement": question}),
            '{"intent":"answer_supplied"}',
            json.dumps({"tool_calls": [{"tool_name": "calculate_spring_force_tool", "arguments": {
                "spring_data": {"spring_constant": 90, "displacement": 0.20}}}]}),
            '{"answer":"The force is 18 N opposite the stretch.","tool_calls":[]}',
        ])
        result = await agent.solve_problem(question, context=self.context(question, self.quiz_history()))
        self.assertTrue(result["success"], result)
        self.assertIn("calculate_spring_force_tool", result["tools_used"])
        self.assertNotIn("quiz_reflection_mode", result["metadata"])
        self.assertEqual(agent._request_ollama_generate.call_args_list[1].args[0], question)

    def test_only_one_question_is_displayed(self):
        self.assertEqual(ForcesAgent._one_question("What is acceleration? What happens to speed?"), "What is acceleration?")
        self.assertEqual(ForcesAgent._one_question("What is acceleration? HIDDEN final answer."), "What is acceleration?")
        for invalid in (None, "", {}, 5):
            with self.assertRaises(RuntimeError):
                ForcesAgent._one_question(invalid)

    def test_full_agent_uses_same_reflection_workflow(self):
        prompt = self.make_agent([])._get_system_prompt()
        self.assertIn(QUIZ_REFLECTION_GUIDANCE, prompt)
        self.assertIn("Do not require the correct answer or an answer key", prompt)

    def test_transport_keeps_generate_compatible_and_native_chat_ordered(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        for messages in (None, [], self.quiz_history()):
            response = Mock()
            response.json.return_value = {"response": "generate reply", "message": {"content": "chat reply", "thinking": "PRIVATE"}}
            with patch("strands_agents.base_physics_agent.requests.post", return_value=response) as post:
                result = agent._request_ollama_generate("zero", 20, 900, system="Tutor instructions", messages=messages)
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["options"]["temperature"], 0)
            if messages is None:
                self.assertTrue(post.call_args.args[0].endswith("/api/generate"))
                self.assertEqual(result, "generate reply")
            else:
                self.assertTrue(post.call_args.args[0].endswith("/api/chat"))
                self.assertEqual(payload["messages"], [{"role": "system", "content": "Tutor instructions"},
                                                      *messages, {"role": "user", "content": "zero"}])
                self.assertEqual(result, "chat reply")

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
        self.assertIn("Do not request the same tool with the same arguments again", agent._request_ollama_generate.call_args.args[0])

    async def test_unverified_numerical_draft_cannot_skip_calculation_tool(self):
        question = "A different spring has k=90 N/m and extension 0.20 m. Find its force."
        agent = self.make_agent([
            '{"answer":"UNVERIFIED draft: 999 N", "tool_calls":[]}',
            json.dumps({"tool_calls": [{"tool_name": "calculate_spring_force_tool", "arguments": {
                "spring_data": {"spring_constant": 90, "displacement": 0.20}}}]}),
            '{"tool_calls":[]}',
            '{"answer":"The verified force is 18 N opposite the displacement.", "tool_calls":[]}',
        ])
        result = await agent.solve_problem(question, context=self.context(question))
        self.assertTrue(result["success"], result)
        self.assertEqual(result["tools_used"], ["get_force_principles", "calculate_spring_force_tool"])
        self.assertNotIn("999", result["solution"])
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
                agent = self.make_agent([json.dumps({"answer": "Model response for this turn.", "tool_calls": []}),
                                         '{"tool_calls":[]}'])
                result = await agent.solve_problem(question, context=self.context(question, messages))
                self.assertTrue(result["success"], result)
                prompt = agent._request_ollama_generate.call_args_list[0].args[0]
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
