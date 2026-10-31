import re
import json
import os
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock, patch

from strands_agents.math_agent import MathAgent
from strands_agents.base_physics_agent import StrandsPhysicsAgent
from strands_agents.linear_systems import (extract_equations, practice_systems, render_solution,
                                         render_practice, requested_topics)

_core_path = Path(os.getenv("MATH_MCP_TEST_CORE", str(Path(__file__).resolve().parents[2] / "mcp_tools/physics_mcp_tools/linear_systems.py")))
_spec = importlib.util.spec_from_file_location("math_mcp_test_core", _core_path)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
validate_generated_problem = core.validate_generated_problem
equation_fingerprint = core.equation_fingerprint


def solve_system(equations, quantities=None):
    return render_solution(core.solve_system(equations, quantities))


def _fixture_problem(topic, variant):
    """Two SI equations using named physical unknowns, without a scenario."""
    k = variant
    if topic == "momentum":
        return ("Momentum", [f"2v_1 + v_2 = {9+3*k}", "v_2 - v_1 = 3"],
                {"v_1": ("Velocity 1", "m/s"), "v_2": ("Velocity 2", "m/s")})
    if topic == "forces":
        return ("Forces", [f"T - {2+k}a = {10*(2+k)}", f"T + {3+k}a = {10*(3+k)}"],
                {"T": ("Tension", "N"), "a": ("Acceleration", "m/s²")})
    if topic == "kinematics":
        return ("Kinematics", [f"v_i + v_f = {10+2*k}", "v_f - v_i = 6"],
                {"v_i": ("Initial velocity", "m/s"), "v_f": ("Final velocity", "m/s")})
    if topic == "energy":
        return ("Energy and power", [f"2P_1 + 3P_2 = {80+8*k}", f"4P_1 + P_2 = {60+6*k}"],
                {"P_1": ("Power 1", "W"), "P_2": ("Power 2", "W")})
    if topic == "circuits":
        return ("Circuits", [f"I_1 + I_2 = {5+5*k}", "2I_1 - 3I_2 = 0"],
                {"I_1": ("Current 1", "A"), "I_2": ("Current 2", "A")})
    if topic == "torque":
        return ("Torque", [f"F_1 + F_2 = {300+30*k}", f"-2F_1 + 2F_2 = {-100-10*k}"],
                {"F_1": ("Support force 1", "N"), "F_2": ("Support force 2", "N")})
    return ("Thermal physics", [f"m_1 + m_2 = {6+3*k}", "40m_1 - 20m_2 = 0"],
            {"m_1": ("Hot-water mass", "kg"), "m_2": ("Cold-water mass", "kg")})




def fixture_candidates(count, request="", offset=0):
    topics = requested_topics(request)
    candidates = []
    for index in range(count):
        topic = topics[index % len(topics)]
        _, equations, quantities = _fixture_problem(topic, index // len(topics) + offset)
        kinds = {symbol: {"v": "velocity", "T": "tension", "a": "acceleration", "P": "power",
                          "I": "current", "F": "force", "m": "mass"}[symbol.split("_")[0]]
                 for symbol in quantities}
        candidates.append({"topic": topic, "equations": equations, "quantities": kinds})
    return candidates


def build_practice(count, include_answers, request=""):
    # Fixed fixtures exist only in tests. Production obtains candidates from the LLM.
    problems = [validate_generated_problem(item, requested_topics(request))[0]
                for item in fixture_candidates(count, request)]
    return render_practice(problems, include_answers)


class LinearSystemTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = MathAgent(enable_database_logging=False, enable_rag=False)
        self.fixture_offset = 0
        self.mcp_calls = []
        def call_tool_sync(tool_use_id, name, arguments):
            self.mcp_calls.append((name, arguments))
            if name == "validate_physics_equations":
                result = core.validate_physics_equations(**arguments)
            elif name == "solve_linear_systems":
                result = core.solve_linear_systems(**arguments)
            else:
                raise AssertionError(name)
            return {"status": "success", "content": [{"text": json.dumps(result)}]}
        self.agent.mcp_client = Mock()
        self.agent.mcp_client.call_tool_sync.side_effect = call_tool_sync
        def generate_fixture(prompt, *args, **kwargs):
            payload = json.loads(prompt)
            candidates = fixture_candidates(payload["count"], payload["student_request"], self.fixture_offset)
            self.fixture_offset += 10
            return json.dumps({"problems": candidates})
        self.agent._request_ollama_generate = Mock(side_effect=generate_fixture)

    async def test_original_request_produces_five_independent_pairs_without_answers(self):
        response = await self.agent.solve_problem("give me 5 two equations two unknowns problems")
        self.assertTrue(response["success"])
        equations = extract_equations(response["solution"])
        self.assertEqual(len(equations), 10)
        self.assertNotIn("Answer key:", response["solution"])
        for index in range(5):
            pair = equations[index * 2:index * 2 + 2]
            quantities = practice_systems(response["solution"])[index][1]
            self.assertTrue(all(all(symbol in equation for symbol in quantities) for equation in pair))
            solution = solve_system(pair)
            self.assertIn("Answer:", solution)
            self.assertNotIn("No solution", solution)
            self.assertNotIn("Infinitely many", solution)
            self.assertEqual(solution.count("✓"), 2)

    async def test_count_does_not_use_dimensions_and_supports_words(self):
        for request, count in [
            ("Give me two equations with two unknowns problems", 5),
            ("Give me 2 equations 2 unknowns practice", 5),
            ("Give me three simultaneous equations problems", 3),
            ("Create 4 systems of linear equations problems", 4),
            ("Give me 20 two-equations two-unknowns problems", 10),
        ]:
            with self.subTest(request=request):
                result = await self.agent.solve_problem(request)
                self.assertEqual(result["metadata"]["linear_system_mode"]["exercise_count"], count)
                self.assertEqual(len(extract_equations(result["solution"])), count * 2)

    async def test_answer_key_has_verified_solutions(self):
        response = await self.agent.solve_problem("Give me 5 two equations two unknowns problems with solutions")
        self.assertIn("Answer key:", response["solution"])
        self.assertEqual(response["solution"].count("Physical answer:"), 5)
        self.assertEqual(response["solution"].count("✓"), 10)
        hidden = await self.agent.solve_problem("Give me 5 simultaneous equations problems without answers")
        self.assertNotIn("Answer key:", hidden["solution"])

    async def test_supplied_system_is_solved_exactly(self):
        response = await self.agent.solve_problem("Solve 2x + 3y = 7, x - y = 1")
        self.assertIn("Answer: x = 2, y = 1", response["solution"])
        self.assertIn("Elimination:", response["solution"])
        self.assertEqual(response["solution"].count("✓"), 2)

    async def test_numbered_followup_uses_actual_history_equations(self):
        worksheet = build_practice(5, False).replace("v_{i} + v_{f} = 10", "v_{i} + v_{f} = 14")
        context = {"conversation_context": {"previous_assistant_response": worksheet}}
        response = await self.agent.solve_problem("solve problem 3", context)
        # Use the modified worksheet, rather than regenerate a stock answer.
        expected = solve_system(*practice_systems(worksheet)[2])
        self.assertEqual(response["solution"], expected)
        all_answers = await self.agent.solve_problem("give me the answer key", context)
        self.assertEqual(all_answers["solution"].count("Physical answer:"), 5)
        missing = await self.agent.solve_problem("solve problem 8", context)
        self.assertIn("1 to 5", missing["solution"])

    async def test_followup_uses_recent_assistant_worksheet(self):
        context = {"conversation_context": {"recent_messages": [
            {"role": "assistant", "content": build_practice(2, False)},
            {"role": "user", "content": "solve problem 2"},
        ]}}
        response = await self.agent.solve_problem("solve problem 2", context)
        self.assertIn("Answer: T = 24, a = 2", response["solution"])

    async def test_api_augmented_followup_uses_current_question(self):
        from main import _augment_problem_with_conversation_context, _extract_conversation_context

        worksheet = build_practice(5, False)
        raw_context = {"conversation_context": {"recent_messages": [
            {"role": "assistant", "content": worksheet},
        ]}}
        conversation = _extract_conversation_context(raw_context, "math_agent")
        question = "Can you solve problem 3 from above?"
        augmented = _augment_problem_with_conversation_context(question, conversation, "math_agent")
        context = {"conversation_context": conversation, "student_followup": question,
                   "contextual_followup": True}
        result = await self.agent.solve_problem(augmented, context)
        self.assertIn("Answer: v_i = 2, v_f = 8", result["solution"])

    async def test_default_practice_covers_physics_topics_without_units(self):
        response = await self.agent.solve_problem("Give me 7 two equations two unknowns problems with answers")
        worksheet = response["solution"]
        for title in ["Momentum", "Forces", "Kinematics", "Energy and power", "Circuits", "Torque", "Thermal physics"]:
            self.assertIn(title, worksheet)
        systems = practice_systems(worksheet)
        self.assertEqual(len(systems), 7)
        expected = [(2, 5), (24, 2), (2, 8), (10, 20), (3, 2), (175, 125), (2, 4)]
        for (equations, quantities), (x, y) in zip(systems, expected):
            solution = solve_system(equations, quantities)
            first, second = quantities
            self.assertIn(f"Answer: {first} = {x}, {second} = {y}", solution)
            self.assertIn("Physical answer:", solution)
            for label, unit in quantities.values():
                self.assertIn(label, solution)
                self.assertEqual(unit, "")

    async def test_topic_specific_practice_honors_momentum_energy_and_mixed_requests(self):
        for topic, title in [("momentum", "Momentum"), ("energy", "Energy and power"), ("circuits", "Circuits")]:
            response = await self.agent.solve_problem(f"Give me 3 two equations two unknowns {topic} problems")
            worksheet = response["solution"]
            self.assertEqual(worksheet.count(title), 3)
            self.assertEqual(len(practice_systems(worksheet)), 3)
        mixed = await self.agent.solve_problem("Give me 5 two equations two unknowns problems, not limited to momentum")
        self.assertIn("Kinematics", mixed["solution"])

    async def test_last_problem_survives_full_api_history_serialization(self):
        from main import _extract_conversation_context
        worksheet = build_practice(10, False)
        self.assertLess(len(worksheet), 12000)
        conversation = _extract_conversation_context({"conversation_context": {"recent_messages": [
            {"role": "assistant", "content": worksheet},
        ], "previous_assistant_response": worksheet}}, "math_agent")
        self.assertEqual(conversation["recent_messages"][0]["content"], worksheet)
        response = await self.agent.solve_problem("solve problem 10", {"conversation_context": conversation})
        expected = solve_system(*practice_systems(worksheet)[9])
        self.assertEqual(response["solution"], expected)

    def test_equation_only_format_uses_physical_symbols(self):
        worksheet = build_practice(7, False)
        self.assertEqual(len(extract_equations(worksheet)), 14)
        for symbol in ["v_{i}", "v_{f}", "m_{1}", "m_{2}", "T", "a", "P_{1}", "P_{2}"]:
            self.assertIn(symbol, worksheet)
        for block in re.split(r"^\d+\. ", worksheet, flags=re.M)[1:]:
            content = block.split("Send your answers")[0].strip().splitlines()
            self.assertEqual(sum("=" in line for line in content), 2)
            self.assertEqual(sum(line.strip() == "$$" for line in content), 4)
        self.assertEqual(worksheet.count("$$"), 28)
        self.assertNotIn("[", worksheet)
        self.assertNotIn("SI units", worksheet)
        self.assertNotIn("m/s", worksheet)
        self.assertNotIn("Model:", worksheet)
        self.assertNotIn("cart", worksheet)
        self.assertNotIn("x:", worksheet)
        self.assertNotIn("y:", worksheet)

    async def test_student_can_supply_named_physical_equations(self):
        response = await self.agent.solve_problem("Solve v_i + v_f = 10, v_f - v_i = 6")
        self.assertIn("v_i = 2", response["solution"])
        self.assertIn("v_f = 8", response["solution"])
        self.assertEqual(response["solution"].count("✓"), 2)

    async def test_generation_calls_llm_and_reports_calculator_verification(self):
        response = await self.agent.solve_problem("Give me 5 two equations two unknowns problems")
        self.agent._request_ollama_generate.assert_called_once()
        mode = response["metadata"]["linear_system_mode"]
        self.assertEqual(mode["generation_source"], "llm")
        self.assertTrue(mode["calculator_verified"])
        call = self.agent._request_ollama_generate.call_args
        self.assertGreater(call.kwargs["model_options"]["temperature"], 0)
        self.assertTrue(call.kwargs["json_format"])

    async def test_structured_llm_coefficients_are_checked_and_rendered(self):
        candidate = {"topic": "kinematics", "quantities": {"v_i": "velocity", "v_f": "velocity"},
                     "equations": [{"coefficients": {"v_i": 2, "v_f": 3}, "rhs": 21},
                                   {"coefficients": {"v_i": -1, "v_f": 1}, "rhs": 2}]}
        self.agent._request_ollama_generate.side_effect = [json.dumps({"problems": [candidate]})]
        response = await self.agent.solve_problem("Give me 1 two equations two unknowns kinematics problem with answers")
        self.assertTrue(response["success"])
        self.assertIn("Answer: v_i = 3, v_f = 5", response["solution"])
        self.assertEqual(response["solution"].count("✓"), 2)
        self.assertIn(r"\cdot", response["solution"])
        self.assertNotIn("+ -", response["solution"])

    async def test_invalid_model_system_is_retried_before_display(self):
        bad = {"topic": "momentum", "equations": ["v_i + v_f = 10", "2v_i + 2v_f = 21"],
               "quantities": {"v_i": "velocity", "v_f": "velocity"}}
        good = fixture_candidates(1, "momentum")[0]
        self.agent._request_ollama_generate.side_effect = [
            json.dumps({"problems": [bad]}), json.dumps({"problems": [good]}),
        ]
        response = await self.agent.solve_problem("Give me 1 two equations two unknowns momentum problem")
        self.assertTrue(response["success"])
        self.assertNotIn("21", response["solution"])
        self.assertEqual(response["metadata"]["linear_system_mode"]["generation_attempts"], 2)
        self.assertEqual(self.agent._request_ollama_generate.call_count, 2)

    async def test_duplicate_proposals_are_replaced_and_answer_key_is_computed(self):
        candidates = fixture_candidates(2, "momentum")
        candidates[0]["answer"] = "HIDDEN LLM ANSWER IS WRONG"
        self.agent._request_ollama_generate.side_effect = [
            json.dumps({"problems": [candidates[0], candidates[0]]}),
            json.dumps({"problems": [candidates[1]]}),
        ]
        response = await self.agent.solve_problem("Give me 2 two equations two unknowns momentum problems with answers")
        self.assertTrue(response["success"])
        self.assertNotIn("HIDDEN", response["solution"])
        self.assertIn("Answer: v_1 = 2, v_2 = 5", response["solution"])
        self.assertEqual(response["solution"].count("Physical answer:"), 2)
        self.assertEqual(response["solution"].count("✓"), 4)
        self.assertEqual(len(self.agent._recent_practice_fingerprints), 2)

    async def test_model_failure_does_not_fall_back_to_fixed_or_unchecked_questions(self):
        for replies in [[RuntimeError("unavailable")] * 3, ["not json"] * 3]:
            self.agent._request_ollama_generate.side_effect = replies
            with patch.object(StrandsPhysicsAgent, "solve_problem", new_callable=AsyncMock) as fallback:
                response = await self.agent.solve_problem("Give me 5 two equations two unknowns problems")
                self.assertFalse(response["success"])
                self.assertNotIn("$$", response["solution"])
                fallback.assert_not_awaited()

    async def test_incomplete_checked_batch_is_not_shown(self):
        candidates = fixture_candidates(4)
        self.agent._request_ollama_generate.side_effect = [
            json.dumps({"problems": candidates}), json.dumps({"problems": []}), json.dumps({"problems": []}),
        ]
        response = await self.agent.solve_problem("Give me 5 two equations two unknowns problems")
        self.assertFalse(response["success"])
        self.assertEqual(len(extract_equations(response["solution"])), 0)
        self.assertEqual(len(self.agent._recent_practice_fingerprints), 0)

    def test_reordered_and_scaled_equations_are_detected_as_duplicates(self):
        first = ["2v_i + v_f = 9", "v_f - v_i = 3"]
        second = ["2v_f - 2v_i = 6", "4v_i + 2v_f = 18"]
        self.assertEqual(equation_fingerprint(first), equation_fingerprint(second))

    async def test_more_practice_avoids_earlier_worksheet(self):
        first = await self.agent.solve_problem("Give me 2 two equations two unknowns momentum problems")
        context = {"conversation_context": {"previous_assistant_response": first["solution"]}}
        second = await self.agent.solve_problem("Give me 2 more momentum problems", context)
        self.assertTrue(second["success"])
        self.assertNotEqual(extract_equations(first["solution"]), extract_equations(second["solution"]))
        self.assertEqual(self.agent._request_ollama_generate.call_count, 2)

    def test_validator_rejects_unsafe_nonphysical_and_unsolvable_equations(self):
        bad_proposals = [
            {"topic": "momentum", "equations": ["v_i + v_f = 2", "2v_i + 2v_f = 4"],
             "quantities": {"v_i": "velocity", "v_f": "velocity"}},
            {"topic": "momentum", "equations": ["v_i*v_f = 2", "v_i - v_f = 1"],
             "quantities": {"v_i": "velocity", "v_f": "velocity"}},
            {"topic": "momentum", "equations": ["x + y = 2", "x - y = 1"],
             "quantities": {"x": "velocity", "y": "velocity"}},
            {"topic": "momentum", "equations": ["v_i + v_f + a = 2", "v_i - v_f = 1"],
             "quantities": {"v_i": "velocity", "v_f": "velocity"}},
            {"topic": "thermal", "equations": ["m_1 + m_2 = 2", "m_1 - m_2 = 4"],
             "quantities": {"m_1": "mass", "m_2": "mass"}},
            {"topic": "momentum", "equations": ["__import__('os') = v_f", "v_i-v_f=1"],
             "quantities": {"v_i": "velocity", "v_f": "velocity"}},
        ]
        for candidate in bad_proposals:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_generated_problem(candidate, requested_topics("mixed"))

    async def test_remaining_candidates_can_be_repaired_on_third_attempt(self):
        good = fixture_candidates(1, "momentum")[0]
        self.agent._request_ollama_generate.side_effect = [
            json.dumps({"problems": []}), json.dumps({"problems": []}),
            json.dumps({"problems": [good]}),
        ]
        result = await self.agent.solve_problem("Give me 1 two equations two unknowns momentum problem")
        self.assertTrue(result["success"])
        self.assertEqual(result["metadata"]["linear_system_mode"]["generation_attempts"], 3)

    async def test_mcp_is_mandatory_for_validation_and_solving(self):
        response = await self.agent.solve_problem("Give me 5 two equations two unknowns physics problems")
        self.assertTrue(response["success"])
        self.assertEqual(response["tools_used"], ["validate_physics_equations"])
        self.assertEqual(self.mcp_calls[0][0], "validate_physics_equations")
        self.assertEqual(len(self.mcp_calls[0][1]["problems"]), 5)
        solved = await self.agent.solve_problem("Solve 2v_i + v_f = 9, v_f - v_i = 3")
        self.assertEqual(solved["tools_used"], ["solve_linear_systems"])
        self.assertEqual(self.mcp_calls[-1][0], "solve_linear_systems")
        self.assertIn("v_i = 2", solved["solution"])

    async def test_mcp_failure_never_uses_local_math_or_unchecked_llm_output(self):
        self.agent.mcp_client.call_tool_sync.side_effect = ConnectionError("MCP unavailable")
        for question in ["Give me 5 two equations two unknowns problems", "Solve x + y = 3, x - y = 1"]:
            with patch.object(StrandsPhysicsAgent, "solve_problem", new_callable=AsyncMock) as fallback:
                response = await self.agent.solve_problem(question)
                self.assertFalse(response["success"])
                self.assertNotIn("$$", response["solution"])
                self.assertNotIn("Answer:", response["solution"])
                self.assertEqual(response["metadata"]["linear_system_mode"]["action"], "calculator_error")
                fallback.assert_not_awaited()

    async def test_answer_key_uses_one_batch_mcp_call(self):
        worksheet = build_practice(5, False)
        context = {"conversation_context": {"previous_assistant_response": worksheet}}
        response = await self.agent.solve_problem("Give me the answer key", context)
        self.assertTrue(response["success"])
        self.assertEqual(len(self.mcp_calls), 1)
        self.assertEqual(len(self.mcp_calls[0][1]["systems"]), 5)

    async def test_other_math_requests_and_unsupported_equations_keep_general_route(self):
        for request in ["Give me 4 kinematics algebra exercises", "Solve x^2 + y = 3, x - y = 1",
                        "Solve x + y = 3, x - y = 1, x = 2", "Find sin(30 degrees)"]:
            with self.subTest(request=request), patch.object(
                StrandsPhysicsAgent, "solve_problem", new_callable=AsyncMock,
                return_value={"general": True},
            ) as fallback:
                self.assertEqual(await self.agent.solve_problem(request), {"general": True})
                fallback.assert_awaited_once()

    def test_fractional_decimal_and_sparse_equations(self):
        for equations, answer in [
            (["2x + y = 1", "x - y = 0"], "x = 1/3, y = 1/3"),
            (["0.5a + b = 2", "a - b = 1"], "a = 2, b = 1"),
            (["x = 2", "y = 3"], "x = 2, y = 3"),
            (["x/2 + y = 2", "x - y = 1"], "x = 2, y = 1"),
            (["2*(x + y) = 6", "x - y = 1"], "x = 2, y = 1"),
        ]:
            with self.subTest(equations=equations):
                self.assertIn(f"Answer: {answer}", solve_system(equations))

    def test_singular_systems(self):
        self.assertIn("Infinitely many solutions", solve_system(["x + y = 2", "2x + 2y = 4"]))
        self.assertIn("No solution", solve_system(["x + y = 2", "2x + 2y = 5"]))
        self.assertIn("No solution", solve_system(["x - x = 1", "x + y = 2"]))

    def test_parser_does_not_extract_partial_nonlinear_equations(self):
        for text in ["x^2 + y = 2, x-y = 1", "x*y = 2, x-y = 1", "x + y = 1/0, x-y = 0"]:
            self.assertEqual(core.extract_equations(text), [])


if __name__ == "__main__":
    unittest.main()
