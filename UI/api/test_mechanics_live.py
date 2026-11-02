"""Opt-in end-to-end conversations with the real Qwen server and domain MCP tools.

RUN_MECHANICS_LIVE_TESTS=1 python -B -m unittest -v test_mechanics_live
"""
import os
import time
import unittest
import uuid
import requests


@unittest.skipUnless(os.getenv("RUN_MECHANICS_LIVE_TESTS") == "1", "Requires local Docker and UConn Qwen")
class MechanicsLiveTests(unittest.TestCase):
    def setUp(self):
        self.url = os.getenv("MECHANICS_LIVE_API_URL", "http://localhost:8501/api")
        self.session = "mechanics-validation-" + uuid.uuid4().hex

    def ask(self, agent, problem, history=None, extra=None):
        start = time.monotonic()
        response = requests.post(f"{self.url}/agent/{agent}/solve", json={
            "problem": problem, "user_id": "mechanics_live_regression", "session_id": self.session,
            "context": {"conversation_context": {"recent_messages": history or []}, **(extra or {})},
        }, timeout=150)
        response.raise_for_status()
        result = response.json()
        print(f"\n{agent} {time.monotonic()-start:.1f}s {problem!r}:\n{result.get('solution')}\n"
              f"TOOLS={result.get('tools_used')} HITL={result.get('hitl')}", flush=True)
        self.assertTrue(result.get("success"), result)
        self.assertIn("get_" + agent.removesuffix("_agent") + "_principles", result.get("tools_used", []))
        self.assertTrue(result.get("metadata", {}).get("llm_response_used"))
        return result

    def test_full_solutions_and_followups(self):
        cases = [
            ("momentum_agent", "A 2 kg cart moving right at 3 m/s sticks to a stationary 1 kg cart on a frictionless track. Find the final velocity.",
             "momentum_conservation_1d", "2", "Is kinetic energy conserved in that collision?", r"not conserved|not.*conserve|\bno\b"),
        ]
        for agent, setup, tool, answer, followup, expected in cases:
            with self.subTest(agent=agent):
                prompt = "Please give the full worked solution: " + setup
                first = self.ask(agent, prompt)
                self.assertIsNone(first.get("hitl"))
                self.assertIn(tool, first["tools_used"])
                self.assertIn(answer, first["solution"])
                second = self.ask(agent, followup, [{"role": "user", "content": prompt},
                                                   {"role": "assistant", "content": first["solution"]}])
                self.assertIsNone(second.get("hitl"))
                self.assertRegex(second["solution"].lower(), expected)

    def test_simple_momentum_calculation_includes_equation_substitution_and_direction(self):
        problem = "A 5.0 kg object moves at 10 m/s east. Calculate its momentum, including magnitude and direction."
        result = self.ask("momentum_agent", problem)
        if result.get("hitl"):
            self.assertEqual(result["hitl"]["status"], "question_required")
            self.assertFalse(result.get("solution"))
            result = self.ask("momentum_agent", "Please give the full worked solution: " + problem)
        self.assertIn("calculate_momentum_1d", result["tools_used"])
        for heading in ("Given", "Equation", "Substitution", "Result"):
            self.assertIn(f"**{heading}**", result["solution"])
        self.assertRegex(result["solution"], r"p\s*=\s*m\s*v")
        self.assertRegex(result["solution"], r"50(?:\.0+)?")
        self.assertIn("east", result["solution"].lower())
        self.assertNotRegex(result["solution"], r"(?i)\bMCP\b|numerical tool|calculate_momentum_1d")

    def test_first_impulse_example_completes_both_supplied_methods(self):
        problem = "Calculate 1D impulse with force 120 N over 0.35 s, initial momentum 10, final momentum 52."
        result = self.ask("momentum_agent", problem)
        if result.get("hitl"):
            self.assertEqual(result["hitl"]["status"], "question_required")
            result = self.ask("momentum_agent", "Please give the full worked solution: " + problem)
        self.assertIn("calculate_impulse_1d", result["tools_used"])
        for heading in ("Equation", "Substitution", "Result", "Check"):
            self.assertIn(f"**{heading}**", result["solution"])
        for value in ("120", "0.35", "42", "52", "10"):
            self.assertIn(value, result["solution"])
        self.assertNotIn("Would you like", result["solution"])

    def test_collision_check_then_conceptual_speed_followup(self):
        problem = "A 2.0 kg cart moving at 4.0 m/s collides with and sticks to a 3.0 kg cart initially at rest. Find the final velocity."
        result = self.ask("momentum_agent", problem)
        self.assertEqual(result["hitl"]["status"], "question_required")
        history = [{"role": "user", "content": problem}]
        if result.get("hitl"):
            check = result["hitl"]
            option = next(o for o in check["options"] if "momentum" in o["text"].lower()
                          and "angular" not in o["text"].lower())
            history.extend([{"role": "assistant", "content": check["question"] + "\n" + str(check["options"])},
                            {"role": "user", "content": option["text"]}])
            result = self.ask("momentum_agent", problem, history, {"knowledge_transfer_response": {
                "check_id": check["check_id"], "selected_option_id": option["id"]}})
            self.assertTrue(result["hitl"]["was_correct"])
        self.assertIn("momentum_conservation_1d", result["tools_used"])
        self.assertIn("1.6", result["solution"])
        history.append({"role": "assistant", "content": result["solution"]})
        followup = self.ask("momentum_agent", "why vf is smaller than vi?", history)
        self.assertIsNone(followup.get("hitl"))
        self.assertEqual(followup["tools_used"], ["get_momentum_principles"])
        self.assertRegex(followup["solution"].lower(),
                         r"momentum.*(?:conserv|shared|unchanged)|(?:conserv|redistribution of).*momentum")
        self.assertRegex(followup["solution"].lower(), r"combined|together|shared|redistribut")
        self.assertNotRegex(followup["solution"].lower(),
                            r"(?:total|system(?:'s)?) mass(?: of the system)? (?:increases|increased)")

    def test_short_equation_in_momentum_quiz_review(self):
        cases = [
            ("momentum_agent", "Two equal-mass carts move in opposite directions at equal speeds and stick. Predict the direction of their combined motion. I thought right because one was initially moving right.",
             "Which equation expresses momentum conservation for the two carts?", "m1*v1i + m2*v2i = (m1+m2)*vf"),
        ]
        for agent, quiz, tutor, equation in cases:
            with self.subTest(agent=agent):
                history = [{"role": "user", "content": "Help me review this quiz mistake with hints, one question at a time. " + quiz},
                           {"role": "assistant", "content": tutor}]
                result = self.ask(agent, equation, history)
                self.assertIsNone(result.get("hitl"))
                self.assertTrue(result["solution"].startswith("That step is correct."), result)
                self.assertEqual(result["solution"].count("?"), 1)
                self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")
                self.assertEqual(result["tools_used"], ["get_" + agent.removesuffix("_agent") + "_principles"])
                self.assertNotRegex(result["solution"], r"factor of [3456789]")

    def test_new_problem_and_return_to_earlier_problem(self):
        history = [
            {"role": "user", "content": "Explain the collision of two carts that stick on a frictionless track."},
            {"role": "assistant", "content": "Their total momentum is conserved; their kinetic energy need not be."},
        ]
        new = "Different problem: a ball rebounds from a wall at the same speed. Is the ball's own momentum conserved?"
        result = self.ask("momentum_agent", new, history)
        self.assertIsNone(result.get("hitl"))
        self.assertRegex(result["solution"].lower(), r"no|not conserved|changes")
        history += [{"role": "user", "content": new}, {"role": "assistant", "content": result["solution"]}]
        result = self.ask("momentum_agent", "Back to the carts: can their total momentum stay constant while kinetic energy decreases?", history)
        self.assertIsNone(result.get("hitl"))
        self.assertRegex(result["solution"].lower(), r"yes|can.*conserv")

    def test_momentum_rebound_uses_signed_velocity_and_external_impulse(self):
        problem = ("Please give the full worked solution: A 0.2 kg ball moves right at 5 m/s, "
                   "hits a wall, and rebounds left at 3 m/s. Take right as positive. Find the impulse on the ball.")
        result = self.ask("momentum_agent", problem)
        self.assertIsNone(result.get("hitl"))
        self.assertTrue({"momentum_impulse_theorem", "calculate_impulse_1d", "calculate_impulse_2d"}
                        .intersection(result["tools_used"]))
        self.assertRegex(result["solution"], r"-1\.6|−1\.6|1\.6[\s\S]*left")
        followup = self.ask("momentum_agent", "Is the ball's own momentum conserved during that rebound?", [
            {"role": "user", "content": problem}, {"role": "assistant", "content": result["solution"]},
        ])
        self.assertIsNone(followup.get("hitl"))
        self.assertEqual(followup["tools_used"], ["get_momentum_principles"])
        self.assertRegex(followup["solution"].lower(), r"not conserved|\bno\b")
        self.assertRegex(followup["solution"].lower(), r"external|wall.*(?:force|impulse)")
        self.assertNotRegex(followup["solution"].replace("\t" + "ext{", r"\text{"),
                            r"[+-]?\d+(?:\.\d+)?\s*(?:\\text\{\s*)?kg")

    def test_wrong_check_answer_then_guided_feedback_and_full_solution(self):
        agent = "momentum_agent"
        problem = "A 2 kg cart moving right at 3 m/s sticks to a stationary 1 kg cart on a frictionless track. Find their final velocity."
        first = self.ask(agent, problem)
        check = first["hitl"]
        self.assertEqual(check["status"], "question_required")
        wrong = next(o for o in check["options"] if "energy" in o["text"].lower()
                     and "momentum" not in o["text"].lower())
        result = self.ask(agent, problem, extra={"knowledge_transfer_response": {
            "check_id": check["check_id"], "selected_option_id": wrong["id"]}})
        self.assertFalse(result["hitl"]["was_correct"])
        self.assertEqual(result["hitl"]["status"], "remediation_required")
        self.assertEqual(result["solution"].count("?"), 1)
        self.assertNotRegex(result["solution"], r"2\.0+ m/s|final velocity is")
        # Simulate a reload: neither the original problem nor the tutor's last
        # question is supplied by the browser on this follow-up.
        uncertain = self.ask(agent, problem, extra={"knowledge_transfer_remediation_followup": {
            "check_id": check["check_id"], "student_message": "I'm lost. Can I have a smaller hint?", "mode": "clarify"}})
        self.assertEqual(uncertain["hitl"]["status"], "remediation_followup")
        self.assertEqual(uncertain["solution"].count("?"), 1)
        self.assertNotIn("Not quite", uncertain["solution"])
        self.assertEqual(uncertain["tools_used"], ["get_momentum_principles"])
        self.assertNotRegex(uncertain["solution"], r"2\.0+\s*(?:m/s|\\text)|final velocity is")
        history = [{"role": "user", "content": problem}, {"role": "assistant", "content": check["question"]},
                   {"role": "user", "content": wrong["text"]}, {"role": "assistant", "content": result["solution"]}]
        reply = "Mechanical energy is not conserved. Momentum of the two-cart system is conserved because external impulse is negligible."
        remediation = {"check_id": check["check_id"], "concept_tag": check["concept_tag"], "original_problem": problem,
                       "student_message": reply, "mode": "clarify"}
        feedback = self.ask(agent, problem, extra={"knowledge_transfer_remediation_followup": remediation})
        self.assertTrue(feedback["solution"].startswith("That step is correct."), feedback)
        self.assertEqual(feedback["hitl"]["status"], "remediation_followup")
        history += [{"role": "user", "content": reply}, {"role": "assistant", "content": feedback["solution"]}]
        complete = self.ask(agent, problem, extra={"knowledge_transfer_remediation_followup": {
            **remediation, "student_message": "The final velocity is 2 m/s to the right because momentum is conserved: (2*3 + 1*0)/(2+1)=2."}})
        self.assertEqual(complete["hitl"]["status"], "remediation_complete")
        self.assertIn("momentum_conservation_1d", complete["tools_used"])
        full = self.ask(agent, problem, extra={"knowledge_transfer_remediation_followup": {
            **remediation, "original_problem": "A different 99 kg cart moving left at 100 m/s.",
            "student_message": "Please give the full solution.", "mode": "full_solution"}})
        self.assertEqual(full["hitl"]["status"], "full_solution_requested")
        self.assertIn("momentum_conservation_1d", full["tools_used"])
        self.assertIn("2", full["solution"])
        self.assertNotIn("99", full["solution"])


if __name__ == "__main__":
    unittest.main()
