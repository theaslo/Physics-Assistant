"""Opt-in Qwen/MCP checks through the student UI proxy, not mocked model replies.

RUN_FORCES_LIVE_TESTS=1 python -B -m unittest -v test_forces_reflection_live
Override FORCES_LIVE_API_URL when running inside Docker (http://student-react-ui/api).
Uses synthetic conversations; requires the local stack and access to UConn Qwen.
"""

import os
import time
import unittest
import uuid

import requests


INTRO = (
    "I am reviewing a quiz about Newton's laws and forces. My answer and reasoning are below. "
    "Help me identify my misunderstanding. Please ask me one question at a time and give hints before revealing the answer."
)
INTAKE = "Please provide the actual quiz question, your original answer, and your reasoning for that answer."
PUCK_QUIZ = (
    "Suppose you are playing hockey on a new-age ice surface for which there is no friction between the ice and the hockey puck. "
    "You wind up and hit the puck as hard as you can. After the puck loses contact with your stick, the puck will "
    "A) start to slow down. B) not slow down or speed up. C) speed up a little, and then slow down. "
    "D) speed up a little, and then move at a constant speed.\n"
    "I said D because it should go faster first because I am hitting it"
)
PUCK_HINT = "What is the net force acting on the puck after it loses contact with your stick?"


@unittest.skipUnless(os.getenv("RUN_FORCES_LIVE_TESTS") == "1", "Opt-in: requires live Qwen and MCP services")
class ForcesReflectionLiveTests(unittest.TestCase):
    def setUp(self):
        self.url = os.getenv("FORCES_LIVE_API_URL", "http://localhost:8501/api").rstrip("/")
        self.session = "short-reply-regression-" + uuid.uuid4().hex

    def history(self, quiz=PUCK_QUIZ, tutor_question=PUCK_HINT):
        return [
            {"role": "user", "content": INTRO},
            {"role": "assistant", "content": INTAKE},
            {"role": "user", "content": quiz},
            {"role": "assistant", "content": tutor_question},
        ]

    def ask(self, reply, history):
        started = time.monotonic()
        response = requests.post(self.url + "/agent/forces_agent/solve", json={
            "problem": reply, "user_id": "forces_live_regression", "session_id": self.session,
            "context": {"conversation_context": {
                "recent_messages": history,
                "previous_user_problem": next((m["content"] for m in reversed(history) if m["role"] == "user"), ""),
                "previous_assistant_response": next((m["content"] for m in reversed(history) if m["role"] == "assistant"), ""),
            }},
        }, timeout=120)
        response.raise_for_status()
        result = response.json()
        self.assertTrue(result.get("success"), result)
        self.assertIsNone(result.get("hitl"), result)
        self.assertIn("get_force_principles", result.get("tools_used", []))
        self.assertTrue(result.get("metadata", {}).get("llm_response_used"))
        print(f"\n{self.id()} [{reply!r}, {time.monotonic() - started:.1f}s]: {result['solution']}", flush=True)
        return result

    def assert_feedback(self, result, correct=True):
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "feedback")
        self.assertNotIn("provide the actual quiz", result["solution"].lower())
        self.assertEqual(result["solution"].count("?"), 1)
        self.assertTrue(result["solution"].startswith("That step is correct." if correct else "Not quite."), result)

    def test_exact_puck_zero_and_numeral(self):
        for reply in ("zero", "0"):
            with self.subTest(reply=reply):
                result = self.ask(reply, self.history())
                self.assert_feedback(result)
                self.assertNotIn("constant velocity", result["solution"].lower())
                self.assertEqual(result["tools_used"], ["get_force_principles"])

    def test_direction_reply_in_different_problem(self):
        result = self.ask("left", self.history(
            "A block slides right on a rough level floor. I chose friction to the right because it is moving right. Help me review my mistake.",
            "Which direction does kinetic friction act on the sliding block?",
        ))
        self.assert_feedback(result)
        self.assertEqual(result["tools_used"], ["get_force_principles"])

    def test_equation_reply_in_different_problem(self):
        result = self.ask("F=-kx", self.history(
            "An ideal spring is stretched twice as far. I chose half the restoring force because I thought the force was inversely proportional to extension.",
            "Which equation relates the spring force to displacement?",
        ))
        self.assert_feedback(result)
        self.assertEqual(result["tools_used"], ["get_force_principles"])
        self.assertNotRegex(result["solution"].lower(), r"why (is|does) (the )?(restoring|spring) force (halv|decreas)")

    def test_incorrect_yes_reply_is_not_accepted(self):
        result = self.ask("yes", self.history(tutor_question="Does the stick continue to push the puck after they lose contact?"))
        self.assert_feedback(result, correct=False)

    def test_missing_quiz_still_requests_intake(self):
        result = self.ask("zero", self.history()[:2])
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "intake")
        self.assertIn("quiz", result["solution"].lower())

    def test_explicit_new_review_can_reset(self):
        result = self.ask("I want to start a different quiz review now. I will paste the new quiz question next.", self.history())
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "intake")
        self.assertNotIn("hockey", result["solution"].lower())
        self.assertRegex(result["solution"].lower(), r"paste|provide|share|send")

    def test_new_numerical_problem_still_uses_mcp(self):
        result = self.ask("New problem, not the quiz: calculate the spring force for k=90 N/m and a 0.20 m extension.", self.history())
        self.assertIn("calculate_spring_force_tool", result["tools_used"])
        self.assertIn("18", result["solution"])

    def test_explicit_full_solution_request(self):
        result = self.ask("I am completely lost. Please give me the full answer and explanation now, not another hint.", self.history())
        self.assertRegex(result["solution"].lower(), r"\bb\b|constant (speed|velocity)|not slow down or speed up")
        self.assertRegex(result["solution"].lower(), r"net (horizontal )?force|zero (horizontal )?force|no (net )?horizontal force")
        self.assertNotRegex(result["solution"].lower(), r"\bno (other )?forces acting on it\b")
        self.assertNotIn("?", result["solution"])

    def test_introduction_and_actual_quiz_submission(self):
        intake = self.ask(INTRO, [])
        self.assertEqual(intake["metadata"]["quiz_reflection_mode"]["stage"], "intake")
        self.assertNotIn("first law", intake["solution"].lower())
        result = self.ask(PUCK_QUIZ, [
            {"role": "user", "content": INTRO}, {"role": "assistant", "content": intake["solution"]},
        ])
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "hint")
        self.assertEqual(result["solution"].count("?"), 1)
        self.assertNotRegex(result["solution"].lower(), r"answer is b|constant (speed|velocity)|choice b")
        history = [
            {"role": "user", "content": INTRO}, {"role": "assistant", "content": intake["solution"]},
            {"role": "user", "content": PUCK_QUIZ}, {"role": "assistant", "content": result["solution"]},
        ]
        reply = "After contact ends the stick exerts no force, and there is no friction, so the net horizontal force is zero."
        feedback = self.ask(reply, history)
        self.assert_feedback(feedback)
        history += [{"role": "user", "content": reply}, {"role": "assistant", "content": feedback["solution"]}]
        completed = self.ask("B. Its velocity stays constant because the net force and therefore acceleration are zero.", history)
        self.assertEqual(completed["metadata"]["quiz_reflection_mode"]["stage"], "complete")
        self.assertNotIn("?", completed["solution"])

    def test_completed_quiz_is_confirmed(self):
        result = self.ask(
            "B: the puck keeps the same velocity because after contact ends the net horizontal force is zero, so acceleration is zero.",
            self.history(),
        )
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "complete")
        self.assertNotIn("?", result["solution"])
        self.assertEqual(result["tools_used"], ["get_force_principles"])

    def test_completed_correction_after_more_than_six_messages(self):
        history = self.history(tutor_question="Does the stick still push after contact ends?") + [
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "Not quite. Can the stick exert a contact force without touching the puck?"},
            {"role": "user", "content": "No. The stick is no longer touching it, and there is no friction."},
            {"role": "assistant", "content": "Correct. What is the net horizontal force?"},
            {"role": "user", "content": "zero"},
            {"role": "assistant", "content": "Correct. What does that imply about acceleration and motion?"},
        ]
        result = self.ask("B. The acceleration is zero, so its velocity stays constant because the net horizontal force is zero.", history)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "complete")
        self.assertNotIn("?", result["solution"])

    def test_static_friction_and_changing_force_questions(self):
        for quiz, tutor, reply in (
            ("A crate stays at rest while pushed horizontally on a rough level floor. I thought the push was larger than friction.",
             "How do the magnitudes of the horizontal push and static friction compare?", "equal"),
            ("An object moves right while its net force points right but decreases. I thought it must slow down.",
             "While the net force is still rightward, which direction is its acceleration?", "right"),
        ):
            with self.subTest(reply=reply):
                self.assert_feedback(self.ask(reply, self.history(quiz, tutor)))

    def test_correct_and_incorrect_numerical_work_uses_calculation_tool(self):
        history = self.history(
            "On my quiz a 5 kg cart experiences a 10 N net force to the right. Find its acceleration and explain why. I chose 50 m/s^2 because I multiplied.",
            "Using F=ma, what acceleration magnitude do you calculate?",
        )
        for reply, correct in (("2 m/s^2", True), ("6 m/s^2", False)):
            with self.subTest(reply=reply):
                result = self.ask(reply, history)
                self.assert_feedback(result, correct=correct)
                self.assertIn("newton_second_law", result["tools_used"])

    def test_transfer_practice_and_completed_attempt(self):
        history = self.history() + [
            {"role": "user", "content": "B. Velocity stays constant because the net horizontal force and acceleration are zero."},
            {"role": "assistant", "content": "Correct. You have answered the question and explained why."},
        ]
        request = "Give me one new practice question on the concept I misunderstood. Change the situation, not just the numbers. Do not show the answer until I attempt it."
        practice = self.ask(request, history)
        self.assertEqual(practice["metadata"]["quiz_reflection_mode"]["stage"], "practice")
        self.assertEqual(practice["solution"].count("?"), 1)
        self.assertNotIn("puck", practice["solution"].lower())
        history += [{"role": "user", "content": request}, {"role": "assistant", "content": practice["solution"]}]
        result = self.ask("It keeps the same velocity because there is no net external force, so it has no acceleration.", history)
        self.assertEqual(result["metadata"]["quiz_reflection_mode"]["stage"], "complete")
        self.assertNotIn("?", result["solution"])

    def test_ordinary_followup_and_topic_change(self):
        history = [
            {"role": "user", "content": "A box slides down a 30 degree ramp with kinetic friction."},
            {"role": "assistant", "content": "Kinetic friction acts up the ramp."},
        ]
        result = self.ask("Is friction down the ramp?", history)
        self.assertNotIn("quiz_reflection_mode", result["metadata"])
        self.assertRegex(result["solution"].lower(), r"\bno\b")
        self.assertIn("up", result["solution"].lower())
        history += [{"role": "user", "content": "A spring is stretched to the right."},
                    {"role": "assistant", "content": "The spring pulls left."}]
        result = self.ask("On a new incline a box slides UP the ramp. Which way is friction?", history)
        self.assertIn("down", result["solution"].lower())
        self.assertNotIn("spring", result["solution"].lower())


if __name__ == "__main__":
    unittest.main()
