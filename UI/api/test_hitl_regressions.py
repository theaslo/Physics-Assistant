import os
import unittest
from typing import Any, Dict, List, Optional

from hitl_knowledge_transfer import KnowledgeTransferGate


class HitlRegressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        os.environ["HITL_GATE_ENABLED"] = "true"
        os.environ["HITL_GATE_THRESHOLD"] = "0.70"
        os.environ["HITL_PILOT_AGENTS"] = "forces_agent,kinematics_agent"
        self.gate = KnowledgeTransferGate(database_api_url="http://database-api:8001")
        self.logged_attempts: List[Dict[str, Any]] = []

        async def fake_fetch_question(agent_id: str, concept_tag: str) -> Optional[Dict[str, Any]]:
            if agent_id != "forces_agent" or concept_tag != "hookes_law":
                return None
            return {
                "question_id": "66666666-6666-4666-8666-666666666666",
                "agent_type": "forces",
                "concept_tag": "hookes_law",
                "question": "For an ideal spring near equilibrium, which relation is Hooke's law?",
                "options": [
                    {"id": "A", "text": "F = kx"},
                    {"id": "B", "text": "F = x/k"},
                    {"id": "C", "text": "F = k/x"},
                ],
                "correct_option_id": "A",
                "correct_feedback": "Magnitude relation is |F| = k|x|.",
                "incorrect_feedback": "Hooke's law is linear in displacement.",
                "distractor_feedback": {
                    "B": "That inverts spring-constant dependence.",
                    "C": "That is not linear in displacement.",
                },
            }

        async def fake_fetch_from_mcp(agent_id: str, concept_tag: str, user_prompt: str) -> Optional[Dict[str, Any]]:
            return None

        def fake_log_attempt(**kwargs: Any) -> None:
            self.logged_attempts.append(kwargs)

        self.gate._fetch_question_from_database = fake_fetch_question
        self.gate._fetch_question_from_mcp = fake_fetch_from_mcp
        self.gate._log_attempt = fake_log_attempt

    async def asyncTearDown(self) -> None:
        await self.gate.cleanup()

    async def test_forces_agent_hitl_first_leg_question_flow(self):
        check, trace = await self.gate.maybe_create_check_with_trace(
            agent_id="forces_agent",
            problem="A spring stretches 0.2 m with k=50 N/m. Use Hooke law to find the force.",
            user_id="student-a",
            session_id="session-a",
            class_identifier="class-a",
        )

        self.assertIsNotNone(check)
        self.assertEqual(check["status"], "question_required")
        self.assertEqual(check["agent_id"], "forces_agent")
        self.assertEqual(check["concept_tag"], "hookes_law")
        self.assertEqual(check["question_id"], "66666666-6666-4666-8666-666666666666")
        self.assertEqual(check["options"][0], {"id": "A", "text": "F = kx"})
        self.assertNotIn("correct_option_id", check)
        self.assertEqual(trace["operation"], "maybe_create_check")
        self.assertIn("database_question_fetch", [stage["stage"] for stage in trace["stages"]])

    async def test_second_hitl_leg_scores_correct_answer_and_logs_attempt(self):
        check, _ = await self.gate.maybe_create_check_with_trace(
            agent_id="forces_agent",
            problem="A spring stretches 0.2 m with k=50 N/m. Use Hooke law to find the force.",
            user_id="student-a",
        )

        result, trace = self.gate.process_answer_with_trace(
            check_id=check["check_id"],
            selected_option_id="A",
            user_id="student-a",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "answer_processed")
        self.assertTrue(result["was_correct"])
        self.assertIn("Knowledge Check: Correct.", result["guidance"])
        self.assertEqual(result["concept_tag"], "hookes_law")
        self.assertEqual(trace["operation"], "process_answer")
        self.assertEqual(len(self.logged_attempts), 1)
        self.assertTrue(self.logged_attempts[0]["was_correct"])
        self.assertEqual(self.logged_attempts[0]["selected_option_id"], "A")

    async def test_second_hitl_leg_scores_incorrect_answer_and_logs_attempt(self):
        check, _ = await self.gate.maybe_create_check_with_trace(
            agent_id="forces_agent",
            problem="A spring stretches 0.2 m with k=50 N/m. Use Hooke law to find the force.",
            user_id="student-a",
        )

        result, _ = self.gate.process_answer_with_trace(
            check_id=check["check_id"],
            selected_option_id="B",
            user_id="student-a",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "remediation_required")
        self.assertFalse(result["was_correct"])
        self.assertIn("Knowledge Check: Not quite.", result["guidance"])
        self.assertIn("That inverts spring-constant dependence.", result["guidance"])
        self.assertIn("do you understand why this answer is not correct", result["guidance"])
        self.assertIn("next step", result["guidance"])
        self.assertNotIn("Proceeding to the full solution.", result["guidance"])
        self.assertEqual(result["remediation"]["can_continue"], False)
        self.assertEqual(len(self.logged_attempts), 1)
        self.assertFalse(self.logged_attempts[0]["was_correct"])
        self.assertEqual(self.logged_attempts[0]["selected_option_id"], "B")


class HitlApiRouteRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import main as api_main

        self.api_main = api_main
        self.original_gate = api_main.knowledge_transfer_gate
        self.original_get_or_create_agent = api_main.get_or_create_agent

    async def asyncTearDown(self) -> None:
        self.api_main.knowledge_transfer_gate = self.original_gate
        self.api_main.get_or_create_agent = self.original_get_or_create_agent

    async def test_wrong_hitl_answer_returns_remediation_without_solver(self):
        class StubGate:
            def is_enabled_for(self, agent_id: str) -> bool:
                return agent_id == "forces_agent"

            def process_answer_with_trace(self, **kwargs: Any):
                return (
                    {
                        "status": "remediation_required",
                        "check_id": kwargs["check_id"],
                        "agent_id": "forces_agent",
                        "original_problem": "A spring stretches 0.2 m with k=50 N/m. Find the force.",
                        "concept_tag": "hookes_law",
                        "was_correct": False,
                        "guidance": (
                            "Knowledge Check: Not quite.\n"
                            "Hooke's law is linear in displacement.\n\n"
                            "Before we continue: do you understand why this answer is not correct?\n"
                            'If you would like help, reply "next step" and I will guide you one step at a time.'
                        ),
                        "confidence": 0.9,
                        "threshold": 0.7,
                        "remediation": {
                            "prompt": "Do you understand why this answer is not correct?",
                            "next_step_prompt": 'Reply "next step" if you want the next step of the solution.',
                            "can_continue": False,
                        },
                    },
                    {"operation": "process_answer", "stages": []},
                )

        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created for incorrect HITL answers")

        self.api_main.knowledge_transfer_gate = StubGate()
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "forces_agent",
            self.api_main.ProblemSolveRequest(
                problem="A spring stretches 0.2 m with k=50 N/m. Find the force.",
                user_id="student-a",
                context={
                    "knowledge_transfer_response": {
                        "check_id": "check-1",
                        "selected_option_id": "B",
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_required")
        self.assertFalse(response.hitl["was_correct"])
        self.assertIn("do you understand why this answer is not correct", response.solution)
        self.assertIsNone(response.tools_used)


if __name__ == "__main__":
    unittest.main()
