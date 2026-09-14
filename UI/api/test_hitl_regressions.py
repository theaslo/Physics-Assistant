import math
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

        question_bank: Dict[tuple[str, str], Dict[str, Any]] = {
            (
                "forces_agent",
                "hookes_law",
            ): {
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
            },
            (
                "forces_agent",
                "newton_second_law",
            ): {
                "question_id": "77777777-7777-4777-8777-777777777777",
                "agent_type": "forces",
                "concept_tag": "newton_second_law",
                "question": "For Newton's Second Law, what equation links net force, mass, and acceleration?",
                "options": [
                    {"id": "A", "text": "F = ma"},
                    {"id": "B", "text": "F = m/a"},
                    {"id": "C", "text": "F = a/m"},
                ],
                "correct_option_id": "A",
                "correct_feedback": "Net force equals mass times acceleration.",
                "incorrect_feedback": "Use the net force, not just one force.",
                "distractor_feedback": {},
            },
            (
                "kinematics_agent",
                "projectile_components",
            ): {
                "question_id": "88888888-8888-4888-8888-888888888888",
                "agent_type": "kinematics",
                "concept_tag": "projectile_components",
                "question": "For projectile launch speed v0 at angle theta, what is the initial vertical component?",
                "options": [
                    {"id": "A", "text": "v0 sin(theta)"},
                    {"id": "B", "text": "v0 cos(theta)"},
                    {"id": "C", "text": "v0 tan(theta)"},
                ],
                "correct_option_id": "A",
                "correct_feedback": "Vertical component uses sine.",
                "incorrect_feedback": "Separate the launch velocity into x and y components.",
                "distractor_feedback": {},
            },
            (
                "kinematics_agent",
                "equation_selection",
            ): {
                "question_id": "99999999-9999-4999-8999-999999999999",
                "agent_type": "kinematics",
                "concept_tag": "equation_selection",
                "question": "Which constant-acceleration equation directly avoids time t?",
                "options": [
                    {"id": "A", "text": "vf^2 = v0^2 + 2a delta x"},
                    {"id": "B", "text": "vf = v0 + at"},
                    {"id": "C", "text": "x = x0 + v0t + 1/2 at^2"},
                ],
                "correct_option_id": "A",
                "correct_feedback": "That relation eliminates time.",
                "incorrect_feedback": "Choose the equation that contains the unknown and knowns.",
                "distractor_feedback": {},
            },
        }

        async def fake_fetch_question(agent_id: str, concept_tag: str) -> Optional[Dict[str, Any]]:
            return question_bank.get((agent_id, concept_tag))

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
        self.assertIsNone(check["question_id"])
        self.assertIn("spring-force problem", check["question"])
        self.assertEqual(check["options"][0], {"id": "A", "text": "|F_s| = k |x|"})
        self.assertNotIn("correct_option_id", check)
        self.assertEqual(trace["operation"], "maybe_create_check")
        self.assertIn("contextual_question_build", [stage["stage"] for stage in trace["stages"]])

    async def test_thrown_ball_projectile_prompt_triggers_kinematics_hitl(self):
        check, trace = await self.gate.maybe_create_check_with_trace(
            agent_id="kinematics_agent",
            problem="Ball thrown at 30 m/s at 45° from 10m height",
            user_id="student-a",
        )

        self.assertIsNotNone(check)
        self.assertEqual(check["status"], "question_required")
        self.assertEqual(check["agent_id"], "kinematics_agent")
        self.assertEqual(check["concept_tag"], "projectile_components")
        self.assertIsNone(check["question_id"])
        self.assertIn("best first setup step", check["question"])
        self.assertIn("Ball thrown at 30 m/s at 45°", check["question"])
        self.assertIn("projectile_components", check["reason_tags"])
        self.assertIn("contextual_question_build", [stage["stage"] for stage in trace["stages"]])

    async def test_one_dimensional_kinematics_prompt_triggers_hitl(self):
        check, trace = await self.gate.maybe_create_check_with_trace(
            agent_id="kinematics_agent",
            problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
            user_id="student-a",
        )

        self.assertIsNotNone(check)
        self.assertEqual(check["status"], "question_required")
        self.assertEqual(check["agent_id"], "kinematics_agent")
        self.assertEqual(check["concept_tag"], "equation_selection")
        self.assertIsNone(check["question_id"])
        self.assertIn("Which kinematics equation fits this question best?", check["question"])
        self.assertEqual(check["options"][0], {"id": "A", "text": "v_f = v_0 + a t"})
        self.assertIn("equation_selection", check["reason_tags"])
        self.assertIn("contextual_question_build", [stage["stage"] for stage in trace["stages"]])

    async def test_one_dimensional_kinematics_question_wording_rotates(self):
        questions: List[str] = []
        first_options: List[Dict[str, str]] = []
        for _ in range(3):
            check, _ = await self.gate.maybe_create_check_with_trace(
                agent_id="kinematics_agent",
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
            )
            questions.append(check["question"])
            first_options.append(check["options"][0])

        self.assertEqual(len(set(questions)), 3)
        self.assertTrue(all("A car starts from rest" in question for question in questions))
        self.assertTrue(all(option == {"id": "A", "text": "v_f = v_0 + a t"} for option in first_options))

    async def test_generic_forces_prompt_triggers_hitl(self):
        check, trace = await self.gate.maybe_create_check_with_trace(
            agent_id="forces_agent",
            problem="A box is pushed across the floor. What should I do first?",
            user_id="student-a",
        )

        self.assertIsNotNone(check)
        self.assertEqual(check["status"], "question_required")
        self.assertEqual(check["agent_id"], "forces_agent")
        self.assertEqual(check["concept_tag"], "newton_second_law")
        self.assertIsNone(check["question_id"])
        self.assertIn("before calculating this force problem", check["question"])
        self.assertIn("sum F = ma", check["options"][0]["text"])
        self.assertIn("newton_second_law", check["reason_tags"])
        self.assertIn("contextual_question_build", [stage["stage"] for stage in trace["stages"]])

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

    async def test_remediation_next_step_followup_returns_scaffold_without_solver(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created for HITL remediation follow-ups")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A projectile is launched at 30 m/s at 45 degrees. Find the maximum height.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-2",
                        "concept_tag": "projectile_peak",
                        "student_message": "next step",
                        "step_index": 0,
                        "mode": "next_step",
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertEqual(response.hitl["concept_tag"], "projectile_peak")
        self.assertEqual(response.hitl["step_index"], 0)
        self.assertIn("Next step 1", response.solution)
        self.assertIn("v_y = 0", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_next_step_prompt_does_not_advance_before_equation_work(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created for HITL remediation follow-ups")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        prompt_response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-next-before-work",
                        "concept_tag": "equation_selection",
                        "student_message": "next step",
                        "step_index": 0,
                        "mode": "next_step",
                    }
                },
            ),
        )

        self.assertTrue(prompt_response.success)
        self.assertEqual(prompt_response.hitl["status"], "remediation_followup")
        self.assertEqual(prompt_response.hitl["step_index"], 0)
        self.assertIn("Next step 1", prompt_response.solution)
        self.assertIn("Write the kinematics equation", prompt_response.solution)

        equation_response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-next-before-work",
                        "concept_tag": "equation_selection",
                        "student_message": "v_f = v_0 + a t",
                        "step_index": prompt_response.hitl["step_index"],
                    }
                },
            ),
        )

        self.assertTrue(equation_response.success)
        self.assertEqual(equation_response.hitl["status"], "remediation_followup")
        self.assertEqual(equation_response.hitl["submission_check"]["status"], "correct")
        self.assertEqual(equation_response.hitl["step_index"], 1)
        self.assertIn("right equation", equation_response.solution)
        self.assertIn("Substitute v_0, a, and t", equation_response.solution)
        self.assertNotIn("I need to see the numerical calculation", equation_response.solution)

    async def test_kinematics_remediation_correct_equation_is_checked_and_advances(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created for HITL remediation follow-ups")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-equation-correct",
                        "concept_tag": "equation_selection",
                        "student_message": "I would use v_f = v_0 + a t",
                        "step_index": 0,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertEqual(response.hitl["submission_check"]["status"], "correct")
        self.assertEqual(response.hitl["step_index"], 1)
        self.assertIn("I checked your answer", response.solution)
        self.assertIn("right equation", response.solution)
        self.assertIn("Next step", response.solution)
        self.assertIn("Substitute v_0, a, and t", response.solution)
        self.assertNotIn("Check your work against this", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_correct_equation_at_substitution_state_is_accepted_without_advancing(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created for HITL remediation follow-ups")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-stale-equation-state",
                        "concept_tag": "equation_selection",
                        "student_message": "v_f = v_0 + a t",
                        "step_index": 1,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertEqual(response.hitl["submission_check"]["status"], "correct")
        self.assertEqual(response.hitl["submission_check"]["advance_step"], False)
        self.assertEqual(response.hitl["step_index"], 1)
        self.assertIn("That equation is correct", response.solution)
        self.assertIn("Replace the symbols with the values", response.solution)
        self.assertNotIn("I need to see the numerical calculation", response.solution)

    async def test_kinematics_remediation_symbol_only_substitution_gets_targeted_feedback(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created for HITL remediation follow-ups")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-substitution-symbols",
                        "concept_tag": "equation_selection",
                        "student_message": "v_0, a, and t",
                        "step_index": 1,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertEqual(response.hitl["submission_check"]["status"], "partial")
        self.assertEqual(response.hitl["step_index"], 1)
        self.assertIn("You identified the right symbols", response.solution)
        self.assertIn("replacing v_0, a, and t with the numbers", response.solution)
        self.assertIn("v_f = 0 + (2)(5)", response.solution)
        self.assertNotIn("I do not see the final-velocity equation", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_kinematics_remediation_numeric_substitution_is_checked_and_advances(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created for HITL remediation follow-ups")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-substitution-values",
                        "concept_tag": "equation_selection",
                        "student_message": "v_0 = 0, a = 2, t = 5",
                        "step_index": 1,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertEqual(response.hitl["submission_check"]["status"], "correct")
        self.assertEqual(response.hitl["step_index"], 2)
        self.assertIn("I checked your answer", response.solution)
        self.assertIn("substituted numerical values", response.solution)
        self.assertIn("Now calculate a t", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_kinematics_remediation_correct_final_calculation_completes(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created after checked final calculation")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-final-calculation",
                        "concept_tag": "equation_selection",
                        "student_message": "a t = 10, so v_f = 10 m/s",
                        "step_index": 2,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_complete")
        self.assertEqual(response.hitl["submission_check"]["status"], "complete")
        self.assertEqual(response.hitl["step_index"], 3)
        self.assertIn("I checked your final answer", response.solution)
        self.assertIn("v_f = 10 m/s", response.solution)
        self.assertIn("closing this step-by-step check", response.solution)
        self.assertNotIn("Now calculate a t", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_kinematics_complete_answer_at_equation_step_completes(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created after complete submitted solution")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-complete-at-equation-step",
                        "concept_tag": "equation_selection",
                        "student_message": "v_f = v_0 + a t = 0 + (2)(5) = 10 m/s",
                        "step_index": 0,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_complete")
        self.assertEqual(response.hitl["submission_check"]["status"], "complete")
        self.assertIn("I checked your final answer", response.solution)
        self.assertIn("v_f = 10 m/s", response.solution)
        self.assertNotIn("Substitute v_0, a, and t", response.solution)
        self.assertNotIn("Now calculate a t", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_kinematics_complete_answer_at_substitution_step_completes(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created after complete submitted solution")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-complete-at-substitution-step",
                        "concept_tag": "equation_selection",
                        "student_message": "v_f = 0 + 2(5) = 10 m/s",
                        "step_index": 1,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_complete")
        self.assertEqual(response.hitl["submission_check"]["status"], "complete")
        self.assertIn("I checked your final answer", response.solution)
        self.assertIn("v_f = 10 m/s", response.solution)
        self.assertNotIn("Now calculate a t", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_hitl_full_solution_request_runs_solver_and_skips_gate(self):
        seen: Dict[str, Any] = {}

        class FailGate:
            def is_enabled_for(self, agent_id: str) -> bool:
                return True

            async def maybe_create_check_with_trace(self, **kwargs: Any):
                raise AssertionError("HITL gate should be skipped for full-solution remediation requests")

        class StubAgent:
            async def solve_problem(
                self,
                problem: str,
                context: Optional[Dict[str, Any]],
                user_id: Optional[str],
                session_id: Optional[str],
            ) -> Dict[str, Any]:
                seen["problem"] = problem
                seen["context"] = context
                seen["user_id"] = user_id
                return {
                    "success": True,
                    "agent_id": "kinematics_agent",
                    "problem": problem,
                    "solution": "Full worked solution body.",
                    "tools_used": ["constant_acceleration_1d"],
                    "metadata": {},
                }

        async def fake_get_or_create_agent(*args: Any, **kwargs: Any):
            return StubAgent()

        self.api_main.knowledge_transfer_gate = FailGate()
        self.api_main.get_or_create_agent = fake_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-full-solution",
                        "concept_tag": "equation_selection",
                        "student_message": "I'm lost. Please show the full solution.",
                        "step_index": 1,
                        "mode": "full_solution",
                    },
                    "class_identifier": "physics-101",
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "full_solution_requested")
        self.assertEqual(response.problem, "A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.")
        self.assertEqual(seen["problem"], response.problem)
        self.assertEqual(seen["context"], {"class_identifier": "physics-101"})
        self.assertIn("leaving step-by-step checking", response.solution)
        self.assertIn("Full worked solution body.", response.solution)
        self.assertEqual(response.tools_used, ["constant_acceleration_1d"])

    async def test_forces_incline_free_body_prompt_uses_incline_diagram_with_friction(self):
        from strands_agents import ForcesAgent

        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        agent.initialized = True

        response = await agent.solve_problem(
            problem=(
                "A 5.0 kg box slides down a 30° incline with coefficient of kinetic friction "
                "μk = 0.30. Draw the free-body diagram and find the acceleration down the ramp."
            ),
            context=None,
            user_id="student-a",
        )

        diagram = response["diagram"]
        acceleration = diagram["net_down_n"] / diagram["mass_kg"]

        self.assertTrue(response["success"])
        self.assertEqual(response["tools_used"], ["analyze_forces_on_incline"])
        self.assertEqual(diagram["type"], "inclined_plane_diagram")
        self.assertTrue(diagram["has_friction"])
        self.assertAlmostEqual(diagram["coefficient_friction"], 0.30)
        self.assertAlmostEqual(diagram["weight_n"], 49.05, places=2)
        self.assertAlmostEqual(diagram["normal_n"], 42.48, places=2)
        self.assertAlmostEqual(diagram["friction_n"], 12.74, places=2)
        self.assertAlmostEqual(acceleration, 2.36, places=2)
        self.assertIn("actual forces", response["solution"])
        self.assertIn("not extra forces", response["solution"])
        self.assertTrue(math.isfinite(response["execution_time_ms"]))

    async def test_math_agent_generates_physics_algebra_practice_without_answers(self):
        from strands_agents import MathAgent

        agent = MathAgent(enable_database_logging=False, enable_rag=False)
        agent.initialized = True

        response = await agent.solve_problem(
            problem="Give me 4 kinematics algebra practice exercises on rearranging formulas, without answers.",
            context=None,
            user_id="student-a",
        )

        solution = response["solution"]
        exercise_mode = response["metadata"]["exercise_mode"]

        self.assertTrue(response["success"])
        self.assertEqual(response["tools_used"], ["physics_algebra_exercise_generator"])
        self.assertEqual(exercise_mode["type"], "physics_algebra")
        self.assertEqual(exercise_mode["topic"], "kinematics")
        self.assertEqual(exercise_mode["exercise_count"], 4)
        self.assertFalse(exercise_mode["includes_answer_key"])
        self.assertIn("Physics Algebra Practice", solution)
        self.assertIn("Formula: v_f = v_0 + a t", solution)
        self.assertIn("Rearrange the equation symbolically first", solution)
        self.assertIn("send your answers", solution)
        self.assertNotIn("Answer key:", solution)

    async def test_math_agent_includes_answer_key_when_requested(self):
        from strands_agents import MathAgent

        agent = MathAgent(enable_database_logging=False, enable_rag=False)
        agent.initialized = True

        response = await agent.solve_problem(
            problem="Give me 3 energy physics algebra exercises with answers.",
            context=None,
            user_id="student-a",
        )

        solution = response["solution"]
        exercise_mode = response["metadata"]["exercise_mode"]

        self.assertTrue(response["success"])
        self.assertEqual(response["tools_used"], ["physics_algebra_exercise_generator"])
        self.assertEqual(exercise_mode["topic"], "energy")
        self.assertEqual(exercise_mode["exercise_count"], 3)
        self.assertTrue(exercise_mode["includes_answer_key"])
        self.assertIn("Answer key:", solution)
        self.assertIn("v = sqrt(2K / m)", solution)
        self.assertIn("m = U_g / (g h)", solution)

    async def test_math_agent_resolves_vector_components_with_mcp_tool(self):
        from strands_agents import MathAgent

        calls: Dict[str, Any] = {}

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id: str, name: str, arguments: Dict[str, Any]):
                calls["tool_use_id"] = tool_use_id
                calls["name"] = name
                calls["arguments"] = arguments
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                "Vector Component Resolution:\n"
                                "x-component = 19.15 N\n"
                                "y-component = 16.07 N\n\n"
                                "DIAGRAM_JSON_START\n"
                                '{"type":"force_components_diagram","title":"Vector Components",'
                                '"vector":{"name":"Vector","magnitude_n":25.0,"angle_deg":40.0,'
                                '"fx_n":19.151111077974452,"fy_n":16.06969024216348}}\n'
                                "DIAGRAM_JSON_END"
                            )
                        }
                    ],
                }

        agent = MathAgent(enable_database_logging=False, enable_rag=False)
        agent.initialized = True
        agent.mcp_client = FakeMcpClient()

        async def fake_llm_plan(problem, catalog, rag_context):
            self.assertIn("resolve_vector_components", [tool["name"] for tool in catalog])
            return {
                "can_solve_with_mcp": True,
                "tool_calls": [
                    {
                        "tool_name": "resolve_vector_components",
                        "arguments": {
                            "magnitude": 25,
                            "angle_degrees": 40,
                            "units": "N",
                        },
                    }
                ],
                "confidence": 0.95,
            }

        agent._build_fast_mcp_plan_with_llm = fake_llm_plan

        response = await agent.solve_problem(
            problem="Resolve a 25 N vector directed 40° above the +x axis into its x and y components",
            context=None,
            user_id="student-a",
        )

        self.assertTrue(response["success"])
        self.assertEqual(response["tools_used"], ["resolve_vector_components"])
        self.assertEqual(calls["name"], "resolve_vector_components")
        self.assertEqual(calls["arguments"], {"magnitude": 25.0, "angle_degrees": 40.0, "units": "N"})
        self.assertEqual(response["diagram"]["type"], "force_components_diagram")
        self.assertAlmostEqual(response["diagram"]["vector"]["fx_n"], 19.15, places=2)
        self.assertAlmostEqual(response["diagram"]["vector"]["fy_n"], 16.07, places=2)
        self.assertIn("Vector Component Resolution", response["solution"])
        self.assertNotIn("DIAGRAM_JSON_START", response["solution"])
        self.assertTrue(response["metadata"]["fast_mcp_pipeline"]["llm_plan_used"])

    async def test_math_agent_unit_speed_distance_uses_general_mcp_pipeline(self):
        from strands_agents import MathAgent

        calls: List[Dict[str, Any]] = []

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id: str, name: str, arguments: Dict[str, Any]):
                calls.append({"name": name, "arguments": arguments})
                if name == "unit_converter":
                    return {
                        "status": "success",
                        "content": [{"text": "Unit Conversion:\nResult: 20 m/s"}],
                    }
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                "Physics Formula Solver:\n"
                                "Formula: d = vt\n"
                                "d = (20.000 m/s)(5.000 s) = 100.000 m\n"
                                "Answer: d = 100.000 m"
                            )
                        }
                    ],
                }

        agent = MathAgent(enable_database_logging=False, enable_rag=False)
        agent.initialized = True
        agent.mcp_client = FakeMcpClient()

        async def fake_llm_plan(problem, catalog, rag_context):
            tool_names = [tool["name"] for tool in catalog]
            self.assertIn("unit_converter", tool_names)
            self.assertIn("physics_formula_solver", tool_names)
            return {
                "can_solve_with_mcp": True,
                "tool_calls": [
                    {
                        "tool_name": "unit_converter",
                        "arguments": {"value": 72, "from_unit": "km/h", "to_unit": "m/s"},
                    },
                    {
                        "tool_name": "physics_formula_solver",
                        "arguments": {
                            "formula_name": "uniform_motion",
                            "known_values": {"speed": 72, "speed_unit": "km/h", "time": 5},
                            "solve_for": "distance",
                        },
                    },
                ],
                "confidence": 0.9,
            }

        agent._build_fast_mcp_plan_with_llm = fake_llm_plan

        response = await agent.solve_problem(
            problem="Convert 72 km/h to m/s, then find how far an object travels in 5 seconds.",
            context=None,
            user_id="student-a",
        )

        self.assertTrue(response["success"])
        self.assertEqual(response["tools_used"], ["unit_converter", "physics_formula_solver"])
        self.assertEqual(calls[0]["name"], "unit_converter")
        self.assertEqual(calls[0]["arguments"], {"value": 72.0, "from_unit": "km/h", "to_unit": "m/s"})
        self.assertEqual(calls[1]["name"], "physics_formula_solver")
        self.assertEqual(
            calls[1]["arguments"]["known_values"],
            '{"speed": 72, "speed_unit": "km/h", "time": 5}',
        )
        self.assertIn("Answer: d = 100.000 m", response["solution"])
        self.assertTrue(response["metadata"]["fast_mcp_pipeline"]["llm_plan_used"])

    async def test_kinematics_remediation_wrong_equation_gets_specific_feedback(self):
        async def fail_get_or_create_agent(*args: Any, **kwargs: Any):
            raise AssertionError("solver should not be created for HITL remediation follow-ups")

        self.api_main.knowledge_transfer_gate = None
        self.api_main.get_or_create_agent = fail_get_or_create_agent

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find final velocity.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-equation-wrong",
                        "concept_tag": "equation_selection",
                        "student_message": "I would use \\Delta x = v_0 t + \\frac{1}{2} a t^2",
                        "step_index": 0,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertEqual(response.hitl["submission_check"]["status"], "incorrect")
        self.assertEqual(response.hitl["step_index"], 0)
        self.assertIn("displacement equation", response.solution)
        self.assertIn("Use v_f = v_0 + at", response.solution)
        self.assertNotIn("Check your work against this", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_projectile_remediation_correct_component_equations_are_checked(self):
        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="Ball thrown at 30 m/s at 45° from 10m height",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-projectile-components",
                        "concept_tag": "projectile_components",
                        "student_message": "v_{0x}=v_0 \\cos\\theta and v_{0y}=v_0 \\sin\\theta",
                        "step_index": 0,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["submission_check"]["status"], "correct")
        self.assertEqual(response.hitl["step_index"], 1)
        self.assertIn("component setup is right", response.solution)
        self.assertIn("Substitute the launch speed and angle", response.solution)

    async def test_forces_remediation_followup_uses_text_submission_wording(self):
        response = await self.api_main.solve_problem(
            "forces_agent",
            self.api_main.ProblemSolveRequest(
                problem="A block slides on a 30 degree incline. Find the acceleration.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-3",
                        "concept_tag": "incline_components",
                        "student_message": "next step",
                        "step_index": 0,
                        "mode": "next_step",
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertEqual(response.hitl["step_index"], 0)
        self.assertIn("tell me which direction you chose as positive", response.solution)
        self.assertNotIn("Draw axes", response.solution)

    async def test_forces_remediation_sketch_submission_uses_typed_checklist(self):
        drawing = {
            "width": 720,
            "height": 420,
            "strokes": [
                {"tool": "pen", "color": "#111827", "points": [{"x": 100, "y": 210}, {"x": 620, "y": 210}]},
                {"tool": "pen", "color": "#111827", "points": [{"x": 360, "y": 360}, {"x": 360, "y": 60}]},
                {"tool": "pen", "color": "#1565C0", "points": [{"x": 220, "y": 320}, {"x": 520, "y": 150}]},
            ],
        }

        response = await self.api_main.solve_problem(
            "forces_agent",
            self.api_main.ProblemSolveRequest(
                problem="A block slides on a 30 degree incline. Find the acceleration.",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-4",
                        "concept_tag": "incline_components",
                        "student_message": "+x is up the ramp, +y is perpendicular outward. Forces: weight and normal.",
                        "has_drawing": True,
                        "drawing": drawing,
                        "step_index": 1,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertTrue(response.hitl["has_drawing"])
        self.assertIn("I checked your work", response.solution)
        self.assertIn("setup is partly there", response.solution)
        self.assertTrue(response.hitl["sketch_analysis"]["axis_like"])
        self.assertEqual(response.hitl["submission_check"]["status"], "partial")
        self.assertGreaterEqual(response.hitl["sketch_analysis"]["diagonal_count"], 1)
        self.assertIn("Sketch analysis:", response.solution)
        self.assertIn("Add the missing axis/component information", response.solution)
        self.assertIsNone(response.tools_used)

    async def test_kinematics_projectile_sketch_analysis_detects_trajectory(self):
        drawing = {
            "width": 720,
            "height": 420,
            "strokes": [
                {"tool": "pen", "color": "#111827", "points": [{"x": 80, "y": 340}, {"x": 640, "y": 340}]},
                {"tool": "pen", "color": "#111827", "points": [{"x": 80, "y": 340}, {"x": 80, "y": 60}]},
                {
                    "tool": "pen",
                    "color": "#1565C0",
                    "points": [
                        {"x": 90, "y": 330},
                        {"x": 190, "y": 120},
                        {"x": 360, "y": 90},
                        {"x": 620, "y": 320},
                    ],
                },
            ],
        }

        response = await self.api_main.solve_problem(
            "kinematics_agent",
            self.api_main.ProblemSolveRequest(
                problem="Ball thrown at 30 m/s at 45° from 10m height",
                user_id="student-a",
                context={
                    "knowledge_transfer_remediation_followup": {
                        "check_id": "check-5",
                        "concept_tag": "projectile_components",
                        "student_message": "I drew x and y axes and the trajectory.",
                        "has_drawing": True,
                        "drawing": drawing,
                        "step_index": 1,
                    }
                },
            ),
        )

        self.assertTrue(response.success)
        self.assertEqual(response.hitl["status"], "remediation_followup")
        self.assertTrue(response.hitl["has_drawing"])
        self.assertTrue(response.hitl["sketch_analysis"]["axis_like"])
        self.assertGreaterEqual(response.hitl["sketch_analysis"]["curved_count"], 1)
        self.assertIn("curved path", response.solution)
        self.assertIsNone(response.tools_used)


if __name__ == "__main__":
    unittest.main()
