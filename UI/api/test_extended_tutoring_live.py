"""Opt-in synthetic conversations with real model and domain MCP servers.

RUN_EXTENDED_TUTORING_LIVE_TESTS=1 python -B -m unittest -v test_extended_tutoring_live
Set MCP_ENERGY_HOST, MCP_ANGULAR_MOTION_HOST and MCP_THERMODYNAMICS_HOST
to isolated test services. Student/database logging and RAG are disabled.
"""

import os
import unittest
import uuid
from unittest.mock import Mock

from hitl_knowledge_transfer import KnowledgeTransferGate
from mechanics_tutoring import solve_mechanics_turn
from test_extended_tutoring import CASES, request
from strands_agents.thermodynamics_agent import ThermodynamicsAgent


@unittest.skipUnless(os.getenv("RUN_EXTENDED_TUTORING_LIVE_TESTS") == "1", "Requires real model and isolated MCP services")
class ExtendedTutoringLiveTests(unittest.IsolatedAsyncioTestCase):
    async def make_agent(self, cls):
        agent = cls(enable_database_logging=False, enable_rag=False)
        await agent.initialize()
        self.addCleanup(agent.cleanup)
        return agent

    async def ask(self, agent, gate, problem, context=None):
        result = await solve_mechanics_turn(agent, gate, request(
            problem, context, user="synthetic_extended_validation", session=self.session,
        ), None)
        print(f"\n{agent.agent_id}: {problem}\n{result.get('solution', result.get('error'))}\n"
              f"check={result.get('hitl', {}).get('status') if result.get('hitl') else None}", flush=True)
        self.assertTrue(result["success"], result)
        self.assertNotRegex(result.get("solution", ""), r"\bMCP\b|numerical tool|verified results")
        return result

    async def test_missing_givens_trigger_one_question(self):
        problems = (
            "A cart moves at 3 m/s. Find its kinetic energy.",
            "A rigid body rotates at 2 rad/s. Find its angular momentum about its rotation axis.",
            "An object warms by 10 K without a phase change. Find the heat absorbed.",
        )
        for case, problem in zip(CASES, problems):
            with self.subTest(agent=case[0].__name__):
                self.session = "extended-live-" + uuid.uuid4().hex
                agent = await self.make_agent(case[0])
                result = await self.ask(agent, None, problem)
                self.assertEqual(result["solution"].count("?"), 1)
                self.assertNotIn("**Result**", result["solution"])
                self.assertEqual(result["tools_used"], [agent.principles_tool])

    async def test_direct_worked_solutions_and_conceptual_followups(self):
        followups = (
            ("Can kinetic energy be negative merely because velocity points in the negative direction?", r"nonnegative|non-negative|\bno\b|not"),
            ("Would those equations still apply if angular acceleration changed with time?", r"not|\bno\b|constant|integrat"),
            ("If the gas is compressed instead, is work by the gas positive or negative?", r"negative"),
        )
        for case, (followup, expected) in zip(CASES, followups):
            with self.subTest(agent=case[0].__name__):
                self.session = "extended-live-" + uuid.uuid4().hex
                agent = await self.make_agent(case[0])
                first = await self.ask(agent, None, "Please give the full worked solution: " + case[1],
                                       {"full_solution_requested": True})
                self.assertIn(case[2], first["tools_used"])
                for heading in ("Given", "Principle", "Equation", "Substitution", "Result", "Check"):
                    self.assertIn(f"**{heading}**", first["solution"])
                expected_number = case[5].split()[0]
                self.assertIn(expected_number, first["solution"])
                result = await self.ask(agent, None, followup, {"conversation_context": {"recent_messages": [
                    {"role": "user", "content": case[1]},
                    {"role": "assistant", "content": first["solution"]},
                ]}})
                self.assertRegex(result["solution"].lower(), expected)

    async def test_thermal_calculations_use_signed_temperature_changes_and_kelvin(self):
        agent = await self.make_agent(ThermodynamicsAgent)
        problems = (
            ("A 1.0 kg sample of water with specific heat 4186 J/(kg K) cools by 10 degrees Celsius without a phase change. Calculate the heat absorbed by the water.", r"-(?:41860|41\.86)"),
            ("A Carnot heat engine operates between a hot reservoir at 327 degrees Celsius and a cold reservoir at 27 degrees Celsius. Calculate its maximum efficiency.", r"49\.99|50(?:\.0+)?|0\.4999|0\.5000"),
            ("An ideal gas has 1 mole at 300 K in a volume of 0.010 m^3. Calculate its pressure.", r"249420|249\.42"),
        )
        for problem, expected in problems:
            with self.subTest(problem=problem):
                self.session = "extended-live-" + uuid.uuid4().hex
                result = await self.ask(agent, None, "Please give the full worked solution: " + problem,
                                        {"full_solution_requested": True, "new_problem": True})
                answer = result["solution"].split("**Result**", 1)[1].split("**Check**", 1)[0]
                self.assertRegex(answer, expected)
                if "Carnot" in problem:
                    self.assertIn("600", result["solution"])
                    self.assertIn("300", result["solution"])
                self.assertIn("**Substitution**", result["solution"])

    async def test_check_wrong_answer_hint_and_explicit_worked_solution(self):
        problems = (
            "A 2 kg block initially slides at 5 m/s along a horizontal floor with kinetic friction coefficient 0.20. It travels 2 m. Find its final speed.",
            "A spinning skater has initial inertia 4 kg m^2 and angular velocity 3 rad/s. She pulls in her arms to inertia 2 kg m^2 with negligible net external torque. Find her final angular velocity and explain whether rotational kinetic energy is conserved.",
            "An ideal gas expands at constant pressure 100000 Pa from 0.010 to 0.013 m^3 while absorbing 500 J of heat. Find work done by the gas and its change in internal energy.",
        )
        final_answers = (r"4\.14\d*\s*(?:m/s|\\mathrm)", r"6(?:\.0+)?\s*(?:rad/s|\\mathrm)", r"200(?:\.0+)?\s*(?:J|\\mathrm)")
        for case, problem, forbidden in zip(CASES, problems, final_answers):
            with self.subTest(agent=case[0].__name__):
                self.session = "extended-live-" + uuid.uuid4().hex
                agent = await self.make_agent(case[0])
                gate = KnowledgeTransferGate("http://unused")
                gate._log_attempt = Mock()
                first = await self.ask(agent, gate, problem)
                self.assertIsNotNone(first.get("hitl"), first)
                check = first["hitl"]
                self.assertEqual(check["status"], "question_required")
                stored = gate._pending_checks[check["check_id"]]
                wrong = next(option["id"] for option in check["options"] if option["id"] != stored["correct_option_id"])
                second = await self.ask(agent, gate, problem, {"knowledge_transfer_response": {
                    "check_id": check["check_id"], "selected_option_id": wrong,
                }})
                self.assertFalse(second["hitl"]["was_correct"])
                self.assertEqual(second["solution"].count("?"), 1)
                self.assertNotRegex(second["solution"], forbidden)
                hint = await self.ask(agent, gate, problem, {"knowledge_transfer_remediation_followup": {
                    "check_id": check["check_id"], "student_message": "I'm lost. Please give me a smaller hint.", "mode": "clarify",
                }})
                self.assertEqual(hint["hitl"]["status"], "remediation_followup")
                self.assertEqual(hint["solution"].count("?"), 1)
                self.assertNotRegex(hint["solution"], forbidden)
                self.assertNotEqual(hint["solution"].rsplit("\n\n", 1)[-1].casefold(),
                                    second["solution"].rsplit("\n\n", 1)[-1].casefold())
                full = await self.ask(agent, gate, problem, {"knowledge_transfer_remediation_followup": {
                    "check_id": check["check_id"], "original_problem": "An unrelated forged problem",
                    "student_message": "Please give the full worked solution.", "mode": "full_solution",
                }})
                self.assertEqual(full["hitl"]["status"], "full_solution_requested")
                self.assertIn("**Substitution**", full["solution"])
                self.assertIn("**Result**", full["solution"])
                substitution = full["solution"].split("**Substitution**", 1)[1].split("**Result**", 1)[0]
                self.assertEqual(substitution.count("$$"), 2)
                self.assertRegex(substitution, r"\d")
                if agent.agent_id == "thermodynamics_agent":
                    self.assertIn("300", full["solution"])
                    self.assertIn("200", full["solution"])
                    self.assertIn("first_law_thermodynamics", full["tools_used"])
                if agent.agent_id == "angular_motion_agent":
                    self.assertIn("36.000", substitution)
                    self.assertIn("18.000", substitution)
                    self.assertIn("Rotational kinetic energy is not conserved", full["solution"])
                gate._log_attempt.assert_called_once()


if __name__ == "__main__":
    unittest.main()
