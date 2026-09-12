import unittest

from strands_agents.forces_agent import ForcesAgent


class ForcesFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_spring_force_fastpath_solves_hookes_law_without_model_call(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)

        async def fake_initialize() -> None:
            agent.initialized = True

        async def forbidden_invoke(*args, **kwargs):
            raise AssertionError("Spring fast path should not invoke the model")

        agent.initialize = fake_initialize
        agent._invoke_agent_with_custom_timeout = forbidden_invoke

        result = await agent.solve_problem(
            problem="A spring stretches 0.2 m with k=50 N/m. Use Hooke law to find the force.",
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["calculate_spring_force_tool"])
        self.assertIn("10 N", result["solution"])
        self.assertIn("F_s = -kx", result["solution"])
        self.assertEqual(result["metadata"]["calculation"]["force_magnitude_n"], 10.0)
        self.assertTrue(result["metadata"]["fastpath_recovery"])


if __name__ == "__main__":
    unittest.main()
