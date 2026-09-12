import unittest

from strands_agents.kinematics_agent import KinematicsAgent


class KinematicsFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_projectile_prompt_without_draw_keyword_uses_fastpath(self):
        agent = KinematicsAgent(enable_database_logging=False, enable_rag=False)

        async def fake_initialize() -> None:
            agent.initialized = True

        async def forbidden_invoke(*args, **kwargs):
            raise AssertionError("Projectile fast path should not invoke the model")

        agent.initialize = fake_initialize
        agent._invoke_agent_with_custom_timeout = forbidden_invoke

        result = await agent.solve_problem(
            problem="Ball thrown at 30 m/s at 45\u00b0 from 10m height. Find the range and maximum height.",
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["projectile_motion_2d"])
        self.assertEqual(result["diagram"]["type"], "projectile_trajectory")
        self.assertEqual(result["diagram"]["launch"]["v0_mps"], 30.0)
        self.assertEqual(result["diagram"]["launch"]["angle_deg"], 45.0)
        self.assertEqual(result["diagram"]["launch"]["h0_m"], 10.0)
        self.assertGreater(result["diagram"]["max_height_m"], 30.0)
        self.assertTrue(result["metadata"]["fastpath_recovery"])

    async def test_initial_velocity_wording_uses_projectile_fastpath(self):
        agent = KinematicsAgent(enable_database_logging=False, enable_rag=False)

        async def fake_initialize() -> None:
            agent.initialized = True

        async def forbidden_invoke(*args, **kwargs):
            raise AssertionError("Projectile fast path should not invoke the model")

        agent.initialize = fake_initialize
        agent._invoke_agent_with_custom_timeout = forbidden_invoke

        result = await agent.solve_problem(
            problem="Projectile motion with initial velocity 30 m/s with launch angle 45 degrees.",
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["projectile_motion_2d"])
        self.assertIn("Initial components", result["solution"])


if __name__ == "__main__":
    unittest.main()
