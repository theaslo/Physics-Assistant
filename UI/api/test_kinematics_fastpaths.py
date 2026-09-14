import unittest

from strands_agents.kinematics_agent import KinematicsAgent


class KinematicsFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_projectile_prompt_without_draw_keyword_uses_llm_guided_mcp_path(self):
        agent = KinematicsAgent(enable_database_logging=False, enable_rag=False)
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
                                "2D Projectile Motion Analysis:\n"
                                "Initial components: vx = 21.21 m/s, vy = 21.21 m/s\n"
                                "Maximum height: 32.94 m\n"
                                "Total range: 106.98 m\n\n"
                                "DIAGRAM_JSON_START\n"
                                '{"type":"projectile_trajectory","launch":{"v0_mps":30.0,'
                                '"angle_deg":45.0,"h0_m":10.0},"max_height_m":32.94,'
                                '"trajectory_points":[{"t_s":0,"x_m":0,"y_m":10}]}\n'
                                "DIAGRAM_JSON_END"
                            )
                        }
                    ],
                }

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def fake_llm_plan(problem, catalog, rag_context):
            self.assertIn("projectile_motion_2d", [tool["name"] for tool in catalog])
            return {
                "can_solve_with_mcp": True,
                "tool_calls": [
                    {
                        "tool_name": "projectile_motion_2d",
                        "arguments": {"launch_conditions": {"v0": 30, "angle": 45, "h0": 10}},
                    }
                ],
                "confidence": 0.95,
            }

        async def forbidden_full_agent(*args, **kwargs):
            raise AssertionError("General fast MCP path should not invoke the full agent")

        agent.initialize = fake_initialize
        agent._build_fast_mcp_plan_with_llm = fake_llm_plan
        agent._call_agent_async = forbidden_full_agent

        result = await agent.solve_problem(
            problem="Ball thrown at 30 m/s at 45\u00b0 from 10m height. Find the range and maximum height.",
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["projectile_motion_2d"])
        self.assertEqual(calls["name"], "projectile_motion_2d")
        self.assertEqual(calls["arguments"]["launch_conditions"], '{"v0": 30, "angle": 45, "h0": 10}')
        self.assertEqual(result["diagram"]["type"], "projectile_trajectory")
        self.assertEqual(result["diagram"]["launch"]["v0_mps"], 30.0)
        self.assertEqual(result["diagram"]["launch"]["angle_deg"], 45.0)
        self.assertEqual(result["diagram"]["launch"]["h0_m"], 10.0)
        self.assertGreater(result["diagram"]["max_height_m"], 30.0)
        self.assertTrue(result["metadata"]["fast_mcp_pipeline"]["llm_plan_used"])

    async def test_initial_velocity_wording_uses_llm_guided_mcp_path(self):
        agent = KinematicsAgent(enable_database_logging=False, enable_rag=False)

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id, name, arguments):
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                "2D Projectile Motion Analysis:\n"
                                "Initial components: vx = 21.21 m/s, vy = 21.21 m/s\n"
                            )
                        }
                    ],
                }

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def fake_llm_plan(problem, catalog, rag_context):
            return {
                "can_solve_with_mcp": True,
                "tool_calls": [
                    {
                        "tool_name": "projectile_motion_2d",
                        "arguments": {"launch_conditions": {"v0": 30, "angle": 45}},
                    }
                ],
                "confidence": 0.95,
            }

        agent.initialize = fake_initialize
        agent._build_fast_mcp_plan_with_llm = fake_llm_plan

        result = await agent.solve_problem(
            problem="Projectile motion with initial velocity 30 m/s with launch angle 45 degrees.",
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["projectile_motion_2d"])
        self.assertIn("Initial components", result["solution"])


if __name__ == "__main__":
    unittest.main()
