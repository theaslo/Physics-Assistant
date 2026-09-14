import unittest

from strands_agents.forces_agent import ForcesAgent


class ForcesFastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_spring_force_uses_llm_guided_mcp_path(self):
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

        async def fake_llm_plan(problem, catalog, rag_context):
            self.assertIn("calculate_spring_force_tool", [tool["name"] for tool in catalog])
            return {
                "can_solve_with_mcp": True,
                "tool_calls": [
                    {
                        "tool_name": "calculate_spring_force_tool",
                        "arguments": {
                            "spring_data": {
                                "spring_constant": 50,
                                "displacement": 0.2,
                            }
                        },
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
            problem="A spring stretches 0.2 m with k=50 N/m. Use Hooke law to find the force.",
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["tools_used"], ["calculate_spring_force_tool"])
        self.assertEqual(calls["name"], "calculate_spring_force_tool")
        self.assertEqual(calls["arguments"]["spring_data"], '{"spring_constant": 50, "displacement": 0.2}')
        self.assertIn("10.00 N", result["solution"])
        self.assertIn("Hooke", result["solution"])
        self.assertTrue(result["metadata"]["fast_mcp_pipeline"]["llm_plan_used"])


if __name__ == "__main__":
    unittest.main()
