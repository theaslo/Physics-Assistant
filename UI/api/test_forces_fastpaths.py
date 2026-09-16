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

    async def test_incline_fast_path_skips_failed_fbd_and_reports_acceleration(self):
        agent = ForcesAgent(enable_database_logging=False, enable_rag=False)
        calls = []

        class FakeMcpClient:
            def call_tool_sync(self, tool_use_id, name, arguments):
                calls.append((name, arguments))
                if name == "create_free_body_diagram":
                    return {
                        "status": "success",
                        "content": [{"text": 'Error: Please provide valid JSON format like [{"name": "Weight"}]'}],
                    }
                if name == "analyze_forces_on_incline":
                    return {
                        "status": "success",
                        "content": [
                            {
                                "text": (
                                    "Forces on Inclined Plane Analysis:\n"
                                    "Net force down incline = 11.78 N\n"
                                    "DIAGRAM_JSON_START\n"
                                    "{\"type\":\"inclined_plane_diagram\",\"title\":\"Inclined Plane Forces\","
                                    "\"mass_kg\":5.0,\"angle_deg\":30.0,\"coefficient_friction\":0.3,"
                                    "\"weight_n\":49.05,\"weight_parallel_n\":24.525,"
                                    "\"weight_perpendicular_n\":42.48,\"normal_n\":42.48,"
                                    "\"net_down_n\":11.78,\"has_friction\":true,\"friction_n\":12.74}"
                                    "\nDIAGRAM_JSON_END"
                                )
                            }
                        ],
                    }
                raise AssertionError(f"Unexpected tool call: {name}")

        async def fake_initialize() -> None:
            agent.initialized = True
            agent.mcp_client = FakeMcpClient()

        async def fake_llm_plan(problem, catalog, rag_context):
            return {
                "can_solve_with_mcp": True,
                "tool_calls": [
                    {
                        "tool_name": "create_free_body_diagram",
                        "arguments": {"object_name": "box", "forces_data": [{"name": "Weight"}]},
                    },
                    {
                        "tool_name": "analyze_forces_on_incline",
                        "arguments": {
                            "mass": 5,
                            "angle_degrees": 30,
                            "coefficient_friction": 0.3,
                            "gravity": 9.81,
                        },
                    },
                ],
                "confidence": 0.95,
            }

        async def forbidden_full_agent(*args, **kwargs):
            raise AssertionError("Fast MCP path should not invoke the full agent when a later MCP tool succeeds")

        agent.initialize = fake_initialize
        agent._build_fast_mcp_plan_with_llm = fake_llm_plan
        agent._call_agent_async = forbidden_full_agent

        result = await agent.solve_problem(
            problem=(
                "A 5.0 kg box slides down a 30 degree incline with coefficient of kinetic friction 0.30. "
                "Draw the free-body diagram and find the acceleration down the ramp."
            ),
            user_id="student-a",
        )

        self.assertTrue(result["success"])
        self.assertEqual([call[0] for call in calls], ["create_free_body_diagram", "analyze_forces_on_incline"])
        self.assertEqual(result["tools_used"], ["analyze_forces_on_incline"])
        self.assertNotIn("Error:", result["solution"])
        self.assertIn("Acceleration magnitude: 2.36 m/s", result["solution"])
        self.assertEqual(result["diagram"]["type"], "inclined_plane_diagram")


if __name__ == "__main__":
    unittest.main()
