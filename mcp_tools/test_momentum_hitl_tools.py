"""Verify the real momentum calculations used to grade student work."""

import json
import unittest
from unittest.mock import patch

from mcp.server.fastmcp import FastMCP
from physics_mcp_tools import momentum_mcp_server


class MomentumHitlToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = FastMCP("momentum-test")
        with patch.object(momentum_mcp_server, "FastMCP", return_value=self.server):
            # Register the real tools without starting a network listener.
            momentum_mcp_server.serve("localhost", 10104, "test")

    async def call(self, name, arguments):
        content = await self.server.call_tool(name, arguments)
        blocks = content[0] if isinstance(content, tuple) else content
        return "\n".join(block.text for block in blocks)

    async def test_rebound_impulse_preserves_signed_initial_and_final_velocities(self):
        result = await self.call("momentum_impulse_theorem", {"problem_data": json.dumps({
            "mass": 0.2, "initial_velocity": 5, "final_velocity": -3})})
        self.assertIn("-1.600 N⋅s", result)
        result = await self.call("calculate_impulse_1d", {"initial_momentum": 1, "final_momentum": -0.6})
        self.assertIn("-1.600 N⋅s", result)
        self.assertNotIn("DIAGRAM_JSON", result)  # No force/time values were supplied.

    async def test_incomplete_inputs_do_not_return_fictitious_zero_impulse(self):
        for name, arguments in (
            ("calculate_impulse_1d", {}),
            ("calculate_impulse_1d", {"force": 2}),
            ("calculate_impulse_1d", {"force": 2, "time": -1}),
            ("momentum_impulse_theorem", {"problem_data": '{"mass": 2, "initial_velocity": 3}'}),
        ):
            with self.subTest(name=name, arguments=arguments):
                result = await self.call(name, arguments)
                self.assertTrue(result.startswith("Error"), result)
                self.assertNotIn("DIAGRAM_JSON", result)

    async def test_force_time_and_initial_momentum_inputs_still_solve(self):
        result = await self.call("calculate_impulse_1d", {"force": -4, "time": 0.5})
        self.assertIn("-2.000 N⋅s", result)
        self.assertIn("DIAGRAM_JSON_START", result)
        result = await self.call("momentum_impulse_theorem", {"problem_data": json.dumps({
            "mass": 1.5, "initial_momentum": 6, "impulse": 4})})
        self.assertIn("6.667 m/s", result)

    async def test_sticking_collision_payload_has_two_real_final_velocities_and_conserves_momentum(self):
        for v1i, v2i, expected in ((3, 0, 2), (0, 0, 0), (-3, 0, -2), (3, -6, 0)):
            with self.subTest(v1i=v1i, v2i=v2i):
                result = await self.call("momentum_conservation_1d", {"collision_data": json.dumps({
                    "m1": 2, "m2": 1, "v1i": v1i, "v2i": v2i, "collision_type": "perfectly_inelastic"})})
                self.assertFalse(result.startswith("Error"), result)
                payload = json.loads(result.split("DIAGRAM_JSON_START\n")[1].split("\nDIAGRAM_JSON_END")[0])
                self.assertEqual([obj["velocity_mps"] for obj in payload["after"]], [expected, expected])
                self.assertAlmostEqual(payload["totals"]["difference_kg_mps"], 0)

    async def test_general_collision_without_final_condition_is_not_a_completed_calculation(self):
        result = await self.call("momentum_conservation_1d", {"collision_data": json.dumps({
            "m1": 2, "m2": 1, "v1i": 3, "v2i": 0, "collision_type": "inelastic"})})
        self.assertTrue(result.startswith("Error"), result)
        self.assertNotIn("DIAGRAM_JSON", result)


if __name__ == "__main__":
    unittest.main()
