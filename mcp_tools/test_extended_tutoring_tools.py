"""Exercise registered domain tools with real numerical and sign conventions."""

import json
import unittest
from unittest.mock import patch

from mcp.server.fastmcp import FastMCP
from physics_mcp_tools import energy_mcp_server, angular_motion_mcp_server, thermodynamics_mcp_server


class ExtendedTutoringToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.servers = {}
        for name, module in (("energy", energy_mcp_server), ("angular", angular_motion_mcp_server),
                             ("thermo", thermodynamics_mcp_server)):
            server = FastMCP("test-" + name)
            with patch.object(module, "FastMCP", return_value=server):
                module.serve("localhost", 0, "test")
            self.servers[name] = server

    async def call(self, domain, tool, **arguments):
        content = await self.servers[domain].call_tool(tool, arguments)
        blocks = content[0] if isinstance(content, tuple) else content
        return "\n".join(block.text for block in blocks)

    async def data(self, domain, tool, field, values):
        return await self.call(domain, tool, **{field: json.dumps(values)})

    async def test_each_domain_registers_its_own_conceptual_reference(self):
        for domain, expected in (("energy", "get_energy_principles"),
                                 ("angular", "get_angular_motion_principles"),
                                 ("thermo", "get_thermodynamics_principles")):
            names = {tool.name for tool in await self.servers[domain].list_tools()}
            self.assertIn(expected, names)

    async def test_kinetic_energy_is_nonnegative_for_signed_velocity(self):
        result = await self.call("energy", "calculate_kinetic_energy_tool", mass=2, velocity=-3)
        self.assertIn("KE = 9.000 J", result)
        zero = await self.call("energy", "calculate_kinetic_energy_tool", mass=2, velocity=0)
        self.assertIn("KE = 0.000 J", zero)
        invalid = await self.call("energy", "calculate_kinetic_energy_tool", mass=-2, velocity=3)
        self.assertTrue(invalid.startswith("Error"), invalid)

    async def test_potential_energy_can_be_negative_relative_to_the_chosen_zero(self):
        result = await self.call("energy", "calculate_gravitational_potential_energy_tool", mass=2, height=-2)
        self.assertIn("PE = -39.240 J", result)
        self.assertNotIn("Impact velocity", result)

    async def test_work_sign_power_and_variable_force_contract(self):
        work = await self.call("energy", "calculate_work_tool", force=4, displacement=3, angle_degrees=180)
        self.assertIn("W = -12.000 J", work)
        variable = await self.call("energy", "calculate_work_tool", force=4, displacement=3, force_type="variable")
        self.assertTrue(variable.startswith("Error"), variable)
        power = await self.call("energy", "calculate_power_tool", work=600, time=3)
        self.assertIn("= 200 W", power)
        invalid = await self.call("energy", "calculate_power_tool", work=600, time=0)
        self.assertTrue(invalid.startswith("Error"), invalid)

    async def test_work_energy_requires_initial_state_and_rejects_impossible_motion(self):
        for values in ({"work_done": 20}, {"mass": 2, "work_done": 20},
                       {"mass": 2, "initial_velocity": 3, "work_done": -20}):
            result = await self.data("energy", "work_energy_theorem", "problem_data", values)
            self.assertTrue(result.startswith("Error"), result)
        result = await self.data("energy", "work_energy_theorem", "problem_data",
                                 {"mass": 2, "initial_velocity": 0, "work_done": 9})
        self.assertIn("= 3.000 m/s", result)

    async def test_zero_friction_and_unreachable_final_height(self):
        result = await self.data("energy", "energy_with_friction", "friction_data",
                                 {"mass": 2, "initial_velocity": 3, "friction_coefficient": 0, "distance": 4})
        self.assertNotIn("Error", result)
        self.assertIn("= 3.000 m/s", result)
        unreachable = await self.data("energy", "energy_conservation", "system_data",
                                      {"mass": 2, "initial_velocity": 0, "initial_height": 0, "final_height": 5})
        self.assertTrue(unreachable.startswith("Error"), unreachable)

    async def test_signed_angular_kinematics_and_zero_elapsed_time(self):
        result = await self.data("angular", "angular_kinematics", "kinematics_data",
                                 {"omega_0": -2, "alpha": -3, "time": 4})
        self.assertIn("= -14 rad/s", result)
        self.assertIn("= -32 rad", result)
        zero = await self.data("angular", "angular_kinematics", "kinematics_data",
                               {"omega_0": 2, "alpha": 0, "time": 0})
        payload = json.loads(zero.split("DIAGRAM_JSON_START\n")[1].split("\nDIAGRAM_JSON_END")[0])
        self.assertEqual(set(payload["times_s"]), {0})
        self.assertEqual(set(payload["omega_series_rad_s"]), {2})

    async def test_angular_kinematics_does_not_invent_rest_or_choose_ambiguous_future(self):
        for values in ({"alpha": 3, "time": 4}, {"omega_0": 2},
                       {"omega_0": 2, "alpha": -1, "theta": 1.5}):
            result = await self.data("angular", "angular_kinematics", "kinematics_data", values)
            self.assertTrue(result.startswith("Error"), result)
        resolved = await self.data("angular", "angular_kinematics", "kinematics_data",
                                   {"omega_0": 2, "alpha": -1, "theta": 1.5, "final_direction": "negative"})
        self.assertIn("= -1 rad/s", resolved)
        self.assertIn("t = 3 s", resolved)

    async def test_angular_acceleration_can_be_solved_from_endpoint_data(self):
        result = await self.data("angular", "angular_kinematics", "kinematics_data",
                                 {"omega_0": 0, "omega_f": 10, "theta": 25})
        self.assertIn("alpha = 2 rad/s^2", result)
        self.assertIn("t = 5 s", result)
        initial = await self.data("angular", "angular_kinematics", "kinematics_data",
                                  {"omega_f": 10, "alpha": 2, "time": 3})
        self.assertIn("omega_0 = 4 rad/s", initial)

    async def test_falling_speed_does_not_need_an_invented_mass(self):
        result = await self.data("energy", "energy_conservation", "system_data",
                                 {"initial_height": 5, "final_height": 0, "initial_velocity": 0})
        self.assertIn("Mass cancels", result)
        self.assertIn("v_f = 9.904", result)
        self.assertNotIn("1 kg", result)

    async def test_torque_needs_an_angle_and_respects_clockwise_direction(self):
        missing = await self.data("angular", "calculate_torque", "torque_data", {"force": 50, "radius": 0.3})
        self.assertTrue(missing.startswith("Error"), missing)
        clockwise = await self.data("angular", "calculate_torque", "torque_data",
                                    {"force": 50, "radius": 0.3, "angle": 90, "direction": "clockwise"})
        self.assertIn("-15.000", clockwise)

    async def test_changing_inertia_conserves_angular_momentum_but_not_kinetic_energy(self):
        values = {"initial": {"I": 4, "omega": 3}, "final": {"I": 2}}
        result = await self.data("angular", "angular_momentum_conservation", "momentum_data", values)
        self.assertIn("= 6.000 rad/s", result)
        self.assertIn("ΔKE = KE₂ - KE₁ = 18.000 J", result)
        values["final"]["omega"] = 1
        invalid = await self.data("angular", "angular_momentum_conservation", "momentum_data", values)
        self.assertTrue(invalid.startswith("Error"), invalid)

    async def test_heat_transfer_handles_cooling_and_zero_without_truthiness_defaults(self):
        for delta, expected in ((-10, "Q = -41860.00 J"), (0, "Q = 0.00 J")):
            result = await self.data("thermo", "heat_transfer", "heat_data",
                                     {"mass": 1, "specific_heat": 4186, "temp_change": delta})
            self.assertIn(expected, result)
        result = await self.data("thermo", "heat_transfer", "heat_data",
                                 {"mass": 1, "specific_heat": 4186, "heat": 0})
        self.assertIn("ΔT = 0.00 K", result)
        unknown = await self.data("thermo", "heat_transfer", "heat_data",
                                  {"specific_heat": 4186, "heat": 0, "temp_change": 0})
        self.assertTrue(unknown.startswith("Error"), unknown)

    async def test_first_law_uses_work_by_sign_and_accepts_zero_or_work_on(self):
        for values, expected in (({"heat": 500, "work_by_system": 200}, "= 300 J"),
                                 ({"heat": 0, "work_on_system": 200}, "= 200 J"),
                                 ({"heat": 0, "work_by_system": 0}, "= 0 J")):
            result = await self.data("thermo", "first_law_thermodynamics", "energy_data", values)
            self.assertIn(expected, result)
        invalid = await self.data("thermo", "first_law_thermodynamics", "energy_data",
                                  {"heat": 500, "work_by_system": 200, "internal_energy_change": 700})
        self.assertTrue(invalid.startswith("Error"), invalid)

    async def test_constant_pressure_compression_work_is_negative(self):
        result = await self.data("thermo", "constant_pressure_work", "process_data",
                                 {"pressure": 100000, "initial_volume": 0.5, "final_volume": 0.3})
        self.assertIn("= -20000 J", result)

    async def test_carnot_requires_actual_kelvin_reservoir_temperatures(self):
        good = await self.data("thermo", "carnot_efficiency", "carnot_data", {"T_hot": 600, "T_cold": 300})
        self.assertIn("50.00%", good)
        for values in ({}, {"T_hot": 600}, {"T_hot": 300, "T_cold": 600},
                       {"T_hot": 300, "T_cold": 300}, {"T_hot": 600, "T_cold": -10}):
            result = await self.data("thermo", "carnot_efficiency", "carnot_data", values)
            self.assertTrue(result.startswith("Error"), result)

    async def test_missing_material_properties_and_nonfinite_states_are_rejected(self):
        cases = (("thermal_expansion", "expansion_data", {"length": 2, "temp_change": 50}),
                 ("heat_conduction", "conduction_data", {"area": 1}),
                 ("ideal_gas_law", "gas_data", {"pressure": 100000, "volume": 0.01, "temperature": -1}),
                 ("ideal_gas_law", "gas_data", {"pressure": float("nan"), "volume": 0.01, "temperature": 300}),
                 ("heat_transfer", "heat_data", {"mass": True, "specific_heat": 4186, "temp_change": 5}))
        for tool, field, values in cases:
            with self.subTest(tool=tool, values=values):
                result = await self.data("thermo", tool, field, values)
                self.assertTrue(result.startswith("Error"), result)


if __name__ == "__main__":
    unittest.main()
