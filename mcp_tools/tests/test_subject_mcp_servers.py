import asyncio
import os
import unittest

os.environ["MCP_DISABLE_DATABASE_LOGGING"] = "1"

from physics_mcp_tools.circuit_mcp_server import create_mcp as create_circuit_mcp
from physics_mcp_tools.electromagnetism_mcp_server import (
    create_mcp as create_electromagnetism_mcp,
)
from physics_mcp_tools.optics_mcp_server import create_mcp as create_optics_mcp
from physics_mcp_tools.waves_mcp_server import create_mcp as create_waves_mcp


def call_tool_text(mcp, name, arguments):
    result = asyncio.run(mcp.call_tool(name, arguments))
    content, metadata = result
    return metadata.get("result") or "\n".join(
        block.text for block in content if hasattr(block, "text")
    )


class SubjectMcpServerTests(unittest.TestCase):
    def test_circuit_mcp_registers_tools_and_solves_ohms_law(self):
        mcp = create_circuit_mcp()

        tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}

        self.assertTrue(
            {
                "ohms_law",
                "resistor_network",
                "voltage_divider",
                "rc_circuit",
            }.issubset(tool_names)
        )

        result = call_tool_text(
            mcp,
            "ohms_law",
            {"circuit_data": '{"voltage": 12, "resistance": 100}'},
        )

        self.assertIn("I = 0.120000 A", result)
        self.assertIn("P = VI", result)

    def test_circuit_mcp_solves_nested_resistor_network(self):
        mcp = create_circuit_mcp()

        result = call_tool_text(
            mcp,
            "resistor_network",
            {"network_data": '{"series": [100, {"parallel": [200, 300]}]}'},
        )

        self.assertIn("R_eq = 220.0000 ohms", result)

    def test_waves_mcp_registers_tools_and_solves_wave_equation(self):
        mcp = create_waves_mcp()

        tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}

        self.assertTrue(
            {
                "wave_equation",
                "doppler_effect",
                "sound_intensity_decibels",
                "standing_waves",
                "wave_interference",
            }.issubset(tool_names)
        )

        result = call_tool_text(
            mcp,
            "wave_equation",
            {"wave_data": '{"frequency": 440, "wavelength": 0.78}'},
        )

        self.assertIn("v = 343.20 m/s", result)

    def test_waves_mcp_solves_standing_waves(self):
        mcp = create_waves_mcp()

        result = call_tool_text(
            mcp,
            "standing_waves",
            {"standing_wave_data": '{"type": "pipe_closed", "length": 0.5}'},
        )

        self.assertIn("Fundamental Frequency", result)
        self.assertIn("171.50 Hz", result)

    def test_electromagnetism_mcp_solves_multicharge_field(self):
        mcp = create_electromagnetism_mcp()

        tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}

        self.assertTrue(
            {
                "coulombs_law",
                "electric_field",
                "electric_potential",
                "capacitance",
                "ohms_law",
                "resistor_network",
                "magnetic_force",
                "magnetic_field_wire",
                "faradays_law",
            }.issubset(tool_names)
        )

        result = call_tool_text(
            mcp,
            "electric_field",
            {
                "field_data": (
                    '{"charges": [{"q": 1e-6, "x": 0, "y": 0}, '
                    '{"q": -1e-6, "x": 0.1, "y": 0}], '
                    '"point": {"x": 0.05, "y": 0.05}}'
                )
            },
        )

        self.assertIn("Multiple Point Charges", result)
        self.assertIn("|E|", result)

    def test_electromagnetism_mcp_reports_missing_faraday_inputs(self):
        mcp = create_electromagnetism_mcp()

        result = call_tool_text(
            mcp, "faradays_law", {"faraday_data": '{"time": 0.1}'}
        )

        self.assertIn("provide either flux_change or both B_change and area", result)

    def test_optics_mcp_registers_tools_and_handles_thin_film_without_substrate(self):
        mcp = create_optics_mcp()

        tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}

        self.assertTrue(
            {
                "snells_law",
                "lens_mirror_equation",
                "diffraction_grating",
                "thin_film_interference",
                "optical_power_diopters",
            }.issubset(tool_names)
        )

        result = call_tool_text(
            mcp,
            "thin_film_interference",
            {
                "film_data": (
                    '{"thickness": 200e-9, "n_film": 1.33, '
                    '"wavelength": 550e-9}'
                )
            },
        )

        self.assertIn("Thin Film Interference Analysis", result)
        self.assertNotIn("Error", result)

    def test_optics_mcp_solves_snells_law(self):
        mcp = create_optics_mcp()

        result = call_tool_text(
            mcp,
            "snells_law",
            {"refraction_data": '{"n1": 1.0, "n2": 1.5, "angle1": 30}'},
        )

        self.assertIn("Snell's Law Analysis", result)
        self.assertIn("19.47", result)


if __name__ == "__main__":
    unittest.main()
