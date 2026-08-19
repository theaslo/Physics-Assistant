import importlib
import os
import unittest
from pathlib import Path

os.environ["MCP_DISABLE_DATABASE_LOGGING"] = "1"

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PACKAGE_DIR = ROOT / "physics_mcp_tools"

SERVER_COMMANDS = {
    "forces-server": "physics_mcp_tools.forces_mcp_server",
    "kinematics-server": "physics_mcp_tools.kinematics_mcp_server",
    "circuit-server": "physics_mcp_tools.circuit_mcp_server",
    "math-server": "physics_mcp_tools.math_mcp_server",
    "momentum-server": "physics_mcp_tools.momentum_mcp_server",
    "energy-server": "physics_mcp_tools.energy_mcp_server",
    "angular-motion-server": "physics_mcp_tools.angular_motion_mcp_server",
    "thermodynamics-server": "physics_mcp_tools.thermodynamics_mcp_server",
    "waves-server": "physics_mcp_tools.waves_mcp_server",
    "electromagnetism-server": "physics_mcp_tools.electromagnetism_mcp_server",
    "optics-server": "physics_mcp_tools.optics_mcp_server",
    "modern-physics-server": "physics_mcp_tools.modern_physics_mcp_server",
}


class McpPackageIntegrationTests(unittest.TestCase):
    def test_all_server_modules_import_and_expose_serve(self):
        for command, module_name in SERVER_COMMANDS.items():
            with self.subTest(command=command):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, "serve", None)))

    def test_all_servers_use_shared_runtime(self):
        for server_file in PACKAGE_DIR.glob("*_mcp_server.py"):
            text = server_file.read_text()
            with self.subTest(server=server_file.name):
                self.assertIn("run_fastmcp_server(mcp, host, port, transport)", text)
                self.assertNotIn("mcp.sse_http_app", text)
                self.assertNotIn("uvicorn.run(mcp.streamable_http_app", text)

    def test_cli_start_script_and_readme_cover_all_server_commands(self):
        cli_text = (PACKAGE_DIR / "__init__.py").read_text()
        start_script = (REPO_ROOT / "docker" / "mcp" / "start.sh").read_text()
        readme = (ROOT / "README.md").read_text()

        for command in SERVER_COMMANDS:
            with self.subTest(command=command):
                self.assertIn(command, cli_text)
                self.assertIn(f"--run {command}", start_script)
                self.assertIn(command, readme)

    def test_docker_compose_has_services_for_server_commands(self):
        compose = (REPO_ROOT / "docker-compose.yaml").read_text()
        service_names = {
            command.removesuffix("-server").replace("-motion", "-motion")
            for command in SERVER_COMMANDS
        }

        for service in service_names:
            with self.subTest(service=service):
                self.assertIn(f"mcp-{service}:", compose)


if __name__ == "__main__":
    unittest.main()
