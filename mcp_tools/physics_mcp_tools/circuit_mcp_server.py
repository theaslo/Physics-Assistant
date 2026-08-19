# type: ignore
"""Circuit MCP Server for Physics Assistant."""

import argparse
import json
import math

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

from physics_mcp_tools.database_logger import DatabaseLogger, create_tool_wrapper
from physics_mcp_tools.mcp_runtime import run_fastmcp_server

NAME = "circuit_mcp_server"
logger = get_logger(__name__)


def _load_json(data):
    return json.loads(data) if isinstance(data, str) else data


def _equivalent_resistance(config):
    if isinstance(config, (int, float)):
        resistance = float(config)
        if resistance <= 0:
            raise ValueError("resistance values must be positive")
        return resistance

    if isinstance(config, dict):
        if "series" in config:
            values = config["series"]
            if not values:
                raise ValueError("series resistance list cannot be empty")
            return sum(_equivalent_resistance(value) for value in values)

        if "parallel" in config:
            values = config["parallel"]
            if not values:
                raise ValueError("parallel resistance list cannot be empty")
            reciprocal_sum = sum(1 / _equivalent_resistance(value) for value in values)
            return 1 / reciprocal_sum

    raise ValueError("resistor network must be a number or contain series/parallel")


def create_mcp():
    """Create the Circuit MCP server."""
    logger.info("Starting Circuit MCP Server")
    mcp = FastMCP(NAME, stateless_http=False)
    db_logger = DatabaseLogger("circuit")

    @mcp.tool()
    @create_tool_wrapper(db_logger, "ohms_law")
    async def ohms_law(circuit_data: str) -> str:
        """
        Apply Ohm's Law and calculate electrical power.

        Args:
            circuit_data: JSON string with two of voltage, current, resistance.
                          Example: '{"voltage": 12, "resistance": 100}'
                          Units: voltage (V), current (A), resistance (ohms)

        Returns:
            str: Complete Ohm's Law and power analysis.
        """
        try:
            data = _load_json(circuit_data)
            voltage = data.get("voltage", data.get("V"))
            current = data.get("current", data.get("I"))
            resistance = data.get("resistance", data.get("R"))

            if voltage is not None:
                voltage = float(voltage)
            if current is not None:
                current = float(current)
            if resistance is not None:
                resistance = float(resistance)

            known = sum(value is not None for value in [voltage, current, resistance])
            if known < 2:
                return "Error: provide at least two of voltage, current, and resistance"

            result = """
Ohm's Law Circuit Analysis:
===========================

Ohm's Law: V = IR
Power: P = VI = I^2 R = V^2 / R

Given:
"""
            if voltage is not None:
                result += f"- Voltage (V): {voltage:.4f} V\n"
            if current is not None:
                result += f"- Current (I): {current:.6f} A\n"
            if resistance is not None:
                result += f"- Resistance (R): {resistance:.4f} ohms\n"

            if voltage is None:
                voltage = current * resistance
                result += f"\nSolving for voltage: V = IR = {current:.6f} x {resistance:.4f}\n"
                result += f"V = {voltage:.4f} V\n"
            elif current is None:
                if resistance == 0:
                    return "Error: resistance cannot be zero when solving for current"
                current = voltage / resistance
                result += f"\nSolving for current: I = V/R = {voltage:.4f} / {resistance:.4f}\n"
                result += f"I = {current:.6f} A = {current * 1000:.3f} mA\n"
            elif resistance is None:
                if current == 0:
                    return "Error: current cannot be zero when solving for resistance"
                resistance = voltage / current
                result += f"\nSolving for resistance: R = V/I = {voltage:.4f} / {current:.6f}\n"
                result += f"R = {resistance:.4f} ohms\n"

            power = voltage * current
            result += f"""
Power:
P = VI = {voltage:.4f} x {current:.6f} = {power:.4f} W
P = I^2 R = {current:.6f}^2 x {resistance:.4f} = {current**2 * resistance:.4f} W
P = V^2 / R = {voltage:.4f}^2 / {resistance:.4f} = {voltage**2 / resistance:.4f} W
"""
            return result

        except Exception as e:
            return f"Error in Ohm's Law calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "resistor_network")
    async def resistor_network(network_data: str) -> str:
        """
        Calculate equivalent resistance for series, parallel, or nested networks.

        Args:
            network_data: JSON string with series/parallel resistor network.
                          Example: '{"series": [100, 200, 300]}'
                          Example: '{"parallel": [100, 200]}'
                          Example: '{"series": [100, {"parallel": [200, 300]}]}'

        Returns:
            str: Equivalent resistance analysis.
        """
        try:
            data = _load_json(network_data)
            equivalent = _equivalent_resistance(data)

            result = f"""
Resistor Network Analysis:
==========================

Network:
{json.dumps(data, indent=2)}

Equivalent Resistance:
R_eq = {equivalent:.4f} ohms

Rules Used:
- Series: R_eq = R1 + R2 + ...
- Parallel: 1/R_eq = 1/R1 + 1/R2 + ...
"""
            return result

        except Exception as e:
            return f"Error in resistor network calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "voltage_divider")
    async def voltage_divider(divider_data: str) -> str:
        """
        Analyze a series voltage divider.

        Args:
            divider_data: JSON string with input voltage and resistor list.
                          Example: '{"input_voltage": 12, "resistors": [1000, 2000]}'

        Returns:
            str: Voltage drop and current through each series resistor.
        """
        try:
            data = _load_json(divider_data)
            input_voltage = float(data.get("input_voltage", data.get("vin")))
            resistors = [float(value) for value in data.get("resistors", [])]

            if not resistors:
                return "Error: provide at least one resistor"
            if any(resistor <= 0 for resistor in resistors):
                return "Error: all resistor values must be positive"

            total_resistance = sum(resistors)
            current = input_voltage / total_resistance

            result = f"""
Voltage Divider Analysis:
=========================

Input Voltage: {input_voltage:.4f} V
Total Resistance: {total_resistance:.4f} ohms
Series Current: I = V_in / R_total = {current:.6f} A

Voltage Drops:
"""
            cumulative = 0.0
            for index, resistor in enumerate(resistors, 1):
                drop = current * resistor
                cumulative += drop
                result += (
                    f"- R{index} = {resistor:.4f} ohms: "
                    f"V{index} = {drop:.4f} V, cumulative = {cumulative:.4f} V\n"
                )

            result += "\nRule: voltage divides in proportion to series resistance.\n"
            return result

        except Exception as e:
            return f"Error in voltage divider calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "rc_circuit")
    async def rc_circuit(rc_data: str) -> str:
        """
        Analyze first-order RC charging or discharging behavior.

        Args:
            rc_data: JSON string with resistance, capacitance, source voltage, and time.
                     Example: '{"resistance": 1000, "capacitance": 1e-6, "voltage": 5, "time": 0.001, "mode": "charging"}'

        Returns:
            str: Time constant and capacitor voltage/current at time t.
        """
        try:
            data = _load_json(rc_data)
            resistance = float(data.get("resistance", data.get("R")))
            capacitance = float(data.get("capacitance", data.get("C")))
            voltage = float(data.get("voltage", data.get("V", 0)))
            time = float(data.get("time", data.get("t", 0)))
            mode = data.get("mode", "charging").lower()

            if resistance <= 0 or capacitance <= 0:
                return "Error: resistance and capacitance must be positive"
            if time < 0:
                return "Error: time cannot be negative"

            tau = resistance * capacitance

            if mode in {"charging", "charge"}:
                capacitor_voltage = voltage * (1 - math.exp(-time / tau))
                current = (voltage / resistance) * math.exp(-time / tau)
                equation = "V_C(t) = V(1 - e^(-t/RC))"
            elif mode in {"discharging", "discharge"}:
                initial_voltage = float(data.get("initial_voltage", voltage))
                capacitor_voltage = initial_voltage * math.exp(-time / tau)
                current = -(initial_voltage / resistance) * math.exp(-time / tau)
                equation = "V_C(t) = V_0 e^(-t/RC)"
            else:
                return "Error: mode must be charging or discharging"

            result = f"""
RC Circuit Analysis:
====================

Mode: {mode}
Resistance: R = {resistance:.4f} ohms
Capacitance: C = {capacitance:.4e} F
Time: t = {time:.6f} s
Time Constant: tau = RC = {tau:.6f} s

Equation:
{equation}

At t = {time:.6f} s:
- Capacitor Voltage: V_C = {capacitor_voltage:.4f} V
- Circuit Current: I = {current:.6f} A
- Elapsed Time: {time / tau:.4f} time constants

Rule of thumb: after 5 time constants, an RC circuit is more than 99% settled.
"""
            return result

        except Exception as e:
            return f"Error in RC circuit calculation: {str(e)}"

    return mcp


def serve(host, port, transport):
    """Initialize and run the Circuit MCP server."""
    mcp = create_mcp()
    logger.info(f"{NAME} MCP Server at {host}:{port} and transport {transport}")
    run_fastmcp_server(mcp, host, port, transport)


def main():
    """CLI entry point for the circuit MCP server."""
    parser = argparse.ArgumentParser(description="Run Physics Circuit MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10102, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")

    args = parser.parse_args()

    if args.run in {"mcp-server", "circuit-server"}:
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(
            f"Unknown run option: {args.run}. Use 'mcp-server' or 'circuit-server'"
        )


if __name__ == "__main__":
    main()
