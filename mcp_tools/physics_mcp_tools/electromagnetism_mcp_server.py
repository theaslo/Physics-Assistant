# type: ignore
"""
Electromagnetism MCP Server for Physics Assistant
Physics 201 - Electricity, Magnetism, and Circuits
"""
import math
import json
import argparse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger
from physics_mcp_tools.database_logger import DatabaseLogger, create_tool_wrapper
from physics_mcp_tools.mcp_runtime import run_fastmcp_server

NAME = "electromagnetism_mcp_server"
logger = get_logger(__name__)

# Physical constants
K_COULOMB = 8.99e9  # N·m²/C² - Coulomb's constant
EPSILON_0 = 8.854e-12  # F/m - Permittivity of free space
MU_0 = 4e-7 * math.pi  # T·m/A - Permeability of free space
E_CHARGE = 1.602e-19  # C - Elementary charge


def create_mcp():
    """Create the Electromagnetism MCP server."""
    logger.info('Starting Electromagnetism MCP Server')
    mcp = FastMCP(NAME, stateless_http=False)
    db_logger = DatabaseLogger("electromagnetism")

    @mcp.tool()
    @create_tool_wrapper(db_logger, "coulombs_law")
    async def coulombs_law(coulomb_data: str) -> str:
        """
        Calculate electric force between point charges using Coulomb's Law.

        Args:
            coulomb_data: JSON string with charge parameters
                         Example: '{"q1": 1e-6, "q2": 2e-6, "distance": 0.1}'
                         Example: '{"q1": -3e-9, "q2": 5e-9, "distance": 0.05}'
                         Units: charges (C), distance (m)

        Returns:
            str: Complete Coulomb's Law analysis
        """
        try:
            data = json.loads(coulomb_data) if isinstance(coulomb_data, str) else coulomb_data

            q1 = float(data.get("q1", data.get("Q1", 1e-6)))
            q2 = float(data.get("q2", data.get("Q2", 1e-6)))
            r = float(data.get("distance", data.get("r", 0.1)))

            # Calculate force magnitude
            F = K_COULOMB * abs(q1 * q2) / (r ** 2)

            # Determine force direction
            same_sign = (q1 * q2) > 0
            force_type = "repulsive" if same_sign else "attractive"

            result = f"""
Coulomb's Law Analysis:
=======================

Given:
- Charge 1 (q₁): {q1:.2e} C = {q1/E_CHARGE:.2f} e
- Charge 2 (q₂): {q2:.2e} C = {q2/E_CHARGE:.2f} e
- Distance (r): {r:.4f} m

Coulomb's Constant: k = {K_COULOMB:.2e} N·m²/C²

Coulomb's Law: F = k|q₁q₂|/r²

Calculation:
F = {K_COULOMB:.2e} × |{q1:.2e} × {q2:.2e}| / {r:.4f}²
F = {K_COULOMB:.2e} × {abs(q1*q2):.2e} / {r**2:.6f}
F = {F:.4e} N

Result:
- Force Magnitude: {F:.4e} N
- Force Type: {force_type.upper()} (charges {"same" if same_sign else "opposite"} sign)
- Direction: {"Away from each other" if same_sign else "Toward each other"}

Comparison:
- Gravitational force between 1kg masses at same distance: ~{6.67e-11 * 1 * 1 / r**2:.2e} N
- Electric force is ~{F / (6.67e-11 / r**2):.2e} times stronger
"""
            return result

        except Exception as e:
            return f"Error in Coulomb's Law calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "electric_field")
    async def electric_field(field_data: str) -> str:
        """
        Calculate electric field from point charge(s).

        Args:
            field_data: JSON string with field parameters
                       Example: '{"charge": 5e-9, "distance": 0.2}'
                       Example: '{"charges": [{"q": 1e-6, "x": 0, "y": 0}, {"q": -1e-6, "x": 0.1, "y": 0}], "point": {"x": 0.05, "y": 0.05}}'
                       Units: charge (C), distances (m)

        Returns:
            str: Complete electric field analysis
        """
        try:
            data = json.loads(field_data) if isinstance(field_data, str) else field_data

            result = """
Electric Field Analysis:
========================

"""
            if "charge" in data:
                # Single point charge
                q = float(data.get("charge", data.get("q")))
                r = float(data.get("distance", data.get("r")))

                E = K_COULOMB * abs(q) / (r ** 2)
                direction = "away from charge" if q > 0 else "toward charge"

                result += f"Single Point Charge:\n"
                result += f"- Charge (q): {q:.2e} C\n"
                result += f"- Distance (r): {r:.4f} m\n\n"
                result += f"Electric Field Formula: E = k|q|/r²\n\n"
                result += f"Calculation:\n"
                result += f"E = {K_COULOMB:.2e} × |{q:.2e}| / {r:.4f}²\n"
                result += f"E = {E:.4e} N/C (or V/m)\n\n"
                result += f"Direction: {direction}\n"

            elif "charges" in data:
                charges = data["charges"]
                point = data.get("point", {"x": 0, "y": 0})
                px = float(point.get("x", 0))
                py = float(point.get("y", 0))
                ex_total = 0.0
                ey_total = 0.0

                result += "Multiple Point Charges (2D superposition):\n"
                result += f"- Field point: ({px:.4f}, {py:.4f}) m\n\n"

                for index, charge in enumerate(charges, 1):
                    q = float(charge.get("q", charge.get("charge")))
                    x = float(charge.get("x", 0))
                    y = float(charge.get("y", 0))
                    dx = px - x
                    dy = py - y
                    r_squared = dx**2 + dy**2

                    if r_squared == 0:
                        return (
                            "Error in electric field calculation: field point cannot "
                            "be at the same location as a point charge"
                        )

                    r = math.sqrt(r_squared)
                    e_magnitude_signed = K_COULOMB * q / r_squared
                    ex = e_magnitude_signed * dx / r
                    ey = e_magnitude_signed * dy / r
                    ex_total += ex
                    ey_total += ey

                    result += f"Charge {index}: q = {q:.2e} C at ({x:.4f}, {y:.4f}) m\n"
                    result += f"  r = {r:.4f} m, E_x = {ex:.4e} N/C, E_y = {ey:.4e} N/C\n"

                e_total = math.sqrt(ex_total**2 + ey_total**2)
                angle = math.degrees(math.atan2(ey_total, ex_total))

                result += f"\nTotal Electric Field:\n"
                result += f"E_x = {ex_total:.4e} N/C\n"
                result += f"E_y = {ey_total:.4e} N/C\n"
                result += f"|E| = {e_total:.4e} N/C\n"
                result += f"Direction = {angle:.2f}° from +x axis\n"

            elif "voltage" in data and "distance" in data:
                # Uniform field
                V = float(data.get("voltage", data.get("V")))
                d = float(data.get("distance", data.get("d")))

                E = V / d

                result += f"Uniform Electric Field (parallel plates):\n"
                result += f"- Voltage (V): {V:.2f} V\n"
                result += f"- Plate Separation (d): {d:.4f} m\n\n"
                result += f"Formula: E = V/d\n\n"
                result += f"E = {V:.2f} / {d:.4f} = {E:.4e} N/C\n"

            result += """
Electric Field Properties:
- Units: N/C = V/m
- Vector quantity (has direction)
- Points away from positive charges
- Points toward negative charges
- Superposition: E_total = E₁ + E₂ + ...
"""
            return result

        except Exception as e:
            return f"Error in electric field calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "electric_potential")
    async def electric_potential(potential_data: str) -> str:
        """
        Calculate electric potential and potential energy.

        Args:
            potential_data: JSON string with potential parameters
                           Example: '{"charge": 2e-6, "distance": 0.5}'
                           Example: '{"q1": 1e-6, "q2": -2e-6, "distance": 0.1}'
                           Units: charges (C), distance (m)

        Returns:
            str: Complete electric potential analysis
        """
        try:
            data = json.loads(potential_data) if isinstance(potential_data, str) else potential_data

            result = """
Electric Potential Analysis:
============================

"""
            if "charge" in data and "distance" in data:
                q = float(data.get("charge", data.get("q")))
                r = float(data.get("distance", data.get("r")))

                V = K_COULOMB * q / r

                result += f"Potential from Point Charge:\n"
                result += f"- Source Charge (q): {q:.2e} C\n"
                result += f"- Distance (r): {r:.4f} m\n\n"
                result += f"Formula: V = kq/r\n\n"
                result += f"V = {K_COULOMB:.2e} × {q:.2e} / {r:.4f}\n"
                result += f"V = {V:.4f} V\n"

            if "q1" in data and "q2" in data and "distance" in data:
                q1 = float(data.get("q1"))
                q2 = float(data.get("q2"))
                r = float(data.get("distance", data.get("r")))

                U = K_COULOMB * q1 * q2 / r

                result += f"\nElectric Potential Energy:\n"
                result += f"- Charge 1 (q₁): {q1:.2e} C\n"
                result += f"- Charge 2 (q₂): {q2:.2e} C\n"
                result += f"- Separation (r): {r:.4f} m\n\n"
                result += f"Formula: U = kq₁q₂/r\n\n"
                result += f"U = {K_COULOMB:.2e} × {q1:.2e} × {q2:.2e} / {r:.4f}\n"
                result += f"U = {U:.4e} J\n\n"
                result += f"Interpretation: {'Bound system (attractive)' if U < 0 else 'Unbound system (repulsive)'}\n"

            return result

        except Exception as e:
            return f"Error in electric potential calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "capacitance")
    async def capacitance(capacitor_data: str) -> str:
        """
        Calculate capacitance, charge, energy, and combinations.

        Args:
            capacitor_data: JSON string with capacitor parameters
                           Example: '{"area": 0.01, "separation": 0.001}'
                           Example: '{"capacitance": 10e-6, "voltage": 12}'
                           Example: '{"series": [10e-6, 20e-6, 30e-6]}'
                           Example: '{"parallel": [10e-6, 20e-6]}'
                           Units: area (m²), separation (m), capacitance (F), voltage (V)

        Returns:
            str: Complete capacitance analysis
        """
        try:
            data = json.loads(capacitor_data) if isinstance(capacitor_data, str) else capacitor_data

            result = """
Capacitor Analysis:
===================

"""
            if "area" in data and "separation" in data:
                A = float(data.get("area", data.get("A")))
                d = float(data.get("separation", data.get("d")))
                kappa = float(data.get("dielectric", data.get("k", 1)))

                C = kappa * EPSILON_0 * A / d

                result += f"Parallel Plate Capacitor:\n"
                result += f"- Plate Area (A): {A:.6f} m²\n"
                result += f"- Separation (d): {d:.6f} m\n"
                result += f"- Dielectric Constant (κ): {kappa}\n\n"
                result += f"Formula: C = κε₀A/d\n\n"
                result += f"C = {kappa} × {EPSILON_0:.2e} × {A:.6f} / {d:.6f}\n"
                result += f"C = {C:.4e} F = {C*1e6:.4f} μF = {C*1e12:.2f} pF\n\n"

            if "capacitance" in data and "voltage" in data:
                C = float(data.get("capacitance", data.get("C")))
                V = float(data.get("voltage", data.get("V")))

                Q = C * V
                U = 0.5 * C * V ** 2

                result += f"Capacitor Charge and Energy:\n"
                result += f"- Capacitance (C): {C:.4e} F = {C*1e6:.4f} μF\n"
                result += f"- Voltage (V): {V:.2f} V\n\n"
                result += f"Charge: Q = CV = {C:.4e} × {V:.2f} = {Q:.4e} C\n"
                result += f"Energy: U = ½CV² = ½ × {C:.4e} × {V:.2f}² = {U:.4e} J\n"

            if "series" in data:
                caps = [float(c) for c in data["series"]]
                C_series = 1 / sum(1/c for c in caps)

                result += f"\nCapacitors in Series:\n"
                result += f"Capacitors: {[f'{c*1e6:.2f} μF' for c in caps]}\n"
                result += f"Formula: 1/C_eq = 1/C₁ + 1/C₂ + ...\n"
                result += f"C_eq = {C_series:.4e} F = {C_series*1e6:.4f} μF\n"

            if "parallel" in data:
                caps = [float(c) for c in data["parallel"]]
                C_parallel = sum(caps)

                result += f"\nCapacitors in Parallel:\n"
                result += f"Capacitors: {[f'{c*1e6:.2f} μF' for c in caps]}\n"
                result += f"Formula: C_eq = C₁ + C₂ + ...\n"
                result += f"C_eq = {C_parallel:.4e} F = {C_parallel*1e6:.4f} μF\n"

            return result

        except Exception as e:
            return f"Error in capacitance calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "ohms_law")
    async def ohms_law(circuit_data: str) -> str:
        """
        Apply Ohm's Law (V = IR) and calculate power.

        Args:
            circuit_data: JSON string with circuit parameters. Provide 2 of 3: voltage, current, resistance
                         Example: '{"voltage": 12, "resistance": 100}'
                         Example: '{"current": 0.5, "resistance": 24}'
                         Units: voltage (V), current (A), resistance (Ω)

        Returns:
            str: Complete Ohm's Law analysis with power calculations
        """
        try:
            data = json.loads(circuit_data) if isinstance(circuit_data, str) else circuit_data

            V = data.get("voltage", data.get("V", None))
            I = data.get("current", data.get("I", None))
            R = data.get("resistance", data.get("R", None))

            if V is not None: V = float(V)
            if I is not None: I = float(I)
            if R is not None: R = float(R)

            result = """
Ohm's Law Analysis (V = IR):
============================

"""
            result += "Given:\n"
            if V is not None: result += f"- Voltage (V): {V:.4f} V\n"
            if I is not None: result += f"- Current (I): {I:.6f} A = {I*1000:.3f} mA\n"
            if R is not None: result += f"- Resistance (R): {R:.4f} Ω\n"

            result += "\nOhm's Law: V = IR\n\n"

            if V is None and I and R:
                V = I * R
                result += f"Solving for Voltage:\n"
                result += f"V = IR = {I:.6f} × {R:.4f} = {V:.4f} V\n"

            elif I is None and V and R:
                I = V / R
                result += f"Solving for Current:\n"
                result += f"I = V/R = {V:.4f} / {R:.4f} = {I:.6f} A = {I*1000:.3f} mA\n"

            elif R is None and V and I:
                R = V / I
                result += f"Solving for Resistance:\n"
                result += f"R = V/I = {V:.4f} / {I:.6f} = {R:.4f} Ω\n"

            # Power calculations
            if V and I:
                P = V * I
                result += f"\nPower Calculations:\n"
                result += f"P = VI = {V:.4f} × {I:.6f} = {P:.4f} W\n"
                result += f"P = I²R = {I:.6f}² × {R:.4f} = {I**2 * R:.4f} W\n"
                result += f"P = V²/R = {V:.4f}² / {R:.4f} = {V**2 / R:.4f} W\n"

            return result

        except Exception as e:
            return f"Error in Ohm's Law calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "resistor_network")
    async def resistor_network(network_data: str) -> str:
        """
        Calculate equivalent resistance for series and parallel combinations.

        Args:
            network_data: JSON string with resistor network
                         Example: '{"series": [100, 200, 300]}'
                         Example: '{"parallel": [100, 200]}'
                         Example: '{"series": [100, {"parallel": [200, 300]}]}'
                         Units: resistance (Ω)

        Returns:
            str: Complete resistor network analysis
        """
        try:
            data = json.loads(network_data) if isinstance(network_data, str) else network_data

            def calc_equivalent(config):
                if isinstance(config, (int, float)):
                    return float(config)
                elif isinstance(config, dict):
                    if "series" in config:
                        return sum(calc_equivalent(r) for r in config["series"])
                    elif "parallel" in config:
                        return 1 / sum(1/calc_equivalent(r) for r in config["parallel"])
                return 0

            result = """
Resistor Network Analysis:
==========================

"""
            if "series" in data:
                resistors = data["series"]
                R_eq = sum(float(r) if isinstance(r, (int, float)) else calc_equivalent(r) for r in resistors)

                result += f"Series Combination:\n"
                result += f"Resistors: {resistors}\n"
                result += f"Formula: R_eq = R₁ + R₂ + R₃ + ...\n"
                result += f"R_eq = {R_eq:.4f} Ω\n\n"
                result += "Note: In series, current is same through all resistors\n"

            if "parallel" in data:
                resistors = data["parallel"]
                R_values = [float(r) if isinstance(r, (int, float)) else calc_equivalent(r) for r in resistors]
                R_eq = 1 / sum(1/r for r in R_values)

                result += f"Parallel Combination:\n"
                result += f"Resistors: {resistors}\n"
                result += f"Formula: 1/R_eq = 1/R₁ + 1/R₂ + ...\n"
                result += f"R_eq = {R_eq:.4f} Ω\n\n"
                result += "Note: In parallel, voltage is same across all resistors\n"

            return result

        except Exception as e:
            return f"Error in resistor network calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "magnetic_force")
    async def magnetic_force(magnetic_data: str) -> str:
        """
        Calculate magnetic force on moving charge or current-carrying wire.

        Args:
            magnetic_data: JSON string with magnetic parameters
                          Example: '{"charge": 1.6e-19, "velocity": 1e6, "B_field": 0.5, "angle": 90}'
                          Example: '{"current": 5, "length": 0.1, "B_field": 0.2, "angle": 90}'
                          Units: charge (C), velocity (m/s), B_field (T), current (A), length (m), angle (degrees)

        Returns:
            str: Complete magnetic force analysis
        """
        try:
            data = json.loads(magnetic_data) if isinstance(magnetic_data, str) else magnetic_data

            B = float(data.get("B_field", data.get("B", 1)))
            angle = float(data.get("angle", 90))
            angle_rad = math.radians(angle)

            result = """
Magnetic Force Analysis:
========================

"""
            if "charge" in data and "velocity" in data:
                q = float(data.get("charge", data.get("q")))
                v = float(data.get("velocity", data.get("v")))

                F = abs(q) * v * B * math.sin(angle_rad)

                result += f"Force on Moving Charge (Lorentz Force):\n"
                result += f"- Charge (q): {q:.2e} C\n"
                result += f"- Velocity (v): {v:.2e} m/s\n"
                result += f"- Magnetic Field (B): {B:.4f} T\n"
                result += f"- Angle between v and B: {angle}°\n\n"
                result += f"Formula: F = |q|vB sin(θ)\n\n"
                result += f"F = |{q:.2e}| × {v:.2e} × {B:.4f} × sin({angle}°)\n"
                result += f"F = {F:.4e} N\n\n"
                result += f"Direction: Use right-hand rule (perpendicular to both v and B)\n"

            elif "current" in data and "length" in data:
                I = float(data.get("current", data.get("I")))
                L = float(data.get("length", data.get("L")))

                F = I * L * B * math.sin(angle_rad)

                result += f"Force on Current-Carrying Wire:\n"
                result += f"- Current (I): {I:.4f} A\n"
                result += f"- Wire Length (L): {L:.4f} m\n"
                result += f"- Magnetic Field (B): {B:.4f} T\n"
                result += f"- Angle between wire and B: {angle}°\n\n"
                result += f"Formula: F = ILB sin(θ)\n\n"
                result += f"F = {I:.4f} × {L:.4f} × {B:.4f} × sin({angle}°)\n"
                result += f"F = {F:.4e} N\n"

            return result

        except Exception as e:
            return f"Error in magnetic force calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "magnetic_field_wire")
    async def magnetic_field_wire(wire_data: str) -> str:
        """
        Calculate magnetic field from current-carrying wire.

        Args:
            wire_data: JSON string with wire parameters
                      Example: '{"current": 10, "distance": 0.05}'
                      Example: '{"current": 5, "radius": 0.02, "type": "solenoid", "turns_per_length": 1000}'
                      Units: current (A), distance (m), radius (m)

        Returns:
            str: Complete magnetic field analysis
        """
        try:
            data = json.loads(wire_data) if isinstance(wire_data, str) else wire_data

            I = float(data.get("current", data.get("I")))
            wire_type = data.get("type", "straight").lower()

            result = """
Magnetic Field Analysis:
========================

"""
            if wire_type == "straight" or "distance" in data:
                r = float(data.get("distance", data.get("r", 0.01)))

                B = MU_0 * I / (2 * math.pi * r)

                result += f"Long Straight Wire:\n"
                result += f"- Current (I): {I:.4f} A\n"
                result += f"- Distance from wire (r): {r:.4f} m\n\n"
                result += f"Formula: B = μ₀I / (2πr)\n\n"
                result += f"B = {MU_0:.2e} × {I:.4f} / (2π × {r:.4f})\n"
                result += f"B = {B:.4e} T = {B*1e6:.4f} μT\n\n"
                result += f"Direction: Circles around wire (right-hand rule)\n"

            elif wire_type == "solenoid":
                n = float(data.get("turns_per_length", data.get("n", 1000)))

                B = MU_0 * n * I

                result += f"Solenoid (inside):\n"
                result += f"- Current (I): {I:.4f} A\n"
                result += f"- Turns per unit length (n): {n:.0f} turns/m\n\n"
                result += f"Formula: B = μ₀nI\n\n"
                result += f"B = {MU_0:.2e} × {n:.0f} × {I:.4f}\n"
                result += f"B = {B:.4e} T = {B*1e3:.4f} mT\n"

            elif wire_type == "loop" or wire_type == "coil":
                R = float(data.get("radius", data.get("R", 0.01)))
                N = int(data.get("turns", data.get("N", 1)))

                B = MU_0 * N * I / (2 * R)

                result += f"Circular Loop/Coil (at center):\n"
                result += f"- Current (I): {I:.4f} A\n"
                result += f"- Radius (R): {R:.4f} m\n"
                result += f"- Number of turns (N): {N}\n\n"
                result += f"Formula: B = μ₀NI / (2R)\n\n"
                result += f"B = {MU_0:.2e} × {N} × {I:.4f} / (2 × {R:.4f})\n"
                result += f"B = {B:.4e} T\n"

            return result

        except Exception as e:
            return f"Error in magnetic field calculation: {str(e)}"

    @mcp.tool()
    @create_tool_wrapper(db_logger, "faradays_law")
    async def faradays_law(faraday_data: str) -> str:
        """
        Calculate induced EMF using Faraday's Law of Induction.

        Args:
            faraday_data: JSON string with induction parameters
                         Example: '{"B_change": 0.5, "area": 0.01, "time": 0.1, "turns": 100}'
                         Example: '{"flux_change": 0.005, "time": 0.02}'
                         Units: B_field (T), area (m²), time (s), flux (Wb)

        Returns:
            str: Complete electromagnetic induction analysis
        """
        try:
            data = json.loads(faraday_data) if isinstance(faraday_data, str) else faraday_data

            N = int(data.get("turns", data.get("N", 1)))
            dt = float(data.get("time", data.get("dt", 1)))

            result = """
Faraday's Law of Induction:
===========================

"""
            if "flux_change" in data:
                d_flux = float(data.get("flux_change", data.get("dPhi")))
                emf = -N * d_flux / dt

                result += f"Given:\n"
                result += f"- Flux Change (ΔΦ): {d_flux:.4e} Wb\n"
                result += f"- Time Interval (Δt): {dt:.4f} s\n"
                result += f"- Number of Turns (N): {N}\n\n"

            elif "B_change" in data and "area" in data:
                dB = float(data.get("B_change", data.get("dB")))
                A = float(data.get("area", data.get("A")))
                d_flux = dB * A
                emf = -N * d_flux / dt

                result += f"Given:\n"
                result += f"- Magnetic Field Change (ΔB): {dB:.4f} T\n"
                result += f"- Area (A): {A:.6f} m²\n"
                result += f"- Time Interval (Δt): {dt:.4f} s\n"
                result += f"- Number of Turns (N): {N}\n\n"
                result += f"Flux Change: ΔΦ = ΔB × A = {dB:.4f} × {A:.6f} = {d_flux:.4e} Wb\n\n"

            else:
                return (
                    "Error in Faraday's Law calculation: provide either "
                    "flux_change or both B_change and area"
                )

            result += f"Faraday's Law: ε = -N(dΦ/dt)\n\n"
            result += f"Induced EMF:\n"
            result += f"ε = -{N} × {d_flux:.4e} / {dt:.4f}\n"
            result += f"ε = {emf:.4f} V\n\n"
            result += f"Magnitude: |ε| = {abs(emf):.4f} V\n\n"
            result += "Note: Negative sign indicates Lenz's Law - induced EMF opposes change\n"

            return result

        except Exception as e:
            return f"Error in Faraday's Law calculation: {str(e)}"

    return mcp


def serve(host, port, transport):
    """Initialize and run the Electromagnetism MCP server."""
    mcp = create_mcp()
    logger.info(f'{NAME} MCP Server at {host}:{port} and transport {transport}')
    run_fastmcp_server(mcp, host, port, transport)


def main():
    """CLI entry point for the electromagnetism MCP server."""
    parser = argparse.ArgumentParser(description="Run Electromagnetism MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10109, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")

    args = parser.parse_args()

    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")


if __name__ == "__main__":
    main()
