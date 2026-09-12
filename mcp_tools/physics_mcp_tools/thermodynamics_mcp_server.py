# type: ignore
"""
Thermodynamics MCP Server for Physics Assistant
Physics 102 - Heat, Temperature, and Thermal Physics
"""
import math
import json
import argparse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

NAME = "thermodynamics_mcp_server"
logger = get_logger(__name__)

# Physical constants
R_UNIVERSAL = 8.314  # J/(mol·K) - Universal gas constant
K_BOLTZMANN = 1.381e-23  # J/K - Boltzmann constant
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴) - Stefan-Boltzmann constant


def serve(host, port, transport):
    """Initialize and run the Thermodynamics MCP server."""
    logger.info('Starting Thermodynamics MCP Server')
    mcp = FastMCP(NAME, stateless_http=False)

    def with_diagram_payload(result_text: str, payload: dict) -> str:
        """Append machine-readable payload for frontend thermodynamics diagrams."""
        return (
            f"{result_text}\n\n"
            "DIAGRAM_JSON_START\n"
            f"{json.dumps(payload)}\n"
            "DIAGRAM_JSON_END"
        )

    @mcp.tool()
    async def ideal_gas_law(gas_data: str) -> str:
        """
        Apply the Ideal Gas Law (PV = nRT) to solve for unknown quantity.

        Args:
            gas_data: JSON string with known values. Provide 3 of 4: pressure, volume, moles, temperature
                     Example: '{"pressure": 101325, "volume": 0.0224, "moles": 1}'
                     Example: '{"pressure": 200000, "volume": 0.01, "temperature": 300}'
                     Units: pressure (Pa), volume (m³), moles (mol), temperature (K)

        Returns:
            str: Complete ideal gas law analysis
        """
        try:
            data = json.loads(gas_data) if isinstance(gas_data, str) else gas_data

            P = data.get("pressure", data.get("P", None))
            V = data.get("volume", data.get("V", None))
            n = data.get("moles", data.get("n", None))
            T = data.get("temperature", data.get("T", None))

            # Convert to float
            if P is not None: P = float(P)
            if V is not None: V = float(V)
            if n is not None: n = float(n)
            if T is not None: T = float(T)

            known_count = sum(1 for v in [P, V, n, T] if v is not None)
            if known_count < 3:
                return "Error: Need at least 3 known values (pressure, volume, moles, temperature)"

            result = """
Ideal Gas Law Analysis (PV = nRT):
==================================

"""
            result += "Given:\n"
            if P is not None: result += f"- Pressure (P): {P:.2f} Pa = {P/1000:.2f} kPa = {P/101325:.3f} atm\n"
            if V is not None: result += f"- Volume (V): {V:.6f} m³ = {V*1000:.3f} L\n"
            if n is not None: result += f"- Amount (n): {n:.4f} mol\n"
            if T is not None: result += f"- Temperature (T): {T:.2f} K = {T-273.15:.2f} °C\n"

            result += f"\nUniversal Gas Constant: R = {R_UNIVERSAL} J/(mol·K)\n\n"
            result += "Ideal Gas Law: PV = nRT\n\n"

            # Solve for unknown
            if T is None:
                T = (P * V) / (n * R_UNIVERSAL)
                result += f"Solving for Temperature:\n"
                result += f"T = PV / (nR)\n"
                result += f"T = ({P:.2f} × {V:.6f}) / ({n:.4f} × {R_UNIVERSAL})\n"
                result += f"T = {T:.2f} K = {T-273.15:.2f} °C\n"

            elif P is None:
                P = (n * R_UNIVERSAL * T) / V
                result += f"Solving for Pressure:\n"
                result += f"P = nRT / V\n"
                result += f"P = ({n:.4f} × {R_UNIVERSAL} × {T:.2f}) / {V:.6f}\n"
                result += f"P = {P:.2f} Pa = {P/101325:.3f} atm\n"

            elif V is None:
                V = (n * R_UNIVERSAL * T) / P
                result += f"Solving for Volume:\n"
                result += f"V = nRT / P\n"
                result += f"V = ({n:.4f} × {R_UNIVERSAL} × {T:.2f}) / {P:.2f}\n"
                result += f"V = {V:.6f} m³ = {V*1000:.3f} L\n"

            elif n is None:
                n = (P * V) / (R_UNIVERSAL * T)
                result += f"Solving for Moles:\n"
                result += f"n = PV / (RT)\n"
                result += f"n = ({P:.2f} × {V:.6f}) / ({R_UNIVERSAL} × {T:.2f})\n"
                result += f"n = {n:.4f} mol\n"

            if P is not None and V is not None and n is not None and T is not None:
                max_p = max(1.0, P * 1.35)
                max_v = max(1e-6, V * 1.35)
                isotherms = []
                for factor in [0.75, 1.0, 1.25]:
                    iso_t = max(1.0, T * factor)
                    curve = []
                    for i in range(40):
                        vol = max_v * (0.2 + 0.8 * i / 39)
                        pressure = (n * R_UNIVERSAL * iso_t) / vol
                        curve.append({"volume_m3": vol, "pressure_pa": pressure})
                    isotherms.append({"temperature_k": iso_t, "points": curve})

                payload = {
                    "type": "pv_diagram",
                    "title": "PV Diagram (Ideal Gas State)",
                    "state": {
                        "pressure_pa": P,
                        "volume_m3": V,
                        "moles": n,
                        "temperature_k": T,
                    },
                    "pv_points": [{"label": "State", "pressure_pa": P, "volume_m3": V}],
                    "isotherms": isotherms,
                    "axis": {"max_pressure_pa": max_p, "max_volume_m3": max_v},
                }
                return with_diagram_payload(result, payload)

            return result

        except Exception as e:
            return f"Error in ideal gas law calculation: {str(e)}"

    @mcp.tool()
    async def heat_transfer(heat_data: str) -> str:
        """
        Calculate heat transfer using Q = mcΔT.

        Args:
            heat_data: JSON string with heat transfer parameters
                      Example: '{"mass": 2, "specific_heat": 4186, "temp_change": 25}'
                      Example: '{"heat": 10000, "mass": 0.5, "specific_heat": 4186}'
                      Units: mass (kg), specific_heat (J/(kg·K)), temp_change (K or °C), heat (J)

        Returns:
            str: Complete heat transfer analysis
        """
        try:
            data = json.loads(heat_data) if isinstance(heat_data, str) else heat_data

            Q = data.get("heat", data.get("Q", None))
            m = data.get("mass", data.get("m", None))
            c = data.get("specific_heat", data.get("c", None))
            dT = data.get("temp_change", data.get("delta_T", data.get("dT", None)))

            if Q is not None: Q = float(Q)
            if m is not None: m = float(m)
            if c is not None: c = float(c)
            if dT is not None: dT = float(dT)

            result = """
Heat Transfer Analysis (Q = mcΔT):
==================================

"""
            result += "Given:\n"
            if Q is not None: result += f"- Heat (Q): {Q:.2f} J = {Q/1000:.3f} kJ\n"
            if m is not None: result += f"- Mass (m): {m:.4f} kg\n"
            if c is not None: result += f"- Specific Heat (c): {c:.2f} J/(kg·K)\n"
            if dT is not None: result += f"- Temperature Change (ΔT): {dT:.2f} K (or °C)\n"

            result += "\nHeat Transfer Equation: Q = mcΔT\n\n"

            if Q is None and m and c and dT:
                Q = m * c * dT
                result += f"Solving for Heat:\n"
                result += f"Q = mcΔT = {m:.4f} × {c:.2f} × {dT:.2f}\n"
                result += f"Q = {Q:.2f} J = {Q/1000:.3f} kJ\n"

            elif m is None and Q and c and dT:
                m = Q / (c * dT)
                result += f"Solving for Mass:\n"
                result += f"m = Q / (cΔT) = {Q:.2f} / ({c:.2f} × {dT:.2f})\n"
                result += f"m = {m:.4f} kg\n"

            elif c is None and Q and m and dT:
                c = Q / (m * dT)
                result += f"Solving for Specific Heat:\n"
                result += f"c = Q / (mΔT) = {Q:.2f} / ({m:.4f} × {dT:.2f})\n"
                result += f"c = {c:.2f} J/(kg·K)\n"

            elif dT is None and Q and m and c:
                dT = Q / (m * c)
                result += f"Solving for Temperature Change:\n"
                result += f"ΔT = Q / (mc) = {Q:.2f} / ({m:.4f} × {c:.2f})\n"
                result += f"ΔT = {dT:.2f} K (or °C)\n"

            result += """
Common Specific Heat Values:
- Water: 4186 J/(kg·K)
- Ice: 2090 J/(kg·K)
- Aluminum: 900 J/(kg·K)
- Copper: 385 J/(kg·K)
- Iron: 450 J/(kg·K)
"""
            if m is not None and c is not None and dT is not None and Q is not None:
                steps = 48
                q_points = []
                for i in range(steps):
                    frac = i / (steps - 1)
                    q_points.append({"q_j": Q * frac, "delta_t_k": dT * frac})

                payload = {
                    "type": "heating_curve_plot",
                    "title": "Heating Curve (Q vs ΔT)",
                    "mass_kg": m,
                    "specific_heat_j_per_kgk": c,
                    "delta_t_k": dT,
                    "heat_j": Q,
                    "q_vs_delta_t": q_points,
                }
                return with_diagram_payload(result, payload)

            return result

        except Exception as e:
            return f"Error in heat transfer calculation: {str(e)}"

    @mcp.tool()
    async def thermal_expansion(expansion_data: str) -> str:
        """
        Calculate linear or volumetric thermal expansion.

        Args:
            expansion_data: JSON string with expansion parameters
                           Example: '{"type": "linear", "length": 2, "alpha": 12e-6, "temp_change": 50}'
                           Example: '{"type": "volume", "volume": 0.001, "beta": 36e-6, "temp_change": 100}'
                           Units: length (m), alpha (1/K), volume (m³), beta (1/K), temp_change (K)

        Returns:
            str: Complete thermal expansion analysis
        """
        try:
            data = json.loads(expansion_data) if isinstance(expansion_data, str) else expansion_data

            exp_type = data.get("type", "linear").lower()
            dT = float(data.get("temp_change", data.get("dT", 0)))

            result = """
Thermal Expansion Analysis:
===========================

"""
            if exp_type == "linear":
                L0 = float(data.get("length", data.get("L0", data.get("L", 1))))
                alpha = float(data.get("alpha", data.get("coefficient", 12e-6)))

                dL = alpha * L0 * dT
                L_final = L0 + dL

                result += f"Type: Linear Expansion\n\n"
                result += f"Given:\n"
                result += f"- Initial Length (L₀): {L0:.6f} m\n"
                result += f"- Coefficient of Linear Expansion (α): {alpha:.2e} /K\n"
                result += f"- Temperature Change (ΔT): {dT:.2f} K\n\n"
                result += f"Formula: ΔL = αL₀ΔT\n\n"
                result += f"Calculation:\n"
                result += f"ΔL = {alpha:.2e} × {L0:.6f} × {dT:.2f}\n"
                result += f"ΔL = {dL:.6f} m = {dL*1000:.4f} mm\n\n"
                result += f"Final Length: L = L₀ + ΔL = {L_final:.6f} m\n"

            else:  # volume
                V0 = float(data.get("volume", data.get("V0", data.get("V", 1))))
                beta = float(data.get("beta", data.get("coefficient", 36e-6)))

                dV = beta * V0 * dT
                V_final = V0 + dV

                result += f"Type: Volumetric Expansion\n\n"
                result += f"Given:\n"
                result += f"- Initial Volume (V₀): {V0:.6f} m³\n"
                result += f"- Coefficient of Volumetric Expansion (β): {beta:.2e} /K\n"
                result += f"- Temperature Change (ΔT): {dT:.2f} K\n\n"
                result += f"Formula: ΔV = βV₀ΔT\n\n"
                result += f"Calculation:\n"
                result += f"ΔV = {beta:.2e} × {V0:.6f} × {dT:.2f}\n"
                result += f"ΔV = {dV:.6f} m³\n\n"
                result += f"Final Volume: V = V₀ + ΔV = {V_final:.6f} m³\n"

            result += """
Common Expansion Coefficients (α):
- Aluminum: 23×10⁻⁶ /K
- Steel: 12×10⁻⁶ /K
- Copper: 17×10⁻⁶ /K
- Glass: 9×10⁻⁶ /K
- Concrete: 12×10⁻⁶ /K

Note: β ≈ 3α for isotropic materials
"""
            return result

        except Exception as e:
            return f"Error in thermal expansion calculation: {str(e)}"

    @mcp.tool()
    async def heat_conduction(conduction_data: str) -> str:
        """
        Calculate heat conduction through a material using Fourier's Law.

        Args:
            conduction_data: JSON string with conduction parameters
                            Example: '{"conductivity": 205, "area": 0.01, "thickness": 0.02, "temp_diff": 80}'
                            Units: conductivity (W/(m·K)), area (m²), thickness (m), temp_diff (K)

        Returns:
            str: Complete heat conduction analysis
        """
        try:
            data = json.loads(conduction_data) if isinstance(conduction_data, str) else conduction_data

            k = float(data.get("conductivity", data.get("k", 1)))
            A = float(data.get("area", data.get("A", 1)))
            L = float(data.get("thickness", data.get("L", data.get("length", 1))))
            dT = float(data.get("temp_diff", data.get("dT", data.get("delta_T", 1))))

            # Heat flow rate
            Q_rate = k * A * dT / L

            result = f"""
Heat Conduction Analysis (Fourier's Law):
=========================================

Given:
- Thermal Conductivity (k): {k:.2f} W/(m·K)
- Cross-sectional Area (A): {A:.6f} m²
- Thickness (L): {L:.4f} m
- Temperature Difference (ΔT): {dT:.2f} K

Fourier's Law: Q/t = kA(ΔT)/L

Calculation:
Q/t = {k:.2f} × {A:.6f} × {dT:.2f} / {L:.4f}
Q/t = {Q_rate:.2f} W

Result: Heat Flow Rate = {Q_rate:.2f} W = {Q_rate:.2f} J/s

Thermal Resistance: R = L/(kA) = {L/(k*A):.4f} K/W

Common Thermal Conductivities:
- Copper: 401 W/(m·K)
- Aluminum: 205 W/(m·K)
- Steel: 50 W/(m·K)
- Glass: 1.0 W/(m·K)
- Wood: 0.15 W/(m·K)
- Styrofoam: 0.033 W/(m·K)
"""
            payload = {
                "type": "heat_transfer_path_diagram",
                "title": "Heat Conduction Path",
                "conductivity_w_mk": k,
                "area_m2": A,
                "thickness_m": L,
                "delta_t_k": dT,
                "heat_rate_w": Q_rate,
                "thermal_resistance_k_per_w": L / (k * A),
                "hot_side_temp_c": dT,
                "cold_side_temp_c": 0.0,
            }
            return with_diagram_payload(result, payload)

        except Exception as e:
            return f"Error in heat conduction calculation: {str(e)}"

    @mcp.tool()
    async def carnot_efficiency(carnot_data: str) -> str:
        """
        Calculate Carnot engine efficiency and related quantities.

        Args:
            carnot_data: JSON string with Carnot engine parameters
                        Example: '{"T_hot": 600, "T_cold": 300}'
                        Example: '{"T_hot": 500, "T_cold": 300, "Q_hot": 1000}'
                        Units: temperatures (K), heat (J), work (J)

        Returns:
            str: Complete Carnot efficiency analysis
        """
        try:
            data = json.loads(carnot_data) if isinstance(carnot_data, str) else carnot_data

            T_hot = float(data.get("T_hot", data.get("Th", 500)))
            T_cold = float(data.get("T_cold", data.get("Tc", 300)))
            Q_hot = data.get("Q_hot", data.get("Qh", None))

            if Q_hot is not None:
                Q_hot = float(Q_hot)

            # Carnot efficiency
            efficiency = 1 - (T_cold / T_hot)

            result = f"""
Carnot Engine Analysis:
=======================

Given:
- Hot Reservoir Temperature (Tₕ): {T_hot:.2f} K = {T_hot-273.15:.2f} °C
- Cold Reservoir Temperature (Tᶜ): {T_cold:.2f} K = {T_cold-273.15:.2f} °C

Carnot Efficiency Formula: η = 1 - Tᶜ/Tₕ

Calculation:
η = 1 - {T_cold:.2f}/{T_hot:.2f}
η = 1 - {T_cold/T_hot:.4f}
η = {efficiency:.4f} = {efficiency*100:.2f}%

Maximum Theoretical Efficiency: {efficiency*100:.2f}%
"""
            if Q_hot is not None:
                W = efficiency * Q_hot
                Q_cold = Q_hot - W
                result += f"""
With Heat Input Qₕ = {Q_hot:.2f} J:
- Work Output: W = ηQₕ = {efficiency:.4f} × {Q_hot:.2f} = {W:.2f} J
- Heat Rejected: Qᶜ = Qₕ - W = {Q_hot:.2f} - {W:.2f} = {Q_cold:.2f} J
"""

            result += """
Key Points:
- Carnot efficiency is the maximum possible efficiency
- Real engines always have lower efficiency
- Efficiency increases with larger temperature difference
- Cannot achieve 100% efficiency (would need Tᶜ = 0 K)
"""
            cycle_points = [
                {"label": "A", "v_norm": 0.25, "p_norm": 0.82},
                {"label": "B", "v_norm": 0.62, "p_norm": 0.64},
                {"label": "C", "v_norm": 0.78, "p_norm": 0.34},
                {"label": "D", "v_norm": 0.36, "p_norm": 0.20},
            ]
            frame_order = [cycle_points[0], cycle_points[1], cycle_points[2], cycle_points[3], cycle_points[0]]
            frames = []
            total_steps = 32
            for i in range(total_steps):
                span = (len(frame_order) - 1) * i / (total_steps - 1)
                seg = min(len(frame_order) - 2, int(math.floor(span)))
                frac = span - seg
                p0 = frame_order[seg]
                p1 = frame_order[seg + 1]
                v_norm = p0["v_norm"] + (p1["v_norm"] - p0["v_norm"]) * frac
                p_norm = p0["p_norm"] + (p1["p_norm"] - p0["p_norm"]) * frac
                frames.append({"step": i, "v_norm": v_norm, "p_norm": p_norm})

            payload = {
                "type": "carnot_cycle_animation",
                "title": "Carnot Cycle (Normalized PV Plane)",
                "t_hot_k": T_hot,
                "t_cold_k": T_cold,
                "efficiency": efficiency,
                "efficiency_percent": efficiency * 100.0,
                "q_hot_j": Q_hot,
                "cycle_points": cycle_points,
                "frames": frames,
            }
            return with_diagram_payload(result, payload)

        except Exception as e:
            return f"Error in Carnot efficiency calculation: {str(e)}"

    logger.info(f'{NAME} MCP Server at {host}:{port} and transport {transport}')
    if transport == "sse":
        mcp.sse_http_app.run(host=host, port=port)
    if transport == "streamable_http":
        import uvicorn
        uvicorn.run(mcp.streamable_http_app, host=host, port=port)


def main():
    """CLI entry point for the thermodynamics MCP server."""
    parser = argparse.ArgumentParser(description="Run Thermodynamics MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10107, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")

    args = parser.parse_args()

    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")


if __name__ == "__main__":
    main()
