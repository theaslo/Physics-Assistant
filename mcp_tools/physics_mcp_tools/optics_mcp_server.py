# type: ignore
"""
Optics MCP Server for Physics Assistant
Physics 202 - Geometric and Wave Optics
"""
import math
import json
import argparse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

NAME = "optics_mcp_server"
logger = get_logger(__name__)

# Physical constants
SPEED_OF_LIGHT = 2.998e8  # m/s


def serve(host, port, transport):
    """Initialize and run the Optics MCP server."""
    logger.info('Starting Optics MCP Server')
    mcp = FastMCP(NAME, stateless_http=False)

    @mcp.tool()
    async def snells_law(refraction_data: str) -> str:
        """
        Apply Snell's Law for refraction at interface between media.

        Args:
            refraction_data: JSON string with refraction parameters
                            Example: '{"n1": 1.0, "n2": 1.5, "angle1": 30}'
                            Example: '{"n1": 1.5, "n2": 1.0, "angle1": 45}'
                            Units: angles in degrees, n is dimensionless

        Returns:
            str: Complete refraction analysis including critical angle
        """
        try:
            data = json.loads(refraction_data) if isinstance(refraction_data, str) else refraction_data

            n1 = float(data.get("n1", 1.0))
            n2 = float(data.get("n2", 1.5))
            theta1 = float(data.get("angle1", data.get("theta1", 30)))

            theta1_rad = math.radians(theta1)

            result = f"""
Snell's Law Analysis:
=====================

Given:
- Medium 1 refractive index (n₁): {n1:.4f}
- Medium 2 refractive index (n₂): {n2:.4f}
- Incident angle (θ₁): {theta1:.2f}°

Snell's Law: n₁ sin(θ₁) = n₂ sin(θ₂)

"""
            # Calculate sin(theta2)
            sin_theta2 = (n1 / n2) * math.sin(theta1_rad)

            if abs(sin_theta2) <= 1:
                theta2_rad = math.asin(sin_theta2)
                theta2 = math.degrees(theta2_rad)

                result += f"Calculation:\n"
                result += f"sin(θ₂) = (n₁/n₂) × sin(θ₁)\n"
                result += f"sin(θ₂) = ({n1:.4f}/{n2:.4f}) × sin({theta1:.2f}°)\n"
                result += f"sin(θ₂) = {sin_theta2:.6f}\n"
                result += f"θ₂ = {theta2:.2f}°\n\n"

                if n1 < n2:
                    result += f"Light bends TOWARD normal (entering denser medium)\n"
                else:
                    result += f"Light bends AWAY from normal (entering less dense medium)\n"
            else:
                result += f"TOTAL INTERNAL REFLECTION occurs!\n"
                result += f"sin(θ₂) = {sin_theta2:.4f} > 1 (impossible)\n"
                result += f"Light cannot enter medium 2 at this angle.\n\n"

            # Critical angle (only exists when going from denser to less dense)
            if n1 > n2:
                theta_c = math.degrees(math.asin(n2 / n1))
                result += f"\nCritical Angle:\n"
                result += f"θc = arcsin(n₂/n₁) = arcsin({n2:.4f}/{n1:.4f}) = {theta_c:.2f}°\n"
                result += f"Total internal reflection occurs for θ₁ > {theta_c:.2f}°\n"

            result += """
Common Refractive Indices:
- Air: 1.000
- Water: 1.333
- Glass (crown): 1.52
- Glass (flint): 1.66
- Diamond: 2.42
"""
            return result

        except Exception as e:
            return f"Error in Snell's Law calculation: {str(e)}"

    @mcp.tool()
    async def lens_mirror_equation(optics_data: str) -> str:
        """
        Apply thin lens/mirror equation to find image properties.

        Args:
            optics_data: JSON string with lens/mirror parameters
                        Example: '{"focal_length": 0.1, "object_distance": 0.3}'
                        Example: '{"focal_length": -0.2, "object_distance": 0.15}'
                        Example: '{"image_distance": 0.2, "object_distance": 0.3}'
                        Units: all distances in meters
                        Convention: + for real (converging), - for virtual (diverging)

        Returns:
            str: Complete image analysis with magnification
        """
        try:
            data = json.loads(optics_data) if isinstance(optics_data, str) else optics_data

            f = data.get("focal_length", data.get("f", None))
            do = data.get("object_distance", data.get("do", data.get("d_o", None)))
            di = data.get("image_distance", data.get("di", data.get("d_i", None)))
            ho = float(data.get("object_height", data.get("ho", 1)))

            if f is not None: f = float(f)
            if do is not None: do = float(do)
            if di is not None: di = float(di)

            result = """
Thin Lens/Mirror Equation:
==========================

Lens Equation: 1/f = 1/dₒ + 1/dᵢ
Magnification: M = -dᵢ/dₒ = hᵢ/hₒ

"""
            result += "Given:\n"
            if f is not None: result += f"- Focal Length (f): {f:.4f} m = {f*100:.2f} cm\n"
            if do is not None: result += f"- Object Distance (dₒ): {do:.4f} m = {do*100:.2f} cm\n"
            if di is not None: result += f"- Image Distance (dᵢ): {di:.4f} m = {di*100:.2f} cm\n"
            result += f"- Object Height (hₒ): {ho:.4f} m\n\n"

            # Solve for unknown
            if di is None and f and do:
                di = 1 / (1/f - 1/do)
                result += f"Solving for Image Distance:\n"
                result += f"1/dᵢ = 1/f - 1/dₒ = 1/{f:.4f} - 1/{do:.4f}\n"
                result += f"dᵢ = {di:.4f} m = {di*100:.2f} cm\n\n"

            elif f is None and do and di:
                f = 1 / (1/do + 1/di)
                result += f"Solving for Focal Length:\n"
                result += f"1/f = 1/dₒ + 1/dᵢ = 1/{do:.4f} + 1/{di:.4f}\n"
                result += f"f = {f:.4f} m = {f*100:.2f} cm\n\n"

            elif do is None and f and di:
                do = 1 / (1/f - 1/di)
                result += f"Solving for Object Distance:\n"
                result += f"1/dₒ = 1/f - 1/dᵢ = 1/{f:.4f} - 1/{di:.4f}\n"
                result += f"dₒ = {do:.4f} m = {do*100:.2f} cm\n\n"

            # Calculate magnification and image properties
            if do and di:
                M = -di / do
                hi = M * ho

                result += f"Magnification:\n"
                result += f"M = -dᵢ/dₒ = -{di:.4f}/{do:.4f} = {M:.4f}\n\n"
                result += f"Image Height: hᵢ = M × hₒ = {M:.4f} × {ho:.4f} = {hi:.4f} m\n\n"

                result += f"Image Properties:\n"
                if di > 0:
                    result += f"- Location: REAL image (di > 0)\n"
                    result += f"- Side: Same side as outgoing light\n"
                else:
                    result += f"- Location: VIRTUAL image (di < 0)\n"
                    result += f"- Side: Same side as object\n"

                if M > 0:
                    result += f"- Orientation: UPRIGHT (M > 0)\n"
                else:
                    result += f"- Orientation: INVERTED (M < 0)\n"

                if abs(M) > 1:
                    result += f"- Size: MAGNIFIED (|M| > 1)\n"
                elif abs(M) < 1:
                    result += f"- Size: REDUCED (|M| < 1)\n"
                else:
                    result += f"- Size: SAME SIZE (|M| = 1)\n"

            return result

        except Exception as e:
            return f"Error in lens/mirror calculation: {str(e)}"

    @mcp.tool()
    async def diffraction_grating(diffraction_data: str) -> str:
        """
        Calculate diffraction pattern for single slit, double slit, or grating.

        Args:
            diffraction_data: JSON string with diffraction parameters
                             Example: '{"type": "double_slit", "slit_separation": 0.1e-3, "wavelength": 500e-9, "order": 1}'
                             Example: '{"type": "single_slit", "slit_width": 0.05e-3, "wavelength": 600e-9, "order": 1}'
                             Example: '{"type": "grating", "lines_per_mm": 600, "wavelength": 550e-9, "order": 2}'
                             Units: distances (m), wavelength (m)

        Returns:
            str: Complete diffraction analysis
        """
        try:
            data = json.loads(diffraction_data) if isinstance(diffraction_data, str) else diffraction_data

            diff_type = data.get("type", "double_slit").lower()
            wavelength = float(data.get("wavelength", data.get("lambda", 500e-9)))
            m = int(data.get("order", data.get("m", 1)))

            result = f"""
Diffraction Analysis:
=====================

Type: {diff_type.replace("_", " ").title()}
Wavelength: λ = {wavelength:.2e} m = {wavelength*1e9:.1f} nm
Order: m = {m}

"""
            if diff_type == "double_slit":
                d = float(data.get("slit_separation", data.get("d", 0.1e-3)))

                # Constructive interference: d sin(θ) = mλ
                sin_theta = m * wavelength / d

                result += f"Slit Separation: d = {d:.2e} m = {d*1e3:.4f} mm\n\n"
                result += f"Double-Slit Interference (constructive maxima):\n"
                result += f"Condition: d sin(θ) = mλ\n\n"

                if abs(sin_theta) <= 1:
                    theta = math.degrees(math.asin(sin_theta))
                    result += f"sin(θ) = mλ/d = {m} × {wavelength:.2e} / {d:.2e} = {sin_theta:.6f}\n"
                    result += f"θ = {theta:.4f}°\n\n"
                    result += f"Maximum #{m} occurs at θ = {theta:.4f}°\n"
                else:
                    result += f"sin(θ) = {sin_theta:.4f} > 1\n"
                    result += f"Order m = {m} is NOT visible (angle too large)\n"

            elif diff_type == "single_slit":
                a = float(data.get("slit_width", data.get("a", 0.05e-3)))

                # Destructive interference (minima): a sin(θ) = mλ
                sin_theta = m * wavelength / a

                result += f"Slit Width: a = {a:.2e} m = {a*1e3:.4f} mm\n\n"
                result += f"Single-Slit Diffraction (dark fringes/minima):\n"
                result += f"Condition: a sin(θ) = mλ (m = ±1, ±2, ...)\n\n"

                if abs(sin_theta) <= 1:
                    theta = math.degrees(math.asin(sin_theta))
                    result += f"sin(θ) = mλ/a = {m} × {wavelength:.2e} / {a:.2e} = {sin_theta:.6f}\n"
                    result += f"θ = {theta:.4f}°\n\n"
                    result += f"Minimum #{m} occurs at θ = ±{theta:.4f}°\n"
                else:
                    result += f"sin(θ) = {sin_theta:.4f} > 1\n"
                    result += f"Order m = {m} minimum is NOT visible\n"

            elif diff_type == "grating":
                lines_per_mm = float(data.get("lines_per_mm", data.get("N", 600)))
                d = 1e-3 / lines_per_mm  # Convert to meters

                sin_theta = m * wavelength / d

                result += f"Grating: {lines_per_mm:.0f} lines/mm\n"
                result += f"Line Spacing: d = {d:.2e} m\n\n"
                result += f"Grating Equation (principal maxima):\n"
                result += f"Condition: d sin(θ) = mλ\n\n"

                if abs(sin_theta) <= 1:
                    theta = math.degrees(math.asin(sin_theta))
                    result += f"sin(θ) = mλ/d = {m} × {wavelength:.2e} / {d:.2e} = {sin_theta:.6f}\n"
                    result += f"θ = {theta:.4f}°\n\n"
                    result += f"Order m = {m} maximum at θ = {theta:.4f}°\n"

                    # Calculate maximum observable order
                    m_max = int(d / wavelength)
                    result += f"\nMaximum Observable Order: m_max = d/λ = {m_max}\n"
                else:
                    result += f"sin(θ) = {sin_theta:.4f} > 1\n"
                    result += f"Order m = {m} is NOT visible\n"

            return result

        except Exception as e:
            return f"Error in diffraction calculation: {str(e)}"

    @mcp.tool()
    async def thin_film_interference(film_data: str) -> str:
        """
        Analyze thin film interference (soap bubbles, oil slicks, coatings).

        Args:
            film_data: JSON string with thin film parameters
                      Example: '{"thickness": 200e-9, "n_film": 1.33, "wavelength": 550e-9}'
                      Example: '{"thickness": 100e-9, "n_film": 1.38, "n_substrate": 1.5}'
                      Units: thickness (m), wavelength (m)

        Returns:
            str: Complete thin film interference analysis
        """
        try:
            data = json.loads(film_data) if isinstance(film_data, str) else film_data

            t = float(data.get("thickness", data.get("t", 200e-9)))
            n_film = float(data.get("n_film", data.get("n", 1.33)))
            n_air = float(data.get("n_air", 1.0))
            n_substrate = float(data.get("n_substrate", data.get("n_sub", None)))

            result = f"""
Thin Film Interference Analysis:
================================

Given:
- Film Thickness (t): {t:.2e} m = {t*1e9:.1f} nm
- Film Refractive Index (n): {n_film:.4f}
- Air Refractive Index: {n_air:.4f}
"""
            if n_substrate:
                result += f"- Substrate Refractive Index: {n_substrate:.4f}\n"

            # Optical path difference
            optical_path = 2 * n_film * t

            result += f"""
Optical Path Difference: 2nt = 2 × {n_film:.4f} × {t:.2e} = {optical_path:.2e} m

Phase Changes on Reflection:
- Air→Film interface: {"π phase shift (n_film > n_air)" if n_film > n_air else "No phase shift"}
"""
            if n_substrate:
                result += f"- Film→Substrate: {"π phase shift (n_sub > n_film)" if n_substrate > n_film else "No phase shift"}\n"

            # Determine interference conditions
            phase_shift_count = 0
            if n_film > n_air:
                phase_shift_count += 1
            if n_substrate and n_substrate > n_film:
                phase_shift_count += 1

            result += f"\nInterference Conditions:\n"

            if phase_shift_count % 2 == 0:
                # Even phase shifts (0 or 2) - same as no net phase shift
                result += "Net effect: No additional phase shift\n"
                result += "Constructive: 2nt = mλ (m = 0, 1, 2, ...)\n"
                result += "Destructive: 2nt = (m + ½)λ\n"
            else:
                # Odd phase shift (1) - net π phase shift
                result += "Net effect: π phase shift (half wavelength)\n"
                result += "Constructive: 2nt = (m + ½)λ (m = 0, 1, 2, ...)\n"
                result += "Destructive: 2nt = mλ\n"

            if "wavelength" in data:
                wavelength = float(data.get("wavelength"))
                m_construct = optical_path / wavelength
                result += f"\nFor λ = {wavelength:.2e} m = {wavelength*1e9:.1f} nm:\n"
                result += f"2nt/λ = {m_construct:.4f}\n"

                if phase_shift_count % 2 == 0:
                    if abs(m_construct - round(m_construct)) < 0.1:
                        result += f"→ Near CONSTRUCTIVE interference (m ≈ {round(m_construct)})\n"
                    elif abs(m_construct - round(m_construct) - 0.5) < 0.1:
                        result += f"→ Near DESTRUCTIVE interference\n"
                else:
                    if abs(m_construct - round(m_construct) - 0.5) < 0.1:
                        result += f"→ Near CONSTRUCTIVE interference\n"
                    elif abs(m_construct - round(m_construct)) < 0.1:
                        result += f"→ Near DESTRUCTIVE interference (m ≈ {round(m_construct)})\n"

            result += """
Applications:
- Anti-reflection coatings (cameras, glasses)
- Soap bubbles and oil slicks (colorful patterns)
- Optical filters
- Semiconductor inspection
"""
            return result

        except Exception as e:
            return f"Error in thin film interference calculation: {str(e)}"

    @mcp.tool()
    async def optical_power_diopters(power_data: str) -> str:
        """
        Calculate lens power in diopters and combined lens systems.

        Args:
            power_data: JSON string with lens parameters
                       Example: '{"focal_length": 0.5}'
                       Example: '{"power": 2.5}'
                       Example: '{"lenses": [{"f": 0.2}, {"f": -0.5}]}'
                       Units: focal_length (m), power (diopters)

        Returns:
            str: Complete optical power analysis
        """
        try:
            data = json.loads(power_data) if isinstance(power_data, str) else power_data

            result = """
Optical Power Analysis:
=======================

Power (Diopters): P = 1/f (where f is in meters)

"""
            if "focal_length" in data or "f" in data:
                f = float(data.get("focal_length", data.get("f")))
                P = 1 / f

                result += f"Given Focal Length: f = {f:.4f} m = {f*100:.2f} cm\n\n"
                result += f"Power: P = 1/f = 1/{f:.4f} = {P:.4f} D (diopters)\n\n"

                if f > 0:
                    result += "Lens Type: CONVERGING (positive power, positive focal length)\n"
                else:
                    result += "Lens Type: DIVERGING (negative power, negative focal length)\n"

            elif "power" in data or "P" in data:
                P = float(data.get("power", data.get("P")))
                f = 1 / P

                result += f"Given Power: P = {P:.4f} D (diopters)\n\n"
                result += f"Focal Length: f = 1/P = 1/{P:.4f} = {f:.4f} m = {f*100:.2f} cm\n"

            elif "lenses" in data:
                lenses = data["lenses"]
                total_power = 0

                result += "Combined Lens System (thin lenses in contact):\n\n"

                for i, lens in enumerate(lenses, 1):
                    f = float(lens.get("focal_length", lens.get("f")))
                    P = 1 / f
                    total_power += P
                    result += f"Lens {i}: f = {f:.4f} m, P = {P:.4f} D\n"

                f_combined = 1 / total_power

                result += f"\nCombined Power: P_total = P₁ + P₂ + ... = {total_power:.4f} D\n"
                result += f"Combined Focal Length: f_total = 1/P_total = {f_combined:.4f} m\n"

            result += """
Common Vision Corrections:
- Myopia (nearsighted): Needs diverging lens (negative power)
- Hyperopia (farsighted): Needs converging lens (positive power)
- Typical reading glasses: +1 to +3 D
- Strong nearsighted correction: -3 to -8 D
"""
            return result

        except Exception as e:
            return f"Error in optical power calculation: {str(e)}"

    logger.info(f'{NAME} MCP Server at {host}:{port} and transport {transport}')
    if transport == "sse":
        mcp.sse_http_app.run(host=host, port=port)
    if transport == "streamable_http":
        import uvicorn
        uvicorn.run(mcp.streamable_http_app, host=host, port=port)


def main():
    """CLI entry point for the optics MCP server."""
    parser = argparse.ArgumentParser(description="Run Optics MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10110, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")

    args = parser.parse_args()

    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")


if __name__ == "__main__":
    main()
