# type: ignore
"""
Waves and Sound MCP Server for Physics Assistant
Physics 102 - Wave Motion, Sound, and Acoustics
"""
import math
import json
import argparse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

NAME = "waves_mcp_server"
logger = get_logger(__name__)

# Physical constants
SPEED_OF_SOUND_AIR = 343  # m/s at 20°C
SPEED_OF_LIGHT = 3e8  # m/s
I_REFERENCE = 1e-12  # W/m² - Reference intensity for decibels


def serve(host, port, transport):
    """Initialize and run the Waves MCP server."""
    logger.info('Starting Waves MCP Server')
    mcp = FastMCP(NAME, stateless_http=False)

    @mcp.tool()
    async def wave_equation(wave_data: str) -> str:
        """
        Solve wave equation problems using v = fλ.

        Args:
            wave_data: JSON string with wave parameters. Provide 2 of 3: velocity, frequency, wavelength
                      Example: '{"frequency": 440, "wavelength": 0.78}'
                      Example: '{"velocity": 343, "frequency": 256}'
                      Units: velocity (m/s), frequency (Hz), wavelength (m)

        Returns:
            str: Complete wave analysis
        """
        try:
            data = json.loads(wave_data) if isinstance(wave_data, str) else wave_data

            v = data.get("velocity", data.get("v", data.get("speed", None)))
            f = data.get("frequency", data.get("f", None))
            wavelength = data.get("wavelength", data.get("lambda", data.get("λ", None)))

            if v is not None: v = float(v)
            if f is not None: f = float(f)
            if wavelength is not None: wavelength = float(wavelength)

            result = """
Wave Equation Analysis (v = fλ):
================================

"""
            result += "Given:\n"
            if v is not None: result += f"- Wave Velocity (v): {v:.2f} m/s\n"
            if f is not None: result += f"- Frequency (f): {f:.2f} Hz\n"
            if wavelength is not None: result += f"- Wavelength (λ): {wavelength:.4f} m\n"

            result += "\nWave Equation: v = fλ\n\n"

            if v is None and f and wavelength:
                v = f * wavelength
                result += f"Solving for Velocity:\n"
                result += f"v = fλ = {f:.2f} × {wavelength:.4f}\n"
                result += f"v = {v:.2f} m/s\n"

            elif f is None and v and wavelength:
                f = v / wavelength
                result += f"Solving for Frequency:\n"
                result += f"f = v/λ = {v:.2f} / {wavelength:.4f}\n"
                result += f"f = {f:.2f} Hz\n"

            elif wavelength is None and v and f:
                wavelength = v / f
                result += f"Solving for Wavelength:\n"
                result += f"λ = v/f = {v:.2f} / {f:.2f}\n"
                result += f"λ = {wavelength:.4f} m\n"

            # Additional calculations
            if f is not None:
                T = 1 / f
                omega = 2 * math.pi * f
                result += f"""
Additional Wave Properties:
- Period (T = 1/f): {T:.6f} s
- Angular Frequency (ω = 2πf): {omega:.2f} rad/s
"""
            if wavelength is not None:
                k = 2 * math.pi / wavelength
                result += f"- Wave Number (k = 2π/λ): {k:.4f} rad/m\n"

            return result

        except Exception as e:
            return f"Error in wave equation calculation: {str(e)}"

    @mcp.tool()
    async def doppler_effect(doppler_data: str) -> str:
        """
        Calculate observed frequency due to Doppler effect.

        Args:
            doppler_data: JSON string with Doppler parameters
                         Example: '{"source_freq": 440, "source_velocity": 30, "observer_velocity": 0, "approaching": true}'
                         Example: '{"source_freq": 1000, "source_velocity": 0, "observer_velocity": 20}'
                         Units: frequencies (Hz), velocities (m/s)
                         Note: Use medium_velocity for sound speed (default 343 m/s)

        Returns:
            str: Complete Doppler effect analysis
        """
        try:
            data = json.loads(doppler_data) if isinstance(doppler_data, str) else doppler_data

            f_source = float(data.get("source_freq", data.get("f_s", data.get("fs", 440))))
            v_source = float(data.get("source_velocity", data.get("v_s", data.get("vs", 0))))
            v_observer = float(data.get("observer_velocity", data.get("v_o", data.get("vo", 0))))
            v_medium = float(data.get("medium_velocity", data.get("v", SPEED_OF_SOUND_AIR)))
            approaching = data.get("approaching", True)

            result = f"""
Doppler Effect Analysis:
========================

Given:
- Source Frequency (f_s): {f_source:.2f} Hz
- Source Velocity (v_s): {v_source:.2f} m/s
- Observer Velocity (v_o): {v_observer:.2f} m/s
- Wave Speed in Medium (v): {v_medium:.2f} m/s
- Motion: {"Approaching" if approaching else "Receding"}

Doppler Equation: f' = f × (v ± v_o) / (v ∓ v_s)
  - Upper signs when approaching
  - Lower signs when receding

"""
            if approaching:
                f_observed = f_source * (v_medium + v_observer) / (v_medium - v_source)
                result += f"Approaching - Use: f' = f × (v + v_o) / (v - v_s)\n"
                result += f"f' = {f_source:.2f} × ({v_medium:.2f} + {v_observer:.2f}) / ({v_medium:.2f} - {v_source:.2f})\n"
            else:
                f_observed = f_source * (v_medium - v_observer) / (v_medium + v_source)
                result += f"Receding - Use: f' = f × (v - v_o) / (v + v_s)\n"
                result += f"f' = {f_source:.2f} × ({v_medium:.2f} - {v_observer:.2f}) / ({v_medium:.2f} + {v_source:.2f})\n"

            freq_shift = f_observed - f_source
            percent_shift = (freq_shift / f_source) * 100

            result += f"""
f' = {f_observed:.2f} Hz

Results:
- Observed Frequency: {f_observed:.2f} Hz
- Frequency Shift: {freq_shift:+.2f} Hz ({percent_shift:+.2f}%)
- Pitch Change: {"Higher (blue shift)" if freq_shift > 0 else "Lower (red shift)" if freq_shift < 0 else "No change"}

Applications:
- Police radar guns
- Medical ultrasound
- Astronomical redshift
- Weather radar
"""
            return result

        except Exception as e:
            return f"Error in Doppler effect calculation: {str(e)}"

    @mcp.tool()
    async def sound_intensity_decibels(sound_data: str) -> str:
        """
        Calculate sound intensity level in decibels or convert between intensity and dB.

        Args:
            sound_data: JSON string with sound parameters
                       Example: '{"intensity": 1e-6}' - Calculate dB from intensity
                       Example: '{"decibels": 85}' - Calculate intensity from dB
                       Example: '{"power": 100, "distance": 10}' - Calculate from source
                       Units: intensity (W/m²), decibels (dB), power (W), distance (m)

        Returns:
            str: Complete sound intensity analysis
        """
        try:
            data = json.loads(sound_data) if isinstance(sound_data, str) else sound_data

            result = """
Sound Intensity Analysis:
=========================

Reference Intensity: I₀ = 10⁻¹² W/m² (threshold of hearing)

"""
            if "intensity" in data or "I" in data:
                I = float(data.get("intensity", data.get("I")))
                dB = 10 * math.log10(I / I_REFERENCE)

                result += f"Given Intensity: I = {I:.2e} W/m²\n\n"
                result += f"Decibel Formula: β = 10 log₁₀(I/I₀)\n"
                result += f"β = 10 × log₁₀({I:.2e} / {I_REFERENCE:.0e})\n"
                result += f"β = 10 × {math.log10(I / I_REFERENCE):.2f}\n"
                result += f"β = {dB:.1f} dB\n"

            elif "decibels" in data or "dB" in data or "beta" in data:
                dB = float(data.get("decibels", data.get("dB", data.get("beta"))))
                I = I_REFERENCE * (10 ** (dB / 10))

                result += f"Given Sound Level: β = {dB:.1f} dB\n\n"
                result += f"Intensity Formula: I = I₀ × 10^(β/10)\n"
                result += f"I = {I_REFERENCE:.0e} × 10^({dB:.1f}/10)\n"
                result += f"I = {I:.2e} W/m²\n"

            elif "power" in data and "distance" in data:
                P = float(data.get("power", data.get("P")))
                r = float(data.get("distance", data.get("r")))
                I = P / (4 * math.pi * r**2)
                dB = 10 * math.log10(I / I_REFERENCE)

                result += f"Point Source Analysis:\n"
                result += f"- Source Power: P = {P:.2f} W\n"
                result += f"- Distance: r = {r:.2f} m\n\n"
                result += f"Intensity: I = P/(4πr²) = {P:.2f}/(4π × {r:.2f}²)\n"
                result += f"I = {I:.2e} W/m²\n\n"
                result += f"Sound Level: β = 10 log₁₀(I/I₀) = {dB:.1f} dB\n"

            result += """
Common Sound Levels:
- Threshold of hearing: 0 dB
- Whisper: 20 dB
- Normal conversation: 60 dB
- Busy traffic: 80 dB
- Rock concert: 110 dB
- Threshold of pain: 130 dB

Note: Every 10 dB increase = 10× intensity = ~2× perceived loudness
"""
            return result

        except Exception as e:
            return f"Error in sound intensity calculation: {str(e)}"

    @mcp.tool()
    async def standing_waves(standing_wave_data: str) -> str:
        """
        Calculate standing wave frequencies and patterns.

        Args:
            standing_wave_data: JSON string with standing wave parameters
                               Example: '{"type": "string", "length": 0.65, "velocity": 343}'
                               Example: '{"type": "pipe_open", "length": 1.0}'
                               Example: '{"type": "pipe_closed", "length": 0.5}'
                               Types: "string" (fixed both ends), "pipe_open" (open both ends),
                                      "pipe_closed" (closed one end)

        Returns:
            str: Complete standing wave analysis with harmonics
        """
        try:
            data = json.loads(standing_wave_data) if isinstance(standing_wave_data, str) else standing_data

            wave_type = data.get("type", "string").lower()
            L = float(data.get("length", data.get("L", 1)))
            v = float(data.get("velocity", data.get("v", SPEED_OF_SOUND_AIR)))

            result = f"""
Standing Wave Analysis:
=======================

System Type: {wave_type.replace("_", " ").title()}
Length: L = {L:.4f} m
Wave Velocity: v = {v:.2f} m/s

"""
            if wave_type == "string" or wave_type == "pipe_open":
                # Harmonics at n = 1, 2, 3, ...
                fundamental = v / (2 * L)
                result += """Boundary Conditions: Nodes at both ends (or antinodes for open pipe)
Wavelength Formula: λₙ = 2L/n
Frequency Formula: fₙ = nv/(2L) = n × f₁

"""
                result += f"Fundamental Frequency (n=1):\n"
                result += f"f₁ = v/(2L) = {v:.2f}/(2 × {L:.4f}) = {fundamental:.2f} Hz\n\n"
                result += "Harmonic Series:\n"
                for n in range(1, 6):
                    fn = n * fundamental
                    wavelength = 2 * L / n
                    result += f"  n={n}: f_{n} = {fn:.2f} Hz, λ_{n} = {wavelength:.4f} m\n"

            elif wave_type == "pipe_closed":
                # Only odd harmonics: n = 1, 3, 5, ...
                fundamental = v / (4 * L)
                result += """Boundary Conditions: Node at closed end, antinode at open end
Wavelength Formula: λₙ = 4L/n (n = 1, 3, 5, ...)
Frequency Formula: fₙ = nv/(4L) = n × f₁ (odd n only)

"""
                result += f"Fundamental Frequency (n=1):\n"
                result += f"f₁ = v/(4L) = {v:.2f}/(4 × {L:.4f}) = {fundamental:.2f} Hz\n\n"
                result += "Harmonic Series (odd harmonics only):\n"
                for n in [1, 3, 5, 7, 9]:
                    fn = n * fundamental
                    wavelength = 4 * L / n
                    result += f"  n={n}: f_{n} = {fn:.2f} Hz, λ_{n} = {wavelength:.4f} m\n"

            result += """
Applications:
- Musical instruments (strings, wind instruments)
- Organ pipes
- Acoustic resonators
- Room acoustics
"""
            return result

        except Exception as e:
            return f"Error in standing wave calculation: {str(e)}"

    @mcp.tool()
    async def wave_interference(interference_data: str) -> str:
        """
        Analyze wave interference patterns (constructive and destructive).

        Args:
            interference_data: JSON string with interference parameters
                              Example: '{"wavelength": 0.02, "path_difference": 0.03}'
                              Example: '{"wavelength": 500e-9, "slit_separation": 0.1e-3, "angle": 0.005}'
                              Units: wavelength (m), path_difference (m), angle (rad)

        Returns:
            str: Complete interference analysis
        """
        try:
            data = json.loads(interference_data) if isinstance(interference_data, str) else interference_data

            wavelength = float(data.get("wavelength", data.get("lambda", 500e-9)))

            result = f"""
Wave Interference Analysis:
===========================

Wavelength: λ = {wavelength:.2e} m

Interference Conditions:
- Constructive: Δ = mλ (m = 0, ±1, ±2, ...)
- Destructive: Δ = (m + ½)λ (m = 0, ±1, ±2, ...)

"""
            if "path_difference" in data or "delta" in data:
                delta = float(data.get("path_difference", data.get("delta")))
                m = delta / wavelength

                result += f"Given Path Difference: Δ = {delta:.2e} m\n"
                result += f"Δ/λ = {delta:.2e} / {wavelength:.2e} = {m:.4f}\n\n"

                if abs(m - round(m)) < 0.01:
                    result += f"Result: CONSTRUCTIVE interference (m ≈ {round(m)})\n"
                    result += "Waves add in phase - maximum amplitude\n"
                elif abs(m - round(m) - 0.5) < 0.01 or abs(m - round(m) + 0.5) < 0.01:
                    result += f"Result: DESTRUCTIVE interference (m + ½ ≈ {m:.1f})\n"
                    result += "Waves add out of phase - zero amplitude\n"
                else:
                    result += f"Result: PARTIAL interference\n"
                    result += f"Not at a maximum or minimum - intermediate amplitude\n"

            if "slit_separation" in data and "angle" in data:
                d = float(data.get("slit_separation"))
                theta = float(data.get("angle"))

                # Path difference for double slit
                delta = d * math.sin(theta)
                m = delta / wavelength

                result += f"\nDouble-Slit Analysis:\n"
                result += f"Slit Separation: d = {d:.2e} m\n"
                result += f"Angle: θ = {theta:.6f} rad = {math.degrees(theta):.4f}°\n"
                result += f"Path Difference: Δ = d sin(θ) = {delta:.2e} m\n"
                result += f"Order: m = Δ/λ = {m:.2f}\n"

            return result

        except Exception as e:
            return f"Error in interference analysis: {str(e)}"

    logger.info(f'{NAME} MCP Server at {host}:{port} and transport {transport}')
    if transport == "sse":
        mcp.sse_http_app.run(host=host, port=port)
    if transport == "streamable_http":
        import uvicorn
        uvicorn.run(mcp.streamable_http_app, host=host, port=port)


def main():
    """CLI entry point for the waves MCP server."""
    parser = argparse.ArgumentParser(description="Run Waves MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10108, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")

    args = parser.parse_args()

    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")


if __name__ == "__main__":
    main()
