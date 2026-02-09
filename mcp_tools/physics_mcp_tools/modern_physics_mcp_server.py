# type: ignore
"""
Modern Physics MCP Server for Physics Assistant
Physics 202 - Relativity, Quantum Mechanics, Nuclear Physics
"""
import math
import json
import argparse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

NAME = "modern_physics_mcp_server"
logger = get_logger(__name__)

# Physical constants
C = 2.998e8  # m/s - Speed of light
H = 6.626e-34  # J·s - Planck's constant
H_BAR = H / (2 * math.pi)  # Reduced Planck's constant
H_EV = 4.136e-15  # eV·s - Planck's constant in eV
E_CHARGE = 1.602e-19  # C - Elementary charge
M_ELECTRON = 9.109e-31  # kg - Electron mass
M_PROTON = 1.673e-27  # kg - Proton mass
M_NEUTRON = 1.675e-27  # kg - Neutron mass
K_BOLTZMANN = 1.381e-23  # J/K - Boltzmann constant
RYDBERG = 13.6  # eV - Rydberg energy


def serve(host, port, transport):
    """Initialize and run the Modern Physics MCP server."""
    logger.info('Starting Modern Physics MCP Server')
    mcp = FastMCP(NAME, stateless_http=False)

    @mcp.tool()
    async def time_dilation(relativity_data: str) -> str:
        """
        Calculate relativistic time dilation.

        Args:
            relativity_data: JSON string with relativity parameters
                            Example: '{"proper_time": 1, "velocity": 0.8}'
                            Example: '{"dilated_time": 5, "velocity": 0.9}'
                            velocity as fraction of c (0 to 1) or in m/s

        Returns:
            str: Complete time dilation analysis
        """
        try:
            data = json.loads(relativity_data) if isinstance(relativity_data, str) else relativity_data

            v = float(data.get("velocity", data.get("v", 0.5)))

            # If velocity > 1, assume it's in m/s
            if v > 1:
                beta = v / C
            else:
                beta = v
                v = beta * C

            if beta >= 1:
                return "Error: Velocity must be less than speed of light (β < 1)"

            gamma = 1 / math.sqrt(1 - beta**2)

            result = f"""
Special Relativity - Time Dilation:
===================================

Given:
- Velocity: v = {beta:.6f}c = {v:.4e} m/s
- Speed of light: c = {C:.3e} m/s

Lorentz Factor: γ = 1/√(1 - v²/c²)
γ = 1/√(1 - {beta:.6f}²) = 1/√{1 - beta**2:.6f}
γ = {gamma:.6f}

"""
            if "proper_time" in data or "t0" in data:
                t0 = float(data.get("proper_time", data.get("t0")))
                t = gamma * t0

                result += f"Proper Time (stationary observer): τ = {t0:.6f} s\n\n"
                result += f"Time Dilation Formula: t = γτ\n"
                result += f"Dilated Time: t = {gamma:.6f} × {t0:.6f} = {t:.6f} s\n\n"
                result += f"Time passes {gamma:.4f}× slower for moving observer\n"

            elif "dilated_time" in data or "t" in data:
                t = float(data.get("dilated_time", data.get("t")))
                t0 = t / gamma

                result += f"Dilated Time (moving frame): t = {t:.6f} s\n\n"
                result += f"Formula: τ = t/γ\n"
                result += f"Proper Time: τ = {t:.6f} / {gamma:.6f} = {t0:.6f} s\n"

            result += f"""
Physical Interpretation:
- Proper time τ: Time measured in the rest frame of the clock
- Dilated time t: Time measured by observer seeing clock move
- Moving clocks run slow by factor γ
- At v = 0.866c, time runs at half speed (γ = 2)
- At v = 0.99c, time runs at 1/7 speed (γ ≈ 7)
"""
            return result

        except Exception as e:
            return f"Error in time dilation calculation: {str(e)}"

    @mcp.tool()
    async def length_contraction(contraction_data: str) -> str:
        """
        Calculate relativistic length contraction.

        Args:
            contraction_data: JSON string with parameters
                             Example: '{"proper_length": 100, "velocity": 0.6}'
                             Example: '{"contracted_length": 50, "velocity": 0.8}'
                             velocity as fraction of c or in m/s

        Returns:
            str: Complete length contraction analysis
        """
        try:
            data = json.loads(contraction_data) if isinstance(contraction_data, str) else contraction_data

            v = float(data.get("velocity", data.get("v", 0.5)))

            if v > 1:
                beta = v / C
            else:
                beta = v

            if beta >= 1:
                return "Error: Velocity must be less than speed of light (β < 1)"

            gamma = 1 / math.sqrt(1 - beta**2)

            result = f"""
Special Relativity - Length Contraction:
========================================

Given:
- Velocity: v = {beta:.6f}c
- Lorentz Factor: γ = {gamma:.6f}

"""
            if "proper_length" in data or "L0" in data:
                L0 = float(data.get("proper_length", data.get("L0")))
                L = L0 / gamma

                result += f"Proper Length: L₀ = {L0:.4f} m\n\n"
                result += f"Length Contraction Formula: L = L₀/γ\n"
                result += f"Contracted Length: L = {L0:.4f} / {gamma:.6f} = {L:.4f} m\n\n"
                result += f"Length contracts to {100/gamma:.2f}% of proper length\n"

            elif "contracted_length" in data or "L" in data:
                L = float(data.get("contracted_length", data.get("L")))
                L0 = L * gamma

                result += f"Contracted Length: L = {L:.4f} m\n\n"
                result += f"Formula: L₀ = Lγ\n"
                result += f"Proper Length: L₀ = {L:.4f} × {gamma:.6f} = {L0:.4f} m\n"

            result += """
Key Points:
- Contraction only in direction of motion
- Proper length L₀ is measured in object's rest frame
- Moving objects appear shorter to stationary observers
- Perpendicular dimensions unchanged
"""
            return result

        except Exception as e:
            return f"Error in length contraction calculation: {str(e)}"

    @mcp.tool()
    async def relativistic_energy_momentum(energy_data: str) -> str:
        """
        Calculate relativistic energy and momentum.

        Args:
            energy_data: JSON string with parameters
                        Example: '{"rest_mass": 9.109e-31, "velocity": 0.9}'
                        Example: '{"rest_mass_eV": 0.511e6, "kinetic_energy_eV": 1e6}'
                        Units: mass (kg or eV/c²), energy (J or eV), velocity as β or m/s

        Returns:
            str: Complete relativistic energy-momentum analysis
        """
        try:
            data = json.loads(energy_data) if isinstance(data, str) else data

            result = """
Relativistic Energy-Momentum:
=============================

Key Equations:
- E² = (pc)² + (m₀c²)²
- E = γm₀c²
- KE = (γ-1)m₀c²
- p = γm₀v

"""
            # Get rest mass
            if "rest_mass" in data or "m0" in data:
                m0 = float(data.get("rest_mass", data.get("m0")))
                E0 = m0 * C**2  # Rest energy in J
                E0_eV = E0 / E_CHARGE
            elif "rest_mass_eV" in data:
                E0_eV = float(data.get("rest_mass_eV"))
                E0 = E0_eV * E_CHARGE
                m0 = E0 / C**2
            else:
                m0 = M_ELECTRON
                E0 = m0 * C**2
                E0_eV = E0 / E_CHARGE

            result += f"Rest Mass: m₀ = {m0:.4e} kg\n"
            result += f"Rest Energy: E₀ = m₀c² = {E0:.4e} J = {E0_eV:.4e} eV\n\n"

            if "velocity" in data or "v" in data:
                v = float(data.get("velocity", data.get("v")))
                beta = v / C if v > 1 else v

                if beta >= 1:
                    return "Error: Velocity must be less than c"

                gamma = 1 / math.sqrt(1 - beta**2)

                E_total = gamma * m0 * C**2
                KE = E_total - E0
                p = gamma * m0 * beta * C

                result += f"Velocity: v = {beta:.6f}c\n"
                result += f"Lorentz Factor: γ = {gamma:.6f}\n\n"
                result += f"Total Energy: E = γm₀c² = {E_total:.4e} J = {E_total/E_CHARGE:.4e} eV\n"
                result += f"Kinetic Energy: KE = (γ-1)m₀c² = {KE:.4e} J = {KE/E_CHARGE:.4e} eV\n"
                result += f"Momentum: p = γm₀v = {p:.4e} kg·m/s\n"

            elif "kinetic_energy" in data or "kinetic_energy_eV" in data:
                if "kinetic_energy" in data:
                    KE = float(data.get("kinetic_energy"))
                else:
                    KE = float(data.get("kinetic_energy_eV")) * E_CHARGE

                E_total = E0 + KE
                gamma = E_total / E0
                beta = math.sqrt(1 - 1/gamma**2)

                result += f"Kinetic Energy: KE = {KE:.4e} J = {KE/E_CHARGE:.4e} eV\n"
                result += f"Total Energy: E = E₀ + KE = {E_total:.4e} J\n"
                result += f"Lorentz Factor: γ = E/E₀ = {gamma:.6f}\n"
                result += f"Velocity: β = √(1-1/γ²) = {beta:.6f}c\n"

            return result

        except Exception as e:
            return f"Error in relativistic energy calculation: {str(e)}"

    @mcp.tool()
    async def photoelectric_effect(photo_data: str) -> str:
        """
        Analyze photoelectric effect using Einstein's equation.

        Args:
            photo_data: JSON string with photoelectric parameters
                       Example: '{"wavelength": 400e-9, "work_function_eV": 2.3}'
                       Example: '{"frequency": 6e14, "work_function_eV": 1.8}'
                       Example: '{"photon_energy_eV": 4.5, "work_function_eV": 2.0}'

        Returns:
            str: Complete photoelectric effect analysis
        """
        try:
            data = json.loads(photo_data) if isinstance(photo_data, str) else photo_data

            phi_eV = float(data.get("work_function_eV", data.get("phi", 2.0)))
            phi_J = phi_eV * E_CHARGE

            result = f"""
Photoelectric Effect Analysis:
==============================

Einstein's Equation: E_photon = φ + KE_max
                    hf = φ + ½m_e v²_max

Work Function: φ = {phi_eV:.4f} eV = {phi_J:.4e} J

"""
            # Get photon energy
            if "wavelength" in data or "lambda" in data:
                wavelength = float(data.get("wavelength", data.get("lambda")))
                f = C / wavelength
                E_photon_J = H * f
                E_photon_eV = E_photon_J / E_CHARGE

                result += f"Photon Wavelength: λ = {wavelength:.2e} m = {wavelength*1e9:.1f} nm\n"
                result += f"Photon Frequency: f = c/λ = {f:.4e} Hz\n"

            elif "frequency" in data or "f" in data:
                f = float(data.get("frequency", data.get("f")))
                wavelength = C / f
                E_photon_J = H * f
                E_photon_eV = E_photon_J / E_CHARGE

                result += f"Photon Frequency: f = {f:.4e} Hz\n"
                result += f"Photon Wavelength: λ = c/f = {wavelength:.2e} m = {wavelength*1e9:.1f} nm\n"

            elif "photon_energy_eV" in data:
                E_photon_eV = float(data.get("photon_energy_eV"))
                E_photon_J = E_photon_eV * E_CHARGE
                f = E_photon_J / H
                wavelength = C / f

                result += f"Photon Energy: E = {E_photon_eV:.4f} eV\n"
                result += f"Photon Frequency: f = E/h = {f:.4e} Hz\n"
                result += f"Photon Wavelength: λ = {wavelength:.2e} m = {wavelength*1e9:.1f} nm\n"

            result += f"Photon Energy: E = hf = {E_photon_eV:.4f} eV = {E_photon_J:.4e} J\n\n"

            # Calculate KE_max
            KE_max_eV = E_photon_eV - phi_eV

            if KE_max_eV > 0:
                KE_max_J = KE_max_eV * E_CHARGE
                v_max = math.sqrt(2 * KE_max_J / M_ELECTRON)

                result += f"Maximum Kinetic Energy:\n"
                result += f"KE_max = hf - φ = {E_photon_eV:.4f} - {phi_eV:.4f} = {KE_max_eV:.4f} eV\n"
                result += f"KE_max = {KE_max_J:.4e} J\n\n"
                result += f"Maximum Electron Speed:\n"
                result += f"v_max = √(2·KE_max/m_e) = {v_max:.4e} m/s = {v_max/C:.4f}c\n\n"
                result += f"✓ Photoelectric effect OCCURS (E_photon > φ)\n"
            else:
                result += f"KE_max = hf - φ = {E_photon_eV:.4f} - {phi_eV:.4f} = {KE_max_eV:.4f} eV\n\n"
                result += f"✗ NO photoelectric effect (E_photon < φ)\n"
                result += f"Photon energy insufficient to overcome work function.\n"

            # Threshold wavelength
            lambda_threshold = H * C / phi_J
            result += f"\nThreshold Wavelength: λ_0 = hc/φ = {lambda_threshold:.2e} m = {lambda_threshold*1e9:.1f} nm\n"

            return result

        except Exception as e:
            return f"Error in photoelectric effect calculation: {str(e)}"

    @mcp.tool()
    async def de_broglie_wavelength(matter_wave_data: str) -> str:
        """
        Calculate de Broglie wavelength for matter waves.

        Args:
            matter_wave_data: JSON string with particle parameters
                             Example: '{"mass": 9.109e-31, "velocity": 1e6}'
                             Example: '{"particle": "electron", "kinetic_energy_eV": 100}'
                             Example: '{"particle": "proton", "momentum": 1e-22}'

        Returns:
            str: Complete de Broglie wavelength analysis
        """
        try:
            data = json.loads(matter_wave_data) if isinstance(matter_wave_data, str) else matter_wave_data

            # Get particle mass
            particle = data.get("particle", "electron").lower()
            if particle == "electron":
                m = M_ELECTRON
            elif particle == "proton":
                m = M_PROTON
            elif particle == "neutron":
                m = M_NEUTRON
            else:
                m = float(data.get("mass", data.get("m", M_ELECTRON)))

            result = f"""
de Broglie Wavelength Analysis:
===============================

de Broglie Relation: λ = h/p = h/(mv)

Particle: {particle.title()}
Mass: m = {m:.4e} kg

"""
            if "velocity" in data or "v" in data:
                v = float(data.get("velocity", data.get("v")))
                p = m * v
                wavelength = H / p

                result += f"Velocity: v = {v:.4e} m/s\n"
                result += f"Momentum: p = mv = {p:.4e} kg·m/s\n\n"

            elif "momentum" in data or "p" in data:
                p = float(data.get("momentum", data.get("p")))
                v = p / m
                wavelength = H / p

                result += f"Momentum: p = {p:.4e} kg·m/s\n"
                result += f"Velocity: v = p/m = {v:.4e} m/s\n\n"

            elif "kinetic_energy_eV" in data or "KE" in data:
                KE_eV = float(data.get("kinetic_energy_eV", data.get("KE")))
                KE_J = KE_eV * E_CHARGE

                # p = √(2mKE) for non-relativistic
                p = math.sqrt(2 * m * KE_J)
                v = p / m
                wavelength = H / p

                result += f"Kinetic Energy: KE = {KE_eV:.4f} eV = {KE_J:.4e} J\n"
                result += f"Momentum: p = √(2mKE) = {p:.4e} kg·m/s\n"
                result += f"Velocity: v = {v:.4e} m/s\n\n"

            result += f"de Broglie Wavelength:\n"
            result += f"λ = h/p = {H:.3e} / {p:.4e}\n"
            result += f"λ = {wavelength:.4e} m = {wavelength*1e9:.4f} nm = {wavelength*1e10:.4f} Å\n\n"

            # Compare to atomic scales
            result += f"Comparison:\n"
            result += f"- Atomic size: ~10⁻¹⁰ m (1 Å)\n"
            result += f"- Nuclear size: ~10⁻¹⁵ m (1 fm)\n"

            if wavelength > 1e-10:
                result += f"→ λ > atomic size: Wave effects significant\n"
            else:
                result += f"→ λ < atomic size: Particle behavior dominates\n"

            return result

        except Exception as e:
            return f"Error in de Broglie wavelength calculation: {str(e)}"

    @mcp.tool()
    async def bohr_model(bohr_data: str) -> str:
        """
        Calculate hydrogen atom energy levels and transitions using Bohr model.

        Args:
            bohr_data: JSON string with Bohr model parameters
                      Example: '{"n_initial": 3, "n_final": 2}'
                      Example: '{"n": 4}'
                      Example: '{"wavelength": 656e-9}'

        Returns:
            str: Complete Bohr model analysis
        """
        try:
            data = json.loads(bohr_data) if isinstance(bohr_data, str) else bohr_data

            result = """
Bohr Model of Hydrogen Atom:
============================

Energy Levels: Eₙ = -13.6/n² eV
Radius: rₙ = n² × 0.529 Å (Bohr radius)

"""
            a0 = 5.29e-11  # Bohr radius in meters

            if "n" in data:
                n = int(data.get("n"))

                E_n = -RYDBERG / n**2
                r_n = n**2 * a0

                result += f"Quantum Number: n = {n}\n\n"
                result += f"Energy Level:\n"
                result += f"E_{n} = -13.6/{n}² = {E_n:.4f} eV\n\n"
                result += f"Orbital Radius:\n"
                result += f"r_{n} = {n}² × a₀ = {n**2} × {a0:.2e} m = {r_n:.4e} m = {r_n*1e10:.4f} Å\n"

            if "n_initial" in data and "n_final" in data:
                ni = int(data.get("n_initial"))
                nf = int(data.get("n_final"))

                E_i = -RYDBERG / ni**2
                E_f = -RYDBERG / nf**2
                delta_E = E_f - E_i

                result += f"\nTransition: n = {ni} → n = {nf}\n\n"
                result += f"Initial Energy: E_{ni} = {E_i:.4f} eV\n"
                result += f"Final Energy: E_{nf} = {E_f:.4f} eV\n"
                result += f"Energy Change: ΔE = E_f - E_i = {delta_E:.4f} eV\n\n"

                if delta_E < 0:
                    # Emission
                    photon_E = abs(delta_E)
                    wavelength = H * C / (photon_E * E_CHARGE)
                    result += f"Process: EMISSION (photon released)\n"
                    result += f"Photon Energy: E = {photon_E:.4f} eV\n"
                    result += f"Photon Wavelength: λ = hc/E = {wavelength:.4e} m = {wavelength*1e9:.1f} nm\n"

                    # Identify spectral series
                    if nf == 1:
                        series = "Lyman (UV)"
                    elif nf == 2:
                        series = "Balmer (visible)"
                    elif nf == 3:
                        series = "Paschen (IR)"
                    else:
                        series = "Higher series (IR)"
                    result += f"Spectral Series: {series}\n"
                else:
                    # Absorption
                    photon_E = delta_E
                    wavelength = H * C / (photon_E * E_CHARGE)
                    result += f"Process: ABSORPTION (photon absorbed)\n"
                    result += f"Required Photon Energy: E = {photon_E:.4f} eV\n"
                    result += f"Required Wavelength: λ = {wavelength:.4e} m = {wavelength*1e9:.1f} nm\n"

            result += """
Spectral Series:
- Lyman (n_f = 1): UV, λ = 91-122 nm
- Balmer (n_f = 2): Visible, λ = 365-656 nm
- Paschen (n_f = 3): IR, λ = 820-1875 nm
"""
            return result

        except Exception as e:
            return f"Error in Bohr model calculation: {str(e)}"

    @mcp.tool()
    async def radioactive_decay(decay_data: str) -> str:
        """
        Calculate radioactive decay, half-life, and activity.

        Args:
            decay_data: JSON string with decay parameters
                       Example: '{"N0": 1e6, "half_life": 5730, "time": 11460}'
                       Example: '{"activity": 1000, "half_life_days": 8, "time_days": 24}'
                       Example: '{"N0": 1e10, "decay_constant": 1e-4}'
                       Units: time in years unless specified, activity in Bq

        Returns:
            str: Complete radioactive decay analysis
        """
        try:
            data = json.loads(decay_data) if isinstance(decay_data, str) else decay_data

            result = """
Radioactive Decay Analysis:
===========================

Decay Law: N(t) = N₀ e^(-λt)
Activity: A = λN = A₀ e^(-λt)
Half-life: t½ = ln(2)/λ ≈ 0.693/λ

"""
            # Get half-life and decay constant
            if "half_life" in data:
                t_half = float(data.get("half_life"))
                lambda_decay = math.log(2) / t_half
                time_unit = "years"
            elif "half_life_days" in data:
                t_half = float(data.get("half_life_days"))
                lambda_decay = math.log(2) / t_half
                time_unit = "days"
            elif "half_life_hours" in data:
                t_half = float(data.get("half_life_hours"))
                lambda_decay = math.log(2) / t_half
                time_unit = "hours"
            elif "decay_constant" in data:
                lambda_decay = float(data.get("decay_constant"))
                t_half = math.log(2) / lambda_decay
                time_unit = "s⁻¹ for λ"
            else:
                t_half = 5730  # Carbon-14 default
                lambda_decay = math.log(2) / t_half
                time_unit = "years"

            result += f"Half-life: t½ = {t_half:.4g} {time_unit}\n"
            result += f"Decay Constant: λ = ln(2)/t½ = {lambda_decay:.4e} {time_unit}⁻¹\n\n"

            # Get initial amount and time
            if "time" in data or "time_days" in data or "time_hours" in data:
                t = float(data.get("time", data.get("time_days", data.get("time_hours", 0))))

            if "N0" in data:
                N0 = float(data.get("N0"))
                N = N0 * math.exp(-lambda_decay * t)
                n_half_lives = t / t_half

                result += f"Initial Nuclei: N₀ = {N0:.4e}\n"
                result += f"Time Elapsed: t = {t:.4g} {time_unit}\n"
                result += f"Number of Half-lives: t/t½ = {n_half_lives:.4f}\n\n"
                result += f"Remaining Nuclei:\n"
                result += f"N(t) = N₀ × e^(-λt) = {N0:.4e} × e^(-{lambda_decay:.4e} × {t:.4g})\n"
                result += f"N(t) = {N:.4e}\n\n"
                result += f"Fraction Remaining: N/N₀ = {N/N0:.6f} = {100*N/N0:.4f}%\n"
                result += f"Fraction Decayed: 1 - N/N₀ = {1 - N/N0:.6f} = {100*(1-N/N0):.4f}%\n"

            elif "activity" in data or "A0" in data:
                A0 = float(data.get("activity", data.get("A0")))
                A = A0 * math.exp(-lambda_decay * t)

                result += f"Initial Activity: A₀ = {A0:.4e} Bq\n"
                result += f"Time Elapsed: t = {t:.4g} {time_unit}\n\n"
                result += f"Remaining Activity:\n"
                result += f"A(t) = A₀ × e^(-λt) = {A:.4e} Bq\n"

            result += """
Common Isotopes:
- Carbon-14: t½ = 5,730 years (dating)
- Iodine-131: t½ = 8.02 days (medical)
- Uranium-238: t½ = 4.47 billion years
- Radon-222: t½ = 3.82 days
- Technetium-99m: t½ = 6.01 hours (medical imaging)
"""
            return result

        except Exception as e:
            return f"Error in radioactive decay calculation: {str(e)}"

    logger.info(f'{NAME} MCP Server at {host}:{port} and transport {transport}')
    if transport == "sse":
        mcp.sse_http_app.run(host=host, port=port)
    if transport == "streamable_http":
        import uvicorn
        uvicorn.run(mcp.streamable_http_app, host=host, port=port)


def main():
    """CLI entry point for the modern physics MCP server."""
    parser = argparse.ArgumentParser(description="Run Modern Physics MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10111, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")

    args = parser.parse_args()

    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")


if __name__ == "__main__":
    main()
