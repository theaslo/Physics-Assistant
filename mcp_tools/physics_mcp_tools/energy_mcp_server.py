from typing import Tuple, Optional, Dict, List
import math
import json
import logging
import argparse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger#

# Initialize FastMCP server
#mcp = FastMCP("energy")

from physics_mcp_tools.energy_utils import (
    calculate_kinetic_energy,
    calculate_gravitational_potential_energy,
    calculate_elastic_potential_energy,
    calculate_work,
    degrees_to_radians,
)

NAME= "energy_mcp_server"

logger = get_logger(__name__)

def serve(host, port, transport):  
    """Initializes and runs the Agent Cards MCP server.

    Args:
        host: The hostname or IP address to bind the server to.
        port: The port number to bind the server to.
        transport: The transport mechanism for the MCP server (e.g., 'stdio', 'sse').
    """
    logger.info('Starting Energy MCP Server')
    
    mcp = FastMCP(NAME, stateless_http=False)


    @mcp.tool()
    async def calculate_kinetic_energy_tool(mass: float, velocity: float) -> str:
        """
        Calculate kinetic energy of a moving object.
        
        Args:
            mass: Mass of the object in kg
            velocity: Speed of the object in m/s
            
        Returns:
            str: Complete kinetic energy calculation with explanation
        """
        try:
            ke = calculate_kinetic_energy(mass, velocity)
            
            result = f"""
    Kinetic Energy Calculation:
    ==========================

    Given:
    - Mass (m): {mass:.3f} kg
    - Velocity (v): {velocity:.3f} m/s

    Kinetic Energy Formula: KE = ½mv²
    KE = ½ × {mass:.3f} × {velocity:.3f}²
    KE = ½ × {mass:.3f} × {velocity**2:.3f}
    KE = {ke:.3f} J

    Physical Interpretation:
    - Kinetic energy: {ke:.3f} J (Joules)
    - Energy due to motion of the object
    - Depends on both mass and velocity (velocity squared!)
    - Always positive (energy is a scalar quantity)
    - Represents the work needed to accelerate the object from rest

    Key Concepts:
    - KE ∝ m: Doubling mass doubles kinetic energy
    - KE ∝ v²: Doubling velocity quadruples kinetic energy
    - At rest (v = 0): KE = 0 J
    - Units: Joules (J) = kg⋅m²/s²

    Real-World Context:
    - A {mass:.1f} kg object at {velocity:.1f} m/s has {ke:.1f} J of kinetic energy
    - This could stop a {ke/10:.1f} N force over 1 meter
    - Equivalent to lifting {ke/9.81:.2f} kg by 1 meter against gravity
    """
            
            return result
            
        except Exception as e:
            return f"Error in kinetic energy calculation: {str(e)}"

    @mcp.tool()
    async def calculate_gravitational_potential_energy_tool(mass: float, height: float, gravity: float = 9.81, reference_level: str = "ground") -> str:
        """
        Calculate gravitational potential energy relative to a reference level.
        
        Args:
            mass: Mass of the object in kg
            height: Height above reference level in meters
            gravity: Gravitational acceleration in m/s² (default: 9.81 for Earth)
            reference_level: Description of reference level (default: "ground")
            
        Returns:
            str: Complete gravitational potential energy calculation
        """
        try:
            pe = calculate_gravitational_potential_energy(mass, height, gravity)
            
            result = f"""
    Gravitational Potential Energy Calculation:
    ==========================================

    Given:
    - Mass (m): {mass:.3f} kg
    - Height (h): {height:.3f} m above {reference_level}
    - Gravitational acceleration (g): {gravity:.2f} m/s²

    Gravitational PE Formula: PE = mgh
    PE = {mass:.3f} × {gravity:.2f} × {height:.3f}
    PE = {pe:.3f} J

    Physical Interpretation:
    - Gravitational potential energy: {pe:.3f} J
    - Energy stored due to position in gravitational field
    - Energy available to do work when object falls
    - Relative to chosen reference level ({reference_level})

    Key Concepts:
    - PE is relative to a chosen reference level
    - PE increases with height (positive work done against gravity)
    - PE decreases with depth below reference (negative relative energy)
    - Conservative force: PE depends only on position, not path
    - Energy can be converted to kinetic energy during free fall

    Reference Level Analysis:
    - Current reference: {reference_level}
    - If dropped from this height: gains {pe:.3f} J of kinetic energy
    - Impact velocity (if dropped): v = √(2gh) = {math.sqrt(2*gravity*height):.2f} m/s
    - Work required to lift object: W = {pe:.3f} J

    Applications:
    - Hydroelectric power: water at height stores energy
    - Roller coasters: height determines maximum speed
    - Satellites: orbital energy and escape velocity
    - Pumped hydro storage: energy storage systems
    """
            
            return result
            
        except Exception as e:
            return f"Error in gravitational potential energy calculation: {str(e)}"

    @mcp.tool()
    async def calculate_elastic_potential_energy_tool(spring_constant: float, displacement: float, equilibrium_position: str = "natural length") -> str:
        """
        Calculate elastic potential energy stored in a spring or elastic system.
        
        Args:
            spring_constant: Spring constant k in N/m
            displacement: Displacement from equilibrium in meters (positive = stretched/compressed)
            equilibrium_position: Description of equilibrium position
            
        Returns:
            str: Complete elastic potential energy calculation
        """
        try:
            pe_elastic = calculate_elastic_potential_energy(spring_constant, displacement)
            
            # Calculate force at this displacement
            force = spring_constant * abs(displacement)
            
            result = f"""
    Elastic Potential Energy Calculation:
    ====================================

    Given:
    - Spring constant (k): {spring_constant:.2f} N/m
    - Displacement from equilibrium (x): {displacement:.3f} m
    - Equilibrium position: {equilibrium_position}

    Elastic PE Formula: PE = ½kx²
    PE = ½ × {spring_constant:.2f} × {displacement:.3f}²
    PE = ½ × {spring_constant:.2f} × {displacement**2:.6f}
    PE = {pe_elastic:.3f} J

    Spring Analysis:
    - Displacement: {abs(displacement):.3f} m {'stretched' if displacement > 0 else 'compressed' if displacement < 0 else 'at equilibrium'}
    - Spring force magnitude: F = kx = {spring_constant:.2f} × {abs(displacement):.3f} = {force:.3f} N
    - Force direction: {'Restoring force toward equilibrium' if displacement != 0 else 'No force at equilibrium'}

    Physical Interpretation:
    - Elastic potential energy: {pe_elastic:.3f} J
    - Energy stored in deformed spring/elastic material
    - Always positive (independent of stretch/compression direction)
    - Energy available when spring returns to equilibrium
    - Maximum when displacement is maximum, zero at equilibrium

    Key Concepts:
    - Hooke's Law: F = -kx (restoring force)
    - PE depends on displacement squared (x²)
    - Stiffer spring (larger k) stores more energy for same displacement
    - Energy is conserved in ideal springs (no energy loss)

    Energy Analysis:
    - Work done to stretch/compress: W = ∫F dx = ½kx² = {pe_elastic:.3f} J
    - Energy released if spring returns to equilibrium: {pe_elastic:.3f} J
    - If released, maximum kinetic energy = {pe_elastic:.3f} J

    Applications:
    - Spring-mass oscillators and simple harmonic motion
    - Elastic collisions and rebound mechanisms
    - Energy storage in mechanical systems
    - Shock absorbers and suspension systems
    - Trampolines, bungee cords, and elastic materials
    """
            
            return result
            
        except Exception as e:
            return f"Error in elastic potential energy calculation: {str(e)}"

    @mcp.tool()
    async def calculate_work_tool(force: float, displacement: float, angle_degrees: float = 0, force_type: str = "constant") -> str:
        """
        Calculate work done by a force.
        
        Args:
            force: Applied force magnitude in N
            displacement: Displacement magnitude in m
            angle_degrees: Angle between force and displacement vectors in degrees
            force_type: Type of force ("constant", "variable", "friction", etc.)
            
        Returns:
            str: Complete work calculation with explanation
        """
        try:
            work = calculate_work(force, displacement, angle_degrees)
            
            # Calculate parallel and perpendicular components
            angle_rad = degrees_to_radians(angle_degrees)
            force_parallel = force * math.cos(angle_rad)
            force_perpendicular = force * math.sin(angle_rad)
            
            result = f"""
    Work Calculation:
    ================

    Given:
    - Applied force (F): {force:.3f} N
    - Displacement (d): {displacement:.3f} m
    - Angle between F and d (θ): {angle_degrees:.1f}°
    - Force type: {force_type.title()}

    Work Formula: W = F⃗ · d⃗ = Fd cos(θ)
    W = {force:.3f} × {displacement:.3f} × cos({angle_degrees:.1f}°)
    W = {force:.3f} × {displacement:.3f} × {math.cos(angle_rad):.6f}
    W = {work:.3f} J

    Force Component Analysis:
    - Force parallel to displacement: F∥ = F cos(θ) = {force_parallel:.3f} N
    - Force perpendicular to displacement: F⊥ = F sin(θ) = {force_perpendicular:.3f} N
    - Only parallel component does work!

    Physical Interpretation:
    - Work done: {work:.3f} J
    - {'Positive work: energy added to system' if work > 0 else 'Negative work: energy removed from system' if work < 0 else 'Zero work: no energy transfer'}
    - Energy transferred to/from the object
    - Work equals change in kinetic energy (work-energy theorem)

    Angle Analysis:
    - θ = 0°: Maximum positive work (force helps motion)
    - θ = 90°: Zero work (force perpendicular to motion)  
    - θ = 180°: Maximum negative work (force opposes motion)
    - Current angle {angle_degrees:.1f}°: {'Force helps motion' if angle_degrees < 90 else 'Force perpendicular to motion' if angle_degrees == 90 else 'Force opposes motion'}

    Key Concepts:
    - Work is a scalar quantity (has magnitude but no direction)
    - Work = 0 when force ⊥ displacement (no energy transfer)
    - Work > 0: force does positive work (adds energy)
    - Work < 0: force does negative work (removes energy)
    - Units: Joules (J) = N⋅m = kg⋅m²/s²

    Applications:
    - Lifting objects against gravity: W = mgh
    - Friction work: always negative (removes energy)
    - Engine work: positive (adds kinetic energy to vehicle)
    - Braking work: negative (removes kinetic energy)
    """
            
            return result
            
        except Exception as e:
            return f"Error in work calculation: {str(e)}"

    @mcp.tool()
    async def work_energy_theorem(problem_data: str) -> str:
        """
        Apply the work-energy theorem to solve problems.
        
        Args:
            problem_data: JSON string with problem parameters
                        Examples:
                        '{"mass": 5, "initial_velocity": 10, "final_velocity": 15}'
                        '{"mass": 2, "initial_velocity": 0, "work_done": 50}'
                        '{"force": 20, "displacement": 3, "angle": 30, "mass": 4, "initial_velocity": 5}'
            
        Returns:
            str: Complete work-energy theorem analysis
        """
        try:
            data = json.loads(problem_data)
            
            result = f"""
    Work-Energy Theorem Analysis:
    ============================

    Given Data:
    """
            
            for key, value in data.items():
                result += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            result += f"""
    Work-Energy Theorem: W_net = ΔKE = KE_f - KE_i

    """
            
            if "mass" in data:
                mass = data["mass"]
                
                # Calculate initial kinetic energy
                if "initial_velocity" in data:
                    vi = data["initial_velocity"]
                    ke_i = calculate_kinetic_energy(mass, vi)
                    result += f"Initial kinetic energy: KE_i = ½mv_i² = ½({mass})({vi})² = {ke_i:.3f} J\n"
                else:
                    ke_i = 0
                    result += f"Initial kinetic energy: KE_i = 0 J (starting from rest)\n"
                
                # Case 1: Given final velocity
                if "final_velocity" in data:
                    vf = data["final_velocity"]
                    ke_f = calculate_kinetic_energy(mass, vf)
                    work_net = ke_f - ke_i
                    
                    result += f"Final kinetic energy: KE_f = ½mv_f² = ½({mass})({vf})² = {ke_f:.3f} J\n"
                    result += f"\nUsing W_net = ΔKE:\n"
                    result += f"Net work done: W_net = KE_f - KE_i = {ke_f:.3f} - {ke_i:.3f} = {work_net:.3f} J\n"
                    
                    if "force" in data and "displacement" in data:
                        force = data["force"]
                        displacement = data["displacement"]
                        angle = data.get("angle", 0)
                        work_calculated = calculate_work(force, displacement, angle)
                        result += f"\nVerification with force and displacement:\n"
                        result += f"Work by force: W = Fd cos(θ) = {force}×{displacement}×cos({angle}°) = {work_calculated:.3f} J\n"
                        result += f"Difference: |W_net - W_force| = |{work_net:.3f} - {work_calculated:.3f}| = {abs(work_net - work_calculated):.6f} J\n"
                
                # Case 2: Given work done
                elif "work_done" in data:
                    work_net = data["work_done"]
                    ke_f = ke_i + work_net
                    vf = math.sqrt(2 * ke_f / mass) if ke_f >= 0 else 0
                    
                    result += f"\nUsing W_net = ΔKE:\n"
                    result += f"Final kinetic energy: KE_f = KE_i + W_net = {ke_i:.3f} + {work_net:.3f} = {ke_f:.3f} J\n"
                    result += f"Final velocity: v_f = √(2KE_f/m) = √(2×{ke_f:.3f}/{mass}) = {vf:.3f} m/s\n"
                    
                    if "force" in data and "displacement" in data:
                        force = data["force"]
                        displacement = data["displacement"]
                        angle = data.get("angle", 0)
                        work_by_force = calculate_work(force, displacement, angle)
                        result += f"\nWork by given force: W = {force}×{displacement}×cos({angle}°) = {work_by_force:.3f} J\n"
                        if abs(work_by_force - work_net) < 0.01:
                            result += f"This matches the net work! ✓\n"
                        else:
                            result += f"Additional forces must account for: {work_net - work_by_force:.3f} J\n"
                
                # Case 3: Given force, displacement, and angle
                elif "force" in data and "displacement" in data:
                    force = data["force"]
                    displacement = data["displacement"]
                    angle = data.get("angle", 0)
                    work_net = calculate_work(force, displacement, angle)
                    ke_f = ke_i + work_net
                    vf = math.sqrt(2 * ke_f / mass) if ke_f >= 0 else 0
                    
                    result += f"\nWork by force: W = Fd cos(θ) = {force}×{displacement}×cos({angle}°) = {work_net:.3f} J\n"
                    result += f"Final kinetic energy: KE_f = KE_i + W = {ke_i:.3f} + {work_net:.3f} = {ke_f:.3f} J\n"
                    result += f"Final velocity: v_f = √(2KE_f/m) = {vf:.3f} m/s\n"
                    
                    delta_v = vf - vi if "initial_velocity" in data else vf
                    result += f"Change in velocity: Δv = v_f - v_i = {delta_v:.3f} m/s\n"
            
            result += f"""

    Physical Insights:
    - Work-energy theorem connects force/work to changes in motion
    - Net work = change in kinetic energy (fundamental energy principle)
    - Positive work increases kinetic energy (speeds up object)
    - Negative work decreases kinetic energy (slows down object)
    - Zero net work means constant kinetic energy (constant speed)

    Applications:
    - Braking systems: negative work reduces vehicle kinetic energy
    - Rocket propulsion: engine work increases spacecraft kinetic energy
    - Sports: analyzing energy transfer in throwing, hitting, jumping
    - Machines: calculating efficiency and energy requirements

    Key Relationships:
    - W_net = ΔKE = ½m(v_f² - v_i²)
    - If multiple forces: W_net = W_1 + W_2 + W_3 + ...
    - Conservative forces: can be expressed as potential energy changes
    - Non-conservative forces: irreversibly change mechanical energy
    """
            
            return result
            
        except Exception as e:
            return f"Error in work-energy theorem analysis: {str(e)}"

    @mcp.tool()
    async def energy_conservation(system_data: str) -> str:
        """
        Analyze energy conservation in mechanical systems.
        
        Args:
            system_data: JSON string with system parameters
                        Examples:
                        '{"mass": 2, "initial_height": 10, "final_height": 5, "initial_velocity": 0}'
                        '{"mass": 1, "spring_constant": 100, "compression": 0.2, "final_velocity": 5}'
            
        Returns:
            str: Complete energy conservation analysis
        """
        try:
            data = json.loads(system_data)
            
            result = f"""
    Energy Conservation Analysis:
    ============================

    Given System Data:
    """
            
            for key, value in data.items():
                result += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            result += f"""
    Conservation of Mechanical Energy: E_initial = E_final
    E = KE + PE_gravitational + PE_elastic = constant (no friction)

    """
            
            mass = data.get("mass", 1.0)
            gravity = data.get("gravity", 9.81)
            
            # Calculate initial energy
            initial_energy = 0
            result += f"Initial Energy Components:\n"
            
            if "initial_velocity" in data:
                vi = data["initial_velocity"]
                ke_i = calculate_kinetic_energy(mass, vi)
                initial_energy += ke_i
                result += f"- Initial kinetic energy: KE_i = ½mv_i² = {ke_i:.3f} J\n"
            else:
                result += f"- Initial kinetic energy: KE_i = 0 J (starts from rest)\n"
            
            if "initial_height" in data:
                hi = data["initial_height"]
                pe_grav_i = calculate_gravitational_potential_energy(mass, hi, gravity)
                initial_energy += pe_grav_i
                result += f"- Initial gravitational PE: PE_grav_i = mgh_i = {pe_grav_i:.3f} J\n"
            
            if "initial_compression" in data or "initial_stretch" in data:
                spring_constant = data.get("spring_constant", 100)
                xi = data.get("initial_compression", data.get("initial_stretch", 0))
                pe_elastic_i = calculate_elastic_potential_energy(spring_constant, xi)
                initial_energy += pe_elastic_i
                result += f"- Initial elastic PE: PE_elastic_i = ½kx_i² = {pe_elastic_i:.3f} J\n"
            
            result += f"Total initial energy: E_i = {initial_energy:.3f} J\n\n"
            
            # Calculate final energy
            final_energy = 0
            result += f"Final Energy Components:\n"
            
            if "final_velocity" in data:
                vf = data["final_velocity"]
                ke_f = calculate_kinetic_energy(mass, vf)
                final_energy += ke_f
                result += f"- Final kinetic energy: KE_f = ½mv_f² = {ke_f:.3f} J\n"
            
            if "final_height" in data:
                hf = data["final_height"]
                pe_grav_f = calculate_gravitational_potential_energy(mass, hf, gravity)
                final_energy += pe_grav_f
                result += f"- Final gravitational PE: PE_grav_f = mgh_f = {pe_grav_f:.3f} J\n"
            
            if "final_compression" in data or "final_stretch" in data:
                spring_constant = data.get("spring_constant", 100)
                xf = data.get("final_compression", data.get("final_stretch", 0))
                pe_elastic_f = calculate_elastic_potential_energy(spring_constant, xf)
                final_energy += pe_elastic_f
                result += f"- Final elastic PE: PE_elastic_f = ½kx_f² = {pe_elastic_f:.3f} J\n"
            
            # If some final values are unknown, solve using conservation
            unknown_final_values = []
            if "final_velocity" not in data and "final_height" not in data and "final_compression" not in data and "final_stretch" not in data:
                result += f"- Some final values unknown - using conservation to solve\n"
            
            # Handle specific conservation scenarios
            if "initial_height" in data and "final_height" in data and "final_velocity" not in data:
                # Falling object - solve for final velocity
                hi = data["initial_height"]
                hf = data["final_height"]
                vi = data.get("initial_velocity", 0)
                
                # mgh_i + ½mv_i² = mgh_f + ½mv_f²
                # v_f² = v_i² + 2g(h_i - h_f)
                vf_squared = vi**2 + 2 * gravity * (hi - hf)
                vf = math.sqrt(vf_squared) if vf_squared >= 0 else 0
                
                result += f"\nSolving for final velocity using conservation:\n"
                result += f"E_i = E_f\n"
                result += f"mgh_i + ½mv_i² = mgh_f + ½mv_f²\n"
                result += f"v_f² = v_i² + 2g(h_i - h_f)\n"
                result += f"v_f² = {vi}² + 2×{gravity}×({hi} - {hf}) = {vf_squared:.3f}\n"
                result += f"v_f = {vf:.3f} m/s\n"
                
                ke_f = calculate_kinetic_energy(mass, vf)
                pe_grav_f = calculate_gravitational_potential_energy(mass, hf, gravity)
                final_energy = ke_f + pe_grav_f
            
            elif "spring_constant" in data and "compression" in data and "final_velocity" not in data:
                # Spring release - solve for final velocity
                spring_constant = data["spring_constant"]
                compression = data["compression"]
                
                pe_elastic_i = calculate_elastic_potential_energy(spring_constant, compression)
                # Assuming spring returns to equilibrium: PE_elastic_f = 0
                # ½kx² = ½mv_f²
                vf = math.sqrt(spring_constant * compression**2 / mass)
                
                result += f"\nSolving for final velocity (spring release):\n"
                result += f"½kx² = ½mv_f²\n"
                result += f"v_f = √(kx²/m) = √({spring_constant}×{compression}²/{mass}) = {vf:.3f} m/s\n"
                
                ke_f = calculate_kinetic_energy(mass, vf)
                final_energy = ke_f
            
            result += f"\nTotal final energy: E_f = {final_energy:.3f} J\n"
            
            # Conservation check
            energy_diff = abs(initial_energy - final_energy)
            result += f"\nConservation Check:\n"
            result += f"Initial energy: E_i = {initial_energy:.3f} J\n"
            result += f"Final energy: E_f = {final_energy:.3f} J\n"
            result += f"Energy difference: |E_i - E_f| = {energy_diff:.6f} J\n"
            
            if energy_diff < 0.01:
                result += f"✓ Energy is conserved (difference ≈ 0)\n"
            else:
                result += f"⚠ Energy difference suggests non-conservative forces or calculation error\n"
            
            result += f"""

    Physical Principles:
    - Mechanical energy conserved when only conservative forces act
    - Conservative forces: gravity, elastic (spring) forces
    - Non-conservative forces: friction, air resistance, applied forces
    - Energy can transform between kinetic and potential forms
    - Total mechanical energy remains constant in ideal systems

    Energy Transformations:
    - Gravitational: PE ↔ KE (falling/rising objects)
    - Elastic: PE ↔ KE (springs, bouncing balls)
    - Combined: Complex motion with multiple energy forms

    Applications:
    - Roller coasters: height determines maximum speed
    - Pendulums: energy oscillates between PE and KE
    - Springs: energy storage and release mechanisms
    - Projectile motion: parabolic paths from energy conservation
    - Satellite orbits: balance of kinetic and gravitational PE

    Real-World Considerations:
    - Air resistance reduces mechanical energy
    - Friction converts mechanical energy to heat
    - Elastic collisions conserve kinetic energy
    - Inelastic collisions lose mechanical energy
    """
            
            return result
            
        except Exception as e:
            return f"Error in energy conservation analysis: {str(e)}"

    @mcp.tool()
    async def energy_with_friction(friction_data: str) -> str:
        """
        Analyze energy in systems with friction and non-conservative forces.
        
        Args:
            friction_data: JSON string with friction system parameters
                        Examples:
                        '{"mass": 5, "initial_velocity": 20, "friction_coefficient": 0.3, "distance": 50}'
                        '{"mass": 2, "initial_height": 10, "final_height": 0, "friction_work": -30}'
            
        Returns:
            str: Complete energy analysis including friction effects
        """
        try:
            data = json.loads(friction_data)
            
            result = f"""
    Energy Analysis with Friction:
    =============================

    Given System Data:
    """
            
            for key, value in data.items():
                result += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            result += f"""
    Energy Conservation with Non-Conservative Forces:
    E_initial = E_final + Energy_dissipated
    KE_i + PE_i = KE_f + PE_f + W_friction + W_other_losses

    """
            
            mass = data.get("mass", 1.0)
            gravity = data.get("gravity", 9.81)
            
            # Calculate initial energy
            initial_energy = 0
            result += f"Initial Energy:\n"
            
            if "initial_velocity" in data:
                vi = data["initial_velocity"]
                ke_i = calculate_kinetic_energy(mass, vi)
                initial_energy += ke_i
                result += f"- Initial kinetic energy: KE_i = ½mv_i² = {ke_i:.3f} J\n"
            else:
                ke_i = 0
                result += f"- Initial kinetic energy: KE_i = 0 J\n"
            
            if "initial_height" in data:
                hi = data["initial_height"]
                pe_i = calculate_gravitational_potential_energy(mass, hi, gravity)
                initial_energy += pe_i
                result += f"- Initial potential energy: PE_i = mgh_i = {pe_i:.3f} J\n"
            else:
                pe_i = 0
            
            result += f"Total initial energy: E_i = {initial_energy:.3f} J\n\n"
            
            # Calculate friction work
            friction_work = 0
            if "friction_coefficient" in data and "distance" in data:
                mu = data["friction_coefficient"]
                distance = data["distance"]
                normal_force = mass * gravity  # Assuming horizontal surface
                friction_force = mu * normal_force
                friction_work = -friction_force * distance  # Negative (opposes motion)
                
                result += f"Friction Analysis:\n"
                result += f"- Friction coefficient (μ): {mu:.3f}\n"
                result += f"- Normal force: N = mg = {normal_force:.3f} N\n"
                result += f"- Friction force: f = μN = {friction_force:.3f} N\n"
                result += f"- Distance traveled: d = {distance:.3f} m\n"
                result += f"- Work by friction: W_friction = -f×d = {friction_work:.3f} J\n"
                
            elif "friction_work" in data:
                friction_work = data["friction_work"]
                result += f"Work by friction: W_friction = {friction_work:.3f} J\n"
            
            result += f"\n"
            
            # Calculate final energy
            final_energy = 0
            result += f"Final Energy:\n"
            
            if "final_velocity" in data:
                vf = data["final_velocity"]
                ke_f = calculate_kinetic_energy(mass, vf)
                final_energy += ke_f
                result += f"- Final kinetic energy: KE_f = ½mv_f² = {ke_f:.3f} J\n"
            
            if "final_height" in data:
                hf = data["final_height"]
                pe_f = calculate_gravitational_potential_energy(mass, hf, gravity)
                final_energy += pe_f
                result += f"- Final potential energy: PE_f = mgh_f = {pe_f:.3f} J\n"
            else:
                pe_f = 0
            
            # Solve for unknown final values using energy balance
            if "final_velocity" not in data and friction_work != 0:
                # Solve for final velocity
                # E_i = E_f - W_friction (friction work is negative)
                remaining_energy = initial_energy + friction_work - pe_f
                if remaining_energy >= 0:
                    vf = math.sqrt(2 * remaining_energy / mass)
                    ke_f = calculate_kinetic_energy(mass, vf)
                    final_energy = ke_f + pe_f
                    
                    result += f"\nSolving for final velocity:\n"
                    result += f"E_i + W_friction = KE_f + PE_f\n"
                    result += f"{initial_energy:.3f} + {friction_work:.3f} = ½mv_f² + {pe_f:.3f}\n"
                    result += f"KE_f = {remaining_energy:.3f} J\n"
                    result += f"v_f = √(2×KE_f/m) = {vf:.3f} m/s\n"
                    result += f"- Final kinetic energy: KE_f = {ke_f:.3f} J\n"
                else:
                    result += f"⚠ Object comes to rest before reaching final position\n"
                    # Calculate stopping distance
                    if "friction_coefficient" in data:
                        mu = data["friction_coefficient"]
                        stopping_distance = vi**2 / (2 * mu * gravity)
                        result += f"Stopping distance: d = v_i²/(2μg) = {stopping_distance:.3f} m\n"
                    vf = 0
                    ke_f = 0
                    final_energy = pe_f
            
            result += f"Total final energy: E_f = {final_energy:.3f} J\n\n"
            
            # Energy balance analysis
            energy_dissipated = initial_energy - final_energy
            result += f"Energy Balance:\n"
            result += f"Initial mechanical energy: E_i = {initial_energy:.3f} J\n"
            result += f"Final mechanical energy: E_f = {final_energy:.3f} J\n"
            result += f"Energy dissipated: ΔE = E_i - E_f = {energy_dissipated:.3f} J\n"
            result += f"Work by friction: W_friction = {friction_work:.3f} J\n"
            
            if abs(energy_dissipated + friction_work) < 0.1:
                result += f"✓ Energy balance: Energy dissipated = -Work by friction\n"
            else:
                other_losses = energy_dissipated + friction_work
                result += f"⚠ Additional energy losses: {other_losses:.3f} J\n"
            
            # Efficiency analysis
            if initial_energy > 0:
                efficiency = (final_energy / initial_energy) * 100
                result += f"Mechanical efficiency: η = E_f/E_i × 100% = {efficiency:.1f}%\n"
            
            result += f"""

    Physical Interpretation:
    - Friction always does negative work (removes mechanical energy)
    - Mechanical energy is not conserved (decreases due to friction)
    - Lost energy is converted to heat (thermal energy)
    - Efficiency < 100% in real systems due to energy dissipation

    Friction Effects:
    - Sliding friction: f = μ_k × N (kinetic friction)
    - Rolling friction: generally much smaller than sliding
    - Air resistance: increases with velocity squared
    - Internal friction: in materials, joints, and mechanisms

    Energy Dissipation Mechanisms:
    - Heat generation: dominant mechanism in friction
    - Sound production: friction creates noise/vibrations
    - Deformation: permanent changes in materials
    - Wear: material removal and surface damage

    Applications:
    - Braking systems: convert kinetic energy to heat
    - Lubrication: reduces friction and energy loss
    - Efficiency optimization: minimizing energy losses
    - Thermal management: dealing with heat from friction

    Design Considerations:
    - Brake design: maximize energy dissipation safely
    - Machine efficiency: minimize friction losses
    - Wear resistance: materials that resist friction damage
    - Heat dissipation: cooling systems for high-friction applications
    """
            
            return result
            
        except Exception as e:
            return f"Error in friction energy analysis: {str(e)}"

    @mcp.tool()
    async def analyze_energy_system(system_data: str) -> str:
        """
        Comprehensive energy analysis for complex mechanical systems.
        
        Args:
            system_data: JSON string with complete system description
                        Example:
                        '{
                        "scenario": "roller_coaster",
                        "mass": 500,
                        "track_points": [
                            {"height": 50, "velocity": 0},
                            {"height": 10, "velocity": null},
                            {"height": 30, "velocity": null}
                        ],
                        "friction_coefficient": 0.02,
                        "track_length": 1000
                        }'
            
        Returns:
            str: Comprehensive energy system analysis
        """
        try:
            data = json.loads(system_data)
            
            scenario = data.get("scenario", "general_system")
            mass = data.get("mass", 1.0)
            gravity = data.get("gravity", 9.81)
            
            result = f"""
    Comprehensive Energy System Analysis:
    ====================================

    System: {scenario.replace('_', ' ').title()}
    Mass: {mass:.1f} kg
    """
            
            if "track_points" in data:
                points = data["track_points"]
                friction_coeff = data.get("friction_coefficient", 0)
                track_length = data.get("track_length", 0)
                
                result += f"Friction coefficient: μ = {friction_coeff:.3f}\n"
                result += f"Total track length: {track_length:.1f} m\n\n"
                
                # Calculate energy at each point
                result += f"Point-by-Point Energy Analysis:\n"
                result += f"{'Point':<8} {'Height':<8} {'PE':<12} {'KE':<12} {'Velocity':<10} {'Total E':<12}\n"
                result += f"{'-'*70}\n"
                
                total_friction_work = 0
                if friction_coeff > 0 and track_length > 0:
                    friction_force = friction_coeff * mass * gravity
                    total_friction_work = -friction_force * track_length
                
                initial_energy = None
                
                for i, point in enumerate(points):
                    height = point["height"]
                    velocity = point.get("velocity")
                    
                    # Calculate potential energy
                    pe = calculate_gravitational_potential_energy(mass, height, gravity)
                    
                    if i == 0:
                        # Initial point
                        if velocity is not None:
                            ke = calculate_kinetic_energy(mass, velocity)
                            total_energy = pe + ke
                            initial_energy = total_energy
                        else:
                            velocity = 0
                            ke = 0
                            total_energy = pe
                            initial_energy = total_energy
                    else:
                        # Subsequent points - calculate velocity using energy conservation
                        if initial_energy is not None:
                            # Account for friction losses proportionally
                            fraction_of_track = i / (len(points) - 1)
                            friction_loss = total_friction_work * fraction_of_track
                            available_energy = initial_energy + friction_loss - pe
                            
                            if available_energy >= 0:
                                ke = available_energy
                                velocity = math.sqrt(2 * ke / mass) if ke > 0 else 0
                                total_energy = pe + ke
                            else:
                                # Not enough energy to reach this height
                                velocity = 0
                                ke = 0
                                total_energy = pe
                                result += f"⚠ Point {i+1}: Insufficient energy to reach height {height:.1f} m\n"
                    
                    result += f"Point {i+1:<3} {height:<8.1f} {pe:<12.1f} {ke:<12.1f} {velocity:<10.2f} {total_energy:<12.1f}\n"
                
                result += f"\nEnergy Summary:\n"
                result += f"Initial mechanical energy: {initial_energy:.1f} J\n"
                if total_friction_work != 0:
                    result += f"Energy lost to friction: {-total_friction_work:.1f} J\n"
                    result += f"Final mechanical energy: {initial_energy + total_friction_work:.1f} J\n"
                    efficiency = ((initial_energy + total_friction_work) / initial_energy) * 100
                    result += f"System efficiency: {efficiency:.1f}%\n"
            
            elif "pendulum" in scenario.lower():
                length = data.get("length", 1.0)
                initial_angle = data.get("initial_angle_degrees", 30)
                
                # Calculate pendulum energy
                initial_height = length * (1 - math.cos(degrees_to_radians(initial_angle)))
                pe_max = calculate_gravitational_potential_energy(mass, initial_height, gravity)
                
                result += f"\nPendulum Analysis:\n"
                result += f"Length: {length:.2f} m\n"
                result += f"Initial angle: {initial_angle:.1f}°\n"
                result += f"Maximum height above lowest point: {initial_height:.3f} m\n"
                result += f"Maximum potential energy: {pe_max:.3f} J\n"
                result += f"Maximum kinetic energy (at bottom): {pe_max:.3f} J\n"
                result += f"Maximum velocity (at bottom): {math.sqrt(2*pe_max/mass):.3f} m/s\n"
                
                # Period calculation (small angle approximation)
                period = 2 * math.pi * math.sqrt(length / gravity)
                result += f"Period (small angles): T = 2π√(L/g) = {period:.3f} s\n"
            
            elif "spring_mass" in scenario.lower():
                spring_constant = data.get("spring_constant", 100)
                amplitude = data.get("amplitude", 0.1)
                
                pe_max = calculate_elastic_potential_energy(spring_constant, amplitude)
                ke_max = pe_max  # At equilibrium position
                v_max = math.sqrt(2 * ke_max / mass)
                
                result += f"\nSpring-Mass System Analysis:\n"
                result += f"Spring constant: k = {spring_constant:.1f} N/m\n"
                result += f"Amplitude: A = {amplitude:.3f} m\n"
                result += f"Maximum potential energy: PE_max = ½kA² = {pe_max:.3f} J\n"
                result += f"Maximum kinetic energy: KE_max = {ke_max:.3f} J\n"
                result += f"Maximum velocity: v_max = {v_max:.3f} m/s\n"
                
                # Frequency and period
                omega = math.sqrt(spring_constant / mass)
                frequency = omega / (2 * math.pi)
                period = 1 / frequency
                result += f"Angular frequency: ω = √(k/m) = {omega:.3f} rad/s\n"
                result += f"Frequency: f = ω/(2π) = {frequency:.3f} Hz\n"
                result += f"Period: T = 1/f = {period:.3f} s\n"
            
            result += f"""

    General Energy Principles:
    - Energy conservation: fundamental law of physics
    - Mechanical energy = Kinetic + Potential (gravitational + elastic)
    - Non-conservative forces (friction) convert mechanical energy to heat
    - Oscillatory systems: energy transforms between KE and PE
    - Efficiency: ratio of useful energy output to total energy input

    System Design Considerations:
    - Minimize energy losses through friction and air resistance
    - Optimize energy storage and release mechanisms
    - Ensure adequate energy for system operation
    - Design safety margins for energy requirements
    - Consider energy recovery and regenerative systems

    Applications:
    - Roller coasters: gravitational energy conversion
    - Pendulum clocks: precise energy oscillation
    - Spring systems: energy storage and shock absorption
    - Hydroelectric: gravitational PE to electrical energy
    - Compressed air systems: elastic PE for energy storage
    """
            
            return result
            
        except Exception as e:
            return f"Error in energy system analysis: {str(e)}"

    logger.info(
        f'{NAME} MCP Server at {host}:{port} and transport {transport}'
    )
    if transport == "sse":
        mcp.sse_http_app.run(host=host, port=port)
    if transport == "streamable_http":
        import uvicorn
        # Start the Streamable HTTP server
        uvicorn.run(mcp.streamable_http_app, host=host, port=port)


def main():
    """CLI entry point for the energy-mcp tool."""
    
    
    parser = argparse.ArgumentParser(description="Run Energy MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10105, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")
    
    args = parser.parse_args()
    
    # Support both 'mcp-server' and 'energy-server' for compatibility
    if args.run in ["mcp-server", "energy-server"]:
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}. Use 'mcp-server' or 'energy-server'")

if __name__ == "__main__":
    main()
