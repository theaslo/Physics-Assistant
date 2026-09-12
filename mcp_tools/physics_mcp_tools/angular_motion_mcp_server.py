from typing import Tuple, Optional, Dict, List
import math
import json
import logging
import argparse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

from physics_mcp_tools.angular_motion_utils import (
    degrees_to_radians,
    radians_to_degrees,
    calculate_angular_displacement,
    calculate_angular_velocity_from_acceleration,
    calculate_angular_velocity_from_displacement,
    calculate_moment_of_inertia_rod,
    calculate_moment_of_inertia_disk,
    calculate_moment_of_inertia_sphere,
    calculate_moment_of_inertia_cylinder,
    calculate_parallel_axis_theorem
    )

NAME= "angular_motion_mcp_server"

logger = get_logger(__name__)

def serve(host, port, transport):  
    """Initializes and runs the Agent Cards MCP server.

    Args:
        host: The hostname or IP address to bind the server to.
        port: The port number to bind the server to.
        transport: The transport mechanism for the MCP server (e.g., 'stdio', 'sse').
    """
    logger.info('Starting Angular Motion MCP Server')
# Initialize FastMCP server
    mcp = FastMCP("angular_motion")

    def with_diagram_payload(result_text: str, payload: Dict) -> str:
        """Append machine-readable diagram payload for downstream UI rendering."""
        return (
            f"{result_text}\n\n"
            "DIAGRAM_JSON_START\n"
            f"{json.dumps(payload)}\n"
            "DIAGRAM_JSON_END"
        )


    @mcp.tool()
    async def angular_kinematics(kinematics_data: str) -> str:
        """
        Solve angular kinematics problems using rotational motion equations.
        
        Args:
            kinematics_data: JSON string with kinematics parameters
                            Examples:
                            '{"omega_0": 5, "alpha": 2, "time": 3}'
                            '{"omega_0": 0, "omega_f": 10, "theta": 25}'
                            '{"theta_0": 0, "omega_0": 5, "alpha": 2, "time": 4}'
            
        Returns:
            str: Complete angular kinematics analysis
        """
        try:
            data = json.loads(kinematics_data)
            
            result = f"""
    Angular Kinematics Analysis:
    ===========================

    Given Data:
    """
            
            for key, value in data.items():
                unit = ""
                if "omega" in key.lower():
                    unit = " rad/s"
                elif "alpha" in key.lower():
                    unit = " rad/s²"
                elif "theta" in key.lower():
                    unit = " rad"
                elif "time" in key.lower():
                    unit = " s"
                
                result += f"- {key.replace('_', ' ').title()}: {value}{unit}\n"
            
            result += f"""
    Angular Kinematic Equations:
    1. θ = θ₀ + ω₀t + ½αt²  (angular displacement)
    2. ω = ω₀ + αt           (angular velocity from acceleration)
    3. ω² = ω₀² + 2α(θ-θ₀)   (angular velocity from displacement)

    Analysis:
    """
            
            # Extract known values
            theta_0 = data.get("theta_0", 0)
            omega_0 = data.get("omega_0", 0)
            omega_f = data.get("omega_f")
            alpha = data.get("alpha")
            time = data.get("time")
            theta = data.get("theta")
            time_window = None
            
            # Solve based on given information
            if omega_0 is not None and alpha is not None and time is not None:
                # Calculate final values using equations 1 and 2
                theta_f = theta_0 + calculate_angular_displacement(omega_0, alpha, time)
                omega_f_calc = calculate_angular_velocity_from_acceleration(omega_0, alpha, time)
                
                result += f"""Using ω₀, α, and t:
    θf = θ₀ + ω₀t + ½αt² = {theta_0} + {omega_0}×{time} + ½×{alpha}×{time}² = {theta_f:.3f} rad
    ωf = ω₀ + αt = {omega_0} + {alpha}×{time} = {omega_f_calc:.3f} rad/s

    Total angular displacement: Δθ = {theta_f - theta_0:.3f} rad = {radians_to_degrees(theta_f - theta_0):.1f}°
    Angular acceleration: α = {alpha:.3f} rad/s²

    """
                time_window = max(0.0, float(time))
                
            elif omega_0 is not None and omega_f is not None and alpha is not None:
                # Calculate time and displacement using equation 2
                time_calc = (omega_f - omega_0) / alpha
                theta_change = (omega_f**2 - omega_0**2) / (2 * alpha)
                theta_f = theta_0 + theta_change
                
                result += f"""Using ω₀, ωf, and α:
    Time calculation: t = (ωf - ω₀)/α = ({omega_f} - {omega_0})/{alpha} = {time_calc:.3f} s
    Angular displacement: Δθ = (ωf² - ω₀²)/(2α) = ({omega_f}² - {omega_0}²)/(2×{alpha}) = {theta_change:.3f} rad
    Final position: θf = θ₀ + Δθ = {theta_0} + {theta_change:.3f} = {theta_f:.3f} rad

    """
                time_window = max(0.0, float(time_calc))
                
            elif omega_0 is not None and alpha is not None and theta is not None:
                # Calculate final velocity using equation 3
                theta_change = theta - theta_0 if theta_0 else theta
                omega_f_calc = calculate_angular_velocity_from_displacement(omega_0, alpha, theta_change)
                time_calc = (omega_f_calc - omega_0) / alpha if alpha != 0 else 0
                
                result += f"""Using ω₀, α, and θ:
    Angular displacement: Δθ = {theta_change:.3f} rad
    Final angular velocity: ωf = √(ω₀² + 2αΔθ) = √({omega_0}² + 2×{alpha}×{theta_change:.3f}) = {omega_f_calc:.3f} rad/s
    Time taken: t = (ωf - ω₀)/α = {time_calc:.3f} s

    """
                time_window = max(0.0, float(time_calc))
            
            # Convert to degrees and other units
            if 'omega_f_calc' in locals():
                omega_f = omega_f_calc
            if 'theta_f' in locals():
                result += f"""Unit Conversions:
    Angular displacement: {theta_f - theta_0:.3f} rad = {radians_to_degrees(theta_f - theta_0):.1f}°
    """
            
            if omega_f:
                result += f"""Final angular velocity: {omega_f:.3f} rad/s = {omega_f * 60/(2*math.pi):.1f} rpm

    """
            
            result += f"""Physical Interpretation:
    - Angular displacement (θ): How much the object has rotated
    - Angular velocity (ω): How fast the object is rotating
    - Angular acceleration (α): How rapidly the rotation is changing
    - All quantities follow the same mathematical relationships as linear motion

    Applications:
    - Spinning wheels, gears, and rotors
    - Planetary and satellite motion
    - Gyroscopes and spinning tops
    - Motors and turbines
    - Amusement park rides

    Key Relationships:
    - Linear velocity: v = rω (where r is radius)
    - Linear acceleration: a = rα (tangential acceleration)
    - Period: T = 2π/ω (time for one complete revolution)
    - Frequency: f = ω/(2π) = 1/T (revolutions per second)
    """
            
            # Build a rotation graphs payload for UI if we have enough signal.
            try:
                alpha_plot = float(alpha) if alpha is not None else 0.0
                omega0_plot = float(omega_0) if omega_0 is not None else 0.0
                theta0_plot = float(theta_0) if theta_0 is not None else 0.0
                t_end = float(time_window) if time_window is not None else (max(0.0, float(time)) if time is not None else 5.0)
                if t_end <= 0:
                    t_end = 5.0

                n = 30
                times = [i * t_end / (n - 1) for i in range(n)]
                theta_series = [theta0_plot + omega0_plot * t + 0.5 * alpha_plot * t * t for t in times]
                omega_series = [omega0_plot + alpha_plot * t for t in times]
                alpha_series = [alpha_plot for _ in times]

                payload = {
                    "type": "rotation_graphs_plot",
                    "title": "Angular Kinematics Graphs",
                    "times_s": times,
                    "theta_series_rad": theta_series,
                    "omega_series_rad_s": omega_series,
                    "alpha_series_rad_s2": alpha_series,
                }
                return with_diagram_payload(result, payload)
            except Exception:
                return result
            
        except Exception as e:
            return f"Error in angular kinematics analysis: {str(e)}"

    @mcp.tool()
    async def calculate_moment_of_inertia(object_data: str) -> str:
        """
        Calculate moment of inertia for common geometric shapes.
        
        Args:
            object_data: JSON string with object parameters
                        Examples:
                        '{"shape": "rod", "mass": 2, "length": 1.5, "axis": "center"}'
                        '{"shape": "disk", "mass": 5, "radius": 0.3}'
                        '{"shape": "sphere", "mass": 3, "radius": 0.2, "hollow": false}'
                        '{"shape": "cylinder", "mass": 4, "radius": 0.25, "hollow": true}'
            
        Returns:
            str: Complete moment of inertia calculation with explanation
        """
        try:
            data = json.loads(object_data)
            
            shape = data["shape"].lower()
            mass = data["mass"]
            
            result = f"""
    Moment of Inertia Calculation:
    =============================

    Object: {shape.title()}
    Mass: {mass:.3f} kg

    """
            
            if shape == "rod":
                length = data["length"]
                axis = data.get("axis", "center")
                I = calculate_moment_of_inertia_rod(mass, length, axis)
                
                result += f"""Rod Parameters:
    Length: {length:.3f} m
    Rotation axis: {axis.title()}

    Moment of Inertia Formula:
    """
                if axis.lower() == "center":
                    result += f"I = (1/12)ML² (rotation about center)\n"
                else:
                    result += f"I = (1/3)ML² (rotation about end)\n"
                    
                result += f"""
    Calculation:
    I = {I:.6f} kg⋅m²

    """
                
            elif shape == "disk":
                radius = data["radius"]
                I = calculate_moment_of_inertia_disk(mass, radius)
                
                result += f"""Disk Parameters:
    Radius: {radius:.3f} m

    Moment of Inertia Formula:
    I = (1/2)MR² (solid disk about center)

    Calculation:
    I = (1/2) × {mass:.3f} × {radius:.3f}² = {I:.6f} kg⋅m²

    """
                
            elif shape == "sphere":
                radius = data["radius"]
                hollow = data.get("hollow", False)
                I = calculate_moment_of_inertia_sphere(mass, radius, hollow)
                
                result += f"""Sphere Parameters:
    Radius: {radius:.3f} m
    Type: {'Hollow' if hollow else 'Solid'}

    Moment of Inertia Formula:
    """
                if hollow:
                    result += f"I = (2/3)MR² (hollow sphere)\n"
                else:
                    result += f"I = (2/5)MR² (solid sphere)\n"
                    
                result += f"""
    Calculation:
    I = {I:.6f} kg⋅m²

    """
                
            elif shape == "cylinder":
                radius = data["radius"]
                hollow = data.get("hollow", False)
                I = calculate_moment_of_inertia_cylinder(mass, radius, hollow)
                
                result += f"""Cylinder Parameters:
    Radius: {radius:.3f} m
    Type: {'Hollow (thin-walled)' if hollow else 'Solid'}

    Moment of Inertia Formula:
    """
                if hollow:
                    result += f"I = MR² (hollow cylinder)\n"
                else:
                    result += f"I = (1/2)MR² (solid cylinder)\n"
                    
                result += f"""
    Calculation:
    I = {I:.6f} kg⋅m²

    """
            
            # Apply parallel axis theorem if offset is given
            if "offset" in data:
                offset = data["offset"]
                I_offset = calculate_parallel_axis_theorem(I, mass, offset)
                
                result += f"""Parallel Axis Theorem Application:
    Offset from center of mass: {offset:.3f} m

    Parallel Axis Theorem: I = I_cm + Md²
    I_offset = {I:.6f} + {mass:.3f} × {offset:.3f}² = {I_offset:.6f} kg⋅m²

    """
                I = I_offset
            
            result += f"""Physical Interpretation:
    - Moment of inertia (I): Rotational analog of mass
    - Units: kg⋅m² (kilogram-meters squared)
    - Larger I means more torque needed for same angular acceleration
    - Depends on mass distribution relative to rotation axis

    Key Concepts:
    - Mass farther from axis contributes more to I (r² dependence)
    - Hollow objects have larger I than solid objects of same mass/size
    - Parallel axis theorem: I increases with offset from center of mass
    - Different axes give different moments of inertia

    Applications:
    - Flywheel design (maximize I for energy storage)
    - Gyroscope stability (large I resists angular acceleration)
    - Vehicle dynamics (wheel and axle I affects acceleration)
    - Sports equipment (bat, club, racket swing dynamics)

    Rotational Dynamics:
    - Torque equation: τ = Iα (analogous to F = ma)
    - Rotational kinetic energy: KE_rot = (1/2)Iω²
    - Angular momentum: L = Iω
    """
            
            return result
            
        except Exception as e:
            return f"Error in moment of inertia calculation: {str(e)}"

    @mcp.tool()
    async def calculate_torque(torque_data: str) -> str:
        """
        Calculate torque using various methods.
        
        Args:
            torque_data: JSON string with torque parameters
                        Examples:
                        '{"force": 50, "radius": 0.3, "angle": 90}'
                        '{"moment_of_inertia": 0.5, "angular_acceleration": 4}'
                        '{"forces": [{"force": 30, "radius": 0.2, "angle": 90}, {"force": 20, "radius": 0.4, "angle": 60}]}'
            
        Returns:
            str: Complete torque calculation and analysis
        """
        try:
            data = json.loads(torque_data)
            
            result = f"""
    Torque Calculation:
    ==================

    Given Data:
    """
            
            for key, value in data.items():
                if key != "forces":
                    result += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            result += f"""
    Torque Formulas:
    1. τ = r × F sin(θ) = rF⊥  (force and radius)
    2. τ = Iα                  (moment of inertia and angular acceleration)
    3. τ_net = Στ              (sum of individual torques)

    """
            
            total_torque = 0
            
            # Method 1: Single force calculation
            if "force" in data and "radius" in data:
                force = data["force"]
                radius = data["radius"]
                angle = data.get("angle", 90)  # Default perpendicular
                
                angle_rad = degrees_to_radians(angle)
                torque = radius * force * math.sin(angle_rad)
                total_torque = torque
                
                perpendicular_force = force * math.sin(angle_rad)
                parallel_force = force * math.cos(angle_rad)
                
                result += f"""Method 1: Force and Radius
    Force: {force:.3f} N
    Radius: {radius:.3f} m
    Angle between F and r: {angle:.1f}°

    Force Components:
    - Perpendicular component: F⊥ = F sin(θ) = {force:.3f} × sin({angle:.1f}°) = {perpendicular_force:.3f} N
    - Parallel component: F∥ = F cos(θ) = {force:.3f} × cos({angle:.1f}°) = {parallel_force:.3f} N

    Torque Calculation:
    τ = r × F sin(θ) = {radius:.3f} × {force:.3f} × sin({angle:.1f}°) = {torque:.3f} N⋅m

    Note: Only the perpendicular component creates torque!

    """
            
            # Method 2: Moment of inertia and angular acceleration
            elif "moment_of_inertia" in data and "angular_acceleration" in data:
                I = data["moment_of_inertia"]
                alpha = data["angular_acceleration"]
                
                torque = I * alpha
                total_torque = torque
                
                result += f"""Method 2: Rotational Dynamics
    Moment of inertia: I = {I:.3f} kg⋅m²
    Angular acceleration: α = {alpha:.3f} rad/s²

    Torque Calculation:
    τ = Iα = {I:.3f} × {alpha:.3f} = {torque:.3f} N⋅m

    This is the NET torque required to produce the given angular acceleration.

    """
            
            # Method 3: Multiple forces
            elif "forces" in data:
                forces = data["forces"]
                
                result += f"""Method 3: Multiple Forces
    Number of forces: {len(forces)}

    Individual Torque Calculations:
    """
                
                for i, force_data in enumerate(forces):
                    f = force_data["force"]
                    r = force_data["radius"] 
                    angle = force_data.get("angle", 90)
                    
                    angle_rad = degrees_to_radians(angle)
                    tau_i = r * f * math.sin(angle_rad)
                    total_torque += tau_i
                    
                    result += f"""Force {i+1}:
    - Force: {f:.3f} N at radius {r:.3f} m, angle {angle:.1f}°
    - Torque: τ{i+1} = {r:.3f} × {f:.3f} × sin({angle:.1f}°) = {tau_i:.3f} N⋅m

    """
                
                result += f"""Net Torque:
    τ_net = Στ = {total_torque:.3f} N⋅m

    """
            
            # Determine direction
            direction = "counterclockwise (positive)" if total_torque >= 0 else "clockwise (negative)"
            
            result += f"""Results Summary:
    Net Torque: {total_torque:.3f} N⋅m
    Direction: {direction}
    Magnitude: {abs(total_torque):.3f} N⋅m

    Physical Interpretation:
    - Torque is the rotational analog of force
    - Units: N⋅m (Newton-meters)
    - Positive torque: counterclockwise rotation (right-hand rule)
    - Negative torque: clockwise rotation
    - Only forces perpendicular to radius create torque

    Key Concepts:
    - Lever arm: effective radius for torque production
    - Maximum torque when force is perpendicular to radius (θ = 90°)
    - Zero torque when force is parallel to radius (θ = 0° or 180°)
    - Torque depends on both force magnitude and application point

    Applications:
    - Wrench and bolt tightening (leverage effect)
    - Engine and motor design (torque output)
    - Gear systems (torque multiplication)
    - Door hinges and handles (optimal placement)
    - Sports equipment (bat/club swing mechanics)

    Equilibrium Conditions:
    - Static equilibrium: Στ = 0 (no angular acceleration)
    - Dynamic equilibrium: Στ = Iα (constant angular acceleration)
    - Balanced systems require equal and opposite torques
    """
            
            # Calculate resulting angular acceleration if I is known
            if "moment_of_inertia" in data and "force" in data:
                I = data["moment_of_inertia"]
                alpha_result = total_torque / I
                
                result += f"""
    Resulting Motion:
    If moment of inertia I = {I:.3f} kg⋅m²:
    Angular acceleration: α = τ/I = {total_torque:.3f}/{I:.3f} = {alpha_result:.3f} rad/s²

    Time to reach 1 revolution (2π rad) from rest:
    Using θ = ½αt²: t = √(2θ/α) = √(2×2π/{alpha_result:.3f}) = {math.sqrt(4*math.pi/abs(alpha_result)) if alpha_result != 0 else 'undefined':.2f} s
    """
            
            # Diagram payload for torque lever visualization (best for force-radius cases).
            if "force" in data and "radius" in data:
                force = float(data["force"])
                radius = float(data["radius"])
                angle = float(data.get("angle", 90))
                angle_rad = degrees_to_radians(angle)
                f_perp = force * math.sin(angle_rad)
                payload = {
                    "type": "torque_lever_diagram",
                    "title": "Torque Lever Diagram",
                    "force_n": force,
                    "radius_m": radius,
                    "angle_deg": angle,
                    "force_perpendicular_n": f_perp,
                    "torque_nm": total_torque,
                    "direction": "counterclockwise" if total_torque >= 0 else "clockwise",
                }
                return with_diagram_payload(result, payload)

            return result
            
        except Exception as e:
            return f"Error in torque calculation: {str(e)}"

    @mcp.tool()
    async def angular_momentum_conservation(momentum_data: str) -> str:
        """
        Analyze angular momentum and conservation in rotational systems.
        
        Args:
            momentum_data: JSON string with angular momentum parameters
                        Examples:
                        '{"initial": {"I": 2, "omega": 5}, "final": {"I": 1, "omega": null}}'
                        '{"objects": [{"I": 0.5, "omega": 10}, {"I": 1.2, "omega": -3}]}'
                        '{"figure_skater": {"I_extended": 5, "omega_extended": 2, "I_tucked": 1.5}}'
            
        Returns:
            str: Complete angular momentum analysis
        """
        try:
            data = json.loads(momentum_data)
            
            result = f"""
    Angular Momentum Conservation Analysis:
    =====================================

    """
            
            # Conservation in changing moment of inertia (like figure skater)
            if "figure_skater" in data or ("initial" in data and "final" in data):
                if "figure_skater" in data:
                    skater = data["figure_skater"]
                    I1 = skater["I_extended"]
                    omega1 = skater["omega_extended"] 
                    I2 = skater["I_tucked"]
                    omega2 = skater.get("omega_tucked")
                    
                    scenario = "Figure Skater Spin"
                    initial_desc = "Arms extended"
                    final_desc = "Arms tucked"
                else:
                    initial = data["initial"]
                    final = data["final"]
                    I1 = initial["I"]
                    omega1 = initial["omega"]
                    I2 = final["I"]
                    omega2 = final.get("omega")
                    
                    scenario = "Changing Moment of Inertia"
                    initial_desc = "Initial state"
                    final_desc = "Final state"
                
                # Calculate missing omega using conservation
                L_initial = I1 * omega1
                if omega2 is None:
                    omega2 = L_initial / I2
                L_final = I2 * omega2
                
                result += f"""Scenario: {scenario}

    {initial_desc.title()}:
    - Moment of inertia: I₁ = {I1:.3f} kg⋅m²
    - Angular velocity: ω₁ = {omega1:.3f} rad/s
    - Angular momentum: L₁ = I₁ω₁ = {I1:.3f} × {omega1:.3f} = {L_initial:.3f} kg⋅m²/s

    {final_desc.title()}:
    - Moment of inertia: I₂ = {I2:.3f} kg⋅m²
    - Angular velocity: ω₂ = ? (to be found)

    Conservation of Angular Momentum:
    L₁ = L₂ (no external torques)
    I₁ω₁ = I₂ω₂

    Solving for ω₂:
    ω₂ = (I₁ω₁)/I₂ = ({I1:.3f} × {omega1:.3f})/{I2:.3f} = {omega2:.3f} rad/s

    Final Angular Momentum:
    L₂ = I₂ω₂ = {I2:.3f} × {omega2:.3f} = {L_final:.3f} kg⋅m²/s

    Verification: |L₁ - L₂| = |{L_initial:.3f} - {L_final:.3f}| = {abs(L_initial - L_final):.6f} ≈ 0 ✓

    """
                
                # Energy analysis
                KE_initial = 0.5 * I1 * omega1**2
                KE_final = 0.5 * I2 * omega2**2
                energy_change = KE_final - KE_initial
                
                result += f"""Rotational Energy Analysis:
    Initial kinetic energy: KE₁ = ½I₁ω₁² = ½ × {I1:.3f} × {omega1:.3f}² = {KE_initial:.3f} J
    Final kinetic energy: KE₂ = ½I₂ω₂² = ½ × {I2:.3f} × {omega2:.3f}² = {KE_final:.3f} J

    Energy change: ΔKE = KE₂ - KE₁ = {energy_change:.3f} J

    """
                if energy_change > 0:
                    result += "Energy increased - work was done by internal forces (muscle work)\n"
                elif energy_change < 0:
                    result += "Energy decreased - internal work was done against the system\n"
                else:
                    result += "Energy conserved - no internal work\n"
            
            # Multiple object system
            elif "objects" in data:
                objects = data["objects"]
                
                result += f"""Multiple Object System:
    Number of objects: {len(objects)}

    Individual Angular Momenta:
    """
                
                total_L = 0
                total_KE = 0
                
                for i, obj in enumerate(objects):
                    I = obj["I"]
                    omega = obj["omega"]
                    L = I * omega
                    KE = 0.5 * I * omega**2
                    
                    total_L += L
                    total_KE += KE
                    
                    direction = "counterclockwise" if omega >= 0 else "clockwise"
                    
                    result += f"""Object {i+1}:
    - Moment of inertia: I = {I:.3f} kg⋅m²
    - Angular velocity: ω = {omega:.3f} rad/s ({direction})
    - Angular momentum: L = Iω = {L:.3f} kg⋅m²/s
    - Rotational KE: KE = ½Iω² = {KE:.3f} J

    """
                
                result += f"""System Totals:
    Total angular momentum: L_total = ΣL = {total_L:.3f} kg⋅m²/s
    Total rotational energy: KE_total = ΣKE = {total_KE:.3f} J

    Conservation Analysis:
    - If no external torques act: L_total remains constant
    - Individual objects can exchange angular momentum
    - Energy may or may not be conserved (depends on collision type)

    """
            
            result += f"""Angular Momentum Fundamentals:

    Definition and Formula:
    - Angular momentum: L = Iω (for rigid body rotation)
    - Vector quantity with direction given by right-hand rule
    - Units: kg⋅m²/s (kilogram-meters squared per second)

    Conservation Law:
    - Conserved when net external torque is zero: Στ_ext = 0
    - Individual objects can exchange angular momentum
    - Total system angular momentum remains constant

    Key Relationships:
    - Torque and angular momentum: τ = dL/dt
    - Angular impulse: Jₐ = ∫τ dt = ΔL
    - Relation to linear momentum: L = r × p (for point mass)

    Real-World Applications:

    1. Figure Skating:
    - Skater pulls arms in: I decreases, ω increases
    - Demonstrates conservation dramatically
    - Energy comes from muscle work

    2. Gyroscopes:
    - Large angular momentum resists orientation changes
    - Used in navigation and stabilization
    - Precession when external torque applied

    3. Planetary Motion:
    - Planets conserve angular momentum in elliptical orbits
    - Kepler's second law: equal areas in equal times
    - Angular momentum per unit mass is constant

    4. Rotating Machinery:
    - Flywheels store energy using angular momentum
    - Turbines and generators rely on rotational dynamics
    - Balance wheels in timepieces

    5. Sports Applications:
    - Divers and gymnasts control rotation by changing I
    - Throwing sports use angular momentum transfer
    - Spinning balls curve due to Magnus effect

    Conservation vs. Non-Conservation:
    - Conserved: Ice skater spinning, planet orbiting, isolated rotating system
    - Not conserved: Friction present, external torques, motors/brakes applied
    - Collision analysis: Usually conserved during brief collision time
    """
            
            return result
            
        except Exception as e:
            return f"Error in angular momentum analysis: {str(e)}"

    @mcp.tool()
    async def rotational_energy(energy_data: str) -> str:
        """
        Calculate rotational kinetic energy and analyze energy transformations.
        
        Args:
            energy_data: JSON string with rotational energy parameters
                        Examples:
                        '{"moment_of_inertia": 2, "angular_velocity": 5}'
                        '{"rolling_object": {"mass": 10, "radius": 0.5, "velocity": 3, "shape": "cylinder"}}'
                        '{"energy_transformation": {"I": 1.5, "omega_initial": 8, "omega_final": 12}}'
            
        Returns:
            str: Complete rotational energy analysis
        """
        try:
            data = json.loads(energy_data)
            
            result = f"""
    Rotational Energy Analysis:
    ==========================

    """
            
            # Simple rotational kinetic energy
            if "moment_of_inertia" in data and "angular_velocity" in data:
                I = data["moment_of_inertia"]
                omega = data["angular_velocity"]
                
                KE_rot = 0.5 * I * omega**2
                
                result += f"""Rotational Kinetic Energy Calculation:

    Given:
    - Moment of inertia: I = {I:.3f} kg⋅m²
    - Angular velocity: ω = {omega:.3f} rad/s

    Rotational Kinetic Energy Formula:
    KE_rot = ½Iω²

    Calculation:
    KE_rot = ½ × {I:.3f} × {omega:.3f}² = {KE_rot:.3f} J

    """
                
                # Equivalent linear energy for comparison
                if "equivalent_mass" in data:
                    m_equiv = data["equivalent_mass"]
                    v_equiv = math.sqrt(2 * KE_rot / m_equiv)
                    
                    result += f"""Equivalent Linear Motion:
    If a {m_equiv:.1f} kg object had the same kinetic energy:
    Required velocity: v = √(2KE/m) = √(2×{KE_rot:.3f}/{m_equiv:.3f}) = {v_equiv:.2f} m/s

    """
            
            # Rolling motion analysis
            elif "rolling_object" in data:
                rolling = data["rolling_object"]
                mass = rolling["mass"]
                radius = rolling["radius"]
                velocity = rolling["velocity"]  # Linear velocity of center of mass
                shape = rolling.get("shape", "cylinder")
                
                # Calculate moment of inertia based on shape
                if shape.lower() == "cylinder":
                    I = 0.5 * mass * radius**2
                    shape_factor = "½"
                elif shape.lower() == "sphere":
                    I = (2/5) * mass * radius**2
                    shape_factor = "2/5"
                elif shape.lower() == "disk":
                    I = 0.5 * mass * radius**2
                    shape_factor = "½"
                elif shape.lower() == "hoop":
                    I = mass * radius**2
                    shape_factor = "1"
                else:
                    I = 0.5 * mass * radius**2  # Default to cylinder
                    shape_factor = "½"
                
                # For rolling motion: v = ωr
                omega = velocity / radius
                
                # Calculate energies
                KE_trans = 0.5 * mass * velocity**2
                KE_rot = 0.5 * I * omega**2
                KE_total = KE_trans + KE_rot
                
                result += f"""Rolling Motion Analysis:

    Object: {shape.title()}
    Mass: {mass:.3f} kg
    Radius: {radius:.3f} m
    Linear velocity (center of mass): v = {velocity:.3f} m/s

    Moment of Inertia:
    I = {shape_factor}MR² = {shape_factor} × {mass:.3f} × {radius:.3f}² = {I:.3f} kg⋅m²

    Rolling Condition:
    No slipping: v = ωr
    Angular velocity: ω = v/r = {velocity:.3f}/{radius:.3f} = {omega:.3f} rad/s

    Energy Components:
    1. Translational KE: KE_trans = ½mv² = ½ × {mass:.3f} × {velocity:.3f}² = {KE_trans:.3f} J
    2. Rotational KE: KE_rot = ½Iω² = ½ × {I:.3f} × {omega:.3f}² = {KE_rot:.3f} J

    Total Kinetic Energy:
    KE_total = KE_trans + KE_rot = {KE_trans:.3f} + {KE_rot:.3f} = {KE_total:.3f} J

    Energy Distribution:
    - Translational: {(KE_trans/KE_total)*100:.1f}% of total energy
    - Rotational: {(KE_rot/KE_total)*100:.1f}% of total energy

    """
                
                # Compare with sliding object
                KE_slide = 0.5 * mass * velocity**2
                energy_ratio = KE_total / KE_slide
                
                result += f"""Comparison with Sliding Object:
    If the same object were sliding (not rolling) at {velocity:.3f} m/s:
    KE_sliding = ½mv² = {KE_slide:.3f} J

    Rolling vs. Sliding:
    - Rolling object has {energy_ratio:.2f}× more kinetic energy
    - Extra energy is in rotational motion
    - Rolling object needs more energy to reach same linear speed

    """
            
            # Energy transformation analysis
            elif "energy_transformation" in data:
                transform = data["energy_transformation"]
                I = transform["I"]
                omega_i = transform["omega_initial"]
                omega_f = transform["omega_final"]
                
                KE_i = 0.5 * I * omega_i**2
                KE_f = 0.5 * I * omega_f**2
                energy_change = KE_f - KE_i
                
                result += f"""Rotational Energy Transformation:

    Initial State:
    - Angular velocity: ω₁ = {omega_i:.3f} rad/s
    - Rotational KE: KE₁ = ½Iω₁² = ½ × {I:.3f} × {omega_i:.3f}² = {KE_i:.3f} J

    Final State:
    - Angular velocity: ω₂ = {omega_f:.3f} rad/s  
    - Rotational KE: KE₂ = ½Iω₂² = ½ × {I:.3f} × {omega_f:.3f}² = {KE_f:.3f} J

    Energy Change:
    ΔKE = KE₂ - KE₁ = {KE_f:.3f} - {KE_i:.3f} = {energy_change:.3f} J

    """
                
                if energy_change > 0:
                    result += f"""Energy Analysis:
    - Energy increased by {energy_change:.3f} J
    - Work was done ON the system (motor, applied torque)
    - External energy input required

    """
                elif energy_change < 0:
                    result += f"""Energy Analysis:
    - Energy decreased by {abs(energy_change):.3f} J  
    - Work was done BY the system (generator, friction)
    - Energy was dissipated or extracted

    """
                else:
                    result += f"""Energy Analysis:
    - No change in rotational energy
    - System in equilibrium or conservative transformation

    """
            
            result += f"""Rotational Energy Concepts:

    Fundamental Relationships:
    - Rotational KE: KE_rot = ½Iω²
    - Analogous to linear KE: KE_trans = ½mv²
    - Work-energy theorem: W = ΔKE_rot = ∫τ dθ
    - Power: P = τω (analogous to P = Fv)

    Types of Rotational Motion:
    1. Pure rotation: Only rotational KE
    2. Pure translation: Only translational KE  
    3. Rolling motion: Both translational and rotational KE
    4. General motion: Translation + rotation about center of mass

    Energy Conservation:
    - Conservative systems: Total mechanical energy constant
    - Non-conservative forces: Energy dissipated (friction, air resistance)
    - Rolling friction: Causes gradual energy loss
    - Elastic collisions: Kinetic energy conserved

    Applications:

    1. Flywheels:
    - Store rotational energy for later use
    - High I and ω for maximum energy storage
    - Used in vehicles, power plants, UPS systems

    2. Turbines and Generators:
    - Convert fluid energy to rotational energy
    - Then to electrical energy via electromagnetic induction
    - Efficiency depends on minimizing energy losses

    3. Sports and Recreation:
    - Spinning tops: Convert potential to rotational energy
    - Figure skating: Change I to control ω while conserving L
    - Yo-yos: Transform between gravitational PE and rotational KE

    4. Vehicle Dynamics:
    - Rolling wheels store energy in rotation
    - Affects acceleration and braking performance
    - Lighter wheels improve acceleration

    5. Machinery Design:
    - Rotating shafts, gears, and pulleys
    - Energy transmission and transformation
    - Inertial effects in start-up and shutdown

    Energy Efficiency:
    - Rolling more efficient than sliding (less friction)
    - Bearing design critical for minimizing energy loss
    - Lubrication reduces rotational friction
    - Aerodynamic effects at high speeds
    """
            
            return result
            
        except Exception as e:
            return f"Error in rotational energy analysis: {str(e)}"

    @mcp.tool()
    async def angular_impulse_momentum(impulse_data: str) -> str:
        """
        Analyze angular impulse and its effect on angular momentum.
        
        Args:
            impulse_data: JSON string with angular impulse parameters
                        Examples:
                        '{"torque": 15, "time": 2, "initial_omega": 3, "moment_of_inertia": 1.2}'
                        '{"angular_momentum_change": 25, "time": 1.5}'
                        '{"variable_torque": {"peak": 20, "time": 3, "shape": "triangular"}}'
            
        Returns:
            str: Complete angular impulse-momentum analysis
        """
        try:
            data = json.loads(impulse_data)
            
            result = f"""
    Angular Impulse-Momentum Analysis:
    =================================

    """
            
            # Constant torque case
            if "torque" in data and "time" in data:
                torque = data["torque"]
                time = data["time"]
                angular_impulse = torque * time
                
                result += f"""Constant Torque Analysis:

    Given:
    - Applied torque: τ = {torque:.3f} N⋅m
    - Time interval: Δt = {time:.3f} s

    Angular Impulse Calculation:
    Angular impulse formula: J_angular = τ × Δt
    J_angular = {torque:.3f} × {time:.3f} = {angular_impulse:.3f} N⋅m⋅s

    """
                
                # If moment of inertia and initial omega are given
                if "moment_of_inertia" in data and "initial_omega" in data:
                    I = data["moment_of_inertia"]
                    omega_i = data["initial_omega"]
                    
                    # Calculate angular momentum change
                    delta_L = angular_impulse
                    delta_omega = delta_L / I
                    omega_f = omega_i + delta_omega
                    
                    L_i = I * omega_i
                    L_f = I * omega_f
                    
                    result += f"""Angular Momentum Analysis:

    Initial conditions:
    - Moment of inertia: I = {I:.3f} kg⋅m²
    - Initial angular velocity: ω₁ = {omega_i:.3f} rad/s  
    - Initial angular momentum: L₁ = Iω₁ = {L_i:.3f} kg⋅m²/s

    Angular Impulse-Momentum Theorem:
    J_angular = ΔL = L₂ - L₁

    Change in angular momentum:
    ΔL = J_angular = {angular_impulse:.3f} kg⋅m²/s

    Final angular momentum:
    L₂ = L₁ + ΔL = {L_i:.3f} + {angular_impulse:.3f} = {L_f:.3f} kg⋅m²/s

    Final angular velocity:
    ω₂ = L₂/I = {L_f:.3f}/{I:.3f} = {omega_f:.3f} rad/s

    Change in angular velocity:
    Δω = ω₂ - ω₁ = {omega_f:.3f} - {omega_i:.3f} = {delta_omega:.3f} rad/s

    """
                    
                    # Energy analysis
                    KE_i = 0.5 * I * omega_i**2
                    KE_f = 0.5 * I * omega_f**2
                    work_done = KE_f - KE_i
                    
                    result += f"""Energy Analysis:
    Initial rotational KE: KE₁ = ½Iω₁² = {KE_i:.3f} J
    Final rotational KE: KE₂ = ½Iω₂² = {KE_f:.3f} J
    Work done by torque: W = ΔKE = {work_done:.3f} J

    Average angular displacement during acceleration:
    θ = ω_avg × t = ((ω₁ + ω₂)/2) × t = {(omega_i + omega_f)/2 * time:.3f} rad = {radians_to_degrees((omega_i + omega_f)/2 * time):.1f}°

    """
            
            # Angular momentum change given directly
            elif "angular_momentum_change" in data:
                delta_L = data["angular_momentum_change"]
                time = data.get("time")
                
                result += f"""Angular Momentum Change Analysis:

    Given:
    - Change in angular momentum: ΔL = {delta_L:.3f} kg⋅m²/s
    """
                
                if time:
                    avg_torque = delta_L / time
                    result += f"- Time interval: Δt = {time:.3f} s\n"
                    result += f"""
    Average Torque Calculation:
    Using J_angular = τ_avg × Δt = ΔL
    τ_avg = ΔL/Δt = {delta_L:.3f}/{time:.3f} = {avg_torque:.3f} N⋅m

    """
            
            # Variable torque analysis
            elif "variable_torque" in data:
                var_torque = data["variable_torque"]
                peak = var_torque["peak"]
                time = var_torque["time"]
                shape = var_torque.get("shape", "triangular")
                
                result += f"""Variable Torque Analysis:

    Torque Profile:
    - Shape: {shape.title()}
    - Peak torque: τ_max = {peak:.3f} N⋅m
    - Duration: Δt = {time:.3f} s

    """
                
                if shape.lower() == "triangular":
                    # For triangular pulse: average = peak/2
                    avg_torque = peak / 2
                    angular_impulse = avg_torque * time
                    
                    result += f"""Triangular Torque Pulse:
    - Torque increases linearly from 0 to {peak:.3f} N⋅m
    - Then decreases linearly back to 0
    - Average torque: τ_avg = τ_max/2 = {avg_torque:.3f} N⋅m

    Angular Impulse:
    J_angular = τ_avg × Δt = {avg_torque:.3f} × {time:.3f} = {angular_impulse:.3f} N⋅m⋅s

    """
                elif shape.lower() == "rectangular":
                    # For rectangular pulse: average = peak
                    avg_torque = peak
                    angular_impulse = avg_torque * time
                    
                    result += f"""Rectangular Torque Pulse:
    - Constant torque of {peak:.3f} N⋅m for entire duration
    - Average torque: τ_avg = τ_max = {avg_torque:.3f} N⋅m

    Angular Impulse:
    J_angular = τ × Δt = {avg_torque:.3f} × {time:.3f} = {angular_impulse:.3f} N⋅m⋅s

    """
            
            result += f"""Angular Impulse-Momentum Theorem:

    Fundamental Relationship:
    J_angular = ∫τ dt = ΔL = L_final - L_initial

    Key Concepts:
    - Angular impulse: Product of torque and time (or integral for variable torque)
    - Units: N⋅m⋅s = kg⋅m²/s (same as angular momentum)
    - Direction follows right-hand rule (same as torque and angular momentum)
    - Always equals change in angular momentum (Newton's second law for rotation)

    Comparison with Linear Motion:
    Linear                    Rotational
    Impulse = F × Δt         Angular impulse = τ × Δt
    J = Δp                   J_angular = ΔL  
    F = dp/dt                τ = dL/dt

    Applications:

    1. Spinning Up Machinery:
    - Motors apply torque over time to reach operating speed
    - Angular impulse determines final angular momentum
    - Larger impulse needed for heavier, larger rotating parts

    2. Sports and Recreation:
    - Baseball bat swing: torque applied over swing time
    - Figure skater jump: brief torque impulse creates rotation
    - Yo-yo tricks: quick torque impulses change spin direction

    3. Vehicle Dynamics:
    - Engine torque impulse accelerates wheels
    - Brake torque impulse slows rotation
    - Steering system applies angular impulses for turning

    4. Impact and Collision:
    - Brief, large torques during rotational collisions
    - Angular impulse determines momentum transfer
    - Important in crash analysis and safety design

    5. Control Systems:
    - Precise angular impulses for positioning
    - Spacecraft attitude control using reaction wheels
    - Robotic joint control with servo motors

    Important Considerations:
    - Variable torque requires integration to find impulse
    - Direction matters: positive vs. negative angular impulse
    - External torques can change angular momentum
    - Internal torques (like muscle forces) can redistribute angular momentum
    - Conservation applies when net external torque is zero

    Problem-Solving Strategy:
    1. Identify all torques acting on the system
    2. Determine time interval of application  
    3. Calculate angular impulse (constant or variable torque)
    4. Apply impulse-momentum theorem: J_angular = ΔL
    5. Solve for unknown quantities (final ω, required τ, etc.)
    """
            
            return result
            
        except Exception as e:
            return f"Error in angular impulse-momentum analysis: {str(e)}"

    @mcp.tool()
    async def rolling_motion_analysis(rolling_data: str) -> str:
        """
        Comprehensive analysis of rolling motion without slipping.
        
        Args:
            rolling_data: JSON string with rolling motion parameters
                        Examples:
                        '{"object": "cylinder", "mass": 5, "radius": 0.3, "incline_angle": 30}'
                        '{"sphere_race": {"solid_sphere": {"mass": 2, "radius": 0.1}, "hollow_sphere": {"mass": 2, "radius": 0.1}}}'
                        '{"yo_yo": {"mass": 0.2, "radius": 0.05, "string_length": 1.5}}'
            
        Returns:
            str: Complete rolling motion analysis
        """
        try:
            data = json.loads(rolling_data)
            
            result = f"""
    Rolling Motion Analysis:
    =======================

    """
            
            # Single object rolling down incline
            if "object" in data and "incline_angle" in data:
                obj_type = data["object"]
                mass = data["mass"]
                radius = data["radius"]
                angle = data["incline_angle"]
                height = data.get("height", 2.0)  # Default height
                
                # Determine moment of inertia factor
                if obj_type.lower() in ["cylinder", "disk"]:
                    I_factor = 0.5
                    I_formula = "½MR²"
                elif obj_type.lower() == "sphere":
                    I_factor = 0.4  # 2/5
                    I_formula = "(2/5)MR²"
                elif obj_type.lower() == "hoop":
                    I_factor = 1.0
                    I_formula = "MR²"
                else:
                    I_factor = 0.5  # Default to cylinder
                    I_formula = "½MR²"
                
                I = I_factor * mass * radius**2
                
                # Rolling motion equations
                g = 9.81
                sin_angle = math.sin(degrees_to_radians(angle))
                
                # Acceleration down the incline for rolling object
                a_rolling = g * sin_angle / (1 + I_factor)
                
                # Compare with sliding object
                a_sliding = g * sin_angle
                
                # Distance along incline
                distance = height / sin_angle
                
                # Time to reach bottom
                time_rolling = math.sqrt(2 * distance / a_rolling)
                time_sliding = math.sqrt(2 * distance / a_sliding)
                
                # Final velocities
                v_final_rolling = a_rolling * time_rolling
                v_final_sliding = a_sliding * time_sliding
                
                # Angular velocity
                omega_final = v_final_rolling / radius
                
                result += f"""Rolling Object: {obj_type.title()}

    Object Properties:
    - Mass: {mass:.3f} kg
    - Radius: {radius:.3f} m  
    - Moment of inertia: I = {I_formula} = {I:.6f} kg⋅m²

    Incline Properties:
    - Angle: θ = {angle:.1f}°
    - Height: h = {height:.3f} m
    - Distance along incline: d = h/sin(θ) = {distance:.3f} m

    Rolling Motion Analysis:
    No-slip condition: v = ωr (velocity constraint)

    Forces down the incline:
    - Component of weight: Mg sin(θ) = {mass:.3f} × {g:.2f} × sin({angle:.1f}°) = {mass * g * sin_angle:.3f} N
    - Friction force: Provides torque to maintain rolling condition

    Acceleration Calculation:
    For rolling object: a = g sin(θ)/(1 + I/(MR²))
    a_rolling = {g:.2f} × sin({angle:.1f}°)/(1 + {I_factor:.1f}) = {a_rolling:.3f} m/s²

    Comparison with Sliding:
    a_sliding = g sin(θ) = {a_sliding:.3f} m/s²
    Ratio: a_rolling/a_sliding = {a_rolling/a_sliding:.3f}

    Motion Results:
    Time to reach bottom:
    - Rolling: t = √(2d/a) = √(2×{distance:.3f}/{a_rolling:.3f}) = {time_rolling:.3f} s
    - Sliding: t = √(2d/a) = √(2×{distance:.3f}/{a_sliding:.3f}) = {time_sliding:.3f} s

    Final velocities:
    - Rolling: v = at = {a_rolling:.3f} × {time_rolling:.3f} = {v_final_rolling:.3f} m/s
    - Sliding: v = at = {a_sliding:.3f} × {time_sliding:.3f} = {v_final_sliding:.3f} m/s

    Angular velocity at bottom:
    ω = v/r = {v_final_rolling:.3f}/{radius:.3f} = {omega_final:.3f} rad/s

    """
                
                # Energy analysis
                PE_initial = mass * g * height
                KE_trans_final = 0.5 * mass * v_final_rolling**2
                KE_rot_final = 0.5 * I * omega_final**2
                KE_total_final = KE_trans_final + KE_rot_final
                
                result += f"""Energy Analysis:
    Initial potential energy: PE = mgh = {mass:.3f} × {g:.2f} × {height:.3f} = {PE_initial:.3f} J

    Final kinetic energy components:
    - Translational: KE_trans = ½mv² = {KE_trans_final:.3f} J
    - Rotational: KE_rot = ½Iω² = {KE_rot_final:.3f} J
    - Total: KE_total = {KE_total_final:.3f} J

    Energy verification: |PE_initial - KE_total| = {abs(PE_initial - KE_total_final):.6f} J ≈ 0 ✓

    Energy distribution:
    - Translational: {(KE_trans_final/KE_total_final)*100:.1f}%
    - Rotational: {(KE_rot_final/KE_total_final)*100:.1f}%

    """

                payload = {
                    "type": "rolling_energy_split",
                    "title": f"Rolling Energy Split: {obj_type.title()}",
                    "object_type": obj_type.lower(),
                    "mass_kg": mass,
                    "radius_m": radius,
                    "velocity_mps": v_final_rolling,
                    "omega_rad_s": omega_final,
                    "translational_ke_j": KE_trans_final,
                    "rotational_ke_j": KE_rot_final,
                    "total_ke_j": KE_total_final,
                    "incline_angle_deg": angle,
                    "acceleration_mps2": a_rolling,
                    "time_s": time_rolling,
                }
                return with_diagram_payload(result, payload)
            
            # Sphere race comparison
            elif "sphere_race" in data:
                race = data["sphere_race"]
                
                result += f"""Sphere Race Analysis:
    Comparing different sphere types rolling down the same incline

    """
                
                race_results = []
                
                for sphere_type, sphere_data in race.items():
                    mass = sphere_data["mass"]
                    radius = sphere_data["radius"]
                    
                    if "solid" in sphere_type.lower():
                        I_factor = 2/5
                        I_formula = "(2/5)MR²"
                    elif "hollow" in sphere_type.lower():
                        I_factor = 2/3
                        I_formula = "(2/3)MR²"
                    else:
                        I_factor = 2/5  # Default to solid
                        I_formula = "(2/5)MR²"
                    
                    I = I_factor * mass * radius**2
                    
                    # Acceleration (assuming same incline)
                    g = 9.81
                    sin_angle = math.sin(degrees_to_radians(30))  # Default 30° incline
                    a = g * sin_angle / (1 + I_factor)
                    
                    race_results.append({
                        'type': sphere_type.replace('_', ' ').title(),
                        'mass': mass,
                        'radius': radius,
                        'I_factor': I_factor,
                        'I_formula': I_formula,
                        'acceleration': a
                    })
                    
                    result += f"""{sphere_type.replace('_', ' ').title()}:
    - Mass: {mass:.3f} kg
    - Radius: {radius:.3f} m
    - Moment of inertia: I = {I_formula} = {I:.6f} kg⋅m²
    - Acceleration: a = {a:.3f} m/s²

    """
                
                # Determine winner
                winner = max(race_results, key=lambda x: x['acceleration'])
                
                result += f"""Race Results:
    Winner: {winner['type']} (highest acceleration = {winner['acceleration']:.3f} m/s²)

    Physics Explanation:
    - Both spheres have same mass and radius
    - Solid sphere has smaller moment of inertia
    - Less rotational inertia means more energy available for translation
    - Solid sphere reaches bottom first despite same mass!

    Key Insight: Shape matters more than mass for rolling objects

    """
            
            # Yo-yo analysis
            elif "yo_yo" in data:
                yo_yo = data["yo_yo"]
                mass = yo_yo["mass"]
                radius = yo_yo["radius"]
                string_length = yo_yo["string_length"]
                
                # Yo-yo moment of inertia (approximated as solid disk)
                I = 0.5 * mass * radius**2
                
                # Yo-yo dynamics: 2/3 of weight goes to translation, 1/3 to rotation
                g = 9.81
                a_down = (2/3) * g  # Downward acceleration
                
                # Time to unwind
                time_unwind = math.sqrt(2 * string_length / a_down)
                
                # Velocity when fully unwound
                v_bottom = a_down * time_unwind
                omega_bottom = v_bottom / radius
                
                result += f"""Yo-Yo Analysis:

    Yo-Yo Properties:
    - Mass: {mass:.3f} kg
    - Radius: {radius:.3f} m
    - String length: {string_length:.3f} m
    - Moment of inertia: I ≈ ½MR² = {I:.6f} kg⋅m² (solid disk approximation)

    Yo-Yo Physics:
    The yo-yo is a special case of rolling motion where the "contact point" 
    is the center of the axle, not the rim.

    Forces and Acceleration:
    - Weight: W = mg = {mass * g:.3f} N (downward)
    - String tension: T (upward)
    - Rolling constraint: v = ωr (string unwinds without slipping)

    Acceleration Analysis:
    For a yo-yo: a = (2/3)g = {a_down:.3f} m/s²
    (This is independent of mass and radius!)

    Motion Results:
    Time to unwind: t = √(2L/a) = √(2×{string_length:.3f}/{a_down:.3f}) = {time_unwind:.3f} s
    Velocity at bottom: v = at = {a_down:.3f} × {time_unwind:.3f} = {v_bottom:.3f} m/s
    Angular velocity: ω = v/r = {v_bottom:.3f}/{radius:.3f} = {omega_bottom:.3f} rad/s

    """
                
                # Energy analysis
                PE_initial = mass * g * string_length
                KE_trans = 0.5 * mass * v_bottom**2
                KE_rot = 0.5 * I * omega_bottom**2
                KE_total = KE_trans + KE_rot
                
                result += f"""Energy Analysis:
    Initial potential energy: PE = mgL = {PE_initial:.3f} J
    Final kinetic energy:
    - Translational: KE_trans = ½mv² = {KE_trans:.3f} J  
    - Rotational: KE_rot = ½Iω² = {KE_rot:.3f} J
    - Total: KE_total = {KE_total:.3f} J

    Energy check: {KE_total:.3f} J vs {PE_initial:.3f} J
    Difference: {abs(PE_initial - KE_total):.6f} J ≈ 0 ✓

    Yo-Yo Behavior:
    - Falls slower than free fall due to rotational energy
    - At bottom, has both translational and rotational energy
    - Can "walk" or return up the string due to stored rotational energy
    - Physics is same regardless of yo-yo size (within limits)

    """
            
            result += f"""Rolling Motion Fundamentals:

    Key Principles:
    1. No-slip condition: v = ωr (constraint equation)
    2. Rolling objects have both translational and rotational motion
    3. Friction provides torque but does no work (ideal rolling)
    4. Acceleration depends on moment of inertia distribution

    Types of Rolling Motion:
    1. Rolling down inclines: gravity provides driving force
    2. Rolling on level surfaces: external force needed to maintain motion
    3. Rolling up inclines: initial kinetic energy converted to potential energy

    Important Relationships:
    - Rolling acceleration: a = g sin(θ)/(1 + I/(MR²))
    - Energy distribution depends on I/(MR²) ratio
    - Objects with smaller I/(MR²) roll faster down inclines

    Shape Comparison (I/(MR²) values):
    - Solid sphere: 2/5 = 0.40 (fastest)
    - Solid cylinder/disk: 1/2 = 0.50 (medium)
    - Hollow sphere: 2/3 = 0.67 (slower)
    - Hollow cylinder: 1 = 1.00 (slowest)

    Real-World Applications:
    - Vehicle wheel dynamics and tire performance
    - Ball sports: golf, bowling, tennis ball behavior
    - Industrial machinery: conveyor rollers, ball bearings
    - Recreation: marbles, yo-yos, rolling toys
    - Engineering: gear systems, cam mechanisms

    Key Insights:
    - Mass cancels out in rolling acceleration equations
    - Shape (moment of inertia distribution) determines rolling behavior
    - Rolling friction much less than sliding friction
    - Energy efficiency: rolling preferred over sliding for transport
    """
            
            return result
            
        except Exception as e:
            return f"Error in rolling motion analysis: {str(e)}"

    @mcp.tool()
    async def circular_motion(circular_data: str) -> str:
        """
        Analyze uniform circular motion - centripetal acceleration and force.

        Args:
            circular_data: JSON string with circular motion parameters
                          Example: '{"radius": 0.5, "velocity": 10}'
                          Example: '{"radius": 2, "period": 4}'
                          Example: '{"radius": 0.3, "frequency": 5, "mass": 0.2}'
                          Units: radius (m), velocity (m/s), period (s), frequency (Hz), mass (kg)

        Returns:
            str: Complete circular motion analysis with centripetal force
        """
        try:
            data = json.loads(circular_data) if isinstance(circular_data, str) else circular_data

            r = float(data.get("radius", data.get("r", 1)))
            m = data.get("mass", data.get("m", None))
            if m is not None: m = float(m)

            result = f"""
Uniform Circular Motion Analysis:
=================================

Radius: r = {r:.4f} m
"""
            if m: result += f"Mass: m = {m:.4f} kg\n"

            # Calculate velocity from different inputs
            if "velocity" in data or "v" in data:
                v = float(data.get("velocity", data.get("v")))
                omega = v / r
                T = 2 * math.pi * r / v
                f = 1 / T

            elif "period" in data or "T" in data:
                T = float(data.get("period", data.get("T")))
                f = 1 / T
                omega = 2 * math.pi / T
                v = omega * r

            elif "frequency" in data or "f" in data:
                f = float(data.get("frequency", data.get("f")))
                T = 1 / f
                omega = 2 * math.pi * f
                v = omega * r

            elif "angular_velocity" in data or "omega" in data:
                omega = float(data.get("angular_velocity", data.get("omega")))
                v = omega * r
                T = 2 * math.pi / omega
                f = 1 / T
            else:
                return "Error: Provide velocity, period, frequency, or angular_velocity"

            # Centripetal acceleration
            a_c = v**2 / r  # or omega**2 * r

            result += f"""
Motion Parameters:
- Linear velocity: v = {v:.4f} m/s
- Angular velocity: ω = {omega:.4f} rad/s = {omega * 60 / (2*math.pi):.2f} rpm
- Period: T = {T:.4f} s
- Frequency: f = {f:.4f} Hz

Centripetal Acceleration:
aᶜ = v²/r = {v:.4f}²/{r:.4f} = {a_c:.4f} m/s²
aᶜ = ω²r = {omega:.4f}² × {r:.4f} = {a_c:.4f} m/s²
aᶜ = {a_c/9.81:.2f} g (multiples of Earth gravity)

Direction: Always toward center of circle
"""
            if m:
                F_c = m * a_c
                result += f"""
Centripetal Force:
Fᶜ = maᶜ = mv²/r = {m:.4f} × {a_c:.4f} = {F_c:.4f} N
Fᶜ = mω²r = {m:.4f} × {omega:.4f}² × {r:.4f} = {F_c:.4f} N

This force must be provided by:
- Tension (ball on string)
- Friction (car turning)
- Gravity (satellite orbit)
- Normal force (loop-the-loop)
"""
            result += """
Key Concepts:
- Centripetal = "center-seeking"
- Velocity is tangent to circle, acceleration points to center
- Speed is constant, but velocity direction changes
- Net force toward center required to maintain circular path

Applications:
- Car on curved road (friction provides Fᶜ)
- Ball on string (tension provides Fᶜ)
- Satellite in orbit (gravity provides Fᶜ)
- Centrifuge (apparent outward force)
- Banked curves (component of normal force)
"""
            payload: Dict[str, float | str | None] = {
                "type": "circular_motion_vectors",
                "title": "Circular Motion Vectors",
                "radius_m": r,
                "speed_mps": v,
                "omega_rad_s": omega,
                "period_s": T,
                "frequency_hz": f,
                "centripetal_acc_mps2": a_c,
            }
            if m is not None:
                payload["mass_kg"] = m
                payload["centripetal_force_n"] = m * a_c
            return with_diagram_payload(result, payload)

        except Exception as e:
            return f"Error in circular motion analysis: {str(e)}"

    @mcp.tool()
    async def simple_harmonic_motion(shm_data: str) -> str:
        """
        Analyze simple harmonic motion (springs, pendulums).

        Args:
            shm_data: JSON string with SHM parameters
                     Example: '{"type": "spring", "mass": 0.5, "k": 200}'
                     Example: '{"type": "pendulum", "length": 1.0}'
                     Example: '{"amplitude": 0.1, "omega": 10, "time": 0.5}'
                     Units: mass (kg), k (N/m), length (m), amplitude (m), omega (rad/s), time (s)

        Returns:
            str: Complete SHM analysis with period, frequency, and motion equations
        """
        try:
            data = json.loads(shm_data) if isinstance(shm_data, str) else shm_data

            shm_type = data.get("type", "spring").lower()
            g = 9.81  # m/s²

            result = """
Simple Harmonic Motion Analysis:
================================

"""
            if shm_type == "spring" or "k" in data:
                m = float(data.get("mass", data.get("m", 1)))
                k = float(data.get("k", data.get("spring_constant", 100)))

                omega = math.sqrt(k / m)
                T = 2 * math.pi / omega
                f = 1 / T

                result += f"""Spring-Mass System:
- Mass: m = {m:.4f} kg
- Spring constant: k = {k:.2f} N/m

Angular Frequency: ω = √(k/m) = √({k:.2f}/{m:.4f}) = {omega:.4f} rad/s
Period: T = 2π/ω = 2π/{omega:.4f} = {T:.4f} s
Frequency: f = 1/T = {f:.4f} Hz

"""
            elif shm_type == "pendulum" or "length" in data:
                L = float(data.get("length", data.get("L", 1)))

                omega = math.sqrt(g / L)
                T = 2 * math.pi / omega
                f = 1 / T

                result += f"""Simple Pendulum (small angle):
- Length: L = {L:.4f} m
- Gravity: g = {g:.2f} m/s²

Angular Frequency: ω = √(g/L) = √({g:.2f}/{L:.4f}) = {omega:.4f} rad/s
Period: T = 2π√(L/g) = 2π√({L:.4f}/{g:.2f}) = {T:.4f} s
Frequency: f = 1/T = {f:.4f} Hz

Note: Valid for small angles (θ < 15°)
"""
            # Position, velocity, acceleration at time t
            A = float(data.get("amplitude", data.get("A", 0.1)))
            t = data.get("time", data.get("t", None))
            phi = float(data.get("phase", data.get("phi", 0)))

            result += f"""
Amplitude: A = {A:.4f} m
Maximum velocity: v_max = Aω = {A:.4f} × {omega:.4f} = {A*omega:.4f} m/s
Maximum acceleration: a_max = Aω² = {A:.4f} × {omega:.4f}² = {A*omega**2:.4f} m/s²

Motion Equations:
x(t) = A cos(ωt + φ) = {A:.4f} cos({omega:.4f}t + {phi})
v(t) = -Aω sin(ωt + φ) = -{A*omega:.4f} sin({omega:.4f}t + {phi})
a(t) = -Aω² cos(ωt + φ) = -{A*omega**2:.4f} cos({omega:.4f}t + {phi})
"""
            if t is not None:
                t = float(t)
                x = A * math.cos(omega * t + phi)
                v = -A * omega * math.sin(omega * t + phi)
                a = -A * omega**2 * math.cos(omega * t + phi)

                result += f"""
At t = {t:.4f} s:
- Position: x = {x:.6f} m
- Velocity: v = {v:.6f} m/s
- Acceleration: a = {a:.6f} m/s²
"""
            result += """
Energy in SHM:
- Total Energy: E = ½kA² = ½mω²A² (constant)
- KE = ½mv² = ½mω²(A² - x²)
- PE = ½kx² = ½mω²x²
- At equilibrium: KE = max, PE = 0
- At amplitude: KE = 0, PE = max

Applications:
- Mechanical oscillators, clocks
- Musical instruments
- Molecular vibrations
- Electrical LC circuits
"""
            # Build animation payloads for spring SHM or pendulum SHM.
            frames_count = 40
            t_end = max(2 * T, T)
            times = [i * t_end / (frames_count - 1) for i in range(frames_count)]

            if shm_type == "pendulum" or "length" in data:
                L = float(data.get("length", data.get("L", 1)))
                # Interpret amplitude as arc-length amplitude for pendulum animation.
                theta_amp_rad = (A / L) if L > 0 else 0.0
                frames = []
                for tt in times:
                    theta_rad = theta_amp_rad * math.cos(omega * tt + phi)
                    x = L * math.sin(theta_rad)
                    y = -L * math.cos(theta_rad)
                    frames.append(
                        {
                            "t_s": tt,
                            "theta_deg": radians_to_degrees(theta_rad),
                            "x_m": x,
                            "y_m": y,
                        }
                    )
                payload = {
                    "type": "pendulum_animation",
                    "title": "Pendulum Motion",
                    "length_m": L,
                    "period_s": T,
                    "frequency_hz": f,
                    "theta_max_deg": radians_to_degrees(theta_amp_rad),
                    "frames": frames,
                }
                return with_diagram_payload(result, payload)

            frames = []
            for tt in times:
                x = A * math.cos(omega * tt + phi)
                v = -A * omega * math.sin(omega * tt + phi)
                a = -A * omega**2 * math.cos(omega * tt + phi)
                frames.append({"t_s": tt, "x_m": x, "v_mps": v, "a_mps2": a})
            payload = {
                "type": "shm_spring_animation",
                "title": "Spring SHM",
                "amplitude_m": A,
                "omega_rad_s": omega,
                "period_s": T,
                "frequency_hz": f,
                "frames": frames,
            }
            return with_diagram_payload(result, payload)

        except Exception as e:
            return f"Error in SHM analysis: {str(e)}"

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
    """CLI entry point for the physics-momentum-mcp tool."""
    parser = argparse.ArgumentParser(description="Run Physics Angular Motion MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10106, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")
    
    args = parser.parse_args()
    
    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")

if __name__ == "__main__":
    main()
