from typing import Tuple, Optional, Dict, List
import math
import json
import logging
import argparse
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

from physics_mcp_tools.momentum_utils import (
    degrees_to_radians,
    radians_to_degrees,
    calculate_momentum_magnitude,
    calculate_momentum_components,
    calculate_resultant_momentum
)

# Initialize FastMCP server
#mcp = FastMCP("momentum")

NAME= "momentum_mcp_server"

logger = get_logger(__name__)

def serve(host, port, transport):  
    """Initializes and runs the Agent Cards MCP server.

    Args:
        host: The hostname or IP address to bind the server to.
        port: The port number to bind the server to.
        transport: The transport mechanism for the MCP server (e.g., 'stdio', 'sse').
    """
    logger.info('Starting Momentum MCP Server')
    
    mcp = FastMCP(NAME, stateless_http=False)


    @mcp.tool()
    async def calculate_momentum_1d(mass: float, velocity: float) -> str:
        """
        Calculate 1D momentum for a single object.
        
        Args:
            mass: Mass of the object in kg
            velocity: Velocity of the object in m/s (positive/negative for direction)
            
        Returns:
            str: Complete 1D momentum calculation with explanation
        """
        try:
            momentum = calculate_momentum_magnitude(mass, velocity)
            
            result = f"""
    1D Momentum Calculation:
    =======================

    Given:
    - Mass (m): {mass:.3f} kg
    - Velocity (v): {velocity:.3f} m/s

    Momentum Formula: p = mv
    p = {mass:.3f} × {velocity:.3f}
    p = {momentum:.3f} kg⋅m/s

    Physical Interpretation:
    - Momentum magnitude: {abs(momentum):.3f} kg⋅m/s
    - Direction: {'Positive direction' if momentum >= 0 else 'Negative direction'}
    - Units: kg⋅m/s (kilogram-meters per second)

    Key Concepts:
    - Momentum is a vector quantity (has both magnitude and direction)
    - In 1D, direction is indicated by the sign (+ or -)
    - Momentum depends on both mass and velocity
    - Larger mass or higher velocity means greater momentum
    """
            
            return result
            
        except Exception as e:
            return f"Error in 1D momentum calculation: {str(e)}"

    @mcp.tool()
    async def calculate_momentum_2d(mass: float, velocity: float, angle_degrees: float) -> str:
        """
        Calculate 2D momentum for a single object.
        
        Args:
            mass: Mass of the object in kg
            velocity: Speed of the object in m/s
            angle_degrees: Direction angle in degrees (0° = +x axis, 90° = +y axis)
            
        Returns:
            str: Complete 2D momentum calculation with components and magnitude
        """
        try:
            px, py = calculate_momentum_components(mass, velocity, angle_degrees)
            p_magnitude, p_angle = calculate_resultant_momentum(px, py)
            
            result = f"""
    2D Momentum Calculation:
    =======================

    Given:
    - Mass (m): {mass:.3f} kg
    - Velocity magnitude (v): {velocity:.3f} m/s
    - Direction angle (θ): {angle_degrees:.1f}°

    Momentum Components:
    px = mv cos(θ) = {mass:.3f} × {velocity:.3f} × cos({angle_degrees:.1f}°) = {px:.3f} kg⋅m/s
    py = mv sin(θ) = {mass:.3f} × {velocity:.3f} × sin({angle_degrees:.1f}°) = {py:.3f} kg⋅m/s

    Resultant Momentum:
    |p| = √(px² + py²) = √({px:.3f}² + {py:.3f}²) = {p_magnitude:.3f} kg⋅m/s
    Direction: θ = arctan(py/px) = arctan({py:.3f}/{px:.3f}) = {p_angle:.1f}°

    Vector Representation:
    p⃗ = ({px:.3f}, {py:.3f}) kg⋅m/s
    p⃗ = {p_magnitude:.3f} kg⋅m/s at {p_angle:.1f}°

    Physical Interpretation:
    - Total momentum magnitude: {p_magnitude:.3f} kg⋅m/s
    - Horizontal component (x): {px:.3f} kg⋅m/s
    - Vertical component (y): {py:.3f} kg⋅m/s
    - Direction from +x axis: {p_angle:.1f}°
    """
            
            return result
            
        except Exception as e:
            return f"Error in 2D momentum calculation: {str(e)}"

    @mcp.tool()
    async def calculate_impulse_1d(force: float, time: float, initial_momentum: float = None, final_momentum: float = None) -> str:
        """
        Calculate 1D impulse using force-time or momentum change.
        
        Args:
            force: Applied force in N (use None if calculating from momentum change)
            time: Time interval in s (use None if calculating from momentum change)
            initial_momentum: Initial momentum in kg⋅m/s (optional)
            final_momentum: Final momentum in kg⋅m/s (optional)
            
        Returns:
            str: Complete impulse calculation with explanation
        """
        try:
            result = f"""
    1D Impulse Calculation:
    ======================

    """
            
            # Calculate impulse using different methods
            if force is not None and time is not None:
                impulse = force * time
                result += f"""Method 1: Force × Time
    Given:
    - Force (F): {force:.3f} N
    - Time interval (Δt): {time:.3f} s

    Impulse Formula: J = FΔt
    J = {force:.3f} × {time:.3f}
    J = {impulse:.3f} N⋅s

    """
            
            if initial_momentum is not None and final_momentum is not None:
                momentum_change = final_momentum - initial_momentum
                result += f"""Method 2: Change in Momentum
    Given:
    - Initial momentum (pi): {initial_momentum:.3f} kg⋅m/s
    - Final momentum (pf): {final_momentum:.3f} kg⋅m/s

    Impulse Formula: J = Δp = pf - pi
    J = {final_momentum:.3f} - {initial_momentum:.3f}
    J = {momentum_change:.3f} N⋅s

    """
                if force is not None and time is not None:
                    result += f"""Verification:
    Both methods should give the same result:
    - Force × Time: {impulse:.3f} N⋅s
    - Momentum Change: {momentum_change:.3f} N⋅s
    - Difference: {abs(impulse - momentum_change):.6f} N⋅s

    """
            
            result += f"""Physical Interpretation:
    - Impulse represents the change in momentum
    - Units: N⋅s (Newton-seconds) = kg⋅m/s
    - Direction: {'Positive' if (impulse if 'impulse' in locals() else momentum_change) >= 0 else 'Negative'}
    - Impulse-Momentum Theorem: J = Δp

    Key Concepts:
    - Large force for short time = Large impulse
    - Small force for long time can = Same impulse
    - Impulse equals the area under the Force vs Time graph
    """
            
            return result
            
        except Exception as e:
            return f"Error in 1D impulse calculation: {str(e)}"

    @mcp.tool()
    async def calculate_impulse_2d(force_data: str, time: float = None, momentum_data: str = None) -> str:
        """
        Calculate 2D impulse using force components and time or momentum change.
        
        Args:
            force_data: JSON string with force components or magnitude/angle
                    Examples: '{"fx": 10, "fy": 5}' or '{"magnitude": 15, "angle": 30}'
            time: Time interval in seconds
            momentum_data: JSON string with initial and final momentum vectors (optional)
                        Example: '{"initial": {"px": 5, "py": 3}, "final": {"px": 8, "py": 7}}'
            
        Returns:
            str: Complete 2D impulse calculation
        """
        try:
            result = f"""
    2D Impulse Calculation:
    ======================

    """
            
            # Parse force data
            force_info = json.loads(force_data)
            
            if "fx" in force_info and "fy" in force_info:
                fx = force_info["fx"]
                fy = force_info["fy"]
                f_magnitude = math.sqrt(fx**2 + fy**2)
                f_angle = radians_to_degrees(math.atan2(fy, fx))
            elif "magnitude" in force_info and "angle" in force_info:
                f_magnitude = force_info["magnitude"]
                f_angle = force_info["angle"]
                fx = f_magnitude * math.cos(degrees_to_radians(f_angle))
                fy = f_magnitude * math.sin(degrees_to_radians(f_angle))
            else:
                return "Error: Force data must contain either components (fx, fy) or magnitude and angle"
            
            result += f"""Given Force:
    - Force components: Fx = {fx:.3f} N, Fy = {fy:.3f} N
    - Force magnitude: |F| = {f_magnitude:.3f} N
    - Force direction: θ = {f_angle:.1f}°

    """
            
            if time is not None:
                jx = fx * time
                jy = fy * time
                j_magnitude = math.sqrt(jx**2 + jy**2)
                j_angle = radians_to_degrees(math.atan2(jy, jx))
                
                result += f"""Method 1: Force × Time
    Time interval (Δt): {time:.3f} s

    Impulse Components:
    Jx = Fx × Δt = {fx:.3f} × {time:.3f} = {jx:.3f} N⋅s
    Jy = Fy × Δt = {fy:.3f} × {time:.3f} = {jy:.3f} N⋅s

    Resultant Impulse:
    |J| = √(Jx² + Jy²) = √({jx:.3f}² + {jy:.3f}²) = {j_magnitude:.3f} N⋅s
    Direction: θ = arctan(Jy/Jx) = {j_angle:.1f}°

    """
            
            if momentum_data is not None:
                momentum_info = json.loads(momentum_data)
                pi_x = momentum_info["initial"]["px"]
                pi_y = momentum_info["initial"]["py"]
                pf_x = momentum_info["final"]["px"]
                pf_y = momentum_info["final"]["py"]
                
                delta_px = pf_x - pi_x
                delta_py = pf_y - pi_y
                delta_p_magnitude = math.sqrt(delta_px**2 + delta_py**2)
                delta_p_angle = radians_to_degrees(math.atan2(delta_py, delta_px))
                
                result += f"""Method 2: Change in Momentum
    Initial momentum: pi = ({pi_x:.3f}, {pi_y:.3f}) kg⋅m/s
    Final momentum: pf = ({pf_x:.3f}, {pf_y:.3f}) kg⋅m/s

    Momentum Change:
    Δpx = pf_x - pi_x = {pf_x:.3f} - {pi_x:.3f} = {delta_px:.3f} kg⋅m/s
    Δpy = pf_y - pi_y = {pf_y:.3f} - {pi_y:.3f} = {delta_py:.3f} kg⋅m/s

    Impulse from Momentum Change:
    |J| = |Δp| = √({delta_px:.3f}² + {delta_py:.3f}²) = {delta_p_magnitude:.3f} N⋅s
    Direction: θ = {delta_p_angle:.1f}°

    """
            
            result += f"""Physical Interpretation:
    - 2D impulse is a vector quantity with x and y components  
    - Each component can be calculated independently
    - The resultant impulse has both magnitude and direction
    - Impulse-Momentum Theorem: J⃗ = Δp⃗ (vector equation)

    Vector Representation:
    J⃗ = ({jx if 'jx' in locals() else delta_px:.3f}, {jy if 'jy' in locals() else delta_py:.3f}) N⋅s
    """
            
            return result
            
        except Exception as e:
            return f"Error in 2D impulse calculation: {str(e)}"

    @mcp.tool()
    async def momentum_impulse_theorem(problem_data: str) -> str:
        """
        Apply the momentum-impulse theorem to solve problems.
        
        Args:
            problem_data: JSON string with problem parameters
                        Examples: 
                        '{"mass": 2, "initial_velocity": 5, "force": 10, "time": 3}'
                        '{"mass": 1.5, "initial_momentum": 6, "impulse": 4}'
            
        Returns:
            str: Complete analysis using momentum-impulse theorem
        """
        try:
            data = json.loads(problem_data)
            
            result = f"""
    Momentum-Impulse Theorem Analysis:
    =================================

    Given Data:
    """
            
            for key, value in data.items():
                result += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            result += f"""
    Momentum-Impulse Theorem: J = Δp = pf - pi

    """
            
            if "mass" in data and "initial_velocity" in data:
                mass = data["mass"]
                vi = data["initial_velocity"]
                pi = mass * vi
                result += f"Initial momentum: pi = m × vi = {mass:.3f} × {vi:.3f} = {pi:.3f} kg⋅m/s\n"
                
                if "force" in data and "time" in data:
                    force = data["force"]
                    time = data["time"]
                    impulse = force * time
                    result += f"Impulse: J = F × Δt = {force:.3f} × {time:.3f} = {impulse:.3f} N⋅s\n"
                    
                    pf = pi + impulse
                    vf = pf / mass
                    result += f"""
    Solution using J = Δp:
    Final momentum: pf = pi + J = {pi:.3f} + {impulse:.3f} = {pf:.3f} kg⋅m/s
    Final velocity: vf = pf / m = {pf:.3f} / {mass:.3f} = {vf:.3f} m/s

    Change in velocity: Δv = vf - vi = {vf:.3f} - {vi:.3f} = {vf - vi:.3f} m/s
    """
                    
                elif "impulse" in data:
                    impulse = data["impulse"]
                    pf = pi + impulse
                    vf = pf / mass
                    result += f"""
    Solution using J = Δp:
    Final momentum: pf = pi + J = {pi:.3f} + {impulse:.3f} = {pf:.3f} kg⋅m/s
    Final velocity: vf = pf / m = {pf:.3f} / {mass:.3f} = {vf:.3f} m/s
    """
            
            result += f"""
    Physical Insights:
    - The impulse-momentum theorem connects force and time to momentum change
    - J = FΔt = Δp shows that the same momentum change can result from:
    * Large force for short time
    * Small force for long time
    - This principle explains airbags, crumple zones, and sports techniques
    - The theorem applies to both 1D and 2D motion (as vector equation)

    Applications:
    - Safety design (airbags reduce force by increasing time)
    - Sports (follow-through increases contact time)
    - Collisions (analyzing impact forces and duration)
    """
            
            return result
            
        except Exception as e:
            return f"Error in momentum-impulse theorem analysis: {str(e)}"

    @mcp.tool()
    async def momentum_conservation_1d(collision_data: str) -> str:
        """
        Solve 1D momentum conservation problems (collisions, explosions).
        
        Args:
            collision_data: JSON string with collision parameters
                        Examples:
                        '{"m1": 2, "v1i": 5, "m2": 3, "v2i": -2, "collision_type": "elastic"}'
                        '{"m1": 1, "v1i": 4, "m2": 2, "v2i": 0, "v1f": 1, "collision_type": "inelastic"}'
            
        Returns:
            str: Complete 1D collision analysis with momentum conservation
        """
        try:
            data = json.loads(collision_data)
            
            m1 = data["m1"]
            m2 = data["m2"] 
            v1i = data["v1i"]
            v2i = data["v2i"]
            collision_type = data.get("collision_type", "unknown")
            
            # Calculate initial momenta
            p1i = m1 * v1i
            p2i = m2 * v2i
            pi_total = p1i + p2i
            
            result = f"""
    1D Momentum Conservation Analysis:
    =================================

    Initial Conditions:
    - Object 1: m₁ = {m1:.3f} kg, v₁ᵢ = {v1i:.3f} m/s
    - Object 2: m₂ = {m2:.3f} kg, v₂ᵢ = {v2i:.3f} m/s
    - Collision type: {collision_type.title()}

    Initial Momentum Calculation:
    p₁ᵢ = m₁ × v₁ᵢ = {m1:.3f} × {v1i:.3f} = {p1i:.3f} kg⋅m/s
    p₂ᵢ = m₂ × v₂ᵢ = {m2:.3f} × {v2i:.3f} = {p2i:.3f} kg⋅m/s
    Total initial momentum: pᵢ = p₁ᵢ + p₂ᵢ = {p1i:.3f} + {p2i:.3f} = {pi_total:.3f} kg⋅m/s

    Conservation of Momentum: pᵢ = pf
    m₁v₁ᵢ + m₂v₂ᵢ = m₁v₁f + m₂v₂f

    """
            
            if collision_type.lower() == "elastic":
                # Elastic collision - both momentum and kinetic energy conserved
                v1f = ((m1 - m2) * v1i + 2 * m2 * v2i) / (m1 + m2)
                v2f = ((m2 - m1) * v2i + 2 * m1 * v1i) / (m1 + m2)
                
                p1f = m1 * v1f
                p2f = m2 * v2f  
                pf_total = p1f + p2f
                
                # Calculate kinetic energies
                ke_i = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
                ke_f = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
                
                result += f"""Elastic Collision Solution:
    Using elastic collision formulas:

    v₁f = [(m₁-m₂)v₁ᵢ + 2m₂v₂ᵢ] / (m₁+m₂)
    v₁f = [({m1:.3f}-{m2:.3f})×{v1i:.3f} + 2×{m2:.3f}×{v2i:.3f}] / ({m1:.3f}+{m2:.3f})
    v₁f = {v1f:.3f} m/s

    v₂f = [(m₂-m₁)v₂ᵢ + 2m₁v₁ᵢ] / (m₁+m₂)  
    v₂f = [({m2:.3f}-{m1:.3f})×{v2i:.3f} + 2×{m1:.3f}×{v1i:.3f}] / ({m1:.3f}+{m2:.3f})
    v₂f = {v2f:.3f} m/s

    Final Momentum Verification:
    p₁f = m₁ × v₁f = {m1:.3f} × {v1f:.3f} = {p1f:.3f} kg⋅m/s
    p₂f = m₂ × v₂f = {m2:.3f} × {v2f:.3f} = {p2f:.3f} kg⋅m/s
    Total final momentum: pf = {p1f:.3f} + {p2f:.3f} = {pf_total:.3f} kg⋅m/s

    Momentum Conservation Check: |pᵢ - pf| = |{pi_total:.3f} - {pf_total:.3f}| = {abs(pi_total - pf_total):.6f} ≈ 0 ✓

    Energy Conservation Check:
    Initial KE: KEᵢ = ½m₁v₁ᵢ² + ½m₂v₂ᵢ² = {ke_i:.3f} J
    Final KE: KEf = ½m₁v₁f² + ½m₂v₂f² = {ke_f:.3f} J
    Energy difference: |KEᵢ - KEf| = {abs(ke_i - ke_f):.6f} ≈ 0 ✓

    """
                
            elif collision_type.lower() == "perfectly_inelastic":
                # Perfectly inelastic - objects stick together
                vf_combined = pi_total / (m1 + m2)
                pf_total = (m1 + m2) * vf_combined
                
                ke_i = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
                ke_f = 0.5 * (m1 + m2) * vf_combined**2
                ke_lost = ke_i - ke_f
                
                result += f"""Perfectly Inelastic Collision Solution:
    Objects stick together after collision.

    Combined final velocity:
    vf = pᵢ / (m₁ + m₂) = {pi_total:.3f} / ({m1:.3f} + {m2:.3f}) = {vf_combined:.3f} m/s

    Final momentum: pf = (m₁ + m₂) × vf = ({m1:.3f} + {m2:.3f}) × {vf_combined:.3f} = {pf_total:.3f} kg⋅m/s

    Momentum Conservation Check: |pᵢ - pf| = |{pi_total:.3f} - {pf_total:.3f}| = {abs(pi_total - pf_total):.6f} ≈ 0 ✓

    Energy Analysis:
    Initial KE: KEᵢ = {ke_i:.3f} J
    Final KE: KEf = {ke_f:.3f} J  
    Energy lost: ΔKE = KEᵢ - KEf = {ke_lost:.3f} J
    Energy lost percentage: {(ke_lost/ke_i)*100:.1f}%

    """
                
            elif "v1f" in data or "v2f" in data:
                # Partially given final velocities
                if "v1f" in data:
                    v1f = data["v1f"]
                    # Solve for v2f using momentum conservation
                    v2f = (pi_total - m1 * v1f) / m2
                else:
                    v2f = data["v2f"]
                    # Solve for v1f using momentum conservation  
                    v1f = (pi_total - m2 * v2f) / m1
                    
                p1f = m1 * v1f
                p2f = m2 * v2f
                pf_total = p1f + p2f
                
                result += f"""Collision with Given Final Velocity:
    Given: v₁f = {v1f:.3f} m/s (or v₂f = {v2f:.3f} m/s)

    Using momentum conservation to find unknown velocity:
    m₁v₁ᵢ + m₂v₂ᵢ = m₁v₁f + m₂v₂f
    {pi_total:.3f} = {m1:.3f} × {v1f:.3f} + {m2:.3f} × v₂f
    Solving: v₂f = {v2f:.3f} m/s

    Final velocities:
    - Object 1: v₁f = {v1f:.3f} m/s
    - Object 2: v₂f = {v2f:.3f} m/s

    Momentum Conservation Check:
    pf = {m1:.3f} × {v1f:.3f} + {m2:.3f} × {v2f:.3f} = {pf_total:.3f} kg⋅m/s
    |pᵢ - pf| = |{pi_total:.3f} - {pf_total:.3f}| = {abs(pi_total - pf_total):.6f} ≈ 0 ✓

    """
            
            result += f"""Physical Interpretation:
    - Momentum is always conserved in isolated systems (no external forces)
    - In elastic collisions, kinetic energy is also conserved
    - In inelastic collisions, some kinetic energy is converted to other forms
    - The coefficient of restitution determines the collision type:
    * e = 1: Perfectly elastic
    * 0 < e < 1: Partially inelastic  
    * e = 0: Perfectly inelastic

    Applications:
    - Car crash analysis and safety design
    - Sports ball collisions
    - Atomic and molecular collisions
    - Rocket propulsion (explosion in reverse)
    """
            
            return result
            
        except Exception as e:
            return f"Error in 1D momentum conservation analysis: {str(e)}"

    @mcp.tool()
    async def momentum_conservation_2d(collision_data: str) -> str:
        """
        Solve 2D momentum conservation problems.
        
        Args:
            collision_data: JSON string with 2D collision parameters
                        Example:
                        '{
                            "object1": {"mass": 2, "velocity": 5, "angle": 0},
                            "object2": {"mass": 3, "velocity": 4, "angle": 90},
                            "collision_type": "elastic"
                        }'
            
        Returns:
            str: Complete 2D collision analysis
        """
        try:
            data = json.loads(collision_data)
            
            obj1 = data["object1"]
            obj2 = data["object2"]
            collision_type = data.get("collision_type", "unknown")
            
            m1 = obj1["mass"]
            v1i = obj1["velocity"]
            angle1i = obj1["angle"]
            
            m2 = obj2["mass"]
            v2i = obj2["velocity"] 
            angle2i = obj2["angle"]
            
            # Calculate initial momentum components
            p1xi = m1 * v1i * math.cos(degrees_to_radians(angle1i))
            p1yi = m1 * v1i * math.sin(degrees_to_radians(angle1i))
            
            p2xi = m2 * v2i * math.cos(degrees_to_radians(angle2i))
            p2yi = m2 * v2i * math.sin(degrees_to_radians(angle2i))
            
            # Total initial momentum components
            pxi_total = p1xi + p2xi
            pyi_total = p1yi + p2yi
            pi_magnitude = math.sqrt(pxi_total**2 + pyi_total**2)
            pi_angle = radians_to_degrees(math.atan2(pyi_total, pxi_total))
            
            result = f"""
    2D Momentum Conservation Analysis:
    =================================

    Initial Conditions:
    Object 1: m₁ = {m1:.3f} kg, v₁ᵢ = {v1i:.3f} m/s at {angle1i:.1f}°
    Object 2: m₂ = {m2:.3f} kg, v₂ᵢ = {v2i:.3f} m/s at {angle2i:.1f}°
    Collision type: {collision_type.title()}

    Initial Momentum Components:
    Object 1:
    - p₁ₓᵢ = m₁v₁ᵢ cos(θ₁) = {m1:.3f} × {v1i:.3f} × cos({angle1i:.1f}°) = {p1xi:.3f} kg⋅m/s
    - p₁yᵢ = m₁v₁ᵢ sin(θ₁) = {m1:.3f} × {v1i:.3f} × sin({angle1i:.1f}°) = {p1yi:.3f} kg⋅m/s

    Object 2:  
    - p₂ₓᵢ = m₂v₂ᵢ cos(θ₂) = {m2:.3f} × {v2i:.3f} × cos({angle2i:.1f}°) = {p2xi:.3f} kg⋅m/s
    - p₂yᵢ = m₂v₂ᵢ sin(θ₂) = {m2:.3f} × {v2i:.3f} × sin({angle2i:.1f}°) = {p2yi:.3f} kg⋅m/s

    Total Initial Momentum:
    - pₓᵢ = p₁ₓᵢ + p₂ₓᵢ = {p1xi:.3f} + {p2xi:.3f} = {pxi_total:.3f} kg⋅m/s
    - pyᵢ = p₁yᵢ + p₂yᵢ = {p1yi:.3f} + {p2yi:.3f} = {pyi_total:.3f} kg⋅m/s
    - |pᵢ| = √(pₓᵢ² + pyᵢ²) = {pi_magnitude:.3f} kg⋅m/s
    - Direction: θᵢ = {pi_angle:.1f}°

    Conservation Laws:
    pₓᵢ = pₓf  (x-component momentum conserved)
    pyᵢ = pyf  (y-component momentum conserved)

    """
            
            if collision_type.lower() == "perfectly_inelastic":
                # Objects stick together - same final velocity
                m_total = m1 + m2
                vxf_combined = pxi_total / m_total
                vyf_combined = pyi_total / m_total
                vf_magnitude = math.sqrt(vxf_combined**2 + vyf_combined**2)
                vf_angle = radians_to_degrees(math.atan2(vyf_combined, vxf_combined))
                
                # Energy analysis
                ke_i = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
                ke_f = 0.5 * m_total * vf_magnitude**2
                ke_lost = ke_i - ke_f
                
                result += f"""Perfectly Inelastic Collision Solution:
    Objects stick together after collision.

    Combined final velocity components:
    vₓf = pₓᵢ / (m₁ + m₂) = {pxi_total:.3f} / {m_total:.3f} = {vxf_combined:.3f} m/s
    vyf = pyᵢ / (m₁ + m₂) = {pyi_total:.3f} / {m_total:.3f} = {vyf_combined:.3f} m/s

    Final velocity:
    |vf| = √(vₓf² + vyf²) = {vf_magnitude:.3f} m/s
    Direction: θf = arctan(vyf/vₓf) = {vf_angle:.1f}°

    Momentum Conservation Verification:
    Final momentum: pf = ({pxi_total:.3f}, {pyi_total:.3f}) kg⋅m/s
    Initial momentum: pᵢ = ({pxi_total:.3f}, {pyi_total:.3f}) kg⋅m/s ✓

    Energy Analysis:
    Initial KE: KEᵢ = {ke_i:.3f} J
    Final KE: KEf = {ke_f:.3f} J
    Energy lost: ΔKE = {ke_lost:.3f} J ({(ke_lost/ke_i)*100:.1f}% of initial energy)

    """
                
            elif "final_angles" in data:
                # Given final directions - solve for speeds
                angles_f = data["final_angles"]
                angle1f = angles_f["object1"]
                angle2f = angles_f["object2"]
                
                # Set up system of equations for momentum conservation
                cos1f = math.cos(degrees_to_radians(angle1f))
                sin1f = math.sin(degrees_to_radians(angle1f))
                cos2f = math.cos(degrees_to_radians(angle2f))
                sin2f = math.sin(degrees_to_radians(angle2f))
                
                # Solve system: m1*v1f*cos1f + m2*v2f*cos2f = pxi_total
                #               m1*v1f*sin1f + m2*v2f*sin2f = pyi_total  
                
                # This is a 2x2 linear system - solving using determinants
                det = m1 * cos1f * m2 * sin2f - m1 * sin1f * m2 * cos2f
                
                if abs(det) > 1e-10:  # System has unique solution
                    v1f = (pxi_total * m2 * sin2f - pyi_total * m2 * cos2f) / det
                    v2f = (pyi_total * m1 * cos1f - pxi_total * m1 * sin1f) / det
                    
                    result += f"""Given Final Directions:
    Object 1 final angle: {angle1f:.1f}°
    Object 2 final angle: {angle2f:.1f}°

    Solving momentum conservation equations:
    m₁v₁f cos({angle1f:.1f}°) + m₂v₂f cos({angle2f:.1f}°) = {pxi_total:.3f}
    m₁v₁f sin({angle1f:.1f}°) + m₂v₂f sin({angle2f:.1f}°) = {pyi_total:.3f}

    Solution:
    v₁f = {v1f:.3f} m/s
    v₂f = {v2f:.3f} m/s

    Final momentum verification:
    pₓf = {m1:.3f}×{v1f:.3f}×cos({angle1f:.1f}°) + {m2:.3f}×{v2f:.3f}×cos({angle2f:.1f}°) = {m1*v1f*cos1f + m2*v2f*cos2f:.3f} kg⋅m/s
    pyf = {m1:.3f}×{v1f:.3f}×sin({angle1f:.1f}°) + {m2:.3f}×{v2f:.3f}×sin({angle2f:.1f}°) = {m1*v1f*sin1f + m2*v2f*sin2f:.3f} kg⋅m/s

    """
                else:
                    result += "Error: The given final angles do not allow a unique solution.\n"
                    
            else:
                result += f"""General 2D Collision Analysis:
    For complete solution, additional information needed:
    - Final directions of both objects, OR  
    - One complete final velocity vector, OR
    - Coefficient of restitution for elastic analysis

    Current analysis shows momentum conservation requirements:
    - Final x-momentum must equal {pxi_total:.3f} kg⋅m/s
    - Final y-momentum must equal {pyi_total:.3f} kg⋅m/s

    """
            
            result += f"""Physical Insights:
    - 2D momentum conservation requires both x and y components to be conserved
    - This provides two equations for solving collision problems
    - Additional constraints (energy conservation, given angles, etc.) needed for unique solutions
    - Vector nature of momentum makes 2D problems more complex than 1D

    Applications:
    - Billiard ball collisions
    - Particle physics experiments  
    - Asteroid impacts
    - Sports ball interactions
    - Molecular collision dynamics
    """
            
            return result
            
        except Exception as e:
            return f"Error in 2D momentum conservation analysis: {str(e)}"

    @mcp.tool()
    async def analyze_collision(collision_scenario: str) -> str:
        """
        Comprehensive collision analysis with multiple approaches.
        
        Args:
            collision_scenario: JSON string describing the collision scenario
                            Example:
                            '{
                                "scenario": "car_crash",
                                "car1": {"mass": 1500, "velocity": 20, "direction": 0},
                                "car2": {"mass": 1200, "velocity": 15, "direction": 90},
                                "analysis_type": "safety"
                            }'
            
        Returns:
            str: Comprehensive collision analysis
        """
        try:
            data = json.loads(collision_scenario)
            scenario = data.get("scenario", "general")
            analysis_type = data.get("analysis_type", "basic")
            
            result = f"""
    Comprehensive Collision Analysis:
    ===============================

    Scenario: {scenario.replace('_', ' ').title()}
    Analysis Type: {analysis_type.title()}

    """
            
            if "car1" in data and "car2" in data:
                car1 = data["car1"]
                car2 = data["car2"]
                
                m1, v1, dir1 = car1["mass"], car1["velocity"], car1["direction"]
                m2, v2, dir2 = car2["mass"], car2["velocity"], car2["direction"]
                
                # Calculate momentum vectors
                p1x = m1 * v1 * math.cos(degrees_to_radians(dir1))
                p1y = m1 * v1 * math.sin(degrees_to_radians(dir1))
                p2x = m2 * v2 * math.cos(degrees_to_radians(dir2))
                p2y = m2 * v2 * math.sin(degrees_to_radians(dir2))
                
                # Total momentum
                px_total = p1x + p2x
                py_total = p1y + p2y
                p_total = math.sqrt(px_total**2 + py_total**2)
                
                # Kinetic energies
                ke1 = 0.5 * m1 * v1**2
                ke2 = 0.5 * m2 * v2**2
                ke_total = ke1 + ke2
                
                result += f"""Initial Conditions:
    Vehicle 1: {m1:.0f} kg at {v1:.1f} m/s ({v1*3.6:.1f} km/h) heading {dir1:.0f}°
    Vehicle 2: {m2:.0f} kg at {v2:.1f} m/s ({v2*3.6:.1f} km/h) heading {dir2:.0f}°

    Momentum Analysis:
    Total momentum: {p_total:.0f} kg⋅m/s
    Total kinetic energy: {ke_total:.0f} J

    Assuming Perfectly Inelastic Collision (vehicles stick together):
    Combined mass: {m1 + m2:.0f} kg
    Final velocity: {p_total/(m1+m2):.2f} m/s ({p_total/(m1+m2)*3.6:.2f} km/h)

    Energy Analysis:
    Initial KE: {ke_total:.0f} J
    Final KE: {0.5*(m1+m2)*(p_total/(m1+m2))**2:.0f} J
    Energy dissipated: {ke_total - 0.5*(m1+m2)*(p_total/(m1+m2))**2:.0f} J

    """
                
                if analysis_type == "safety":
                    # Calculate forces assuming crash duration
                    crash_duration = 0.15  # typical crash duration in seconds
                    avg_deceleration1 = v1 / crash_duration
                    avg_deceleration2 = v2 / crash_duration
                    avg_force1 = m1 * avg_deceleration1
                    avg_force2 = m2 * avg_deceleration2
                    
                    result += f"""Safety Analysis:
    Assuming crash duration: {crash_duration:.2f} s

    Average decelerations:
    - Vehicle 1: {avg_deceleration1:.1f} m/s² ({avg_deceleration1/9.81:.1f} g's)
    - Vehicle 2: {avg_deceleration2:.1f} m/s² ({avg_deceleration2/9.81:.1f} g's)

    Average crash forces:
    - On vehicle 1: {avg_force1:.0f} N ({avg_force1/1000:.0f} kN)
    - On vehicle 2: {avg_force2:.0f} N ({avg_force2/1000:.0f} kN)

    Safety Considerations:
    - Forces above 50 kN can cause severe structural damage
    - Decelerations above 20g can be fatal to occupants
    - Crumple zones increase crash duration, reducing peak forces
    - Airbags further extend the deceleration time for occupants

    Energy Dissipation Mechanisms:
    - Deformation of vehicle structure: ~60-70%
    - Heat generation: ~20-30%  
    - Sound and vibration: ~5-10%
    - Other losses: ~5%

    """
            
            result += f"""General Collision Principles:

    1. Conservation Laws:
    - Momentum is ALWAYS conserved (no external forces)
    - Energy is conserved but may change forms
    - Angular momentum conserved (for rotational effects)

    2. Collision Types:
    - Elastic: Both momentum and kinetic energy conserved
    - Inelastic: Only momentum conserved, some KE lost
    - Perfectly Inelastic: Objects stick together

    3. Real-World Applications:
    - Vehicle crash testing and safety design
    - Sports equipment optimization
    - Asteroid impact studies
    - Particle physics experiments
    - Industrial process design

    4. Analysis Techniques:
    - Vector decomposition for 2D/3D problems
    - Energy methods for complex scenarios  
    - Coefficient of restitution for material properties
    - Impulse-momentum theorem for force analysis

    5. Safety Engineering:
    - Increase collision duration (airbags, crumple zones)
    - Redirect momentum (barriers, deflection systems)
    - Absorb energy (foam, honeycomb structures)
    - Distribute forces (seat belts, helmets)
    """
            
            return result
            
        except Exception as e:
            return f"Error in collision analysis: {str(e)}"

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
    parser = argparse.ArgumentParser(description="Run Physics Momentum MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10104, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")
    
    args = parser.parse_args()
    
    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")

if __name__ == "__main__":
    main()
