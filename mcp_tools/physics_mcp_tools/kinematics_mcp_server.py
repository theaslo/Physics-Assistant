# type: ignore
from typing import Any
import math
import json

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

from physics_mcp_tools.kinematics_utils import (
    degrees_to_radians,
    radians_to_degrees,
    solve_quadratic,
    format_time,
    safe_format
)
import uvicorn
import argparse
NAME= "kinematics_mcp_server"

logger = get_logger(__name__)

def serve(host, port, transport):  
    """Initializes and runs the Agent Cards MCP server.

    Args:
        host: The hostname or IP address to bind the server to.
        port: The port number to bind the server to.
        transport: The transport mechanism for the MCP server (e.g., 'stdio', 'sse').
    """
    logger.info('Starting Kinematics MCP Server')
    
    mcp = FastMCP(NAME, stateless_http=False)



    @mcp.tool()
    async def uniform_motion_1d(known_values: str) -> str:
        """Solve 1D uniform motion problems using x = x₀ + vt.
        
        Args:
            known_values: JSON string with known values. At least 2 of 3 values needed.
                        Example: '{"x0": 0, "v": 25, "t": 4}' or '{"x": 100, "v": 25}'
                        
                        Variables:
                        - x0: initial position (m)
                        - x: final position (m) 
                        - v: constant velocity (m/s)
                        - t: time (s)
        
        Returns:
            str: Complete uniform motion analysis with solution
        """
        try:
            data = json.loads(known_values)
            
            # Extract known values
            x0 = data.get('x0', None)
            x = data.get('x', None)  
            v = data.get('v', None)
            t = data.get('t', None)
            
            # Count known values
            known = sum(1 for val in [x0, x, v, t] if val is not None)
            
            if known < 2:
                return "Error: Need at least 2 known values to solve uniform motion problem"
            
            result = """
    1D Uniform Motion Analysis:
    ==========================

    Given Values:
    """
            
            # Display known values
            if x0 is not None:
                result += f"- Initial position (x₀): {x0:.2f} m\n"
            if x is not None:
                result += f"- Final position (x): {x:.2f} m\n"
            if v is not None:
                result += f"- Velocity (v): {v:.2f} m/s\n"
            if t is not None:
                result += f"- Time (t): {format_time(t)}\n"
                
            result += f"\nKinematic Equation: x = x₀ + vt\n\nSolution:\n"
            
            # Solve for missing values
            if x0 is None:
                x0 = x - v * t
                result += f"x₀ = x - vt = {x:.2f} - {v:.2f} × {t:.2f} = {x0:.2f} m\n"
                
            if x is None:
                x = x0 + v * t  
                result += f"x = x₀ + vt = {x0:.2f} + {v:.2f} × {t:.2f} = {x:.2f} m\n"
                
            if v is None:
                v = (x - x0) / t
                result += f"v = (x - x₀)/t = ({x:.2f} - {x0:.2f})/{t:.2f} = {v:.2f} m/s\n"
                
            if t is None:
                t = (x - x0) / v
                result += f"t = (x - x₀)/v = ({x:.2f} - {x0:.2f})/{v:.2f} = {t:.2f} s\n"
            
            # Calculate displacement
            displacement = x - x0
            
            result += f"""
    Complete Solution:
    - Initial position (x₀): {x0:.2f} m
    - Final position (x): {x:.2f} m  
    - Velocity (v): {v:.2f} m/s
    - Time (t): {format_time(t)}
    - Displacement (Δx): {displacement:.2f} m

    Physical Interpretation:
    """
            
            if v > 0:
                result += f"- Object moves in positive direction at constant speed\n"
            elif v < 0:
                result += f"- Object moves in negative direction at constant speed\n" 
            else:
                result += f"- Object remains stationary (no motion)\n"
                
            result += f"- Average velocity = {v:.2f} m/s\n"
            result += f"- Distance traveled = {abs(displacement):.2f} m"
            
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as e:
            return f'Error: {str(e)}\nExpected format: {{"x0": 0, "v": 25, "t": 4}}'

    @mcp.tool()
    async def constant_acceleration_1d(known_values: str) -> str:
        """Solve 1D constant acceleration problems using kinematic equations.
        
        Args:
            known_values: JSON string with known values. Need at least 3 of 5 values.
                        Example: '{"v0": 0, "a": 9.81, "t": 3}' 
                        
                        Variables:
                        - x0: initial position (m)
                        - x: final position (m)
                        - v0: initial velocity (m/s) 
                        - v: final velocity (m/s)
                        - a: acceleration (m/s²)
                        - t: time (s)
        
        Returns:
            str: Complete constant acceleration analysis with solution
        """
        try:
            data = json.loads(known_values)
            
            # Extract known values
            x0 = data.get('x0', 0.0)  # Default to 0 if not provided
            x = data.get('x', None)
            v0 = data.get('v0', None) 
            v = data.get('v', None)
            a = data.get('a', None)
            t = data.get('t', None)
            
            # Count known values (excluding x0 default)
            known_vars = [val for val in [x, v0, v, a, t] if val is not None]
            if x0 != 0.0 or 'x0' in data:
                known_vars.append(x0)
                
            if len(known_vars) < 3:
                return "Error: Need at least 3 known values to solve constant acceleration problem"
            
            result = f"""
    1D Constant Acceleration Analysis:
    =================================

    Given Values:
    - Initial position (x₀): {x0:.2f} m
    """
            
            # Display known values
            if x is not None:
                result += f"- Final position (x): {x:.2f} m\n"
            if v0 is not None:
                result += f"- Initial velocity (v₀): {v0:.2f} m/s\n"
            if v is not None:
                result += f"- Final velocity (v): {v:.2f} m/s\n"  
            if a is not None:
                result += f"- Acceleration (a): {a:.2f} m/s²\n"
            if t is not None:
                result += f"- Time (t): {format_time(t)}\n"
                
            result += f"\nKinematic Equations Used:\n"
            equations_used = []
            
            # Solve using appropriate equations
            if v0 is not None and a is not None and t is not None and v is None:
                # Use v = v₀ + at
                v = v0 + a * t
                result += f"v = v₀ + at = {v0:.2f} + {a:.2f} × {t:.2f} = {v:.2f} m/s\n"
                equations_used.append("v = v₀ + at")
                
            if v is not None and v0 is not None and a is not None and t is None:
                # Use t = (v - v₀)/a
                t = (v - v0) / a
                result += f"t = (v - v₀)/a = ({v:.2f} - {v0:.2f})/{a:.2f} = {t:.2f} s\n"
                equations_used.append("v = v₀ + at")
                
            if v0 is not None and t is not None and a is not None and x is None:
                # Use x = x₀ + v₀t + ½at²
                x = x0 + v0 * t + 0.5 * a * t**2
                result += f"x = x₀ + v₀t + ½at² = {x0:.2f} + {v0:.2f} × {t:.2f} + 0.5 × {a:.2f} × {t:.2f}² = {x:.2f} m\n"
                equations_used.append("x = x₀ + v₀t + ½at²")
                
            if v0 is not None and v is not None and a is not None and x is None:
                # Use v² = v₀² + 2a(x - x₀)
                x = x0 + (v**2 - v0**2) / (2 * a)
                result += f"x = x₀ + (v² - v₀²)/(2a) = {x0:.2f} + ({v:.2f}² - {v0:.2f}²)/(2 × {a:.2f}) = {x:.2f} m\n"
                equations_used.append("v² = v₀² + 2a(x - x₀)")
                
            # Handle cases where we need to solve quadratic equations
            if x is not None and v0 is not None and a is not None and t is None:
                # Use x = x₀ + v₀t + ½at² to solve for t
                # Rearrange to: ½at² + v₀t + (x₀ - x) = 0
                A, B, C = 0.5 * a, v0, x0 - x
                t1, t2 = solve_quadratic(A, B, C)
                
                if t1 is not None:
                    # Choose positive time solution
                    t = t1 if t1 >= 0 else t2 if t2 is not None and t2 >= 0 else t1
                    result += f"t = {t:.2f} s (from quadratic equation)\n"
                    equations_used.append("x = x₀ + v₀t + ½at²")
                    
                    # Calculate final velocity
                    if v is None:
                        v = v0 + a * t
                        
            # Fill in any remaining unknowns
            if v is None and v0 is not None and a is not None and t is not None:
                v = v0 + a * t
            if a is None and v is not None and v0 is not None and t is not None:
                a = (v - v0) / t
                
            # Calculate displacement and distance
            displacement = x - x0 if x is not None else None
            
            result += f"""
    Complete Solution:
    - Initial position (x₀): {x0:.2f} m
    - Final position (x): {x:.2f} m
    - Initial velocity (v₀): {v0:.2f} m/s  
    - Final velocity (v): {v:.2f} m/s
    - Acceleration (a): {a:.2f} m/s²
    - Time (t): {format_time(t)}
    - Displacement (Δx): {displacement:.2f} m

    Physical Interpretation:
    """
            
            if a > 0:
                result += f"- Object accelerates in positive direction\n"
            elif a < 0:
                result += f"- Object decelerates (or accelerates in negative direction)\n"
            else:
                result += f"- Object moves at constant velocity (no acceleration)\n"
                
            if v0 is not None and v is not None:
                avg_velocity = (v0 + v) / 2
                result += f"- Average velocity: {avg_velocity:.2f} m/s\n"
                
            if displacement is not None:
                result += f"- Distance traveled: {abs(displacement):.2f} m"
            
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as e:
            return f'Error: {str(e)}\nExpected format: {{"v0": 0, "a": 9.81, "t": 3}}'

    # @mcp.tool()
    # async def free_fall_motion(known_values: str, gravity: float = 9.81) -> str:
    #     """Analyze free fall motion (special case of constant acceleration).
        
    #     Args:
    #         known_values: JSON string with known values. Need at least 2 values.
    #                     Example: '{"h0": 100, "t": 4}' or '{"v0": 20, "h": 0}'
                        
    #                     Variables:
    #                     - h0: initial height (m)
    #                     - h: final height (m) 
    #                     - v0: initial velocity (m/s, positive = upward)
    #                     - v: final velocity (m/s, positive = upward)
    #                     - t: time (s)
                        
    #         gravity: Acceleration due to gravity (default: 9.81 m/s²)
        
    #     Returns:
    #         str: Complete free fall analysis with trajectory
    #     """
    #     try:
    #         data = json.loads(known_values)
            
    #         # Extract known values
    #         h0 = data.get('h0', None)
    #         h = data.get('h', None)
    #         v0 = data.get('v0', 0.0)  # Default to 0 if not provided
    #         v = data.get('v', None)
    #         t = data.get('t', None)
            
    #         # Convert to standard kinematic variables (y-axis, positive up)
    #         # a = -g (negative because gravity pulls down)
    #         a = -gravity
            
    #         result = f"""
    # Free Fall Motion Analysis:
    # =========================

    # Given Values:
    # - Gravity (g): {gravity:.2f} m/s²
    # - Acceleration (a = -g): {a:.2f} m/s²
    # """
            
    #         if h0 is not None:
    #             result += f"- Initial height (h₀): {h0:.2f} m\n"
    #         if h is not None:
    #             result += f"- Final height (h): {h:.2f} m\n"
    #         result += f"- Initial velocity (v₀): {v0:.2f} m/s {'(upward)' if v0 > 0 else '(downward)' if v0 < 0 else '(dropped from rest)'}\n"
    #         if v is not None:
    #             result += f"- Final velocity (v): {v:.2f} m/s\n"
    #         if t is not None:
    #             result += f"- Time (t): {format_time(t)}\n"
                
    #         # Solve using kinematic equations
    #         result += f"\nSolution Process:\n"
            
    #         # If we have initial conditions and time
    #         if h0 is not None and t is not None:
    #             if h is None:
    #                 h = h0 + v0 * t + 0.5 * a * t**2
    #                 result += f"h = h₀ + v₀t + ½at² = {h0:.2f} + {v0:.2f} × {t:.2f} + 0.5 × {a:.2f} × {t:.2f}² = {h:.2f} m\n"
                
    #             if v is None:
    #                 v = v0 + a * t
    #                 result += f"v = v₀ + at = {v0:.2f} + {a:.2f} × {t:.2f} = {v:.2f} m/s\n"
                    
    #         # If we need to find time when object hits ground
    #         elif h0 is not None and h is not None and t is None:
    #             # Use h = h₀ + v₀t + ½at²
    #             # Rearrange: ½at² + v₀t + (h₀ - h) = 0
    #             A, B, C = 0.5 * a, v0, h0 - h
    #             t1, t2 = solve_quadratic(A, B, C)
                
    #             if t1 is not None and t2 is not None:
    #                 # Choose the positive solution that makes physical sense
    #                 if t1 >= 0 and t2 >= 0:
    #                     t = max(t1, t2) if h < h0 else min(t1, t2)
    #                 elif t1 >= 0:
    #                     t = t1
    #                 elif t2 >= 0:
    #                     t = t2
    #                 else:
    #                     return "Error: No valid time solution found"
                        
    #                 result += f"t = {t:.2f} s (from quadratic equation)\n"
                    
    #                 if v is None:
    #                     v = v0 + a * t
    #                     result += f"v = v₀ + at = {v0:.2f} + {a:.2f} × {t:.2f} = {v:.2f} m/s\n"
                        
    #         # Calculate maximum height if object is thrown upward
    #         max_height = None
    #         time_to_max = None
    #         if v0 > 0 and h0 is not None:
    #             time_to_max = -v0 / a  # Time when v = 0
    #             max_height = h0 + v0 * time_to_max + 0.5 * a * time_to_max**2
                
    #         result += f"""
    # Complete Solution:
    # - Initial height (h₀): {h0:.2f} m
    # - Final height (h): {h:.2f} m
    # - Initial velocity (v₀): {v0:.2f} m/s
    # - Final velocity (v): {v:.2f} m/s  
    # - Time (t): {format_time(t)}
    # - Displacement (Δh): {h - h0:.2f} m
    # """

    #         if max_height is not None and time_to_max is not None:
    #             result += f"- Maximum height: {max_height:.2f} m (at t = {time_to_max:.2f} s)\n"
                
    #         result += f"""
    # Physical Interpretation:
    # """
            
    #         if v0 > 0:
    #             result += f"- Object thrown upward with initial speed {v0:.2f} m/s\n"
    #             if max_height is not None:
    #                 result += f"- Rises to maximum height of {max_height:.2f} m\n"
    #             result += f"- Then falls under gravity\n"
    #         elif v0 < 0:
    #             result += f"- Object thrown downward with initial speed {abs(v0):.2f} m/s\n"
    #         else:
    #             result += f"- Object dropped from rest\n"
                
    #         result += f"- Final velocity: {abs(v):.2f} m/s {'downward' if v < 0 else 'upward'}\n"
            
    #         # Impact velocity if hitting ground
    #         if h is not None and h <= 0:
    #             result += f"- Impact velocity: {abs(v):.2f} m/s"
            
    #         return result
            
    #     except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as e:
    #         return f'Error: {str(e)}\nExpected format: {{"h0": 100, "v0": 0, "t": 4}}'
 
    @mcp.tool()
    async def free_fall_motion(known_values: str, gravity: float = 9.81) -> str:
        """Analyze free fall motion (special case of constant acceleration).
        
        Args:
            known_values: JSON string with known values. Need at least 2 values.
                        Example: '{"h0": 100, "t": 4}' or '{"v0": 20, "h": 0}'
                        
                        Variables:
                        - h0: initial height (m)
                        - h: final height (m) 
                        - v0: initial velocity (m/s, positive = upward)
                        - v: final velocity (m/s, positive = upward)
                        - t: time (s)
                        
            gravity: Acceleration due to gravity (default: 9.81 m/s²)
        
        Returns:
            str: Complete free fall analysis with trajectory
        """
        try:
            data = json.loads(known_values)
            
            # Extract known values
            h0 = data.get('h0')
            h = data.get('h') 
            v0 = data.get('v0', 0.0)  # Default to 0 if not provided
            v = data.get('v')
            t = data.get('t')
            
            # Convert to standard kinematic variables (y-axis, positive up)
            a = -gravity  # negative because gravity pulls down
            
            # Build initial result
            result = []
            result.append("Free Fall Motion Analysis:")
            result.append("=========================")
            result.append("")
            result.append("Given Values:")
            result.append(f"- Gravity (g): {gravity:.2f} m/s²")
            result.append(f"- Acceleration (a = -g): {a:.2f} m/s²")
            
            if h0 is not None:
                result.append(f"- Initial height (h₀): {h0:.2f} m")
            if h is not None:
                result.append(f"- Final height (h): {h:.2f} m")
                
            velocity_desc = f"- Initial velocity (v₀): {v0:.2f} m/s"
            if v0 > 0:
                velocity_desc += " (upward)"
            elif v0 < 0:
                velocity_desc += " (downward)" 
            else:
                velocity_desc += " (dropped from rest)"
            result.append(velocity_desc)
                
            if v is not None:
                result.append(f"- Final velocity (v): {v:.2f} m/s")
            if t is not None:
                result.append(f"- Time (t): {format_time(t)}")
                
            result.append("")
            result.append("Solution Process:")
            
            # Determine which case we're in and solve
            solved = False
            
            # CASE 1: We have h0 and t
            if h0 is not None and t is not None and not solved:
                if h is None:
                    h = h0 + v0 * t + 0.5 * a * t * t
                    result.append(f"h = h₀ + v₀t + ½at² = {h0:.2f} + ({v0:.2f})({t:.2f}) + 0.5({a:.2f})({t:.2f})² = {h:.2f} m")
                
                if v is None:
                    v = v0 + a * t
                    result.append(f"v = v₀ + at = {v0:.2f} + ({a:.2f})({t:.2f}) = {v:.2f} m/s")
                solved = True
            
            # CASE 2: We have h0 and h
            if h0 is not None and h is not None and t is None and not solved:
                # Use h = h₀ + v₀t + ½at²
                # Rearrange: ½at² + v₀t + (h₀ - h) = 0
                A, B, C = 0.5 * a, v0, h0 - h
                t1, t2 = solve_quadratic(A, B, C)
                
                # Choose the positive solution
                if t1 is not None and t1 >= 0:
                    t = t1
                elif t2 is not None and t2 >= 0:
                    t = t2
                
                if t is not None:
                    result.append(f"Solving: {h:.2f} = {h0:.2f} + ({v0:.2f})t + 0.5({a:.2f})t²")
                    result.append(f"Time: t = {t:.2f} s")
                    
                    if v is None:
                        v = v0 + a * t
                        result.append(f"v = v₀ + at = {v0:.2f} + ({a:.2f})({t:.2f}) = {v:.2f} m/s")
                    solved = True
            
            # CASE 3: We have h0 but no h (assume falling to ground)
            if h0 is not None and h is None and not solved:
                h = 0  # Assume hitting the ground
                
                # Use h = h₀ + v₀t + ½at²
                # 0 = h₀ + v₀t + ½at²
                A, B, C = 0.5 * a, v0, h0
                result.append(f"DEBUG: Quadratic coefficients A={A:.3f}, B={B:.3f}, C={C:.3f}")
                
                t1, t2 = solve_quadratic(A, B, C)
                result.append(f"DEBUG: Quadratic solutions t1={t1}, t2={t2}")
                
                # Choose the positive solution
                if t1 is not None and t1 > 0:
                    t = t1
                    result.append(f"DEBUG: Chose t1 = {t:.3f}")
                elif t2 is not None and t2 > 0:
                    t = t2
                    result.append(f"DEBUG: Chose t2 = {t:.3f}")
                else:
                    result.append("DEBUG: No valid positive solution found!")
                    t = None
                
                if t is not None:
                    result.append("Solving for time to hit ground (h = 0):")
                    result.append(f"0 = {h0:.2f} + ({v0:.2f})t + 0.5({a:.2f})t²")
                    result.append(f"Time to ground: t = {t:.2f} s")
                    
                    if v is None:
                        v = v0 + a * t
                        result.append(f"Final velocity: v = {v0:.2f} + ({a:.2f})({t:.2f}) = {v:.2f} m/s")
                        result.append(f"DEBUG: Calculated v = {v}")
                    solved = True
                else:
                    result.append("ERROR: Could not solve for time!")
            
            # Calculate maximum height if object is thrown upward
            max_height = None
            time_to_max = None
            if v0 > 0 and h0 is not None:
                time_to_max = -v0 / a  # Time when v = 0
                max_height = h0 + v0 * time_to_max + 0.5 * a * time_to_max * time_to_max
                
            # Build final results
            result.append("")
            result.append("Complete Solution:")
            result.append(f"- Initial height (h₀): {safe_format(h0)} m")
            result.append(f"- Final height (h): {safe_format(h)} m")
            result.append(f"- Initial velocity (v₀): {safe_format(v0)} m/s")
            result.append(f"- Final velocity (v): {safe_format(v)} m/s")
            result.append(f"- Time (t): {format_time(t) if t is not None else 'Unknown'}")
            
            if h is not None and h0 is not None:
                displacement = h - h0
                result.append(f"- Displacement (Δh): {displacement:.2f} m")
            else:
                result.append("- Displacement (Δh): Unknown m")

            if max_height is not None and time_to_max is not None:
                result.append(f"- Maximum height: {max_height:.2f} m (at t = {time_to_max:.2f} s)")
                
            result.append("")
            result.append("Physical Interpretation:")
            
            if v0 > 0:
                result.append(f"- Object thrown upward with initial speed {v0:.2f} m/s")
                if max_height is not None:
                    result.append(f"- Rises to maximum height of {max_height:.2f} m")
                result.append("- Then falls under gravity")
            elif v0 < 0:
                result.append(f"- Object thrown downward with initial speed {abs(v0):.2f} m/s")
            else:
                result.append("- Object dropped from rest")
                
            if v is not None:
                direction = "downward" if v < 0 else "upward"
                result.append(f"- Final velocity: {abs(v):.2f} m/s {direction}")
            
            # Impact velocity if hitting ground
            if h is not None and h <= 0 and v is not None:
                result.append(f"- Impact velocity: {abs(v):.2f} m/s")
            
            return "\n".join(result)
            
        except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as e:
            return f'Error: {str(e)}\nExpected format: {{"h0": 100, "v0": 0, "t": 4}}'
        
    @mcp.tool()
    async def projectile_motion_2d(launch_conditions: str, target_info: str = None) -> str:
        """Analyze 2D projectile motion.
        
        Args:
            launch_conditions: JSON string with launch parameters.
                            Example: '{"v0": 50, "angle": 45, "h0": 10}'
                            
                            Variables:
                            - v0: initial speed (m/s)
                            - angle: launch angle in degrees (from horizontal)
                            - h0: initial height (m, default: 0)
                            - x0: initial horizontal position (m, default: 0)
                            
            target_info: Optional JSON string with target parameters.
                        Example: '{"x_target": 200}' or '{"x_target": 150, "h_target": 20}'
        
        Returns:
            str: Complete 2D projectile motion analysis
        """
        try:
            launch_data = json.loads(launch_conditions)
            
            v0 = launch_data['v0']
            angle_deg = launch_data['angle'] 
            h0 = launch_data.get('h0', 0.0)
            x0 = launch_data.get('x0', 0.0)
            gravity = launch_data.get('gravity', 9.81)
            
            # Convert angle to radians
            angle_rad = degrees_to_radians(angle_deg)
            
            # Initial velocity components
            v0x = v0 * math.cos(angle_rad)
            v0y = v0 * math.sin(angle_rad)
            
            result = f"""
    2D Projectile Motion Analysis:
    =============================

    Launch Conditions:
    - Initial speed (v₀): {v0:.2f} m/s
    - Launch angle (θ): {angle_deg:.1f}°
    - Initial height (h₀): {h0:.2f} m
    - Initial horizontal position (x₀): {x0:.2f} m
    - Gravity (g): {gravity:.2f} m/s²

    Initial Velocity Components:
    - Horizontal (v₀ₓ): v₀ cos(θ) = {v0:.2f} × cos({angle_deg:.1f}°) = {v0x:.2f} m/s
    - Vertical (v₀ᵧ): v₀ sin(θ) = {v0:.2f} × sin({angle_deg:.1f}°) = {v0y:.2f} m/s

    Motion Equations:
    - Horizontal: x(t) = x₀ + v₀ₓt = {x0:.2f} + {v0x:.2f}t
    - Vertical: y(t) = h₀ + v₀ᵧt - ½gt² = {h0:.2f} + {v0y:.2f}t - {0.5*gravity:.2f}t²

    Key Trajectory Points:
    """
            
            # Time to reach maximum height
            if v0y > 0:
                t_max_height = v0y / gravity
                max_height = h0 + v0y * t_max_height - 0.5 * gravity * t_max_height**2
                x_at_max = x0 + v0x * t_max_height
                
                result += f"Maximum Height:\n"
                result += f"- Time to max height: {t_max_height:.2f} s\n"
                result += f"- Maximum height: {max_height:.2f} m\n"
                result += f"- Horizontal distance at max height: {x_at_max:.2f} m\n\n"
            else:
                max_height = h0
                t_max_height = 0
                
            # Time to hit ground (y = 0)
            # Solve: 0 = h₀ + v₀ᵧt - ½gt²
            # Rearrange: ½gt² - v₀ᵧt - h₀ = 0
            A, B, C = 0.5 * gravity, -v0y, -h0
            t1, t2 = solve_quadratic(A, B, C)
            
            if t1 is not None and t2 is not None:
                # Choose positive solution
                t_flight = max(t1, t2) if max(t1, t2) > 0 else None
            elif t1 is not None:
                t_flight = t1 if t1 > 0 else None
            else:
                t_flight = None
                
            if t_flight is not None:
                # Range and impact conditions
                x_range = x0 + v0x * t_flight
                impact_vx = v0x  # Horizontal velocity constant
                impact_vy = v0y - gravity * t_flight
                impact_speed = math.sqrt(impact_vx**2 + impact_vy**2)
                impact_angle = radians_to_degrees(math.atan2(-abs(impact_vy), impact_vx))
                
                result += f"Impact Conditions (when projectile hits ground):\n"
                result += f"- Flight time: {format_time(t_flight)}\n"
                result += f"- Range (horizontal distance): {x_range - x0:.2f} m\n"
                result += f"- Impact position: ({x_range:.2f}, 0) m\n"
                result += f"- Impact velocity components: ({impact_vx:.2f}, {impact_vy:.2f}) m/s\n"
                result += f"- Impact speed: {impact_speed:.2f} m/s\n"
                result += f"- Impact angle: {impact_angle:.1f}° below horizontal\n\n"
            
            # Handle target analysis if provided
            if target_info:
                try:
                    target_data = json.loads(target_info)
                    x_target = target_data.get('x_target', None)
                    h_target = target_data.get('h_target', 0)
                    
                    if x_target is not None:
                        # Find time when projectile reaches target x-position
                        t_target = (x_target - x0) / v0x
                        
                        if t_target >= 0:
                            h_at_target = h0 + v0y * t_target - 0.5 * gravity * t_target**2
                            vy_at_target = v0y - gravity * t_target
                            
                            result += f"Target Analysis (x = {x_target:.2f} m):\n"
                            result += f"- Time to reach target: {format_time(t_target)}\n"
                            result += f"- Height at target: {h_at_target:.2f} m\n"
                            result += f"- Vertical velocity at target: {vy_at_target:.2f} m/s\n"
                            
                            if h_target is not None:
                                height_diff = h_at_target - h_target
                                result += f"- Target height: {h_target:.2f} m\n"
                                if abs(height_diff) < 0.1:
                                    result += f"✓ Target HIT! (within 0.1 m)\n"
                                else:
                                    result += f"✗ Target MISSED by {abs(height_diff):.2f} m {'above' if height_diff > 0 else 'below'}\n"
                        else:
                            result += f"Target Analysis: Target is behind launch position\n"
                            
                except json.JSONDecodeError:
                    result += f"Warning: Could not parse target_info\n"
            
            # Summary
            result += f"""
    Trajectory Summary:
    - Launch speed: {v0:.2f} m/s at {angle_deg:.1f}°
    - Maximum height: {max_height:.2f} m"""
            
            if t_flight is not None:
                result += f"\n- Total flight time: {format_time(t_flight)}"
                result += f"\n- Total range: {x_range - x0:.2f} m"
                result += f"\n- Impact speed: {impact_speed:.2f} m/s"
                
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as e:
            return f'Error: {str(e)}\nExpected format: {{"v0": 50, "angle": 45, "h0": 10}}'

    @mcp.tool()
    async def motion_graphs(motion_type: str, parameters: str, time_range: str = "0,10,0.5") -> str:
        """Generate position, velocity, and acceleration vs time data for graphing.
        
        Args:
            motion_type: Type of motion - "uniform", "constant_acceleration", or "projectile"
            parameters: JSON string with motion parameters
            time_range: String "start,end,step" for time values (default: "0,10,0.5")
        
        Returns:
            str: Tabulated motion data for graphing with analysis
        """
        try:
            params = json.loads(parameters)
            t_start, t_end, t_step = map(float, time_range.split(','))
            
            result = f"""
    Motion Graph Data:
    =================

    Motion Type: {motion_type.replace('_', ' ').title()}
    Time Range: {t_start:.1f} to {t_end:.1f} s (step: {t_step:.1f} s)

    """
            
            # Generate time array
            times = []
            t = t_start
            while t <= t_end:
                times.append(t)
                t += t_step
                
            if motion_type == "uniform":
                x0 = params.get('x0', 0)
                v = params['v']
                
                result += f"Parameters: x₀ = {x0:.2f} m, v = {v:.2f} m/s\n\n"
                result += f"{'Time (s)':<8} {'Position (m)':<12} {'Velocity (m/s)':<14} {'Acceleration (m/s²)':<16}\n"
                result += f"{'-'*50}\n"
                
                for t in times:
                    x = x0 + v * t
                    result += f"{t:<8.1f} {x:<12.2f} {v:<14.2f} {0.0:<16.2f}\n"
                    
            elif motion_type == "constant_acceleration":
                x0 = params.get('x0', 0)
                v0 = params['v0']
                a = params['a']
                
                result += f"Parameters: x₀ = {x0:.2f} m, v₀ = {v0:.2f} m/s, a = {a:.2f} m/s²\n\n"
                result += f"{'Time (s)':<8} {'Position (m)':<12} {'Velocity (m/s)':<14} {'Acceleration (m/s²)':<16}\n"
                result += f"{'-'*50}\n"
                
                for t in times:
                    x = x0 + v0 * t + 0.5 * a * t**2
                    v = v0 + a * t
                    result += f"{t:<8.1f} {x:<12.2f} {v:<14.2f} {a:<16.2f}\n"
                    
            elif motion_type == "projectile":
                x0 = params.get('x0', 0)
                h0 = params.get('h0', 0)
                v0 = params['v0']
                angle = params['angle']
                g = params.get('gravity', 9.81)
                
                angle_rad = degrees_to_radians(angle)
                v0x = v0 * math.cos(angle_rad)
                v0y = v0 * math.sin(angle_rad)
                
                result += f"Parameters: v₀ = {v0:.2f} m/s, θ = {angle:.1f}°, h₀ = {h0:.2f} m\n"
                result += f"Components: v₀ₓ = {v0x:.2f} m/s, v₀ᵧ = {v0y:.2f} m/s\n\n"
                result += f"{'Time (s)':<8} {'X (m)':<10} {'Y (m)':<10} {'Vx (m/s)':<10} {'Vy (m/s)':<10} {'Speed (m/s)':<12}\n"
                result += f"{'-'*60}\n"
                
                for t in times:
                    x = x0 + v0x * t
                    y = h0 + v0y * t - 0.5 * g * t**2
                    vx = v0x
                    vy = v0y - g * t
                    speed = math.sqrt(vx**2 + vy**2)
                    
                    # Only show data while projectile is above ground
                    if y >= 0:
                        result += f"{t:<8.1f} {x:<10.2f} {y:<10.2f} {vx:<10.2f} {vy:<10.2f} {speed:<12.2f}\n"
                    else:
                        break
            else:
                return "Error: motion_type must be 'uniform', 'constant_acceleration', or 'projectile'"
                
            result += f"\nGraph Analysis Tips:\n"
            result += f"- Position vs Time shows the trajectory\n"
            result += f"- Velocity vs Time shows how speed changes\n"
            result += f"- Acceleration vs Time shows forces acting on object\n"
            result += f"- Slope of position graph = velocity\n"
            result += f"- Slope of velocity graph = acceleration"
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            return f'Error: {str(e)}\nCheck parameters and time_range format'

    @mcp.tool()
    async def relative_motion_1d(objects_data: str) -> str:
        """Analyze relative motion between two objects in 1D.
        
        Args:
            objects_data: JSON string with two objects' motion data.
                        Example: '{"object1": {"x0": 0, "v": 20}, "object2": {"x0": 100, "v": -15}}'
                        Each object needs position and velocity info.
        
        Returns:
            str: Relative motion analysis including when/where objects meet
        """
        try:
            data = json.loads(objects_data)
            
            obj1 = data['object1']
            obj2 = data['object2']
            
            # Extract data for object 1
            x1_0 = obj1.get('x0', 0)
            v1 = obj1.get('v', 0)
            
            # Extract data for object 2  
            x2_0 = obj2.get('x0', 0)
            v2 = obj2.get('v', 0)
            
            result = f"""
    1D Relative Motion Analysis:
    ===========================

    Object 1:
    - Initial position: {x1_0:.2f} m
    - Velocity: {v1:.2f} m/s
    - Position equation: x₁(t) = {x1_0:.2f} + {v1:.2f}t

    Object 2:
    - Initial position: {x2_0:.2f} m  
    - Velocity: {v2:.2f} m/s
    - Position equation: x₂(t) = {x2_0:.2f} + {v2:.2f}t

    Relative Motion Analysis:
    """
            
            # Initial separation
            initial_separation = abs(x2_0 - x1_0)
            result += f"- Initial separation: {initial_separation:.2f} m\n"
            
            # Relative velocity
            v_rel = v1 - v2
            result += f"- Relative velocity (v₁ - v₂): {v_rel:.2f} m/s\n"
            
            # Determine if objects will meet
            if abs(v_rel) < 0.001:  # Essentially zero relative velocity
                if abs(x1_0 - x2_0) < 0.001:
                    result += f"- Objects are moving together (same position and velocity)\n"
                else:
                    result += f"- Objects maintain constant separation (parallel motion)\n"
                    result += f"- Separation remains: {abs(x1_0 - x2_0):.2f} m\n"
            else:
                # Find meeting time: x1(t) = x2(t)
                # x1_0 + v1*t = x2_0 + v2*t
                # (v1 - v2)*t = x2_0 - x1_0
                t_meet = (x2_0 - x1_0) / v_rel
                
                if t_meet >= 0:
                    x_meet = x1_0 + v1 * t_meet
                    result += f"\nObjects WILL meet:\n"
                    result += f"- Meeting time: {format_time(t_meet)}\n"
                    result += f"- Meeting position: {x_meet:.2f} m\n"
                    
                    # Verify
                    x2_meet = x2_0 + v2 * t_meet
                    result += f"- Verification: x₂({t_meet:.2f}) = {x2_meet:.2f} m ✓\n"
                    
                else:
                    result += f"\nObjects will NOT meet (would have met {format_time(-t_meet)} ago)\n"
                    
                    # Find minimum separation if objects are moving apart
                    # This occurs when relative velocity changes sign (not applicable in uniform motion)
                    result += f"- Objects are moving away from each other\n"
                    
            # Distance between objects as function of time
            result += f"\nSeparation Distance:\n"
            result += f"d(t) = |x₁(t) - x₂(t)| = |{x1_0:.2f} + {v1:.2f}t - {x2_0:.2f} - {v2:.2f}t|\n"
            result += f"d(t) = |{x1_0 - x2_0:.2f} + {v_rel:.2f}t|\n"
            
            # Sample distances at different times
            result += f"\nSample Separations:\n"
            result += f"{'Time (s)':<10} {'Separation (m)':<15}\n"
            result += f"{'-'*25}\n"
            
            for t in [0, 1, 2, 5, 10]:
                if t_meet is not None and abs(t - t_meet) < 0.1:
                    continue  # Skip time near meeting point
                separation = abs((x1_0 - x2_0) + v_rel * t)
                result += f"{t:<10.1f} {separation:<15.2f}\n"
                
            # Physical interpretation
            result += f"\nPhysical Interpretation:\n"
            
            if v1 > v2:
                result += f"- Object 1 is faster than Object 2\n"
            elif v1 < v2:
                result += f"- Object 2 is faster than Object 1\n"
            else:
                result += f"- Both objects have the same speed\n"
                
            if v_rel > 0:
                result += f"- Object 1 is catching up to Object 2\n"
            elif v_rel < 0:
                result += f"- Object 2 is catching up to Object 1\n"
            else:
                result += f"- Objects maintain constant relative position"
                
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as e:
            return f'Error: {str(e)}\nExpected format: {{"object1": {{"x0": 0, "v": 20}}, "object2": {{"x0": 100, "v": -15}}}}'


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
    """CLI entry point for the physics-forces-mcp tool."""
    parser = argparse.ArgumentParser(description="Run Physics Kinematics MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10101, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")
    
    args = parser.parse_args()
    
    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")

if __name__ == "__main__":
    main()
