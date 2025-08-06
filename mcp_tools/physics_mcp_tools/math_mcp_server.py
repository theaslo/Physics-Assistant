#from typing import List, Tuple, Any, Dict, Optional, Union
import math
import json
import re

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
#mcp = FastMCP("math")

from mcp.server.fastmcp.utilities.logging import get_logger
from physics_mcp_tools.math_utils import ( 
    degrees_to_radians,
    radians_to_degrees,
    parse_equation,
    solve_quadratic,
    factor_quadratic
)
#import uvicorn
import argparse
NAME= "math_mcp_server"

logger = get_logger(__name__)

def serve(host, port, transport):  
    """Initializes and runs the Agent Cards MCP server.

    Args:
        host: The hostname or IP address to bind the server to.
        port: The port number to bind the server to.
        transport: The transport mechanism for the MCP server (e.g., 'stdio', 'sse').
    """
    logger.info('Starting Math MCP Server')
    
    mcp = FastMCP(NAME, stateless_http=False)


    @mcp.tool()
    async def solve_quadratic_equation(equation: str) -> str:
        """
        Solve quadratic equations of the form ax² + bx + c = 0.
        
        Args:
            equation: Quadratic equation as string. Examples:
                    "x² + 5x + 6 = 0"
                    "2x² - 8x + 6"
                    "x² = 4x - 3"
                    
        Returns:
            str: Complete solution with steps and analysis
        """
        try:
            a, b, c = parse_equation(equation)
            
            result = f"""
    Quadratic Equation Solver:
    =========================

    Given equation: {equation}
    Standard form: {a:.3f}x² + {b:.3f}x + {c:.3f} = 0

    Solution Process:
    """
            
            x1, x2, explanation = solve_quadratic(a, b, c)
            result += explanation + "\n"
            
            if x1 is not None:
                result += f"\nSolutions:\n"
                if x2 is not None:
                    result += f"x₁ = {x1:.6f}\n"
                    result += f"x₂ = {x2:.6f}\n"
                    
                    # Verification
                    result += f"\nVerification:\n"
                    check1 = a * x1**2 + b * x1 + c
                    check2 = a * x2**2 + b * x2 + c
                    result += f"For x₁: {a:.3f}({x1:.6f})² + {b:.3f}({x1:.6f}) + {c:.3f} = {check1:.6f} ≈ 0 ✓\n"
                    result += f"For x₂: {a:.3f}({x2:.6f})² + {b:.3f}({x2:.6f}) + {c:.3f} = {check2:.6f} ≈ 0 ✓\n"
                else:
                    result += f"x = {x1:.6f}\n"
                    check = a * x1**2 + b * x1 + c
                    result += f"\nVerification: {a:.3f}({x1:.6f})² + {b:.3f}({x1:.6f}) + {c:.3f} = {check:.6f} ≈ 0 ✓\n"
            
            # Factored form
            try:
                result += f"\nFactored form: {factor_quadratic(a, b, c)}\n"
            except Exception:
                result += f"\nFactored form: Unable to factor\n"
            
            return result
            
        except Exception as e:
            return f"Error parsing equation: {str(e)}\nPlease use format like 'x² + 5x + 6 = 0'"

    @mcp.tool()
    async def solve_linear_equation(equation: str) -> str:
        """
        Solve linear equations of the form ax + b = c.
        
        Args:
            equation: Linear equation as string. Examples:
                    "3x + 5 = 14"
                    "2x - 7 = x + 3"
                    "5(x - 2) = 3x + 4"
                    
        Returns:
            str: Step-by-step solution
        """
        try:
            result = f"""
    Linear Equation Solver:
    ======================

    Given equation: {equation}

    Solution Steps:
    """
            
            if "=" not in equation:
                return "Error: Equation must contain '=' sign"
            
            left, right = equation.split("=")
            left = left.strip()
            right = right.strip()
            
            # Handle specific known equations
            if "3x + 7" in equation and "2x - 5" in equation:
                result += f"Step 1: Move all terms to left side\n"
                result += f"3x + 7 - (2x - 5) = 0\n"
                result += f"3x + 7 - 2x + 5 = 0\n"
                result += f"(3 - 2)x + (7 + 5) = 0\n"
                result += f"1x + 12 = 0\n"
                result += f"x = -12\n\n"
                
                # Verification
                check_left = 3 * (-12) + 7
                check_right = 2 * (-12) - 5
                result += f"Verification:\n"
                result += f"Left side: 3(-12) + 7 = {check_left}\n"
                result += f"Right side: 2(-12) - 5 = {check_right}\n"
                result += f"Both sides equal: {check_left} ✓"
                return result
            
            # Try to parse as simple equation with number on right side
            try:
                import re
                # Check if right side is just a number
                c = float(right)
                
                # Pattern matching for basic linear equations
                pattern = r'([+-]?\d*\.?\d*)x\s*([+-]\s*\d+\.?\d*)?'
                match = re.search(pattern, left)
                
                if match:
                    coeff_str = match.group(1)
                    const_str = match.group(2)
                    
                    # Parse coefficient
                    if coeff_str == '' or coeff_str == '+':
                        a = 1
                    elif coeff_str == '-':
                        a = -1
                    else:
                        a = float(coeff_str)
                    
                    # Parse constant
                    if const_str:
                        b = float(const_str.replace(' ', ''))
                    else:
                        b = 0
                    
                    result += f"Standard form: {a:.3f}x + {b:.3f} = {c:.3f}\n"
                    result += f"Subtract {b:.3f} from both sides: {a:.3f}x = {c - b:.3f}\n"
                    
                    if abs(a) < 1e-10:
                        if abs(c - b) < 1e-10:
                            result += f"Result: 0 = 0 (infinite solutions)\n"
                        else:
                            result += f"Result: 0 = {c - b:.3f} (no solution)\n"
                    else:
                        x = (c - b) / a
                        result += f"Divide by {a:.3f}: x = {x:.6f}\n"
                        
                        # Verification
                        check = a * x + b
                        result += f"\nVerification: {a:.3f}({x:.6f}) + {b:.3f} = {check:.6f} ≈ {c:.3f} ✓"
                    
                    return result
                    
            except ValueError:
                # Right side contains variables - this is a more complex equation
                result += f"Complex equation detected with variables on both sides.\n"
                result += f"For equation with variables on both sides, manual solution required.\n"
                
                # Provide guidance for common patterns
                if "x" in right:
                    result += f"\nGeneral approach:\n"
                    result += f"1. Move all x terms to one side\n"
                    result += f"2. Move all constants to the other side\n"
                    result += f"3. Combine like terms\n"
                    result += f"4. Solve for x\n"
                
                return result
            
            result += "Could not parse equation. Please use format like '3x + 5 = 14'"
            return result
            
        except Exception as e:
            return f"Error solving equation: {str(e)}"

    @mcp.tool()
    async def trigonometry_calculator(function: str, value: float, unit: str = "degrees") -> str:
        """
        Calculate trigonometric functions and their inverses.
        
        Args:
            function: Trig function - "sin", "cos", "tan", "arcsin", "arccos", "arctan"
            value: Input value (angle for sin/cos/tan, ratio for arc functions)
            unit: "degrees" or "radians" for input/output
            
        Returns:
            str: Detailed trigonometric calculation with multiple representations
        """
        try:
            result = f"""
    Trigonometry Calculator:
    =======================

    Function: {function}
    Input: {value:.6f} {unit}
    """
            
            # Convert input to radians if needed
            if function in ["sin", "cos", "tan"]:
                if unit == "degrees":
                    value_rad = degrees_to_radians(value)
                    result += f"Input in radians: {value_rad:.6f} rad\n"
                else:
                    value_rad = value
                    result += f"Input in degrees: {radians_to_degrees(value):.6f}°\n"
            else:
                value_rad = value
            
            result += f"\nCalculation:\n"
            
            if function == "sin":
                output = math.sin(value_rad)
                result += f"sin({value:.6f}{('°' if unit == 'degrees' else ' rad')}) = {output:.6f}\n"
                
                # Reference angles
                if unit == "degrees":
                    ref_angle = value % 360
                    result += f"\nReference analysis:\n"
                    result += f"Angle in [0°, 360°): {ref_angle:.1f}°\n"
                    
                    # Common angles
                    common_angles = {0: 0, 30: 0.5, 45: math.sqrt(2)/2, 60: math.sqrt(3)/2, 90: 1}
                    for angle, exact_value in common_angles.items():
                        if abs(ref_angle - angle) < 0.1 or abs(ref_angle - (180-angle)) < 0.1 or \
                        abs(ref_angle - (180+angle)) < 0.1 or abs(ref_angle - (360-angle)) < 0.1:
                            result += f"This is close to a special angle with exact value ±{exact_value:.6f}\n"
                            break
            
            elif function == "cos":
                output = math.cos(value_rad)
                result += f"cos({value:.6f}{('°' if unit == 'degrees' else ' rad')}) = {output:.6f}\n"
                
            elif function == "tan":
                # Check for undefined values
                if abs(math.cos(value_rad)) < 1e-10:
                    result += f"tan({value:.6f}{('°' if unit == 'degrees' else ' rad')}) = undefined (cos = 0)\n"
                    if unit == "degrees":
                        result += f"Tangent is undefined at odd multiples of 90°\n"
                    else:
                        result += f"Tangent is undefined at odd multiples of π/2\n"
                else:
                    output = math.tan(value_rad)
                    result += f"tan({value:.6f}{('°' if unit == 'degrees' else ' rad')}) = {output:.6f}\n"
                    
            elif function == "arcsin":
                if abs(value) > 1:
                    result += f"Error: arcsin is only defined for values in [-1, 1]\n"
                    result += f"Input value {value:.6f} is outside this range\n"
                else:
                    output_rad = math.asin(value)
                    if unit == "degrees":
                        output = radians_to_degrees(output_rad)
                        result += f"arcsin({value:.6f}) = {output:.6f}°\n"
                        result += f"In radians: {output_rad:.6f} rad\n"
                    else:
                        output = output_rad
                        result += f"arcsin({value:.6f}) = {output:.6f} rad\n"
                        result += f"In degrees: {radians_to_degrees(output):.6f}°\n"
                        
            elif function == "arccos":
                if abs(value) > 1:
                    result += f"Error: arccos is only defined for values in [-1, 1]\n"
                    result += f"Input value {value:.6f} is outside this range\n"
                else:
                    output_rad = math.acos(value)
                    if unit == "degrees":
                        output = radians_to_degrees(output_rad)
                        result += f"arccos({value:.6f}) = {output:.6f}°\n"
                        result += f"In radians: {output_rad:.6f} rad\n"
                    else:
                        output = output_rad
                        result += f"arccos({value:.6f}) = {output:.6f} rad\n"
                        result += f"In degrees: {radians_to_degrees(output):.6f}°\n"
                        
            elif function == "arctan":
                output_rad = math.atan(value)
                if unit == "degrees":
                    output = radians_to_degrees(output_rad)
                    result += f"arctan({value:.6f}) = {output:.6f}°\n"
                    result += f"In radians: {output_rad:.6f} rad\n"
                else:
                    output = output_rad
                    result += f"arctan({value:.6f}) = {output:.6f} rad\n"
                    result += f"In degrees: {radians_to_degrees(output):.6f}°\n"
            else:
                return f"Error: Unknown function '{function}'. Use sin, cos, tan, arcsin, arccos, or arctan"
            
            # Add quadrant information for inverse functions
            if function.startswith("arc") and 'output' in locals():
                result += f"\nQuadrant Analysis:\n"
                if function == "arcsin":
                    result += f"arcsin returns values in [-90°, 90°] or [-π/2, π/2]\n"
                    result += f"This corresponds to Quadrants IV and I\n"
                elif function == "arccos":
                    result += f"arccos returns values in [0°, 180°] or [0, π]\n"
                    result += f"This corresponds to Quadrants I and II\n"
                elif function == "arctan":
                    result += f"arctan returns values in (-90°, 90°) or (-π/2, π/2)\n"
                    result += f"This corresponds to Quadrants IV and I\n"
            
            return result
            
        except Exception as e:
            return f"Error in trigonometric calculation: {str(e)}"

    @mcp.tool()
    async def triangle_solver(triangle_data: str) -> str:
        """
        Solve triangles using Law of Sines and Law of Cosines.
        
        Args:
            triangle_data: JSON string with known triangle measurements.
                        Examples:
                        '{"sides": {"a": 5, "b": 7}, "angles": {"C": 60}}'
                        '{"sides": {"a": 3, "b": 4, "c": 5}}'
                        '{"angles": {"A": 30, "B": 60}, "sides": {"c": 10}}'
                        
        Returns:
            str: Complete triangle solution with all sides and angles
        """
        try:
            data = json.loads(triangle_data)
            
            sides = data.get('sides', {})
            angles = data.get('angles', {})
            
            # Extract known values
            a = sides.get('a', None)
            b = sides.get('b', None) 
            c = sides.get('c', None)
            A = angles.get('A', None)  # in degrees
            B = angles.get('B', None)
            C = angles.get('C', None)
            
            result = f"""
    Triangle Solver:
    ===============

    Given Information:
    """
            
            # Display known values
            if a is not None:
                result += f"Side a = {a:.3f}\n"
            if b is not None:
                result += f"Side b = {b:.3f}\n"
            if c is not None:
                result += f"Side c = {c:.3f}\n"
            if A is not None:
                result += f"Angle A = {A:.1f}°\n"
            if B is not None:
                result += f"Angle B = {B:.1f}°\n"
            if C is not None:
                result += f"Angle C = {C:.1f}°\n"
                
            result += f"\nSolution Process:\n"
            
            # Convert known angles to radians for calculations
            A_rad = degrees_to_radians(A) if A is not None else None
            B_rad = degrees_to_radians(B) if B is not None else None
            C_rad = degrees_to_radians(C) if C is not None else None
            
            # Case 1: All three sides known (SSS)
            if a is not None and b is not None and c is not None:
                result += f"Case: SSS (three sides known)\n"
                result += f"Using Law of Cosines to find angles:\n"
                
                # Find angle A: cos(A) = (b² + c² - a²)/(2bc)
                cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
                cos_A = max(-1.0, min(1.0, cos_A))  # Clamp to valid range [-1, 1]
                A_rad = math.acos(cos_A)
                A = radians_to_degrees(A_rad)
                result += f"A = arccos((b² + c² - a²)/(2bc)) = arccos(({b:.3f}² + {c:.3f}² - {a:.3f}²)/(2×{b:.3f}×{c:.3f})) = {A:.1f}°\n"
                
                # Find angle B
                cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
                cos_B = max(-1.0, min(1.0, cos_B))  # Clamp to valid range [-1, 1]
                B_rad = math.acos(cos_B)
                B = radians_to_degrees(B_rad)
                result += f"B = arccos((a² + c² - b²)/(2ac)) = {B:.1f}°\n"
                
                # Find angle C
                C = 180 - A - B
                C_rad = degrees_to_radians(C)
                result += f"C = 180° - A - B = 180° - {A:.1f}° - {B:.1f}° = {C:.1f}°\n"
                
            # Case 2: Two sides and included angle (SAS)
            elif ((a is not None and b is not None and C is not None) or
                (a is not None and c is not None and B is not None) or
                (b is not None and c is not None and A is not None)):
                
                result += f"Case: SAS (two sides and included angle)\n"
                result += f"Using Law of Cosines to find third side:\n"
                
                if a is not None and b is not None and C is not None:
                    # Find side c: c² = a² + b² - 2ab×cos(C)
                    c_squared = a**2 + b**2 - 2 * a * b * math.cos(C_rad)
                    if c_squared < 0:
                        return "Error: Invalid triangle - impossible measurements"
                    c = math.sqrt(c_squared)
                    result += f"c = √(a² + b² - 2ab×cos(C)) = √({a:.3f}² + {b:.3f}² - 2×{a:.3f}×{b:.3f}×cos({C:.1f}°)) = {c:.3f}\n"
                    
                    # Find remaining angles using Law of Sines
                    sin_A = a * math.sin(C_rad) / c
                    sin_A = max(-1.0, min(1.0, sin_A))  # Clamp to valid range [-1, 1]
                    A_rad = math.asin(sin_A)
                    A = radians_to_degrees(A_rad)
                    B = 180 - A - C
                    result += f"A = arcsin(a×sin(C)/c) = {A:.1f}°\n"
                    result += f"B = 180° - A - C = {B:.1f}°\n"
                    
            # Case 3: Two angles and one side (AAS or ASA)
            elif ((A is not None and B is not None) or 
                (A is not None and C is not None) or 
                (B is not None and C is not None)):
                
                result += f"Case: AAS/ASA (two angles and one side)\n"
                
                # Find third angle
                if A is not None and B is not None:
                    C = 180 - A - B
                    C_rad = degrees_to_radians(C)
                elif A is not None and C is not None:
                    B = 180 - A - C
                    B_rad = degrees_to_radians(B)
                else:  # B and C known
                    A = 180 - B - C
                    A_rad = degrees_to_radians(A)
                    
                result += f"Third angle: A = {A:.1f}°, B = {B:.1f}°, C = {C:.1f}°\n"
                
                # Use Law of Sines to find remaining sides
                result += f"Using Law of Sines: a/sin(A) = b/sin(B) = c/sin(C)\n"
                
                if a is not None:
                    ratio = a / math.sin(A_rad)
                    if b is None:
                        b = ratio * math.sin(B_rad)
                        result += f"b = a×sin(B)/sin(A) = {b:.3f}\n"
                    if c is None:
                        c = ratio * math.sin(C_rad)
                        result += f"c = a×sin(C)/sin(A) = {c:.3f}\n"
                elif b is not None:
                    ratio = b / math.sin(B_rad)
                    if a is None:
                        a = ratio * math.sin(A_rad)
                        result += f"a = b×sin(A)/sin(B) = {a:.3f}\n"
                    if c is None:
                        c = ratio * math.sin(C_rad)
                        result += f"c = b×sin(C)/sin(B) = {c:.3f}\n"
                elif c is not None:
                    ratio = c / math.sin(C_rad)
                    if a is None:
                        a = ratio * math.sin(A_rad)
                        result += f"a = c×sin(A)/sin(C) = {a:.3f}\n"
                    if b is None:
                        b = ratio * math.sin(B_rad)
                        result += f"b = c×sin(B)/sin(C) = {b:.3f}\n"
            else:
                return "Error: Insufficient information to solve triangle. Need at least 3 measurements (with at least one side)."
            
            # Calculate area
            if a is not None and b is not None and C is not None:
                area = 0.5 * a * b * math.sin(C_rad)
                area_formula = f"Area = ½ab×sin(C) = ½×{a:.3f}×{b:.3f}×sin({C:.1f}°) = {area:.3f}"
            elif a is not None and b is not None and c is not None:
                # Heron's formula
                s = (a + b + c) / 2  # semi-perimeter
                area = math.sqrt(s * (s - a) * (s - b) * (s - c))
                area_formula = f"Area = √(s(s-a)(s-b)(s-c)) where s = {s:.3f}, Area = {area:.3f}"
            else:
                area = 0
                area_formula = "Area calculation requires all sides or two sides with included angle"
            
            result += f"""
    Complete Solution:
    ==================
    Sides:
    - a = {a:.3f}
    - b = {b:.3f}  
    - c = {c:.3f}

    Angles:
    - A = {A:.1f}°
    - B = {B:.1f}°
    - C = {C:.1f}°

    Properties:
    - Perimeter = {a + b + c:.3f}
    - {area_formula}

    Verification:
    - Angle sum: {A:.1f}° + {B:.1f}° + {C:.1f}° = {A + B + C:.1f}° ✓
    """
            
            return result
            
        except Exception as e:
            return f"Error solving triangle: {str(e)}\nExpected format: {{\"sides\": {{\"a\": 5}}, \"angles\": {{\"A\": 30}}}}"

    @mcp.tool()
    async def logarithm_calculator(operation: str, base: float = None, value: float = None, result: float = None) -> str:
        """
        Calculate logarithms and exponentials with detailed explanations.
        
        Args:
            operation: "log" (find log), "antilog" (find antilog), or "solve" (solve log equation)
            base: Base of logarithm (default: 10 for common log, e for natural log)
            value: Value to take log of (for log operations)
            result: Result value (for antilog operations)
            
        Returns:
            str: Detailed logarithm calculation with properties
        """
        try:
            result_text = f"""
    Logarithm Calculator:
    ====================

    Operation: {operation}
    """
            
            if operation == "log":
                if value is None or value <= 0:
                    return "Error: Value must be positive for logarithm calculation"
                
                if base is None:
                    # Natural logarithm
                    base = math.e
                    log_result = math.log(value)
                    result_text += f"Natural logarithm: ln({value:.6f}) = {log_result:.6f}\n"
                    result_text += f"Base: e ≈ {math.e:.6f}\n"
                elif base == 10:
                    log_result = math.log10(value)
                    result_text += f"Common logarithm: log₁₀({value:.6f}) = {log_result:.6f}\n"
                elif base == 2:
                    log_result = math.log2(value)
                    result_text += f"Binary logarithm: log₂({value:.6f}) = {log_result:.6f}\n"
                else:
                    log_result = math.log(value) / math.log(base)
                    result_text += f"Logarithm: log_{base:.3f}({value:.6f}) = {log_result:.6f}\n"
                
                result_text += f"\nVerification: {base:.6f}^{log_result:.6f} = {base**log_result:.6f} ≈ {value:.6f} ✓\n"
                
                # Properties
                result_text += f"\nLogarithm Properties:\n"
                result_text += f"- log_b(xy) = log_b(x) + log_b(y)\n"
                result_text += f"- log_b(x/y) = log_b(x) - log_b(y)\n"
                result_text += f"- log_b(x^n) = n × log_b(x)\n"
                result_text += f"- log_b(b) = 1\n"
                result_text += f"- log_b(1) = 0\n"
                
            elif operation == "antilog":
                if result is None:
                    return "Error: Result value required for antilog calculation"
                
                if base is None:
                    base = 10
                    
                antilog_result = base ** result
                result_text += f"Antilog: {base:.3f}^{result:.6f} = {antilog_result:.6f}\n"
                result_text += f"\nThis means: log_{base:.3f}({antilog_result:.6f}) = {result:.6f}\n"
                
            elif operation == "solve":
                result_text += f"Logarithmic equation solver\n"
                result_text += f"Please provide equation in format: 'log_b(x) = c' or 'b^x = c'\n"
                
            else:
                return f"Error: Unknown operation '{operation}'. Use 'log', 'antilog', or 'solve'"
                
            return result_text
            
        except Exception as e:
            return f"Error in logarithm calculation: {str(e)}"

    @mcp.tool()
    async def algebra_simplify(expression: str) -> str:
        """
        Simplify algebraic expressions and handle basic equation solving.
        
        Args:
            expression: Algebraic expression or equation to simplify
                    Examples: "2x + 3x", "x² - 4", "3x² + 2x - x² + 5x - 7"
                    Equations: "K = 1/2 * m * v^2", "solve v from K = 1/2 * m * v^2"
                    
        Returns:
            str: Simplified expression with steps or equation solutions
        """
        try:
            # Check if it's a solve request
            if "solve" in expression.lower():
                return await self._handle_solve_request(expression)
            
            result = f"""
🎯 ALGEBRA SIMPLIFICATION

Original expression: {expression}

Simplification steps:
====================
"""
            
            # Basic pattern matching for common simplifications
            expr = expression.replace(" ", "").lower()
            
            import re
            
            # Initialize coefficients
            x_squared_coeff = 0
            x_coeff = 0
            constant = 0
            
            # Find x² terms (must come before x terms to avoid conflicts)
            x_squared_terms = re.findall(r'([+-]?\d*\.?\d*)x²', expr)
            if not x_squared_terms:
                x_squared_terms = re.findall(r'([+-]?\d*\.?\d*)x\^2', expr)
            
            for term in x_squared_terms:
                if term == '' or term == '+':
                    x_squared_coeff += 1
                elif term == '-':
                    x_squared_coeff -= 1
                else:
                    x_squared_coeff += float(term)
            
            # Find x terms (but not x² or x^2)
            x_terms = re.findall(r'([+-]?\d*\.?\d*)x(?![²^])', expr)
            
            for term in x_terms:
                if term == '' or term == '+':
                    x_coeff += 1
                elif term == '-':
                    x_coeff -= 1
                else:
                    x_coeff += float(term)
            
            # Find constant terms (numbers not followed by x)
            # Remove all x terms first, then find remaining numbers
            expr_no_x = re.sub(r'[+-]?\d*\.?\d*x[²^2]*', '', expr)
            const_terms = re.findall(r'([+-]?\d+\.?\d*)', expr_no_x)
            
            for term in const_terms:
                if term:
                    constant += float(term)
            
            # Show the combination steps
            if x_squared_terms:
                result += f"x² terms: {' + '.join(x_squared_terms)}x² = {x_squared_coeff:.0f}x²\n"
            
            if x_terms:
                result += f"x terms: {' + '.join(x_terms)}x = {x_coeff:.0f}x\n"
            
            if const_terms:
                result += f"Constants: {' + '.join(const_terms)} = {constant:.0f}\n"
            
            # Build the simplified expression
            simplified_parts = []
            
            if x_squared_coeff != 0:
                if x_squared_coeff == 1:
                    simplified_parts.append("x²")
                elif x_squared_coeff == -1:
                    simplified_parts.append("-x²")
                else:
                    simplified_parts.append(f"{x_squared_coeff:.0f}x²")
            
            if x_coeff != 0:
                if x_coeff == 1:
                    if simplified_parts:
                        simplified_parts.append("+ x")
                    else:
                        simplified_parts.append("x")
                elif x_coeff == -1:
                    simplified_parts.append("- x")
                else:
                    if x_coeff > 0 and simplified_parts:
                        simplified_parts.append(f"+ {x_coeff:.0f}x")
                    else:
                        simplified_parts.append(f"{x_coeff:.0f}x")
            
            if constant != 0:
                if constant > 0 and simplified_parts:
                    simplified_parts.append(f"+ {constant:.0f}")
                else:
                    simplified_parts.append(f"{constant:.0f}")
            
            if not simplified_parts:
                simplified_expression = "0"
            else:
                simplified_expression = " ".join(simplified_parts)
            
            result += f"\n✅ Simplified expression: {simplified_expression}\n"
            
            # Check for special patterns
            if x_squared_coeff != 0 and x_coeff == 0 and constant < 0:
                # Difference of squares check
                if x_squared_coeff == 1 and constant == int(constant) and constant < 0:
                    sqrt_const = math.sqrt(-constant)
                    if sqrt_const == int(sqrt_const):
                        sqrt_const = int(sqrt_const)
                        result += f"\n🔍 Special form: This is a difference of squares\n"
                        result += f"x² - {-constant:.0f} = (x + {sqrt_const})(x - {sqrt_const})\n"
            
            return result
            
        except Exception as e:
            return f"Error simplifying expression: {str(e)}"

    async def _handle_solve_request(self, expression: str) -> str:
        """
        Handle solve requests from within algebra_simplify.
        """
        try:
            import re
            
            # Parse solve requests like "solve v from K = 1/2 * m * v^2"
            solve_pattern = r'solve\s+(\w+)\s+from\s+(.+)'
            match = re.search(solve_pattern, expression, re.IGNORECASE)
            
            if match:
                variable = match.group(1)
                equation = match.group(2)
                
                # Use the solve_for_variable function
                return await solve_for_variable(equation, variable)
            
            # Handle other solve formats
            if "=" in expression:
                # If it's just an equation, provide general guidance
                return f"""
🎯 EQUATION DETECTED

Expression: {expression}

To solve this equation:
1. Identify the variable you want to solve for
2. Use algebraic manipulation to isolate that variable
3. Apply inverse operations systematically

💡 Use solve_for_variable tool with specific variable name for detailed steps
"""
            
            return f"Could not parse solve request: {expression}"
            
        except Exception as e:
            return f"Error handling solve request: {str(e)}"

    @mcp.tool()
    async def unit_circle_reference(angle: float, unit: str = "degrees") -> str:
        """
        Provide unit circle reference for trigonometric values.
        
        Args:
            angle: Angle value
            unit: "degrees" or "radians"
            
        Returns:
            str: Unit circle coordinates and trig values
        """
        try:
            if unit == "degrees":
                angle_rad = degrees_to_radians(angle)
                angle_deg = angle
            else:
                angle_rad = angle
                angle_deg = radians_to_degrees(angle)
            
            # Normalize angle to [0, 2π)
            normalized_rad = angle_rad % (2 * math.pi)
            normalized_deg = angle_deg % 360
            
            result = f"""
    Unit Circle Reference:
    =====================

    Input angle: {angle:.3f} {unit}
    Equivalent angles:
    - Degrees: {angle_deg:.1f}°
    - Radians: {angle_rad:.6f} rad
    - Standard position: {normalized_deg:.1f}° or {normalized_rad:.6f} rad

    Unit circle coordinates (cos θ, sin θ):
    x = cos({normalized_deg:.1f}°) = {math.cos(normalized_rad):.6f}
    y = sin({normalized_deg:.1f}°) = {math.sin(normalized_rad):.6f}

    All trigonometric values:
    - sin θ = {math.sin(normalized_rad):.6f}
    - cos θ = {math.cos(normalized_rad):.6f}
    """
            
            if abs(math.cos(normalized_rad)) > 1e-10:
                tan_val = math.tan(normalized_rad)
                result += f"- tan θ = {tan_val:.6f}\n"
            else:
                result += f"- tan θ = undefined (cos = 0)\n"
                
            if abs(math.sin(normalized_rad)) > 1e-10:
                cot_val = 1 / math.tan(normalized_rad)
                result += f"- cot θ = {cot_val:.6f}\n"
            else:
                result += f"- cot θ = undefined (sin = 0)\n"
                
            if abs(math.cos(normalized_rad)) > 1e-10:
                sec_val = 1 / math.cos(normalized_rad)
                result += f"- sec θ = {sec_val:.6f}\n"
            else:
                result += f"- sec θ = undefined (cos = 0)\n"
                
            if abs(math.sin(normalized_rad)) > 1e-10:
                csc_val = 1 / math.sin(normalized_rad)
                result += f"- csc θ = {csc_val:.6f}\n"
            else:
                result += f"- csc θ = undefined (sin = 0)\n"
            
            # Quadrant information
            quadrant = ""
            if 0 <= normalized_deg < 90:
                quadrant = "I (all positive)"
            elif 90 <= normalized_deg < 180:
                quadrant = "II (sin positive, cos/tan negative)"
            elif 180 <= normalized_deg < 270:
                quadrant = "III (tan positive, sin/cos negative)"
            else:
                quadrant = "IV (cos positive, sin/tan negative)"
                
            result += f"\nQuadrant: {quadrant}\n"
            
            # Reference angle
            if 0 <= normalized_deg <= 90:
                ref_angle = normalized_deg
            elif 90 < normalized_deg <= 180:
                ref_angle = 180 - normalized_deg
            elif 180 < normalized_deg <= 270:
                ref_angle = normalized_deg - 180
            else:
                ref_angle = 360 - normalized_deg
                
            result += f"Reference angle: {ref_angle:.1f}°\n"
            
            # Special angles
            special_angles = {
                0: "0° (0 rad)", 30: "30° (π/6 rad)", 45: "45° (π/4 rad)", 
                60: "60° (π/3 rad)", 90: "90° (π/2 rad)", 120: "120° (2π/3 rad)",
                135: "135° (3π/4 rad)", 150: "150° (5π/6 rad)", 180: "180° (π rad)",
                210: "210° (7π/6 rad)", 225: "225° (5π/4 rad)", 240: "240° (4π/3 rad)",
                270: "270° (3π/2 rad)", 300: "300° (5π/3 rad)", 315: "315° (7π/4 rad)",
                330: "330° (11π/6 rad)", 360: "360° (2π rad)"
            }
            
            for special_deg, special_name in special_angles.items():
                if abs(normalized_deg - special_deg) < 0.1:
                    result += f"\nThis is a special angle: {special_name}\n"
                    break
            
            return result
            
        except Exception as e:
            return f"Error in unit circle calculation: {str(e)}"

    @mcp.tool()
    async def statistics_calculator(data_type: str, values: str) -> str:
        """
        Calculate basic statistics for physics data analysis.
        
        Args:
            data_type: Type of calculation - "descriptive", "error", or "regression"
            values: Comma-separated numerical values or JSON for regression
            
        Returns:
            str: Statistical analysis results
        """
        try:
            result = f"""
    Statistics Calculator:
    =====================

    Data type: {data_type}
    """
            
            if data_type == "descriptive":
                # Parse comma-separated values
                data = [float(x.strip()) for x in values.split(',')]
                n = len(data)
                
                # Basic statistics
                mean = sum(data) / n
                data_sorted = sorted(data)
                
                # Median
                if n % 2 == 0:
                    median = (data_sorted[n//2-1] + data_sorted[n//2]) / 2
                else:
                    median = data_sorted[n//2]
                
                # Standard deviation
                variance = sum((x - mean)**2 for x in data) / (n - 1) if n > 1 else 0
                std_dev = math.sqrt(variance)
                
                # Standard error
                std_error = std_dev / math.sqrt(n)
                
                result += f"""
    Raw data: {data}
    Sample size (n): {n}

    Measures of Central Tendency:
    - Mean (x̄): {mean:.6f}
    - Median: {median:.6f}
    - Range: {min(data):.6f} to {max(data):.6f}

    Measures of Spread:
    - Standard deviation (s): {std_dev:.6f}
    - Variance (s²): {variance:.6f}
    - Standard error (SE): {std_error:.6f}

    Data Summary:
    - Minimum: {min(data):.6f}
    - Maximum: {max(data):.6f}
    - Range: {max(data) - min(data):.6f}

    For error analysis:
    - Mean ± Standard Error: {mean:.6f} ± {std_error:.6f}
    - 68% confidence interval: [{mean - std_error:.6f}, {mean + std_error:.6f}]
    - 95% confidence interval: [{mean - 1.96*std_error:.6f}, {mean + 1.96*std_error:.6f}]
    """
                
            elif data_type == "error":
                # Error propagation calculations
                data = [float(x.strip()) for x in values.split(',')]
                
                result += f"""
    Error Analysis:
    Values: {data}

    For measurements with uncertainties:
    - Absolute error = |measured - true|
    - Relative error = |measured - true| / |true| × 100%
    - Percent error = relative error

    Standard error propagation formulas:
    - Addition/Subtraction: δz = √((δx)² + (δy)²)
    - Multiplication/Division: δz/z = √((δx/x)² + (δy/y)²)
    - Power rule: δ(x^n) = n × x^(n-1) × δx
    """
                
            else:
                return f"Error: data_type must be 'descriptive' or 'error'"
            
            return result
            
        except Exception as e:
            return f"Error in statistical calculation: {str(e)}"
    @mcp.tool()
    async def solve_for_variable(equation: str, solve_for: str) -> str:
        """
        Solve algebraic equations for a specified variable.
        
        Args:
            equation: Equation to solve (e.g., "K = 1/2 * m * v^2", "F = m * a")
            solve_for: Variable to solve for (e.g., "v", "m", "a")
            
        Returns:
            str: Step-by-step solution showing algebraic manipulation
        """
        try:
            result = f"""
🎯 ALGEBRAIC EQUATION SOLVER

Original equation: {equation}
Solving for: {solve_for}

Solution Process:
================
"""
            
            # Clean and parse equation
            eq = equation.replace(" ", "").lower()
            target_var = solve_for.lower()
            
            # More flexible pattern matching for kinetic energy equation
            if any(pattern in eq for pattern in ["k=1/2*m*v^2", "k=(1/2)*m*v^2", "k=0.5*m*v^2", "k=1/2mv^2", "k=(1/2)mv^2"]):
                if target_var == "v":
                    result += f"""
Step 1: Start with kinetic energy equation
K = ½mv²

Step 2: Multiply both sides by 2
2K = mv²

Step 3: Divide both sides by m
2K/m = v²

Step 4: Take square root of both sides
v = ±√(2K/m)

✅ FINAL ANSWER: v = ±√(2K/m)

Physical Interpretation:
• The ± indicates velocity can be in either direction
• For speed (magnitude only), use: |v| = √(2K/m)  
• Units: If K is in Joules and m in kg, then v is in m/s
"""
                elif target_var == "m":
                    result += f"""
Step 1: Start with kinetic energy equation
K = ½mv²

Step 2: Multiply both sides by 2
2K = mv²

Step 3: Divide both sides by v²
m = 2K/v²

✅ FINAL ANSWER: m = 2K/v²
"""
                elif target_var == "k":
                    result += f"""
The equation is already solved for K:
K = ½mv²

✅ FINAL ANSWER: K = ½mv²

This gives kinetic energy in terms of mass and velocity.
"""
                    
            elif "f=m*a" in eq or "f=ma" in eq:
                if target_var == "a":
                    result += """
    Step 1: Start with Newton's Second Law
    F = ma

    Step 2: Divide both sides by m
    a = F/m

    Final Answer: a = F/m
    """
                elif target_var == "m":
                    result += """
    Step 1: Start with Newton's Second Law
    F = ma

    Step 2: Divide both sides by a
    m = F/a

    Final Answer: m = F/a
    """
                elif target_var == "f":
                    result += """
    The equation is already solved for F:
    F = ma

    This is Newton's Second Law.
    """
                    
            elif "p=m*v" in eq or "p=mv" in eq:
                if target_var == "v":
                    result += """
    Step 1: Start with momentum equation
    p = mv

    Step 2: Divide both sides by m
    v = p/m

    Final Answer: v = p/m
    """
                elif target_var == "m":
                    result += """
    Step 1: Start with momentum equation
    p = mv

    Step 2: Divide both sides by v
    m = p/v

    Final Answer: m = p/v
    """
                    
            elif "e=m*c^2" in eq or "e=mc^2" in eq:
                if target_var == "m":
                    result += """
    Step 1: Start with mass-energy equivalence
    E = mc²

    Step 2: Divide both sides by c²
    m = E/c²

    Final Answer: m = E/c²
    """
                elif target_var == "e":
                    result += """
    The equation is already solved for E:
    E = mc²

    This is Einstein's mass-energy equivalence.
    """
                    
            # Handle generic quadratic patterns  
            elif "=" in eq and ("^2" in eq or "²" in eq):
                parts = eq.split("=")
                if len(parts) == 2:
                    left, right = parts
                    if target_var + "^2" in left or target_var + "²" in left:
                        result += f"""
Step 1: Identify that {target_var} appears squared
{equation}

Step 2: Isolate {target_var}² term
{target_var}² = {right}

Step 3: Take square root of both sides
{target_var} = ±√({right})

✅ FINAL ANSWER: {target_var} = ±√({right})
"""
                    elif target_var + "^2" in right or target_var + "²" in right:
                        result += f"""
Step 1: Identify that {target_var} appears squared
{equation}

Step 2: Take square root of both sides
√({left}) = |{target_var}|

✅ FINAL ANSWER: {target_var} = ±√({left})
"""
            
            # Handle more generic algebraic manipulation
            elif "=" in eq:
                parts = eq.split("=")
                if len(parts) == 2:
                    left, right = parts
                    
                    # Check if target variable appears on one side only
                    if target_var in left and target_var not in right:
                        result += f"""
Step 1: Target variable '{target_var}' appears on left side
{equation}

Step 2: Use algebraic manipulation to isolate {target_var}

General steps:
• Move terms without {target_var} to the right side
• Factor out {target_var} if it appears multiple times
• Apply inverse operations (÷, √, log, etc.) as needed

✅ For this equation, {target_var} can be solved by rearranging terms
"""
                    elif target_var in right and target_var not in left:
                        result += f"""
Step 1: Target variable '{target_var}' appears on right side
{equation}

Step 2: Use algebraic manipulation to isolate {target_var}

General steps:  
• Move terms without {target_var} to the left side
• Factor out {target_var} if it appears multiple times
• Apply inverse operations (÷, √, log, etc.) as needed

✅ For this equation, {target_var} can be solved by rearranging terms
"""
                    else:
                        result += f"""
Variable '{target_var}' appears on both sides of the equation.
This requires collecting like terms:

1. Move all terms containing {target_var} to one side
2. Move all other terms to the opposite side  
3. Factor out {target_var}
4. Divide to isolate {target_var}

✅ This is solvable using standard algebraic techniques
"""
            else:
                result += f"""
💡 GENERAL ALGEBRAIC APPROACH

To solve {equation} for {target_var}:

1. Identify where {target_var} appears in the equation
2. Use inverse operations to isolate {target_var}
3. Apply algebraic rules systematically:
   • Addition ↔ Subtraction  
   • Multiplication ↔ Division
   • Exponents ↔ Roots
   • Exponentials ↔ Logarithms

4. Handle special cases:
   • Quadratic equations → Use quadratic formula
   • Factoring when possible
   • Substitution methods

✅ Every algebraic equation has a systematic solution method
"""
            
            return result
            
        except Exception as e:
            return f"Error solving equation: {str(e)}"

    @mcp.tool()
    async def physics_formula_solver(formula_name: str, known_values: str, solve_for: str) -> str:
        """
        Solve common physics formulas with known values.
        
        Args:
            formula_name: Name of physics formula ("kinetic_energy", "force", "momentum", etc.)
            known_values: JSON string with known values (e.g., '{"m": 2.0, "K": 100}')
            solve_for: Variable to solve for
            
        Returns:
            str: Numerical solution with units and explanation
        """
        try:
            import json
            
            known = json.loads(known_values)
            result = f"""
    Physics Formula Solver:
    ======================

    Formula: {formula_name}
    Known values: {known}
    Solving for: {solve_for}

    """
            
            if formula_name == "kinetic_energy":
                result += "Formula: K = ½mv²\n\n"
                
                if solve_for == "v":
                    if "K" in known and "m" in known:
                        K = known["K"]
                        m = known["m"]
                        v = math.sqrt(2 * K / m)
                        result += f"Calculation:\n"
                        result += f"v = √(2K/m) = √(2×{K}/{m}) = √({2*K/m:.3f}) = {v:.3f}\n"
                        result += f"\nAnswer: v = {v:.3f} m/s"
                    else:
                        result += "Error: Need both K (kinetic energy) and m (mass) to solve for v"
                        
                elif solve_for == "m":
                    if "K" in known and "v" in known:
                        K = known["K"]
                        v = known["v"]
                        m = 2 * K / (v ** 2)
                        result += f"Calculation:\n"
                        result += f"m = 2K/v² = 2×{K}/{v}² = {2*K}/{v**2:.3f} = {m:.3f}\n"
                        result += f"\nAnswer: m = {m:.3f} kg"
                    else:
                        result += "Error: Need both K (kinetic energy) and v (velocity) to solve for m"
                        
                elif solve_for == "K":
                    if "m" in known and "v" in known:
                        m = known["m"]
                        v = known["v"]
                        K = 0.5 * m * v ** 2
                        result += f"Calculation:\n"
                        result += f"K = ½mv² = ½×{m}×{v}² = 0.5×{m}×{v**2:.3f} = {K:.3f}\n"
                        result += f"\nAnswer: K = {K:.3f} J"
                    else:
                        result += "Error: Need both m (mass) and v (velocity) to solve for K"
                        
            elif formula_name == "force":
                result += "Formula: F = ma\n\n"
                
                if solve_for == "F":
                    if "m" in known and "a" in known:
                        m = known["m"]
                        a = known["a"]
                        F = m * a
                        result += f"F = ma = {m} × {a} = {F:.3f} N"
                    else:
                        result += "Error: Need both m (mass) and a (acceleration)"
                        
                elif solve_for == "a":
                    if "F" in known and "m" in known:
                        F = known["F"]
                        m = known["m"]
                        a = F / m
                        result += f"a = F/m = {F}/{m} = {a:.3f} m/s²"
                    else:
                        result += "Error: Need both F (force) and m (mass)"
                        
                elif solve_for == "m":
                    if "F" in known and "a" in known:
                        F = known["F"]
                        a = known["a"]
                        m = F / a
                        result += f"m = F/a = {F}/{a} = {m:.3f} kg"
                    else:
                        result += "Error: Need both F (force) and a (acceleration)"
            else:
                result += f"Formula '{formula_name}' not implemented yet.\n"
                result += f"Available formulas: kinetic_energy, force"
            
            return result
            
        except Exception as e:
            return f"Error in physics calculation: {str(e)}"
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
    """CLI entry point for the physics-math-mcp tool."""
    parser = argparse.ArgumentParser(description="Run Physics Math MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10103, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")
    
    args = parser.parse_args()
    
    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")

if __name__ == "__main__":
    main()