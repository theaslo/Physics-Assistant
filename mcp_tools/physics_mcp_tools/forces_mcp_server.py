# type: ignore
import json
from typing import List, Dict
import math
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger
from physics_mcp_tools.forces_utils import (
    calculate_force_components,
    calculate_resultant_force,
    calculate_spring_force,
    calculate_friction_force,
    calculate_gravitational_force,
    degrees_to_radians,
    radians_to_degrees,
)
import uvicorn
import argparse
NAME = "forces_mcp_server"

logger = get_logger(__name__)

def serve(host, port, transport):  # noqa: PLR0915
    """Initializes and runs the Agent Cards MCP server.

    Args:
        host: The hostname or IP address to bind the server to.
        port: The port number to bind the server to.
        transport: The transport mechanism for the MCP server (e.g., 'stdio', 'sse').
    """
    logger.info('Starting Forces MCP Server')
    mcp = FastMCP(NAME, stateless_http=False)



    @mcp.tool()
    async def add_forces_1d(forces: str) -> str:
        """Add forces in one dimension (along a line).

        Args:
            forces: Comma-separated list of forces in Newtons.
                    Use positive values for forces in positive direction,
                    negative values for forces in negative direction.
                    Example: "10, -5, 15, -3"

        Returns:
            str: Net force calculation and explanation
        """
        try:
            force_list = [float(f.strip()) for f in forces.split(',')]
            net_force = sum(force_list)

            result = f"""
    1D Force Addition:
    ================
    Individual Forces: {', '.join([f'{f:.1f} N' for f in force_list])}
    Net Force: {net_force:.2f} N

    Calculation:
    Net Force = {' + '.join([f'({f:.1f})' for f in force_list])} = {net_force:.2f} N

    Result:"""

            if net_force > 0:
                result += f"\n- The net force acts in the positive direction with magnitude {abs(net_force):.2f} N"
            elif net_force < 0:
                result += f"\n- The net force acts in the negative direction with magnitude {abs(net_force):.2f} N"
            else:
                result += "\n- The forces are balanced (net force = 0 N) - object is in equilibrium"

            return result

        except ValueError:
            return "Error: Please provide valid numbers separated by commas (e.g., '10, -5, 15')"

    @mcp.tool()
    async def add_forces_2d(forces_data: List[Dict]) -> str:#(forces_data: str) -> str:
        """
        Add multiple 2D forces to find the resultant.

        Args:
            forces_data: List of objects, each with 'magnitude' and 'angle' fields.
                        Example: [{"magnitude": 10, "angle": 30}, {"magnitude": 15, "angle": 120}]
                        Angles are in degrees, measured counterclockwise from positive x-axis.
        Returns:
            str: Complete 2D force addition analysis with resultant
        """
        print(f"[TOOL INVOKED] add_forces_2d called with: {forces_data}")
        # """Add multiple 2D forces to find the resultant.

        # Args:
        #     forces_data: JSON string with list of forces containing magnitude and angle.
        #                 Example: '[{"magnitude": 10, "angle": 30}, {"magnitude": 15, "angle": 120}]'
        #                 Angles are in degrees, measured counterclockwise from positive x-axis.

        # Returns:
        #     str: Complete 2D force addition analysis with resultant
        # """

        #result += f"🔧 REAL MCP TOOL VERIFICATION: {forces_data}\n\n"
        try:
            if isinstance(forces_data, str):
                forces = json.loads(forces_data)
            else:
                forces = forces_data

            total_fx, total_fy = 0, 0
            force_details = []

            result = "2D Force Addition:\n================\n\nIndividual Forces:\n"
            result += f"🔧 REAL MCP TOOL VERIFICATION: {forces_data}\n\n"

            for i, force in enumerate(forces, 1):
                magnitude = float(force['magnitude'])
                angle = float(force['angle'])
                fx, fy = calculate_force_components(magnitude, angle)
                total_fx += fx
                total_fy += fy

                force_details.append({
                    'num': i,
                    'magnitude': magnitude,
                    'angle': angle,
                    'fx': fx,
                    'fy': fy
                })

                result += f"Force {i}: {magnitude:.1f} N at {angle:.1f}°\n"
                result += f"  → Fx{i} = {magnitude:.1f} × cos({angle:.1f}°) = {fx:.2f} N\n"
                result += f"  → Fy{i} = {magnitude:.1f} × sin({angle:.1f}°) = {fy:.2f} N\n\n"

            net_magnitude, net_angle = calculate_resultant_force(total_fx, total_fy)

            result += f"Net Force Components:\n"
            # Create component strings to avoid nested f-string issues
            fx_components = [f"{fd['fx']:.2f}" for fd in force_details]
            fy_components = [f"{fd['fy']:.2f}" for fd in force_details]

            result += f"Total Fx = {' + '.join(fx_components)} = {total_fx:.2f} N\n"
            result += f"Total Fy = {' + '.join(fy_components)} = {total_fy:.2f} N\n\n"

            result += f"Resultant Force:\n"
            result += f"🔧 REAL MCP TOOL VERIFICATION: Tool called with {len(forces)} forces\n"
            result += f"Magnitude = √(Fx² + Fy²) = √({total_fx:.2f}² + {total_fy:.2f}²) = {net_magnitude:.2f} N\n"
            result += f"Direction = arctan(Fy/Fx) = arctan({total_fy:.2f}/{total_fx:.2f}) = {net_angle:.1f}°\n\n"

            if abs(net_magnitude) < 0.01:
                result += "The forces are balanced (net force ≈ 0) - system is in equilibrium!"
            else:
                result += f"The resultant force is {net_magnitude:.2f} N at {net_angle:.1f}° from the positive x-axis."

            return result.strip()

        except Exception as e:
            return f"Error: {str(e)}\nExpected format: '[{{\"magnitude\": 10, \"angle\": 30}}]'"

    @mcp.tool()
    async def resolve_force_components(magnitude: float, angle_degrees: float) -> str:
        """Break down a force into its x and y components.

        Args:
            magnitude: Force magnitude in Newtons
            angle_degrees: Angle in degrees (counterclockwise from positive x-axis)

        Returns:
            str: Force components calculation and explanation
        """
        try:
            force_x, force_y = calculate_force_components(magnitude, angle_degrees)

            result = f"""
    Force Component Resolution:
    ==========================

    Given Force:
    - Magnitude: {magnitude:.2f} N
    - Angle: {angle_degrees:.1f}° (from positive x-axis)

    Component Calculations:
    Fx = F × cos(θ) = {magnitude:.2f} × cos({angle_degrees:.1f}°) = {force_x:.2f} N
    Fy = F × sin(θ) = {magnitude:.2f} × sin({angle_degrees:.1f}°) = {force_y:.2f} N

    Results:
    - X-component (horizontal): {force_x:.2f} N {'→' if force_x >= 0 else '←'}
    - Y-component (vertical): {force_y:.2f} N {'↑' if force_y >= 0 else '↓'}

    Verification:
    Magnitude = √(Fx² + Fy²) = √({force_x:.2f}² + {force_y:.2f}²) = {magnitude:.2f} N ✓
    """
            return result

        except Exception as e:
            return f"Error in calculation: {str(e)}"

    @mcp.tool()
    async def find_resultant_force(force_components: str) -> str:
        """Find the resultant force from x and y components.

        Args:
            force_components: JSON string with x and y components.
                            Example: '{"fx": 8.66, "fy": 5.0}'

        Returns:
            str: Resultant force magnitude and direction
        """
        try:
            components = json.loads(force_components)
            fx = float(components['fx'])
            fy = float(components['fy'])

            magnitude, angle = calculate_resultant_force(fx, fy)

            result = f"""
    Resultant Force Calculation:
    ===========================

    Given Components:
    - Fx = {fx:.2f} N
    - Fy = {fy:.2f} N

    Magnitude Calculation:
    |F| = √(Fx² + Fy²) = √({fx:.2f}² + {fy:.2f}²) = √{fx**2 + fy**2:.2f} = {magnitude:.2f} N

    Direction Calculation:
    θ = arctan(Fy/Fx) = arctan({fy:.2f}/{fx:.2f}) = {angle:.1f}°

    Resultant Force:
    - Magnitude: {magnitude:.2f} N
    - Direction: {angle:.1f}° (counterclockwise from +x axis)

    Physical Interpretation:
    The resultant force points in the direction {angle:.1f}° with magnitude {magnitude:.2f} N.
    """

            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return 'Error: Please provide valid JSON format like {"fx": 8.66, "fy": 5.0}'

    @mcp.tool()
    async def create_free_body_diagram(object_name: str, forces_data: str) -> str:
        """Generate a text-based free body diagram description.

        Args:
            object_name: Name of the object (e.g., "box", "ball", "car")
            forces_data: JSON string with list of forces acting on the object.
                        Example: '[{"name": "Weight", "magnitude": 50, "angle": 270},
                                {"name": "Normal", "magnitude": 50, "angle": 90},
                                {"name": "Applied", "magnitude": 20, "angle": 0}]'

        Returns:
            str: Text description of the free body diagram with force analysis
        """
        try:
            forces = json.loads(forces_data)

            diagram_text = f"""
    FREE BODY DIAGRAM: {object_name.upper()}
    {'='*40}

    Object: {object_name} (represented as a point mass)

    Forces Acting on {object_name}:
    """

            total_fx = 0
            total_fy = 0

            for force in forces:
                name = force['name']
                magnitude = force['magnitude']
                angle = force['angle']

                fx, fy = calculate_force_components(magnitude, angle)
                total_fx += fx
                total_fy += fy

                # Describe force direction
                if angle == 0:
                    direction = "→ (rightward)"
                elif angle == 90:
                    direction = "↑ (upward)"
                elif angle == 180:
                    direction = "← (leftward)"
                elif angle == 270:
                    direction = "↓ (downward)"
                else:
                    direction = f"at {angle:.1f}° from +x axis"

                diagram_text += f"• {name}: {magnitude:.1f} N {direction}\n"
                diagram_text += f"  Components: Fx = {fx:.2f} N, Fy = {fy:.2f} N\n\n"

            net_magnitude, net_angle = calculate_resultant_force(total_fx, total_fy)

            diagram_text += f"""Force Analysis:
    ===============
    Net Force Components:
    - ΣFx = {total_fx:.2f} N
    - ΣFy = {total_fy:.2f} N
    - Net Force Magnitude = {net_magnitude:.2f} N
    - Net Force Direction = {net_angle:.1f}°

    Equilibrium Status:
    """

            if abs(net_magnitude) < 0.01:
                diagram_text += "✓ Object is in EQUILIBRIUM (net force = 0)\n"
                diagram_text += "  → Object will remain at rest or continue moving at constant velocity"
            else:
                diagram_text += f"✗ Object is NOT in equilibrium (net force = {net_magnitude:.2f} N)\n"
                diagram_text += f"  → Object will accelerate in the direction {net_angle:.1f}°"

            return diagram_text

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return 'Error: Please provide valid JSON format like [{"name": "Weight", "magnitude": 50, "angle": 270}]'

    @mcp.tool()
    async def check_equilibrium(forces_data: str) -> str:
        """Check if forces are in equilibrium and suggest balancing force if needed.

        Args:
            forces_data: JSON string with list of forces.
                        Example: '[{"magnitude": 10, "angle": 0}, {"magnitude": 15, "angle": 120}]'

        Returns:
            str: Equilibrium analysis and balancing force if needed
        """
        try:
            forces = json.loads(forces_data)

            total_fx = 0
            total_fy = 0

            result = "Equilibrium Analysis:\n====================\n\nForces Applied:\n"

            for i, force in enumerate(forces, 1):
                magnitude = force['magnitude']
                angle = force['angle']
                fx, fy = calculate_force_components(magnitude, angle)
                total_fx += fx
                total_fy += fy
                result += f"- Force {i}: {magnitude:.1f} N at {angle:.1f}° → ({fx:.2f}, {fy:.2f}) N\n"

            net_magnitude, net_angle = calculate_resultant_force(total_fx, total_fy)

            result += f"\nNet Force Components:\n"
            result += f"ΣFx = {total_fx:.2f} N\n"
            result += f"ΣFy = {total_fy:.2f} N\n"
            result += f"Net Force = {net_magnitude:.2f} N at {net_angle:.1f}°\n\n"

            if abs(net_magnitude) < 0.01:
                result += "✓ EQUILIBRIUM ACHIEVED!\n"
                result += "The forces are perfectly balanced. The object will:\n"
                result += "- Remain at rest if initially at rest\n"
                result += "- Continue moving at constant velocity if already in motion"
            else:
                # Calculate balancing force
                balance_fx = -total_fx
                balance_fy = -total_fy
                balance_magnitude, balance_angle = calculate_resultant_force(balance_fx, balance_fy)

                result += f"✗ NOT IN EQUILIBRIUM\n\n"
                result += f"To achieve equilibrium, add a balancing force:\n"
                result += f"- Magnitude: {balance_magnitude:.2f} N\n"
                result += f"- Direction: {balance_angle:.1f}° (counterclockwise from +x axis)\n"
                result += f"- Components: Fx = {balance_fx:.2f} N, Fy = {balance_fy:.2f} N\n\n"
                result += f"This balancing force will exactly cancel the net force and restore equilibrium."

            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return 'Error: Please provide valid JSON format like [{"magnitude": 10, "angle": 30}]'

    @mcp.tool()
    async def calculate_spring_force_tool(spring_constant: float, displacement: float) -> str:
        """Calculate spring force using Hooke's Law.

        Args:
            spring_constant: Spring constant k in N/m
            displacement: Displacement from equilibrium position in meters (positive = stretched, negative = compressed)

        Returns:
            str: Spring force calculation and explanation
        """
        try:
            spring_force = calculate_spring_force(spring_constant, displacement)

            result = f"""
    Spring Force Calculation (Hooke's Law):
    ======================================

    Given:
    - Spring constant (k): {spring_constant:.2f} N/m
    - Displacement (x): {displacement:.2f} m

    Hooke's Law: F = -kx
    F = -({spring_constant:.2f}) × ({displacement:.2f})
    F = {spring_force:.2f} N

    Physical Interpretation:
    """

            if displacement > 0:
                result += f"- Spring is STRETCHED by {displacement:.2f} m\n"
                result += f"- Spring force acts OPPOSITE to displacement (restoring force)\n"
                result += f"- Force magnitude: {abs(spring_force):.2f} N in the negative direction"
            elif displacement < 0:
                result += f"- Spring is COMPRESSED by {abs(displacement):.2f} m\n"
                result += f"- Spring force acts OPPOSITE to displacement (restoring force)\n"
                result += f"- Force magnitude: {abs(spring_force):.2f} N in the positive direction"
            else:
                result += f"- Spring is at equilibrium position\n"
                result += f"- No spring force (F = 0 N)"

            return result

        except Exception as e:
            return f"Error in calculation: {str(e)}"

    @mcp.tool()
    async def calculate_friction_force_tool(coefficient: float, normal_force: float, force_type: str = "kinetic") -> str:
        """Calculate friction force.

        Args:
            coefficient: Coefficient of friction (μ)
            normal_force: Normal force in Newtons
            force_type: Type of friction - "kinetic" or "static"

        Returns:
            str: Friction force calculation and explanation
        """
        try:
            is_kinetic = force_type.lower() == "kinetic"
            friction_force = calculate_friction_force(coefficient, normal_force, is_kinetic)

            result = f"""
    Friction Force Calculation:
    ==========================

    Given:
    - Coefficient of {force_type} friction (μ): {coefficient:.3f}
    - Normal force (N): {normal_force:.2f} N

    Friction Formula: f = μN
    f = {coefficient:.3f} × {normal_force:.2f}
    f = {friction_force:.2f} N

    Physical Interpretation:
    - Type: {force_type.title()} friction
    - Magnitude: {friction_force:.2f} N
    - Direction: Opposite to the direction of motion (or intended motion)
    """

            if force_type.lower() == "static":
                result += f"\nNote: This is the MAXIMUM static friction force.\n"
                result += f"Actual static friction ≤ {friction_force:.2f} N (depends on applied force)"
            else:
                result += f"\nNote: Kinetic friction acts when object is sliding.\n"
                result += f"This force opposes the relative motion between surfaces."

            return result

        except Exception as e:
            return f"Error in calculation: {str(e)}"

    @mcp.tool()
    async def calculate_weight_force(mass: float, gravity: float = 9.81) -> str:
        """Calculate gravitational/weight force.

        Args:
            mass: Mass of the object in kilograms
            gravity: Acceleration due to gravity in m/s² (default: 9.81 m/s² for Earth)

        Returns:
            str: Weight force calculation and explanation
        """
        try:
            weight = calculate_gravitational_force(mass, gravity)

            result = f"""
    Weight Force Calculation:
    ========================

    Given:
    - Mass (m): {mass:.2f} kg
    - Gravitational acceleration (g): {gravity:.2f} m/s²

    Weight Formula: W = mg
    W = {mass:.2f} × {gravity:.2f}
    W = {weight:.2f} N

    Physical Interpretation:
    - Weight force magnitude: {weight:.2f} N
    - Direction: Vertically downward (toward center of Earth)
    - This is the gravitational force acting on the object
    """

            if gravity == 9.81:
                result += f"\nNote: Using standard Earth gravity (g = 9.81 m/s²)"
            else:
                result += f"\nNote: Using custom gravity value (g = {gravity:.2f} m/s²)"

            return result

        except Exception as e:
            return f"Error in calculation: {str(e)}"

    @mcp.tool()
    async def analyze_forces_on_incline(mass: float, angle_degrees: float, coefficient_friction: float = 0.0, gravity: float = 9.81) -> str:
        """Analyze forces acting on an object on an inclined plane.

        Args:
            mass: Mass of the object in kg
            angle_degrees: Angle of the incline in degrees (from horizontal)
            coefficient_friction: Coefficient of kinetic friction (default: 0.0 for frictionless)
            gravity: Acceleration due to gravity in m/s² (default: 9.81)

        Returns:
            str: Complete force analysis on inclined plane
        """
        try:
            weight = mass * gravity
            angle_rad = degrees_to_radians(angle_degrees)

            # Weight components
            weight_parallel = weight * math.sin(angle_rad)  # Down the incline
            weight_perpendicular = weight * math.cos(angle_rad)  # Into the incline

            # Normal force (perpendicular to incline, outward)
            normal_force = weight_perpendicular

            # Friction force (if any)
            friction_force = coefficient_friction * normal_force if coefficient_friction > 0 else 0

            result = f"""
    Forces on Inclined Plane Analysis:
    =================================

    Given:
    - Mass: {mass:.2f} kg
    - Incline angle: {angle_degrees:.1f}° (from horizontal)
    - Coefficient of friction: {coefficient_friction:.3f}
    - Gravity: {gravity:.2f} m/s²

    Weight Force (W = mg):
    W = {mass:.2f} × {gravity:.2f} = {weight:.2f} N (vertically downward)

    Weight Components:
    - Parallel to incline (down): W∥ = mg sin(θ) = {weight:.2f} × sin({angle_degrees:.1f}°) = {weight_parallel:.2f} N
    - Perpendicular to incline (into): W⊥ = mg cos(θ) = {weight:.2f} × cos({angle_degrees:.1f}°) = {weight_perpendicular:.2f} N

    Normal Force:
    N = W⊥ = {normal_force:.2f} N (perpendicular to incline, outward)
    """

            if coefficient_friction > 0:
                result += f"""
    Friction Force:
    f = μN = {coefficient_friction:.3f} × {normal_force:.2f} = {friction_force:.2f} N (up the incline)

    Net Force Analysis:
    Net force down incline = W∥ - f = {weight_parallel:.2f} - {friction_force:.2f} = {weight_parallel - friction_force:.2f} N
    """

                if weight_parallel > friction_force:
                    result += f"\nResult: Object will slide DOWN the incline (net force = {weight_parallel - friction_force:.2f} N)"
                elif weight_parallel < friction_force:
                    result += f"\nResult: Friction prevents sliding (friction > gravitational component)"
                else:
                    result += f"\nResult: Object is in equilibrium (forces balanced)"
            else:
                result += f"""
    No Friction (frictionless incline)

    Net Force Analysis:
    Net force down incline = W∥ = {weight_parallel:.2f} N

    Result: Object will slide down the incline with net force {weight_parallel:.2f} N
    """

            return result

        except Exception as e:
            return f"Error in calculation: {str(e)}"

    @mcp.tool()
    async def force_vector_operations(operation: str, vector_data: str) -> str:
        """Perform vector operations on forces (addition, subtraction, dot product, cross product magnitude).

        Args:
            operation: Type of operation - "add", "subtract", "dot_product", "cross_product"
            vector_data: JSON string with vector information.
                        For add/subtract: '{"vectors": [{"fx": 10, "fy": 5}, {"fx": -3, "fy": 8}]}'
                        For dot product: '{"vector1": {"fx": 4, "fy": 3}, "vector2": {"fx": 2, "fy": 1}}'
                        For cross product: '{"vector1": {"fx": 4, "fy": 3}, "vector2": {"fx": 2, "fy": 1}}'

        Returns:
            str: Vector operation results and explanation
        """
        try:
            data = json.loads(vector_data)

            if operation == "add":
                vectors = data['vectors']
                total_fx = sum(v['fx'] for v in vectors)
                total_fy = sum(v['fy'] for v in vectors)

                result = f"""
    Vector Addition:
    ===============

    Vectors:
    """
                for i, v in enumerate(vectors, 1):
                    result += f"Vector {i}: ({v['fx']:.2f}, {v['fy']:.2f}) N\n"

                result += f"""
    Sum:
    Fx_total = {' + '.join([f"{v['fx']:.2f}" for v in vectors])} = {total_fx:.2f} N
    Fy_total = {' + '.join([f"{v['fy']:.2f}" for v in vectors])} = {total_fy:.2f} N

    Resultant Vector: ({total_fx:.2f}, {total_fy:.2f}) N
    Magnitude: {math.sqrt(total_fx**2 + total_fy**2):.2f} N
    Direction: {radians_to_degrees(math.atan2(total_fy, total_fx)):.1f}°
    """

            elif operation == "subtract":
                vectors = data['vectors']
                if len(vectors) != 2:
                    return "Error: Subtraction requires exactly 2 vectors"

                result_fx = vectors[0]['fx'] - vectors[1]['fx']
                result_fy = vectors[0]['fy'] - vectors[1]['fy']

                result = f"""
    Vector Subtraction:
    ==================

    Vector 1: ({vectors[0]['fx']:.2f}, {vectors[0]['fy']:.2f}) N
    Vector 2: ({vectors[1]['fx']:.2f}, {vectors[1]['fy']:.2f}) N

    Subtraction (Vector 1 - Vector 2):
    Fx = {vectors[0]['fx']:.2f} - {vectors[1]['fx']:.2f} = {result_fx:.2f} N
    Fy = {vectors[0]['fy']:.2f} - {vectors[1]['fy']:.2f} = {result_fy:.2f} N

    Result Vector: ({result_fx:.2f}, {result_fy:.2f}) N
    Magnitude: {math.sqrt(result_fx**2 + result_fy**2):.2f} N
    Direction: {radians_to_degrees(math.atan2(result_fy, result_fx)):.1f}°
    """

            elif operation == "dot_product":
                v1 = data['vector1']
                v2 = data['vector2']

                dot_product = v1['fx'] * v2['fx'] + v1['fy'] * v2['fy']

                # Calculate magnitudes
                mag1 = math.sqrt(v1['fx']**2 + v1['fy']**2)
                mag2 = math.sqrt(v2['fx']**2 + v2['fy']**2)

                # Calculate angle between vectors
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot_product / (mag1 * mag2)
                    # Clamp to avoid floating point errors
                    cos_angle = max(-1, min(1, cos_angle))
                    angle_rad = math.acos(cos_angle)
                    angle_deg = radians_to_degrees(angle_rad)
                else:
                    angle_deg = 0

                result = f"""
    Vector Dot Product:
    ==================

    Vector 1: ({v1['fx']:.2f}, {v1['fy']:.2f}) N, |v1| = {mag1:.2f} N
    Vector 2: ({v2['fx']:.2f}, {v2['fy']:.2f}) N, |v2| = {mag2:.2f} N

    Dot Product Calculation:
    v1 · v2 = ({v1['fx']:.2f})({v2['fx']:.2f}) + ({v1['fy']:.2f})({v2['fy']:.2f})
    v1 · v2 = {v1['fx'] * v2['fx']:.2f} + {v1['fy'] * v2['fy']:.2f} = {dot_product:.2f}

    Angle Between Vectors:
    cos(θ) = (v1 · v2) / (|v1| × |v2|) = {dot_product:.2f} / ({mag1:.2f} × {mag2:.2f}) = {cos_angle:.3f}
    θ = {angle_deg:.1f}°

    Physical Interpretation:
    - Dot product: {dot_product:.2f} (scalar quantity)
    - Angle between forces: {angle_deg:.1f}°
    """

            elif operation == "cross_product":
                v1 = data['vector1']
                v2 = data['vector2']

                # For 2D vectors, cross product gives magnitude of z-component
                cross_product_z = v1['fx'] * v2['fy'] - v1['fy'] * v2['fx']

                result = f"""
    Vector Cross Product (2D):
    =========================

    Vector 1: ({v1['fx']:.2f}, {v1['fy']:.2f}) N
    Vector 2: ({v2['fx']:.2f}, {v2['fy']:.2f}) N

    Cross Product Calculation (z-component):
    v1 × v2 = ({v1['fx']:.2f})({v2['fy']:.2f}) - ({v1['fy']:.2f})({v2['fx']:.2f})
    v1 × v2 = {v1['fx'] * v2['fy']:.2f} - {v1['fy'] * v2['fx']:.2f} = {cross_product_z:.2f}

    Result:
    - Cross product magnitude: {abs(cross_product_z):.2f}
    - Direction: {'Into page (⊗)' if cross_product_z < 0 else 'Out of page (⊙)'}

    Physical Interpretation:
    The cross product represents the magnitude and direction of torque if one force
    were applied at a position vector equal to the other force vector.
    """
            else:
                return "Error: Operation must be 'add', 'subtract', 'dot_product', or 'cross_product'"

            return result

        except Exception as e:
            return f"Error: {str(e)}\nPlease check your input format."

    @mcp.tool()
    async def analyze_tension_forces(masses: str, angles: str = "0", gravity: float = 9.81) -> str:
        """Analyze tension forces in rope/string systems.

        Args:
            masses: Comma-separated list of masses in kg (e.g., "5, 10, 3")
            angles: Comma-separated list of angles in degrees for each mass (default: "0" for all hanging vertically)
            gravity: Acceleration due to gravity in m/s² (default: 9.81)

        Returns:
            str: Tension force analysis for the system
        """
        try:
            mass_list = [float(m.strip()) for m in masses.split(',')]

            # Parse angles - if only one angle given, use it for first mass, rest are vertical
            angle_str_list = [a.strip() for a in angles.split(',')]
            angle_list = []
            for i in range(len(mass_list)):
                if i < len(angle_str_list):
                    angle_list.append(float(angle_str_list[i]))
                else:
                    angle_list.append(0.0)  # Default to vertical

            result = f"""
    Tension Force Analysis:
    ======================

    System Configuration:
    - Number of masses: {len(mass_list)}
    - Gravity: {gravity:.2f} m/s²

    Masses and Angles:
    """

            total_weight = 0
            for i, (mass, angle) in enumerate(zip(mass_list, angle_list), 1):
                weight = mass * gravity
                total_weight += weight
                result += f"- Mass {i}: {mass:.2f} kg, Weight: {weight:.2f} N, Angle: {angle:.1f}°\n"

            result += f"\nForce Analysis:\n"

            if len(mass_list) == 1:
                # Single hanging mass
                mass = mass_list[0]
                angle = angle_list[0]
                weight = mass * gravity

                if angle == 0:
                    # Simple vertical hanging
                    result += f"Single mass hanging vertically:\n"
                    result += f"Tension = Weight = {weight:.2f} N\n"
                else:
                    # Mass at an angle
                    angle_rad = degrees_to_radians(angle)
                    tension_vertical = weight / math.cos(angle_rad)
                    tension_horizontal = weight * math.tan(angle_rad)

                    result += f"Mass at {angle:.1f}° from vertical:\n"
                    result += f"Vertical tension component: {weight:.2f} N\n"
                    result += f"Horizontal tension component: {tension_horizontal:.2f} N\n"
                    result += f"Total tension in rope: {tension_vertical:.2f} N\n"

            elif len(mass_list) == 2:
                # Two masses - could be pulley system
                m1, m2 = mass_list
                w1, w2 = m1 * gravity, m2 * gravity

                result += f"Two-mass system analysis:\n"
                result += f"Weight 1: {w1:.2f} N, Weight 2: {w2:.2f} N\n"

                if angle_list[0] == 0 and angle_list[1] == 0:
                    # Simple Atwood machine
                    if w1 != w2:
                        tension = (2 * m1 * m2 * gravity) / (m1 + m2)
                        acceleration = abs(w1 - w2) / (m1 + m2)
                        result += f"\nAtwood Machine Configuration:\n"
                        result += f"Tension in rope: {tension:.2f} N\n"
                        result += f"System acceleration: {acceleration:.2f} m/s²\n"
                        result += f"Direction: {'Mass 1 down' if w1 > w2 else 'Mass 2 down'}\n"
                    else:
                        result += f"\nBalanced system:\n"
                        result += f"Tension = Weight = {w1:.2f} N\n"
                        result += f"System acceleration: 0 m/s² (equilibrium)\n"
                else:
                    result += f"\nFor angled configurations, more complex analysis required.\n"
                    result += f"This involves resolving forces in multiple directions.\n"

            else:
                # Multiple masses - general case
                result += f"Multi-mass system:\n"
                result += f"Total system weight: {total_weight:.2f} N\n"
                result += f"For complex multi-mass systems, detailed equilibrium\n"
                result += f"analysis of each connection point is required.\n"

            return result

        except Exception as e:
            return f"Error in calculation: {str(e)}"

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
    parser = argparse.ArgumentParser(description="Run Physics Forces MCP Server")
    parser.add_argument("--run", default="mcp-server", help="Command to run")
    parser.add_argument("--host", default="localhost", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=10100, help="Port to bind server to")
    parser.add_argument("--transport", default="streamable_http", help="Transport type")
    
    args = parser.parse_args()
    
    if args.run == "mcp-server":
        serve(args.host, args.port, args.transport)
    else:
        raise ValueError(f"Unknown run option: {args.run}")

if __name__ == "__main__":
    main()
