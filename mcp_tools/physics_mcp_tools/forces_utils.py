from typing import List, Tuple, Any, Dict
import math
import json

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians for internal calculations."""
    return math.radians(degrees)

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees for output."""
    return math.degrees(radians)

def calculate_force_components(magnitude: float, angle_degrees: float) -> Tuple[float, float]:
    """Calculate x and y components from magnitude and angle."""
    angle_rad = degrees_to_radians(angle_degrees)
    force_x = magnitude * math.cos(angle_rad)
    force_y = magnitude * math.sin(angle_rad)
    return force_x, force_y

def calculate_resultant_force(force_x_total: float, force_y_total: float) -> Tuple[float, float]:
    """Calculate magnitude and angle from x and y components."""
    magnitude = math.sqrt(force_x_total**2 + force_y_total**2)
    if force_x_total == 0:
        angle = 90.0 if force_y_total > 0 else 270.0
    else:
        angle = radians_to_degrees(math.atan2(force_y_total, force_x_total))
        if angle < 0:
            angle += 360  # Convert to positive angle
    return magnitude, angle

def calculate_spring_force(k: float, displacement: float) -> float:
    """Calculate spring force using Hooke's law: F = -kx"""
    return -k * displacement

def calculate_friction_force(coefficient: float, normal_force: float, kinetic: bool = True) -> float:
    """Calculate friction force: F = μN"""
    return coefficient * normal_force

def calculate_gravitational_force(mass: float, g: float = 9.81) -> float:
    """Calculate gravitational force: F = mg"""
    return mass * g
