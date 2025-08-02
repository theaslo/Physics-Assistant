from typing import Tuple
import math


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians for internal calculations."""
    return math.radians(degrees)

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees for output."""
    return math.degrees(radians)

def calculate_momentum_magnitude(mass: float, velocity: float) -> float:
    """Calculate momentum magnitude: p = mv"""
    return mass * velocity

def calculate_momentum_components(mass: float, velocity: float, angle_degrees: float) -> Tuple[float, float]:
    """Calculate momentum components in 2D"""
    angle_rad = degrees_to_radians(angle_degrees)
    px = mass * velocity * math.cos(angle_rad)
    py = mass * velocity * math.sin(angle_rad)
    return px, py

def calculate_resultant_momentum(px: float, py: float) -> Tuple[float, float]:
    """Calculate resultant momentum magnitude and direction"""
    magnitude = math.sqrt(px**2 + py**2)
    if px == 0:
        angle = 90.0 if py > 0 else 270.0
    else:
        angle = radians_to_degrees(math.atan2(py, px))
        if angle < 0:
            angle += 360
    return magnitude, angle