from typing import Tuple, Optional, Dict, List
import math
from physics_mcp_tools.input_validation import number

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees for output."""
    return math.degrees(radians)

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians for internal calculations."""
    return math.radians(degrees)

def calculate_kinetic_energy(mass: float, velocity: float) -> float:
    """Calculate kinetic energy: KE = ½mv²"""
    mass = number({"mass": mass}, "mass", positive=True)
    velocity = number({"velocity": velocity}, "velocity")
    return 0.5 * mass * velocity**2

def calculate_gravitational_potential_energy(mass: float, height: float, gravity: float = 9.81) -> float:
    """Calculate gravitational potential energy: PE = mgh"""
    mass = number({"mass": mass}, "mass", positive=True)
    height = number({"height": height}, "height")
    gravity = number({"gravity": gravity}, "gravity", nonnegative=True)
    return mass * gravity * height

def calculate_elastic_potential_energy(spring_constant: float, displacement: float) -> float:
    """Calculate elastic potential energy: PE = ½kx²"""
    spring_constant = number({"spring_constant": spring_constant}, "spring_constant", positive=True)
    displacement = number({"displacement": displacement}, "displacement")
    return 0.5 * spring_constant * displacement**2

def calculate_work(force: float, displacement: float, angle_degrees: float = 0) -> float:
    """Calculate work: W = F·d·cos(θ)"""
    force = number({"force": force}, "force")
    displacement = number({"displacement": displacement}, "displacement")
    angle_degrees = number({"angle": angle_degrees}, "angle")
    angle_rad = degrees_to_radians(angle_degrees)
    return force * displacement * math.cos(angle_rad)
