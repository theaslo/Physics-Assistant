from typing import Tuple, Optional, Dict, List
import math

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees for output."""
    return math.degrees(radians)

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians for internal calculations."""
    return math.radians(degrees)

def calculate_kinetic_energy(mass: float, velocity: float) -> float:
    """Calculate kinetic energy: KE = ½mv²"""
    return 0.5 * mass * velocity**2

def calculate_gravitational_potential_energy(mass: float, height: float, gravity: float = 9.81) -> float:
    """Calculate gravitational potential energy: PE = mgh"""
    return mass * gravity * height

def calculate_elastic_potential_energy(spring_constant: float, displacement: float) -> float:
    """Calculate elastic potential energy: PE = ½kx²"""
    return 0.5 * spring_constant * displacement**2

def calculate_work(force: float, displacement: float, angle_degrees: float = 0) -> float:
    """Calculate work: W = F·d·cos(θ)"""
    angle_rad = degrees_to_radians(angle_degrees)
    return force * displacement * math.cos(angle_rad)
