from typing import Tuple, Optional, Dict, List
import math
import json

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians for internal calculations."""
    return math.radians(degrees)

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees for output."""
    return math.degrees(radians)

def calculate_angular_displacement(omega_0: float, alpha: float, time: float) -> float:
    """Calculate angular displacement: θ = ω₀t + ½αt²"""
    return omega_0 * time + 0.5 * alpha * time**2

def calculate_angular_velocity_from_acceleration(omega_0: float, alpha: float, time: float) -> float:
    """Calculate angular velocity: ω = ω₀ + αt"""
    return omega_0 + alpha * time

def calculate_angular_velocity_from_displacement(omega_0: float, alpha: float, theta: float) -> float:
    """Calculate angular velocity: ω² = ω₀² + 2αθ"""
    omega_squared = omega_0**2 + 2 * alpha * theta
    return math.sqrt(abs(omega_squared)) * (1 if omega_squared >= 0 else -1)

def calculate_moment_of_inertia_rod(mass: float, length: float, axis: str = "center") -> float:
    """Calculate moment of inertia for a rod"""
    if axis.lower() == "center":
        return (1/12) * mass * length**2
    elif axis.lower() == "end":
        return (1/3) * mass * length**2
    else:
        raise ValueError("Axis must be 'center' or 'end'")

def calculate_moment_of_inertia_disk(mass: float, radius: float) -> float:
    """Calculate moment of inertia for a solid disk: I = ½MR²"""
    return 0.5 * mass * radius**2

def calculate_moment_of_inertia_sphere(mass: float, radius: float, hollow: bool = False) -> float:
    """Calculate moment of inertia for a sphere"""
    if hollow:
        return (2/3) * mass * radius**2  # Hollow sphere
    else:
        return (2/5) * mass * radius**2  # Solid sphere

def calculate_moment_of_inertia_cylinder(mass: float, radius: float, hollow: bool = False) -> float:
    """Calculate moment of inertia for a cylinder"""
    if hollow:
        return mass * radius**2  # Hollow cylinder (thin-walled)
    else:
        return 0.5 * mass * radius**2  # Solid cylinder

def calculate_parallel_axis_theorem(I_cm: float, mass: float, distance: float) -> float:
    """Apply parallel axis theorem: I = I_cm + Md²"""
    return I_cm + mass * distance**2