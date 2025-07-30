from typing import List, Tuple, Any, Dict, Optional
import math
import json

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians for internal calculations."""
    return math.radians(degrees)

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees for output."""
    return math.degrees(radians)

def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float]]:
    """Solve quadratic equation ax² + bx + c = 0. Returns (solution1, solution2) or (None, None) if no real solutions."""
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None, None
    elif discriminant == 0:
        return -b / (2*a), None
    else:
        sqrt_disc = math.sqrt(discriminant)
        return (-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)

def format_time(t: float) -> str:
    """Format time with appropriate units."""
    if abs(t) < 0.001:
        return f"{t*1000:.2f} ms"
    elif abs(t) < 1:
        return f"{t*1000:.1f} ms"
    elif abs(t) < 60:
        return f"{t:.2f} s"
    else:
        minutes = int(t // 60)
        seconds = t % 60
        return f"{minutes}m {seconds:.1f}s"

def safe_format(value, precision=2):
    """Safely format a number, returning 'Unknown' if None."""
    if value is None:
        return "Unknown"
    return f"{value:.{precision}f}"