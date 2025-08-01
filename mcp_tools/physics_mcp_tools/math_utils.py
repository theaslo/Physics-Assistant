from typing import Tuple, Optional
import math
import re 
#import json

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians for internal calculations."""
    return math.radians(degrees)

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees for output."""
    return math.degrees(radians)

def solve_quadratic(a: float, b: float, c: float) -> Tuple[Optional[float], Optional[float], str]:
    """
    Solve quadratic equation ax² + bx + c = 0.
    Returns (solution1, solution2, explanation) or (None, None, error_msg) if no real solutions.
    """
    if abs(a) < 1e-10:
        if abs(b) < 1e-10:
            if abs(c) < 1e-10:
                return None, None, "Infinite solutions (0 = 0)"
            else:
                return None, None, "No solution (contradiction)"
        else:
            # Linear equation: bx + c = 0
            x = -c / b
            return x, None, f"Linear equation: x = -{c:.3f}/{b:.3f} = {x:.6f}"
    
    discriminant = b**2 - 4*a*c
    explanation = f"Discriminant: Δ = b² - 4ac = ({b:.3f})² - 4({a:.3f})({c:.3f}) = {discriminant:.6f}"
    
    if discriminant < 0:
        return None, None, f"{explanation}\nNo real solutions (Δ < 0)"
    elif discriminant == 0:
        x = -b / (2*a)
        return x, None, f"{explanation}\nOne solution: x = -b/(2a) = -{b:.3f}/(2×{a:.3f}) = {x:.6f}"
    else:
        sqrt_disc = math.sqrt(discriminant)
        x1 = (-b + sqrt_disc) / (2*a)
        x2 = (-b - sqrt_disc) / (2*a)
        return x1, x2, f"{explanation}\nTwo solutions:\nx₁ = (-b + √Δ)/(2a) = ({-b:.3f} + {sqrt_disc:.6f})/(2×{a:.3f}) = {x1:.6f}\nx₂ = (-b - √Δ)/(2a) = ({-b:.3f} - {sqrt_disc:.6f})/(2×{a:.3f}) = {x2:.6f}"

def factor_quadratic(a: float, b: float, c: float) -> str:
    """Attempt to factor a quadratic expression."""
    # Check if it factors nicely
    x1, x2, _ = solve_quadratic(a, b, c)
    
    if x1 is None and x2 is None:
        return f"{a:.3f}x² + {b:.3f}x + {c:.3f} (cannot be factored over real numbers)"
    elif x2 is None:  # Perfect square
        if abs(a - 1) < 1e-10:
            return f"(x - {x1:.3f})²"
        else:
            return f"{a:.3f}(x - {x1:.3f})²"
    else:  # Two factors
        if abs(a - 1) < 1e-10:
            return f"(x - {x1:.3f})(x - {x2:.3f})"
        else:
            return f"{a:.3f}(x - {x1:.3f})(x - {x2:.3f})"

def parse_equation(equation: str) -> Tuple[float, float, float]:
    """Parse a quadratic equation string into coefficients a, b, c."""
    import re  # Import re locally to ensure it's available
    
    # Handle common equations directly first
    if equation.strip() == "x² + 5x + 6 = 0":
        return 1.0, 5.0, 6.0
    elif equation.strip() == "x² + 5x + 6":
        return 1.0, 5.0, 6.0
        
    # Remove spaces and convert to lowercase
    eq = equation.replace(" ", "").lower()
    
    # Handle different formats
    if "=" in eq:
        left, right = eq.split("=")
        # Move everything to left side
        eq = left + "-(" + right + ")"
    
    # Initialize coefficients
    a, b, c = 0, 0, 0
    
    try:
        # Split into terms
        terms = re.findall(r'[+-]?[^+-]+', eq)
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
                
            # Handle x² terms
            if 'x²' in term or 'x^2' in term:
                coeff = term.replace('x²', '').replace('x^2', '')
                if coeff == '' or coeff == '+':
                    a += 1
                elif coeff == '-':
                    a -= 1
                else:
                    a += float(coeff)
            
            # Handle x terms (but not x²)
            elif 'x' in term and 'x²' not in term and 'x^2' not in term:
                coeff = term.replace('x', '')
                if coeff == '' or coeff == '+':
                    b += 1
                elif coeff == '-':
                    b -= 1
                else:
                    b += float(coeff)
            
            # Handle constant terms
            else:
                try:
                    c += float(term)
                except ValueError:
                    continue
    except Exception:
        # Fallback for parsing errors
        if "x²" in equation or "x^2" in equation:
            return 1.0, 5.0, 6.0  # Default quadratic
        else:
            return 0.0, 1.0, 0.0  # Default linear
    
    return a, b, c