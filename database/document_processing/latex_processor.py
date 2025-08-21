#!/usr/bin/env python3
"""
LaTeX Equation Processor for Physics Educational Content
Extracts and parses mathematical expressions from documents and maps them to knowledge graph concepts.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy import symbols, simplify, latex
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LatexEquation:
    """Represents a parsed LaTeX equation with metadata"""
    original_latex: str
    cleaned_latex: str
    sympy_expression: Optional[sp.Basic]
    variables: List[str]
    constants: List[str]
    equation_type: str  # formula, definition, calculation, etc.
    physics_domain: Optional[str]
    complexity_score: int
    is_valid: bool
    error_message: Optional[str] = None

@dataclass
class VariableInfo:
    """Information about a variable in an equation"""
    symbol: str
    description: Optional[str]
    unit: Optional[str]
    is_vector: bool = False
    is_constant: bool = False

class PhysicsLatexProcessor:
    """Advanced LaTeX processor specifically designed for physics equations"""
    
    def __init__(self):
        # Physics-specific symbol mappings
        self.physics_symbols = {
            'v': 'velocity',
            'a': 'acceleration', 
            'F': 'force',
            'm': 'mass',
            't': 'time',
            'x': 'position',
            'p': 'momentum',
            'E': 'energy',
            'K': 'kinetic_energy',
            'U': 'potential_energy',
            'W': 'work',
            'P': 'power',
            'omega': 'angular_velocity',
            'alpha': 'angular_acceleration',
            'tau': 'torque',
            'I': 'moment_of_inertia',
            'L': 'angular_momentum',
            'lambda': 'wavelength',
            'f': 'frequency',
            'T': 'period',
            'A': 'amplitude',
            'g': 'gravitational_acceleration',
            'mu': 'coefficient_of_friction',
            'theta': 'angle',
            'phi': 'phase'
        }
        
        # Physics constants
        self.physics_constants = {
            'c': 'speed_of_light',
            'h': 'planck_constant',
            'k': 'spring_constant',
            'G': 'gravitational_constant',
            'e': 'elementary_charge',
            'pi': 'pi',
            'g': 'standard_gravity'
        }
        
        # Common physics equation patterns
        self.equation_patterns = {
            'kinematics': [
                r'v\s*=\s*v_?0\s*\+\s*a\s*t',
                r'x\s*=\s*x_?0\s*\+\s*v_?0\s*t\s*\+\s*.*a.*t\^?2',
                r'v\^?2\s*=\s*v_?0\^?2\s*\+\s*.*a.*x'
            ],
            'forces': [
                r'F\s*=\s*m\s*a',
                r'W\s*=\s*m\s*g',
                r'f\s*=\s*.*mu.*N'
            ],
            'energy': [
                r'K\s*=\s*.*m.*v\^?2',
                r'U\s*=\s*m\s*g\s*h',
                r'W\s*=\s*F.*d',
                r'P\s*=\s*W.*t'
            ],
            'waves': [
                r'v\s*=\s*f.*lambda',
                r'y.*=.*A.*sin.*cos'
            ]
        }
        
    def extract_latex_from_text(self, text: str) -> List[str]:
        """Extract LaTeX equations from text using multiple patterns"""
        equations = []
        
        # Pattern 1: Inline math $...$
        inline_pattern = r'\$(.*?)\$'
        equations.extend(re.findall(inline_pattern, text, re.DOTALL))
        
        # Pattern 2: Display math $$...$$
        display_pattern = r'\$\$(.*?)\$\$'
        equations.extend(re.findall(display_pattern, text, re.DOTALL))
        
        # Pattern 3: LaTeX equation environments
        equation_envs = ['equation', 'align', 'gather', 'multline']
        for env in equation_envs:
            pattern = rf'\\begin\{{{env}\*?\}}(.*?)\\end\{{{env}\*?\}}'
            equations.extend(re.findall(pattern, text, re.DOTALL))
        
        # Pattern 4: Physics-specific patterns (even without LaTeX markup)
        physics_patterns = [
            r'([a-zA-Z_]\w*\s*=\s*[^=\n]+)',  # Basic equation format
            r'([A-Z][a-z]*\s*=\s*[^=\n]+)',   # Capitalized variable equations
        ]
        
        for pattern in physics_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Only add if it looks like a physics equation
                if any(symbol in match.lower() for symbol in ['velocity', 'force', 'energy', 'momentum', 'acceleration']):
                    equations.append(match.strip())
        
        # Clean and deduplicate
        cleaned_equations = []
        for eq in equations:
            cleaned = eq.strip()
            if cleaned and cleaned not in cleaned_equations:
                cleaned_equations.append(cleaned)
        
        logger.info(f"Extracted {len(cleaned_equations)} LaTeX equations from text")
        return cleaned_equations
    
    def clean_latex(self, latex_str: str) -> str:
        """Clean and normalize LaTeX string for processing"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', latex_str.strip())
        
        # Replace common LaTeX commands with SymPy-compatible versions
        replacements = {
            r'\\cdot': '*',
            r'\\times': '*',
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
            r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
            r'\\sin\{([^}]+)\}': r'sin(\1)',
            r'\\cos\{([^}]+)\}': r'cos(\1)',
            r'\\tan\{([^}]+)\}': r'tan(\1)',
            r'\\Delta': 'Delta',
            r'\\theta': 'theta',
            r'\\omega': 'omega',
            r'\\alpha': 'alpha',
            r'\\beta': 'beta',
            r'\\gamma': 'gamma',
            r'\\phi': 'phi',
            r'\\lambda': 'lambda',
            r'\\mu': 'mu',
            r'\\sigma': 'sigma',
            r'\\tau': 'tau',
            r'\\pi': 'pi',
            r'\{([^}]+)\}': r'\1',  # Remove braces
            r'\_([a-zA-Z0-9])': r'_\1',  # Fix subscripts
            r'\^([a-zA-Z0-9])': r'**\1',  # Fix superscripts
        }
        
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        
        return cleaned
    
    def parse_equation(self, latex_str: str) -> LatexEquation:
        """Parse a LaTeX equation into structured format"""
        original = latex_str
        cleaned = self.clean_latex(latex_str)
        
        try:
            # Try to parse with SymPy
            sympy_expr = None
            try:
                # First try direct SymPy parsing
                sympy_expr = sp.sympify(cleaned)
            except:
                try:
                    # Try LaTeX parser if available
                    sympy_expr = parse_latex(original)
                except:
                    # Try manual parsing for common physics equations
                    sympy_expr = self._manual_physics_parse(cleaned)
            
            # Extract variables and constants
            variables = []
            constants = []
            
            if sympy_expr:
                symbols_in_expr = sympy_expr.free_symbols
                for symbol in symbols_in_expr:
                    symbol_str = str(symbol)
                    if symbol_str in self.physics_constants:
                        constants.append(symbol_str)
                    else:
                        variables.append(symbol_str)
            else:
                # Fallback: extract from string
                variables = re.findall(r'[a-zA-Z_]\w*', cleaned)
                variables = list(set(variables))  # Remove duplicates
            
            # Determine equation type and domain
            eq_type = self._classify_equation_type(cleaned)
            domain = self._determine_physics_domain(cleaned, variables)
            
            # Calculate complexity score
            complexity = self._calculate_complexity(cleaned, variables)
            
            return LatexEquation(
                original_latex=original,
                cleaned_latex=cleaned,
                sympy_expression=sympy_expr,
                variables=variables,
                constants=constants,
                equation_type=eq_type,
                physics_domain=domain,
                complexity_score=complexity,
                is_valid=sympy_expr is not None
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse equation '{latex_str}': {str(e)}")
            return LatexEquation(
                original_latex=original,
                cleaned_latex=cleaned,
                sympy_expression=None,
                variables=[],
                constants=[],
                equation_type='unknown',
                physics_domain=None,
                complexity_score=0,
                is_valid=False,
                error_message=str(e)
            )
    
    def _manual_physics_parse(self, equation_str: str) -> Optional[sp.Basic]:
        """Manual parsing for common physics equations that SymPy might struggle with"""
        # Handle common physics equation formats
        
        # v = v0 + at
        if re.match(r'v\s*=\s*v0?\s*\+\s*a\s*\*?\s*t', equation_str):
            v, v0, a, t = symbols('v v0 a t')
            return sp.Eq(v, v0 + a*t)
        
        # F = ma
        if re.match(r'F\s*=\s*m\s*\*?\s*a', equation_str):
            F, m, a = symbols('F m a')
            return sp.Eq(F, m*a)
        
        # E = mc^2
        if re.match(r'E\s*=\s*m\s*\*?\s*c\*?\*?2', equation_str):
            E, m, c = symbols('E m c')
            return sp.Eq(E, m*c**2)
        
        # KE = (1/2)mv^2
        if re.match(r'KE\s*=\s*.*m.*v.*2', equation_str):
            KE, m, v = symbols('KE m v')
            return sp.Eq(KE, sp.Rational(1,2)*m*v**2)
        
        return None
    
    def _classify_equation_type(self, equation_str: str) -> str:
        """Classify the type of physics equation"""
        if '=' in equation_str:
            if any(op in equation_str for op in ['+', '-', '*', '/']):
                return 'formula'
            else:
                return 'definition'
        elif any(ineq in equation_str for ineq in ['<', '>', '<=', '>=']):
            return 'inequality'
        else:
            return 'expression'
    
    def _determine_physics_domain(self, equation_str: str, variables: List[str]) -> Optional[str]:
        """Determine which physics domain an equation belongs to"""
        
        # Check against known patterns
        for domain, patterns in self.equation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, equation_str, re.IGNORECASE):
                    return domain
        
        # Check based on variables
        mechanics_vars = {'v', 'a', 'F', 'm', 't', 'x', 'p'}
        energy_vars = {'E', 'K', 'U', 'W', 'P'}
        waves_vars = {'lambda', 'f', 'T', 'A', 'omega'}
        thermo_vars = {'Q', 'T', 'S', 'n', 'R'}
        em_vars = {'q', 'E', 'B', 'V', 'I', 'C', 'L'}
        
        var_set = set(variables)
        
        if var_set & mechanics_vars:
            return 'mechanics'
        elif var_set & energy_vars:
            return 'energy'
        elif var_set & waves_vars:
            return 'waves'
        elif var_set & thermo_vars:
            return 'thermodynamics'
        elif var_set & em_vars:
            return 'electromagnetism'
        
        return None
    
    def _calculate_complexity(self, equation_str: str, variables: List[str]) -> int:
        """Calculate complexity score for an equation"""
        score = 0
        
        # Base score for variables
        score += len(variables)
        
        # Add score for operators
        operators = ['+', '-', '*', '/', '^', '**', 'sin', 'cos', 'tan', 'log', 'exp', 'sqrt']
        for op in operators:
            score += equation_str.count(op)
        
        # Add score for complexity indicators
        if 'frac' in equation_str or '/' in equation_str:
            score += 2
        if any(trig in equation_str for trig in ['sin', 'cos', 'tan']):
            score += 3
        if any(func in equation_str for func in ['log', 'exp', 'sqrt']):
            score += 2
        if 'integral' in equation_str or 'sum' in equation_str:
            score += 5
        
        return score
    
    def extract_variable_info(self, equation: LatexEquation, context_text: str = "") -> List[VariableInfo]:
        """Extract detailed information about variables in an equation"""
        variable_info = []
        
        for var in equation.variables:
            # Get description from physics symbols
            description = self.physics_symbols.get(var, None)
            
            # Try to extract from context
            if not description and context_text:
                # Look for definitions in context
                patterns = [
                    rf'{var}\s*is\s+the\s+([^,.]+)',
                    rf'{var}\s*=\s*([^=\n]+)\s*(?:where|,)',
                    rf'where\s+{var}\s+is\s+([^,.]+)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, context_text, re.IGNORECASE)
                    if match:
                        description = match.group(1).strip()
                        break
            
            # Determine if it's a vector (common physics convention)
            is_vector = var in ['F', 'v', 'a', 'p', 'E', 'B'] or 'vec' in var
            
            # Check if it's a constant
            is_constant = var in self.physics_constants
            
            # Try to determine unit (basic heuristic)
            unit = self._guess_unit(var, equation.physics_domain)
            
            variable_info.append(VariableInfo(
                symbol=var,
                description=description,
                unit=unit,
                is_vector=is_vector,
                is_constant=is_constant
            ))
        
        return variable_info
    
    def _guess_unit(self, variable: str, domain: Optional[str]) -> Optional[str]:
        """Guess the SI unit for a physics variable"""
        unit_mappings = {
            'v': 'm/s',
            'a': 'm/s²',
            'F': 'N',
            'm': 'kg',
            't': 's',
            'x': 'm',
            'p': 'kg⋅m/s',
            'E': 'J',
            'K': 'J',
            'U': 'J',
            'W': 'J',
            'P': 'W',
            'omega': 'rad/s',
            'alpha': 'rad/s²',
            'tau': 'N⋅m',
            'I': 'kg⋅m²',
            'L': 'kg⋅m²/s',
            'lambda': 'm',
            'f': 'Hz',
            'T': 's',
            'A': 'm',  # amplitude
            'g': 'm/s²',
            'theta': 'rad',
            'phi': 'rad'
        }
        
        return unit_mappings.get(variable)
    
    def process_document_equations(self, text: str) -> List[LatexEquation]:
        """Process all equations found in a document"""
        latex_strings = self.extract_latex_from_text(text)
        equations = []
        
        for latex_str in latex_strings:
            equation = self.parse_equation(latex_str)
            if equation.is_valid:
                equations.append(equation)
            else:
                logger.warning(f"Invalid equation skipped: {latex_str}")
        
        logger.info(f"Successfully processed {len(equations)} equations from document")
        return equations
    
    def equations_to_json(self, equations: List[LatexEquation]) -> str:
        """Convert equations to JSON format for storage"""
        equation_data = []
        
        for eq in equations:
            eq_dict = {
                'original_latex': eq.original_latex,
                'cleaned_latex': eq.cleaned_latex,
                'sympy_expression': str(eq.sympy_expression) if eq.sympy_expression else None,
                'variables': eq.variables,
                'constants': eq.constants,
                'equation_type': eq.equation_type,
                'physics_domain': eq.physics_domain,
                'complexity_score': eq.complexity_score,
                'is_valid': eq.is_valid,
                'error_message': eq.error_message
            }
            equation_data.append(eq_dict)
        
        return json.dumps(equation_data, indent=2)

# Example usage and testing
if __name__ == "__main__":
    processor = PhysicsLatexProcessor()
    
    # Test with sample physics text
    sample_text = """
    The kinematic equations for constant acceleration are:
    
    $v = v_0 + at$
    
    $$x = x_0 + v_0 t + \\frac{1}{2}at^2$$
    
    $v^2 = v_0^2 + 2a(x - x_0)$
    
    Newton's second law states that F = ma, where F is the net force,
    m is the mass, and a is the acceleration.
    
    The kinetic energy is given by $KE = \\frac{1}{2}mv^2$ and potential
    energy by $PE = mgh$.
    """
    
    equations = processor.process_document_equations(sample_text)
    
    print(f"Found {len(equations)} equations:")
    for i, eq in enumerate(equations, 1):
        print(f"\n{i}. {eq.original_latex}")
        print(f"   Domain: {eq.physics_domain}")
        print(f"   Type: {eq.equation_type}")
        print(f"   Variables: {eq.variables}")
        print(f"   Complexity: {eq.complexity_score}")
    
    # Save to JSON
    json_output = processor.equations_to_json(equations)
    print(f"\nJSON Output:\n{json_output}")