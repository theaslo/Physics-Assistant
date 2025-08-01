"""
Data formatting utilities for the Physics Assistant UI
"""

import re
import math
from typing import Any, Union, Optional, Tuple
from datetime import datetime, timedelta
from utils.constants import UNIT_CONVERSIONS, UNIT_SYMBOLS

class PhysicsFormatter:
    """Formatter for physics-related data display"""
    
    @staticmethod
    def format_scientific_notation(value: float, precision: int = 3) -> str:
        """Format number in scientific notation with specified precision"""
        if value == 0:
            return "0"
        
        # Handle very small or very large numbers
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.{precision}e}"
        else:
            return f"{value:.{precision}f}".rstrip('0').rstrip('.')
    
    @staticmethod
    def format_with_units(value: float, unit: str, precision: int = 3) -> str:
        """Format value with appropriate units"""
        formatted_value = PhysicsFormatter.format_scientific_notation(value, precision)
        unit_name = UNIT_SYMBOLS.get(unit, unit)
        return f"{formatted_value} {unit}"
    
    @staticmethod
    def format_vector(components: list, unit: str = "", precision: int = 3) -> str:
        """Format vector components"""
        formatted_components = [
            PhysicsFormatter.format_scientific_notation(comp, precision) 
            for comp in components
        ]
        
        if len(components) == 2:
            result = f"({formatted_components[0]}, {formatted_components[1]})"
        elif len(components) == 3:
            result = f"({formatted_components[0]}, {formatted_components[1]}, {formatted_components[2]})"
        else:
            result = f"({', '.join(formatted_components)})"
        
        if unit:
            result += f" {unit}"
        
        return result
    
    @staticmethod
    def format_equation(equation: str) -> str:
        """Format physics equation for LaTeX rendering"""
        # Replace common variable patterns with LaTeX formatting
        replacements = {
            r'\*': r' \cdot ',
            r'\^(\w)': r'^{\1}',
            r'\^(\d+)': r'^{\1}',
            r'_(\w)': r'_{\1}',
            r'_(\d+)': r'_{\1}',
            r'sqrt\(([^)]+)\)': r'\\sqrt{\1}',
            r'pi': r'\\pi',
            r'theta': r'\\theta',
            r'alpha': r'\\alpha',
            r'beta': r'\\beta',
            r'gamma': r'\\gamma',
            r'delta': r'\\delta',
            r'omega': r'\\omega',
            r'mu': r'\\mu',
            r'sigma': r'\\sigma'
        }
        
        formatted = equation
        for pattern, replacement in replacements.items():
            formatted = re.sub(pattern, replacement, formatted)
        
        return formatted
    
    @staticmethod
    def format_percentage(value: float, precision: int = 1) -> str:
        """Format value as percentage"""
        return f"{value:.{precision}f}%"
    
    @staticmethod
    def format_ratio(numerator: float, denominator: float, precision: int = 2) -> str:
        """Format ratio with proper formatting"""
        if denominator == 0:
            return "∞"
        
        ratio = numerator / denominator
        return PhysicsFormatter.format_scientific_notation(ratio, precision)

class TimeFormatter:
    """Formatter for time-related data"""
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """Format timestamp to readable date/time"""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def format_relative_time(timestamp: float) -> str:
        """Format timestamp relative to now (e.g., '5 minutes ago')"""
        now = datetime.now()
        dt = datetime.fromtimestamp(timestamp)
        diff = now - dt
        
        if diff.total_seconds() < 60:
            return "Just now"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = diff.days
            return f"{days} day{'s' if days != 1 else ''} ago"

class UnitConverter:
    """Handle unit conversions for physics calculations"""
    
    @staticmethod
    def convert_units(value: float, from_unit: str, to_unit: str, unit_type: str) -> Tuple[float, bool]:
        """
        Convert between units
        Returns: (converted_value, success)
        """
        try:
            conversions = UNIT_CONVERSIONS.get(unit_type, {})
            conversion_key = f"{from_unit}_to_{to_unit}"
            
            if conversion_key in conversions:
                converted_value = value * conversions[conversion_key]
                return converted_value, True
            else:
                # Try reverse conversion
                reverse_key = f"{to_unit}_to_{from_unit}"
                if reverse_key in conversions:
                    converted_value = value / conversions[reverse_key]
                    return converted_value, True
            
            return value, False
        except:
            return value, False
    
    @staticmethod
    def temperature_convert(value: float, from_unit: str, to_unit: str) -> Tuple[float, bool]:
        """Special handling for temperature conversions"""
        try:
            if from_unit == to_unit:
                return value, True
            
            # Convert to Celsius first
            if from_unit == "°F":
                celsius = (value - 32) * 5/9
            elif from_unit == "K":
                celsius = value - 273.15
            else:  # Already Celsius
                celsius = value
            
            # Convert from Celsius to target
            if to_unit == "°F":
                result = celsius * 9/5 + 32
            elif to_unit == "K":
                result = celsius + 273.15
            else:  # Target is Celsius
                result = celsius
            
            return result, True
        except:
            return value, False
    
    @staticmethod
    def get_unit_type(unit: str) -> Optional[str]:
        """Determine the type of unit (length, mass, etc.)"""
        for unit_type, conversions in UNIT_CONVERSIONS.items():
            for conversion_key in conversions.keys():
                if unit in conversion_key:
                    return unit_type
        
        # Special case for temperature
        if unit in ["°C", "°F", "K"]:
            return "temperature"
        
        return None

class DataTableFormatter:
    """Formatter for data tables and statistics"""
    
    @staticmethod
    def format_statistics_table(data: dict) -> dict:
        """Format statistics data for display in Streamlit"""
        formatted_data = {}
        
        for key, value in data.items():
            if isinstance(value, float):
                if key.endswith('_rate') or key.endswith('_percentage'):
                    formatted_data[key] = PhysicsFormatter.format_percentage(value * 100)
                elif key.endswith('_time') or key.endswith('_duration'):
                    formatted_data[key] = TimeFormatter.format_duration(value)
                else:
                    formatted_data[key] = PhysicsFormatter.format_scientific_notation(value)
            elif isinstance(value, int):
                formatted_data[key] = str(value)
            elif isinstance(value, list):
                formatted_data[key] = ", ".join(map(str, value))
            else:
                formatted_data[key] = str(value)
        
        return formatted_data

class MessageFormatter:
    """Formatter for chat messages and responses"""
    
    @staticmethod
    def format_physics_response(response: str) -> str:
        """Format physics response with proper mathematical notation"""
        # Replace common physics notation with LaTeX
        formatted = response
        
        # Mathematical operators
        formatted = re.sub(r'\*', ' × ', formatted)
        formatted = re.sub(r'/', ' ÷ ', formatted)
        
        # Greek letters (if not already LaTeX)
        greek_letters = {
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'Δ',
            'epsilon': 'ε', 'theta': 'θ', 'lambda': 'λ', 'mu': 'μ',
            'pi': 'π', 'sigma': 'σ', 'omega': 'ω', 'phi': 'φ'
        }
        
        for word, symbol in greek_letters.items():
            if f'\\{word}' not in formatted:  # Don't replace LaTeX
                formatted = re.sub(rf'\b{word}\b', symbol, formatted, flags=re.IGNORECASE)
        
        return formatted
    
    @staticmethod
    def format_equation_explanation(equation: str, variables: dict) -> str:
        """Format equation with variable explanations"""
        explanation = f"**Equation:** {equation}\n\n**Where:**\n"
        
        for var, description in variables.items():
            explanation += f"- {var}: {description}\n"
        
        return explanation
    
    @staticmethod
    def truncate_message(message: str, max_length: int = 500) -> str:
        """Truncate long messages for display"""
        if len(message) <= max_length:
            return message
        
        truncated = message[:max_length - 3]
        # Try to break at a word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # If we can break at a reasonable point
            truncated = truncated[:last_space]
        
        return truncated + "..."

class ColorFormatter:
    """Color formatting utilities for UI elements"""
    
    @staticmethod
    def get_agent_color(agent_id: str) -> str:
        """Get consistent color for each physics agent"""
        colors = {
            "kinematics": "#FF6B6B",     # Red
            "forces": "#4ECDC4",         # Teal  
            "energy": "#45B7D1",         # Blue
            "momentum": "#96CEB4",       # Green
            "rotation": "#FFEAA7",       # Yellow
            "math_helper": "#DDA0DD"     # Plum
        }
        return colors.get(agent_id, "#888888")  # Default gray
    
    @staticmethod
    def get_difficulty_color(difficulty: int) -> str:
        """Get color based on difficulty level (1-5)"""
        colors = {
            1: "#4CAF50",  # Green - Easy
            2: "#8BC34A",  # Light Green
            3: "#FFC107",  # Amber - Medium
            4: "#FF9800",  # Orange
            5: "#F44336"   # Red - Hard
        }
        return colors.get(difficulty, "#9E9E9E")  # Default gray
    
    @staticmethod
    def get_success_color(success_rate: float) -> str:
        """Get color based on success rate percentage"""
        if success_rate >= 80:
            return "#4CAF50"  # Green
        elif success_rate >= 60:
            return "#FFC107"  # Amber
        elif success_rate >= 40:
            return "#FF9800"  # Orange
        else:
            return "#F44336"  # Red