"""
Input validation utilities for the Physics Assistant UI
"""

import re
import math
from typing import Tuple, Optional, List, Any
from utils.constants import UNIT_SYMBOLS, PHYSICS_TOPICS

class InputValidator:
    """General input validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, email):
            return True, ""
        return False, "Invalid email format"
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """Validate username format"""
        if len(username) < 3:
            return False, "Username must be at least 3 characters long"
        if len(username) > 20:
            return False, "Username must be no more than 20 characters long"
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        if len(password) > 100:
            return False, "Password is too long"
        return True, ""
    
    @staticmethod
    def validate_file_upload(file, allowed_types: List[str], max_size_mb: float = 5) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if not file:
            return False, "No file selected"
        
        # Check file extension
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in allowed_types:
            return False, f"File type '{file_extension}' not allowed. Allowed types: {', '.join(allowed_types)}"
        
        # Check file size
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size ({file_size_mb:.1f} MB) exceeds limit ({max_size_mb} MB)"
        
        return True, ""

class PhysicsValidator:
    """Validation for physics-related inputs"""
    
    @staticmethod
    def validate_numerical_input(value: str, allow_negative: bool = True, allow_zero: bool = True) -> Tuple[bool, Optional[float], str]:
        """
        Validate numerical physics input
        Returns: (is_valid, parsed_value, error_message)
        """
        if not value.strip():
            return False, None, "Please enter a value"
        
        try:
            # Handle scientific notation
            parsed_value = float(value.replace(',', ''))
            
            if not allow_negative and parsed_value < 0:
                return False, None, "Value cannot be negative"
            
            if not allow_zero and parsed_value == 0:
                return False, None, "Value cannot be zero"
            
            # Check for reasonable physics values
            if abs(parsed_value) > 1e20:
                return False, None, "Value is unreasonably large"
            
            return True, parsed_value, ""
            
        except ValueError:
            return False, None, "Please enter a valid number"
    
    @staticmethod
    def validate_vector_input(value: str) -> Tuple[bool, Optional[List[float]], str]:
        """Validate vector input (comma-separated values)"""
        if not value.strip():
            return False, None, "Please enter vector components"
        
        try:
            # Remove parentheses and split by comma
            cleaned = value.strip().strip('()[]{}')
            components = [float(x.strip()) for x in cleaned.split(',')]
            
            if len(components) < 2:
                return False, None, "Vector must have at least 2 components"
            if len(components) > 3:
                return False, None, "Vector cannot have more than 3 components"
            
            return True, components, ""
            
        except ValueError:
            return False, None, "Please enter valid numerical components separated by commas"
    
    @staticmethod
    def validate_unit(unit: str) -> Tuple[bool, str]:
        """Validate physics unit"""
        if not unit.strip():
            return False, "Please specify a unit"
        
        # Check against known units
        if unit in UNIT_SYMBOLS:
            return True, ""
        
        # Check common variations
        common_variations = {
            'meters': 'm', 'meter': 'm', 'metre': 'm', 'metres': 'm',
            'seconds': 's', 'second': 's', 'sec': 's',
            'kilograms': 'kg', 'kilogram': 'kg', 'kg': 'kg',
            'newtons': 'N', 'newton': 'N',
            'joules': 'J', 'joule': 'J',
            'watts': 'W', 'watt': 'W',
            'degrees': 'deg', 'degree': 'deg'
        }
        
        if unit.lower() in common_variations:
            return True, f"Recognized as {common_variations[unit.lower()]}"
        
        return False, f"Unknown unit '{unit}'. Please use standard physics units."
    
    @staticmethod
    def validate_equation_input(equation: str) -> Tuple[bool, str]:
        """Validate physics equation input"""
        if not equation.strip():
            return False, "Please enter an equation"
        
        # Check for basic equation structure
        if '=' not in equation:
            return False, "Equation must contain an equals sign (=)"
        
        # Check for balanced parentheses
        if equation.count('(') != equation.count(')'):
            return False, "Unbalanced parentheses in equation"
        
        # Check for dangerous characters (basic security)
        dangerous_chars = ['<', '>', '&', '|', ';', '`']
        if any(char in equation for char in dangerous_chars):
            return False, "Equation contains invalid characters"
        
        return True, ""
    
    @staticmethod
    def validate_physics_range(value: float, physics_type: str) -> Tuple[bool, str]:
        """Validate if value is within reasonable physics ranges"""
        ranges = {
            'velocity': {'min': -3e8, 'max': 3e8, 'unit': 'm/s', 'note': 'cannot exceed speed of light'},
            'acceleration': {'min': -1e10, 'max': 1e10, 'unit': 'm/s²', 'note': 'extremely high acceleration'},
            'force': {'min': -1e15, 'max': 1e15, 'unit': 'N', 'note': 'extremely large force'},
            'energy': {'min': -1e20, 'max': 1e20, 'unit': 'J', 'note': 'extremely large energy'},
            'power': {'min': -1e15, 'max': 1e15, 'unit': 'W', 'note': 'extremely large power'},
            'mass': {'min': 0, 'max': 1e50, 'unit': 'kg', 'note': 'mass cannot be negative'},
            'temperature': {'min': 0, 'max': 1e8, 'unit': 'K', 'note': 'temperature cannot be below absolute zero'}
        }
        
        if physics_type not in ranges:
            return True, ""  # No specific validation for this type
        
        range_info = ranges[physics_type]
        
        if value < range_info['min']:
            return False, f"Value too small - {range_info['note']}"
        
        if value > range_info['max']:
            return False, f"Value too large - {range_info['note']}"
        
        return True, ""

class MessageValidator:
    """Validation for chat messages and content"""
    
    @staticmethod
    def validate_chat_message(message: str) -> Tuple[bool, str]:
        """Validate chat message content"""
        if not message.strip():
            return False, "Message cannot be empty"
        
        if len(message) > 2000:
            return False, "Message is too long (max 2000 characters)"
        
        # Check for inappropriate content (basic)
        inappropriate_patterns = [
            r'\b(password|secret|key)\s*[:=]\s*\w+',  # Potential credentials
            r'<script.*?>',  # Script tags
            r'javascript:',  # JavaScript URLs
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return False, "Message contains inappropriate content"
        
        return True, ""
    
    @staticmethod
    def validate_latex_input(latex: str) -> Tuple[bool, str]:
        """Validate LaTeX equation input"""
        if not latex.strip():
            return False, "LaTeX input cannot be empty"
        
        # Check for balanced braces
        if latex.count('{') != latex.count('}'):
            return False, "Unbalanced braces in LaTeX"
        
        # Check for dangerous LaTeX commands
        dangerous_commands = [
            '\\input', '\\include', '\\write', '\\openout',
            '\\read', '\\openin', '\\immediate', '\\special'
        ]
        
        for cmd in dangerous_commands:
            if cmd in latex:
                return False, f"LaTeX command '{cmd}' not allowed"
        
        return True, ""
    
    @staticmethod
    def detect_physics_topic(message: str) -> Optional[str]:
        """Detect physics topic from message content"""
        message_lower = message.lower()
        
        topic_scores = {}
        for topic, keywords in PHYSICS_TOPICS.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            # Return topic with highest score
            return max(topic_scores, key=topic_scores.get)
        
        return None
    
    @staticmethod
    def extract_numbers_from_text(text: str) -> List[float]:
        """Extract numerical values from text"""
        # Pattern to match numbers (including scientific notation)
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers

class SecurityValidator:
    """Security-related validation"""
    
    @staticmethod
    def validate_session_data(data: Any) -> Tuple[bool, str]:
        """Validate session data for security issues"""
        if isinstance(data, str):
            # Check for potential XSS
            xss_patterns = [
                r'<script.*?>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe.*?>'
            ]
            
            for pattern in xss_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    return False, "Data contains potentially malicious content"
        
        return True, ""
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Sanitize user input"""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&"\']', '', input_str)
        # Limit length
        sanitized = sanitized[:1000]
        return sanitized.strip()
    
    @staticmethod
    def validate_file_content(file_content: bytes, expected_type: str) -> Tuple[bool, str]:
        """Validate uploaded file content"""
        # Basic file signature validation
        signatures = {
            'png': b'\x89PNG\r\n\x1a\n',
            'jpg': b'\xff\xd8\xff',
            'jpeg': b'\xff\xd8\xff',
            'gif': b'GIF8',
            'pdf': b'%PDF'
        }
        
        if expected_type in signatures:
            expected_sig = signatures[expected_type]
            if not file_content.startswith(expected_sig):
                return False, f"File content doesn't match expected {expected_type} format"
        
        return True, ""