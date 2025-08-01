import os
from typing import Dict, List

# Application Configuration
class Config:
    # App Settings
    APP_NAME = "Physics Assistant UI"
    APP_VERSION = "1.0.0"
    PAGE_TITLE = "Physics Assistant"
    PAGE_ICON = "🔬"
    LAYOUT = "wide"
    
    # Authentication Settings
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    MAX_LOGIN_ATTEMPTS = 3
    
    # MCP Server Configuration
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    MCP_WEBSOCKET_URL = os.getenv("MCP_WEBSOCKET_URL", "ws://localhost:8000/ws")
    MCP_API_KEY = os.getenv("MCP_API_KEY", "")
    
    # Physics Agents Configuration
    PHYSICS_AGENTS = {
        "kinematics": {
            "name": "Kinematics Agent",
            "description": "1D and 2D motion problems",
            "icon": "🚀"
        },
        "forces": {
            "name": "Forces Agent", 
            "description": "Newton's laws and vector analysis",
            "icon": "⚡"
        },
        "energy": {
            "name": "Work and Energy Agent",
            "description": "Work, energy, and conservation laws",
            "icon": "🔋"
        },
        "momentum": {
            "name": "Momentum Agent",
            "description": "Momentum and impulse problems",
            "icon": "💥"
        },
        "rotation": {
            "name": "Rotation Agent",
            "description": "Rotational motion and dynamics",
            "icon": "🌀"
        },
        "math_helper": {
            "name": "Math Helper Agent",
            "description": "Trigonometry and algebra support",
            "icon": "📐"
        }
    }
    
    # UI Configuration
    SIDEBAR_WIDTH = 300
    MAX_CHAT_HISTORY = 100
    MAX_FILE_SIZE_MB = 5
    ALLOWED_FILE_TYPES = ['png', 'jpg', 'jpeg', 'gif', 'pdf']
    
    # Cache Configuration
    CACHE_TTL = 300  # 5 minutes
    MAX_CACHE_ENTRIES = 1000
    
    # Database Configuration (if using local storage)
    DB_PATH = os.getenv("DB_PATH", "data/physics_assistant.db")
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/physics_assistant.log")

# Physics Constants for calculations
PHYSICS_CONSTANTS = {
    "g": 9.81,  # gravitational acceleration (m/s²)
    "c": 299792458,  # speed of light (m/s)
    "e": 1.602176634e-19,  # elementary charge (C)
    "h": 6.62607015e-34,  # Planck constant (J⋅s)
    "k_B": 1.380649e-23,  # Boltzmann constant (J/K)
    "N_A": 6.02214076e23,  # Avogadro constant (mol⁻¹)
}

# Common physics units for conversion
PHYSICS_UNITS = {
    "length": ["m", "cm", "mm", "km", "in", "ft", "yd"],
    "time": ["s", "ms", "min", "h", "day"],
    "mass": ["kg", "g", "mg", "lb", "oz"],
    "force": ["N", "kN", "lbf", "dyne"],
    "energy": ["J", "kJ", "cal", "kcal", "eV", "kWh"],
    "power": ["W", "kW", "MW", "hp"],
    "pressure": ["Pa", "kPa", "MPa", "atm", "psi", "torr"],
    "temperature": ["K", "°C", "°F"],
    "angle": ["rad", "deg", "grad"]
}