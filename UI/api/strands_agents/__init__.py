"""
Strands-based Physics Agents
============================
All physics agents (101, 102, 201, 202) now use the Strands Agents SDK
for agent orchestration with MCP tool integration and Ollama LLM.
"""

from .base_physics_agent import StrandsPhysicsAgent

# Physics 101 Agents
from .forces_agent import ForcesAgent
from .kinematics_agent import KinematicsAgent
from .math_agent import MathAgent
from .momentum_agent import MomentumAgent
from .energy_agent import EnergyAgent
from .angular_motion_agent import AngularMotionAgent

# Physics 102 Agents
from .thermodynamics_agent import ThermodynamicsAgent
from .waves_agent import WavesAgent

# Physics 201 Agents
from .electromagnetism_agent import ElectromagnetismAgent

# Physics 202 Agents
from .optics_agent import OpticsAgent
from .modern_physics_agent import ModernPhysicsAgent

__all__ = [
    # Base
    'StrandsPhysicsAgent',
    # Physics 101
    'ForcesAgent',
    'KinematicsAgent',
    'MathAgent',
    'MomentumAgent',
    'EnergyAgent',
    'AngularMotionAgent',
    # Physics 102
    'ThermodynamicsAgent',
    'WavesAgent',
    # Physics 201
    'ElectromagnetismAgent',
    # Physics 202
    'OpticsAgent',
    'ModernPhysicsAgent',
]
