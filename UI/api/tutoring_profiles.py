"""Capabilities shared by the API and conversational mechanics agents."""

import re
from force_quiz_reflection import is_force_quiz_reflection_prompt

PRINCIPLES_TOOLS = {
    "forces_agent": "get_force_principles",
    "momentum_agent": "get_momentum_principles",
    "energy_agent": "get_energy_principles",
    "angular_motion_agent": "get_angular_motion_principles",
    "thermodynamics_agent": "get_thermodynamics_principles",
}
CONVERSATION_AGENTS = frozenset(PRINCIPLES_TOOLS)
MECHANICS_AGENTS = CONVERSATION_AGENTS - {"forces_agent"}


def is_guided_reflection(agent_id, text):
    if agent_id == "forces_agent":
        return is_force_quiz_reflection_prompt(agent_id, text)
    if agent_id not in MECHANICS_AGENTS:
        return False
    return bool(re.search(
        r"\b(quiz|exam|test|reflection|review|misunderstanding|misconception|hint)\b"
        r"|one (question|step) at a time|step.by.step|help me (understand|identify).*mistake",
        text, re.IGNORECASE,
    ))
