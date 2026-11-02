import os
import unittest
from unittest.mock import patch

from hitl_knowledge_transfer import KnowledgeTransferGate
from tutoring_profiles import MECHANICS_AGENTS
from strands_agents.conversation_agent import ConversationPhysicsAgent
from strands_agents.energy_agent import EnergyAgent
from strands_agents.angular_motion_agent import AngularMotionAgent
from strands_agents.thermodynamics_agent import ThermodynamicsAgent


class MomentumReleaseScopeTests(unittest.TestCase):
    def test_only_momentum_uses_the_new_contextual_gate_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            gate = KnowledgeTransferGate('http://unused')
        self.assertEqual(gate.contextual_agents, {'momentum_agent'})
        self.assertEqual(MECHANICS_AGENTS, {'momentum_agent'})
        for cls in (EnergyAgent, AngularMotionAgent, ThermodynamicsAgent):
            self.assertFalse(issubclass(cls, ConversationPhysicsAgent))
