import os
import unittest
from unittest.mock import patch

from hitl_knowledge_transfer import KnowledgeTransferGate
from tutoring_profiles import MECHANICS_AGENTS
from strands_agents.conversation_agent import ConversationPhysicsAgent
from strands_agents.energy_agent import EnergyAgent
from strands_agents.angular_motion_agent import AngularMotionAgent
from strands_agents.thermodynamics_agent import ThermodynamicsAgent


class ContextualRolloutScopeTests(unittest.TestCase):
    def test_deployments_can_disable_or_limit_contextual_checks(self):
        for setting, expected in (("", set()), ("energy_agent", {"energy_agent"})):
            with patch.dict(os.environ, {"HITL_CONTEXTUAL_AGENTS": setting}, clear=True):
                gate = KnowledgeTransferGate('http://unused')
            self.assertEqual(gate.contextual_agents, expected)

    def test_all_four_domains_use_contextual_checks_and_worked_solutions(self):
        with patch.dict(os.environ, {}, clear=True):
            gate = KnowledgeTransferGate('http://unused')
        self.assertEqual(gate.contextual_agents, {'momentum_agent', 'energy_agent', 'angular_motion_agent', 'thermodynamics_agent'})
        self.assertEqual(MECHANICS_AGENTS, gate.contextual_agents)
        for cls in (EnergyAgent, AngularMotionAgent, ThermodynamicsAgent):
            self.assertTrue(issubclass(cls, ConversationPhysicsAgent))
            self.assertTrue(cls.worked_solution_required)
