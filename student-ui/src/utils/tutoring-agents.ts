const contextualAgents = new Set([
  'momentum_agent',
  'energy_agent',
  'angular_motion_agent',
  'thermodynamics_agent',
])

export function usesContextualTutoring(agentId: string): boolean {
  return contextualAgents.has(agentId)
}

export function conversationHistoryLimit(agentId: string): number {
  return agentId === 'forces_agent' || usesContextualTutoring(agentId) ? 40 : 6
}
