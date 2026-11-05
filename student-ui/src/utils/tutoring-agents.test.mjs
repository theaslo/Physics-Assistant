import test from 'node:test'
import assert from 'node:assert/strict'
import { conversationHistoryLimit, usesContextualTutoring } from './tutoring-agents.ts'

test('all four contextual tutors keep enough history for check-in follow-ups', () => {
  for (const agent of ['momentum_agent', 'energy_agent', 'angular_motion_agent', 'thermodynamics_agent']) {
    assert.equal(usesContextualTutoring(agent), true, agent)
    assert.equal(conversationHistoryLimit(agent), 40, agent)
  }
})

test('existing forces and math history behavior is retained', () => {
  assert.equal(usesContextualTutoring('forces_agent'), false)
  assert.equal(conversationHistoryLimit('forces_agent'), 40)
  for (const agent of ['math_agent', 'kinematics_agent', 'unknown_agent']) {
    assert.equal(usesContextualTutoring(agent), false)
    assert.equal(conversationHistoryLimit(agent), 6)
  }
})
