import test from 'node:test'
import assert from 'node:assert/strict'
import { isFullSolutionRequest } from './tutoring-intent.ts'

test('explicit requests release the full solution', () => {
  for (const text of ['Please give the full solution.', 'Show me the solution',
    "I'm lost. Please give the full worked solution.", 'Provide a complete worked solution']) {
    assert.equal(isFullSolutionRequest(text), true, text)
  }
})

test('uncertainty and requests for hints keep the solution hidden', () => {
  for (const text of ["I'm lost", 'I am completely lost', "I'm stuck. Can I have a smaller hint?",
    'What is the next step?', 'Which equation should I use?']) {
    assert.equal(isFullSolutionRequest(text), false, text)
  }
})

test('negative requests mentioning a full solution do not reveal it', () => {
  for (const text of ["Don't give me the full solution", 'Do not show the answer. Give me a hint.',
    'I don’t want the full solution yet', 'Help without the full solution', 'No full solution please',
    'Hints only, not the full solution']) {
    assert.equal(isFullSolutionRequest(text), false, text)
  }
})
