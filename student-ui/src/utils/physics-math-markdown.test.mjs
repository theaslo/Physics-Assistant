import test from 'node:test'
import assert from 'node:assert/strict'
import { formatPhysicsMathMarkdown as format } from './physics-math-markdown.ts'

test('mechanics palette symbols retain subscripts and render as complete math', () => {
  for (const symbol of ['p_i', 'p_f', 'K_i', 'K_f', 'U_i', 'U_f', '\\vec{p}', '\\Delta p',
    '\\Delta K', '\\theta_i', '\\theta_f', '\\omega_i', '\\omega_f', '\\alpha', '\\sum \\tau', '\\vec{L}',
    '\\Delta \\omega', '\\omega_{f}', '\\omega_f^2', 'K_{rot}', 'r^{2}', 'v_i^2', 'y^x', 'x^{}']) {
    assert.equal(format(symbol), `$${symbol}$`, symbol)
  }
})

test('legacy braced subscripts do not leave a closing brace outside math', () => {
  for (const symbol of ['x_{f}', 'v_{0}', 'F_{net}', 'x_f', 'v_0^2']) {
    assert.equal(format(symbol), `$${symbol}$`)
  }
})

test('existing math, code, prose and arithmetic operators stay unchanged', () => {
  for (const input of ['$K_i + U_i = K_f + U_f$', '`p_i`', '```tex\n\\omega_i\n```',
    'Is momentum conserved?', '+ - * / ()']) assert.equal(format(input), input)
  assert.equal(format('Use p_i = p_f'), 'Use $p_i$ = $p_f$')
})

test('momentum units and multiplication survive malformed JSON math escapes', () => {
  const malformed = JSON.parse(String.raw`{"answer":"Use $p=mv$. Given $m=5.0\text{ kg}$ and $v=10\text{ m/s}$ east, $p=5.0\times10=50\text{ kg m/s}$ east."}`).answer
  // Commands immediately followed by a number are valid TeX command boundaries.
  const expected = String.raw`Use $p=mv$. Given $m=5.0\text{ kg}$ and $v=10\text{ m/s}$ east, $p=5.0\times10=50\text{ kg m/s}$ east.`
  assert.equal(format(malformed), expected)
  assert.equal(format(expected), expected)
})

test('common corrupted commands are repaired only in math', () => {
  const malformed = '$\frac{1}{2}mv^2 + \theta + \tau + \beta + \rho + \\left(x\right)$'
  assert.equal(format(malformed), String.raw`$\frac{1}{2}mv^2 + \theta + \tau + \beta + \rho + \left(x\right)$`)
  for (const input of ['a\tb\nNext paragraph', '`\text{kg}`', '```json\n"\times"\n```',
    '~~~tex\n\\theta\n~~~', '``\\frac{1}{2}``', '$x\t y$', '$x\ttexture$']) {
    assert.equal(format(input), input)
  }
})

test('LaTeX delimiters are normalized without fragmenting full equations', () => {
  assert.equal(format(String.raw`Use \(p_f = m v_f\) here.`), 'Use $p_f = m v_f$ here.')
  assert.equal(format(String.raw`\[K_f = \frac{1}{2}m v_f^2\]`), '\n\n$$\nK_f = \\frac{1}{2}m v_f^2\n$$\n\n')
  assert.equal(format('`\\(p_f\\)`'), '`\\(p_f\\)`')
  assert.equal(format(String.raw`\(p=mv`), String.raw`\(p=mv`)
  const displayed = format('Substitute: $$p=5.0 \\times 10=50\\text{ kg m/s}$$ east.')
  assert.equal(displayed, 'Substitute: \n\n$$\np=5.0 \\times 10=50\\text{ kg m/s}\n$$\n\n east.')
  assert.equal(format(displayed), displayed)
})
