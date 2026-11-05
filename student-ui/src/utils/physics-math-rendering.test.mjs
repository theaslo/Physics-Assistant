import test from 'node:test'
import assert from 'node:assert/strict'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { formatPhysicsMathMarkdown } from './physics-math-markdown.ts'

function render(content) {
  return renderToStaticMarkup(React.createElement(ReactMarkdown, {
    remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex],
    children: formatPhysicsMathMarkdown(content),
  }))
}

test('momentum answer renders real math, units and multiplication rather than code', () => {
  for (const content of [
    String.raw`Use $p = mv$. Given $m = 5.0\text{ kg}$ and $v = 10\text{ m/s}$ east, $p=5.0\times 10=50\text{ kg m/s}$ east.`,
    'Use $p = mv$. Given $m = 5.0\text{ kg}$ and $v = 10\text{ m/s}$ east, $p=5.0\times 10=50\text{ kg m/s}$ east.',
    String.raw`Use \(p = mv\). Given \(m = 5.0\text{ kg}\) and \(v = 10\text{ m/s}\) east, \(p=5.0\times 10=50\text{ kg m/s}\) east.`,
  ]) {
    const html = render(content)
    assert.match(html, /class="katex"/)
    assert.match(html, /<mtext>[^<]*kg[^<]*<\/mtext>/)
    assert.match(html, /<mo>\u00d7<\/mo>/)
    assert.doesNotMatch(html, /katex-error|<code|<mi>e<\/mi><mi>x<\/mi><mi>t<\/mi>/)
  }
})

test('display equations render fractions, subscripts and units for other agents', () => {
  for (const content of [
    '$$\nK_f = \frac{1}{2}m v_f^2 = 25\text{ J}\n$$',
    String.raw`\[\omega_f = \omega_i + \alpha t = 14\text{ rad/s}\]`,
  ]) {
    const html = render(content)
    assert.match(html, /class="katex-display"/)
    assert.match(html, /<msub>/)
    assert.match(html, /<mtext>/)
    assert.doesNotMatch(html, /katex-error|<code/)
  }
})

test('inline-positioned display math from the live model is placed on its own line', () => {
  const html = render(String.raw`Substituting gives $$ p = (5.0\text{ kg})(10\text{ m/s}) = 50\text{ kg m/s} $$ The direction is east.`)
  assert.match(html, /class="katex-display"/)
  assert.match(html, /<p>The direction is east\.<\/p>/)
  assert.doesNotMatch(html, /katex-error|<code/)
})

test('rotation and thermodynamics worked substitutions render signed values and units', () => {
  for (const content of [
    String.raw`$$\begin{gathered}\omega_f=2+(3)(4)=14\,\mathrm{rad/s}\\\theta_f=(2)(4)+\frac12(3)(4)^2=32\,\mathrm{rad}\end{gathered}$$`,
    String.raw`$$\begin{gathered}\Delta U=(0\,\mathrm{J})-(-200\,\mathrm{J})=200\,\mathrm{J}\end{gathered}$$`,
    String.raw`$$\eta_{\max}=1-\frac{300\,\mathrm{K}}{600\,\mathrm{K}}=0.5$$`,
  ]) {
    const html = render(content)
    assert.match(html, /class="katex/)
    assert.doesNotMatch(html, /katex-error|<code/)
  }
})

test('a heat result with bare LaTeX units renders as quantities instead of raw commands', () => {
  const html = render(String.raw`-41860 \, \text{J} = -41.86 \, \text{kJ}`)
  assert.match(html, /class="katex"/)
  assert.match(html, /<mtext>J<\/mtext>/)
  assert.match(html, /<mtext>kJ<\/mtext>/)
  assert.doesNotMatch(html, /katex-error|<code/)
})
