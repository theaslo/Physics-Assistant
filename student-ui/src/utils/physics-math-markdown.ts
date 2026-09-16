type PhysicsMathReplacement = {
  pattern: RegExp
  latex: string | ((match: string, ...groups: string[]) => string)
}

const PROTECTED_MARKDOWN_SEGMENT = /(```[\s\S]*?```|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$)/g

const PHYSICS_MATH_REPLACEMENTS: PhysicsMathReplacement[] = [
  { pattern: /\\frac\{[^{}]*\}\{[^{}]*\}/g, latex: (match) => match },
  { pattern: /\\sqrt\{[^{}]*\}/g, latex: (match) => match },
  { pattern: /\\(?:sin|cos)\s*\([^)]*\)/g, latex: (match) => match },
  { pattern: /\\Delta\s*x\b/g, latex: '\\Delta x' },
  { pattern: /\bDelta\s+x\b/g, latex: '\\Delta x' },
  { pattern: /\\Delta\s*t\b/g, latex: '\\Delta t' },
  { pattern: /\bDelta\s+t\b/g, latex: '\\Delta t' },
  { pattern: /\\theta\b/g, latex: '\\theta' },
  { pattern: /\btheta\b/g, latex: '\\theta' },
  { pattern: /\\sum\s*F\b/g, latex: '\\sum F' },
  { pattern: /\bsum\s*F\b/g, latex: '\\sum F' },
  { pattern: /\bF_\{?net\}?\b/g, latex: 'F_{net}' },
  { pattern: /\bf_\{?k\}?\b/g, latex: 'f_k' },
  { pattern: /\\mu_\{?k\}?\b/g, latex: '\\mu_k' },
  { pattern: /\\vec\{([A-Za-z])\}/g, latex: (_match, symbol) => `\\vec{${symbol}}` },
  {
    pattern: /\b([xv])_\{?(0x|0y|fx|fy|0|f|i|x|y)\}?(?:\^2)?\b/g,
    latex: (match, symbol, subscript) => `${symbol}_{${subscript}}${match.endsWith('^2') ? '^2' : ''}`,
  },
  { pattern: /\bt\^2\b/g, latex: 't^2' },
]

function isProtectedMarkdownSegment(segment: string) {
  return segment.startsWith('```') || segment.startsWith('`') || segment.startsWith('$$') || (
    segment.startsWith('$') && segment.endsWith('$')
  )
}

function addPhysicsMathDelimiters(segment: string) {
  const mathSnippets: string[] = []
  const stashMath = (latex: string) => {
    const token = `@@PHYSICS_MATH_${mathSnippets.length}@@`
    mathSnippets.push(`$${latex}$`)
    return token
  }

  let formatted = segment
  PHYSICS_MATH_REPLACEMENTS.forEach((replacement) => {
    formatted = formatted.replace(replacement.pattern, (match, ...groups) => {
      const latex = typeof replacement.latex === 'function'
        ? replacement.latex(match, ...groups.map(String))
        : replacement.latex
      return stashMath(latex)
    })
  })

  return formatted.replace(/@@PHYSICS_MATH_(\d+)@@/g, (token, index) => mathSnippets[Number(index)] || token)
}

export function formatPhysicsMathMarkdown(content: string) {
  return content
    .split(PROTECTED_MARKDOWN_SEGMENT)
    .map((segment) => (isProtectedMarkdownSegment(segment) ? segment : addPhysicsMathDelimiters(segment)))
    .join('')
}
