type PhysicsMathReplacement = {
  pattern: RegExp
  latex: string | ((match: string, ...groups: string[]) => string)
}

const PROTECTED_MARKDOWN_SEGMENT = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`+[^`\n]*`+|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))/g

const PHYSICS_MATH_REPLACEMENTS: PhysicsMathReplacement[] = [
  { pattern: /\\frac\{[^{}]*\}\{[^{}]*\}/g, latex: (match) => match },
  { pattern: /\\sqrt\{[^{}]*\}/g, latex: (match) => match },
  { pattern: /\\(?:sin|cos)\s*\([^)]*\)/g, latex: (match) => match },
  { pattern: /\\Delta\s*(?:\\(?:omega|theta)|[pKUEtxy])(?:_(?:\{[^{}]+\}|[a-z0-9]+))?/g, latex: (match) => match },
  { pattern: /\\sum\s*\\tau(?:_(?:\{[^{}]+\}|[a-z0-9]+))?/g, latex: (match) => match },
  { pattern: /\\(?:omega|alpha|theta|tau)(?:_(?:\{[^{}]+\}|[a-z0-9]+))?(?:\^(?:\{[^{}]+\}|[0-9]+))?(?![A-Za-z])/g, latex: (match) => match },
  { pattern: /\b[A-Za-z](?:_(?:\{[^{}]+\}|[a-z0-9]+))?\^(?:\{[^{}]*\}|[0-9]+|[A-Za-z]\b)/g, latex: (match) => match },
  { pattern: /\b[pKULIEJW]_(?:\{[^{}]+\}|[a-z0-9]+)(?:\^(?:\{[^{}]+\}|[0-9]+))?/g, latex: (match) => match },
  { pattern: /\\Delta\s*x\b/g, latex: '\\Delta x' },
  { pattern: /\bDelta\s+x\b/g, latex: '\\Delta x' },
  { pattern: /\\Delta\s*t\b/g, latex: '\\Delta t' },
  { pattern: /\bDelta\s+t\b/g, latex: '\\Delta t' },
  { pattern: /\\theta\b/g, latex: '\\theta' },
  { pattern: /\btheta\b/g, latex: '\\theta' },
  { pattern: /\\sum\s*F\b/g, latex: '\\sum F' },
  { pattern: /\bsum\s*F\b/g, latex: '\\sum F' },
  { pattern: /\bF_(?:\{net\}|net\b)/g, latex: 'F_{net}' },
  { pattern: /\bf_(?:\{k\}|k\b)/g, latex: 'f_k' },
  { pattern: /\\mu_(?:\{k\}|k\b)/g, latex: '\\mu_k' },
  { pattern: /\\vec\{([A-Za-z])\}/g, latex: (_match, symbol) => `\\vec{${symbol}}` },
  {
    pattern: /\b([xv])_(?:\{(?:0x|0y|fx|fy|0|f|i|x|y)\}|(?:0x|0y|fx|fy|0|f|i|x|y)\b)(?:\^2)?/g,
    latex: (match) => match,
  },
  { pattern: /\bt\^2\b/g, latex: 't^2' },
]

function normalizeMathSegment(segment: string) {
  // An LLM can emit a single JSON backslash: \text becomes a tab + "ext".
  // Repair only recognizable commands inside explicit math, never prose/code.
  const repaired = segment
    .replace(/\t(?=(?:ext(?:bf|it|normal|rm)?|imes|heta|au)(?![A-Za-z]))/g, '\\t')
    .replace(/\f(?=rac(?![A-Za-z]))/g, '\\f')
    .replace(/\x08(?=(?:egin|eta|oxed|oldsymbol)(?![A-Za-z]))/g, '\\b')
    .replace(/\r(?=(?:ight|ho)(?![A-Za-z]))/g, '\\r')
  if (repaired.startsWith('\\[')) return `\n\n$$\n${repaired.slice(2, -2).trim()}\n$$\n\n`
  if (repaired.startsWith('\\(')) return `$${repaired.slice(2, -2).trim()}$`
  if (repaired.startsWith('$$') && !(repaired.startsWith('$$\n') && repaired.endsWith('\n$$'))) {
    return `\n\n$$\n${repaired.slice(2, -2).trim()}\n$$\n\n`
  }
  return repaired
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
    .map((segment, index) => {
      if (index % 2 === 0) return addPhysicsMathDelimiters(segment)
      if (segment.startsWith('`') || segment.startsWith('~~~')) return segment
      return normalizeMathSegment(segment)
    })
    .join('')
}
