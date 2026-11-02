/** Keep uncertainty and hint-only requests in guided tutoring. */
export function isFullSolutionRequest(message: string): boolean {
  const text = message.trim().replace(/’/g, "'")
  if (/\b(?:(?:don't|do not|dont)\s+(?:show|give|reveal|provide|tell|want)|without|no)\b[^.!?]{0,60}\b(?:solution|answer)\b|\b(?:hints? only|only (?:a )?hint|no spoilers)\b/i.test(text)) {
    return false
  }
  return /\bfull(?: worked)? solution\b|\b(?:show|give|provide)(?: me)?(?: the| a)?(?: full| complete| worked)* (?:solution|answer)\b/i.test(text)
}
