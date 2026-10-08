"""Helpers for Newton's-laws quiz reflection prompts."""

import re
from typing import Optional


def is_force_quiz_reflection_prompt(agent_id: str, problem: str) -> bool:
    """Detect Newton-law quiz reflection prompts that should not get an MCQ gate."""
    if agent_id != "forces_agent":
        return False

    lower = (problem or "").lower()
    if not lower.strip():
        return False

    newton_context = any(
        re.search(pattern, lower)
        for pattern in (
            r"newton'?s?\s*(?:1st|first|2nd|second)\s+law",
            r"\bfirst\s+law\b",
            r"\bsecond\s+law\b",
            r"\binertia\b",
            r"\bnet\s+force\b",
            r"\bf\s*=\s*m\s*a\b",
            r"\bsum\s*f\b",
            r"\bΣ\s*f\b",
        )
    )
    reflection_context = any(
        re.search(pattern, lower)
        for pattern in (
            r"\bquiz\b",
            r"\btest\b",
            r"\bexam\b",
            r"\bmistake\b",
            r"\bwrong\b",
            r"\bincorrect\b",
            r"\bmissed\b",
            r"\breflection\b",
            r"\banaly[sz]e\b",
            r"\bmy\s+answer\b",
            r"\bcorrect\s+answer\b",
            r"\bwhy\s+(?:i|did|was)\b",
        )
    )

    return newton_context and reflection_context


def infer_newton_reflection_focus(problem: str) -> str:
    lower = (problem or "").lower()
    if re.search(
        r"newton'?s?\s*(?:1st|first)\s+law|\bfirst\s+law\b|\binertia\b|constant\s+velocity|at\s+rest",
        lower,
    ):
        return "newton_first_law"
    if re.search(
        r"newton'?s?\s*(?:2nd|second)\s+law|\bsecond\s+law\b|\bf\s*=\s*m\s*a\b|\bsum\s*f\b|\bnet\s+force\b|\baccelerat",
        lower,
    ):
        return "newton_second_law"
    return "newton_laws"


def build_force_quiz_reflection_response(
    problem: str,
    *,
    concept_tag: Optional[str] = None,
    tool_note: Optional[str] = None,
) -> str:
    """Build a reflection-first response for Newton quiz mistake analysis."""
    focus = concept_tag or infer_newton_reflection_focus(problem)
    lower = (problem or "").lower()
    has_student_answer = any(marker in lower for marker in ("my answer", "i answered", "i chose", "i picked", "i said"))
    has_correct_answer = "correct answer" in lower or "right answer" in lower or "answer key" in lower
    has_question = "question" in lower or "prompt" in lower or "asked" in lower

    if focus == "newton_first_law":
        concept_focus = (
            "Newton's First Law: an object at rest stays at rest, and an object moving at constant velocity keeps moving "
            "unless a nonzero net external force changes its motion."
        )
        common_misconception = (
            "A common mistake is thinking motion requires a forward net force. Constant velocity means the net force is zero, "
            "not that there must be a force in the direction of motion."
        )
    elif focus == "newton_second_law":
        concept_focus = (
            "Newton's Second Law: acceleration is determined by the net external force on the chosen object, "
            "so the useful equation is sum F = ma."
        )
        common_misconception = (
            "A common mistake is using one individual force as ma instead of adding forces with directions first. "
            "Balanced forces give zero acceleration even if the object is moving."
        )
    else:
        concept_focus = (
            "Newton's laws reflection: first decide whether the question is about constant motion/no net force "
            "or about acceleration caused by a net force."
        )
        common_misconception = (
            "A common mistake is mixing up velocity and acceleration: motion itself does not prove there is a nonzero net force."
        )

    targeted_feedback = _targeted_reflection_feedback(lower, focus)
    details_prompt = ""
    if not (has_question and has_student_answer and has_correct_answer):
        details_prompt = (
            "\n\nTo analyze your exact mistake, paste these four parts:\n"
            "1. The quiz question.\n"
            "2. Your original answer.\n"
            "3. The correct answer or answer-key reasoning.\n"
            "4. Why you chose your answer at the time."
        )

    sections = [
        "Newton's Laws Quiz Reflection Mode",
        "I will not start by giving you a new multiple-choice force check. Instead, we will analyze the quiz mistake you already made.",
        f"Concept focus: {concept_focus}",
        f"Likely misconception to check: {common_misconception}",
    ]
    if targeted_feedback:
        sections.append(f"For your answer: {targeted_feedback}")
    if tool_note:
        sections.append(f"MCP force-balance check: {tool_note}")
    sections.append(
        "Use this reflection structure:\n"
        "- My original thinking was: ...\n"
        "- The correct physics idea is: ...\n"
        "- The mistake in my reasoning was: ...\n"
        "- Next time, I will check: net force, acceleration, and whether the motion is constant or changing."
    )
    sections.append(
        "If you already included the quiz question, your answer, and the correct answer, send them in that format and I will help you tighten the reflection."
        f"{details_prompt}"
    )
    return "\n\n".join(sections)


def _targeted_reflection_feedback(lower: str, focus: str) -> str:
    if "biggest force" in lower or "largest force" in lower:
        return (
            "that points to the key issue: Newton's Second Law uses the vector sum of all external forces, "
            "not the biggest single force."
        )
    if "forward force" in lower and ("constant velocity" in lower or focus == "newton_first_law"):
        return (
            "that points to the key issue: constant velocity does not require a forward net force; "
            "it requires zero net external force."
        )
    if "moving" in lower and ("force" in lower or "net force" in lower) and focus == "newton_first_law":
        return "watch the difference between moving and changing motion; only changing motion requires a nonzero net force."
    if "mass" in lower and "acceleration" in lower and focus == "newton_second_law":
        return "make sure the acceleration comes from the net external force on the object, using sum F = ma."
    return ""
