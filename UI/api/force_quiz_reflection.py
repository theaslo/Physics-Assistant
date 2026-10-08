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
            r"\bnewton(?:['’]s|s)?\s+laws?\b",
            r"newton(?:['’]s|s)?\s*(?:1st|first|2nd|second)\s+laws?",
            r"\bfirst\s+law\b",
            r"\bsecond\s+law\b",
            r"\binertia\b",
            r"\bnet\s+force\b",
            r"\bf\s*=\s*m\s*a\b",
            r"\bsum\s*f\b",
            r"\b[σΣ]\s*f\b",
        )
    )
    physics_quiz_context = any(
        re.search(pattern, lower)
        for pattern in (
            r"\bhockey\b",
            r"\bpuck\b",
            r"\bstick\b",
            r"\bfrictionless\b",
            r"\bno\s+friction\b",
            r"\bconstant\s+speed\b",
            r"\bslow\s+down\b",
            r"\bspeed\s+up\b",
            r"\bforces?\s+balance\b",
            r"\bcrate\b",
            r"\bthree\s+forces\b",
            r"\bcomponents?\b",
        )
    )
    reflection_context = any(
        re.search(pattern, lower)
        for pattern in (
            r"\bquiz\b",
            r"\btest\b",
            r"\bexam\b",
            r"\breview(?:ing)?\b",
            r"\bmistake\b",
            r"\bwrong\b",
            r"\bincorrect\b",
            r"\bmissed\b",
            r"\bmisunderstanding\b",
            r"\breasoning\b",
            r"\breflection\b",
            r"\banaly[sz]e\b",
            r"\bmy\s+answer\b",
            r"\bcorrect\s+answer\b",
            r"\bhelp\s+me\s+identify\b",
            r"\bwhy\s+(?:i|did|was)\b",
        )
    )
    submitted_quiz_answer = (
        bool(re.search(r"\b(?:i\s+said|i\s+answered|i\s+chose|my\s+answer)\s+[a-d]\b", lower))
        or bool(re.search(r"\b[abcd]\)\s+", lower))
        or bool(re.search(r"\(\s*\d+\s*pts?\s*\)", lower))
    ) and any(keyword in lower for keyword in ("because", "reasoning", "i said", "my answer", "i answered", "i chose"))

    return (newton_context and reflection_context) or (physics_quiz_context and submitted_quiz_answer)


def infer_newton_reflection_focus(problem: str) -> str:
    lower = (problem or "").lower()
    if re.search(
        r"newton(?:['’]s|s)?\s*(?:1st|first)\s+laws?|\bfirst\s+law\b|\binertia\b|constant\s+velocity|at\s+rest|\bpuck\b|\bfrictionless\b|\bno\s+friction\b|\bconstant\s+speed\b|\bslow\s+down\b|\bspeed\s+up\b",
        lower,
    ):
        return "newton_first_law"
    if re.search(
        r"newton(?:['’]s|s)?\s*(?:2nd|second)\s+laws?|\bsecond\s+law\b|\bf\s*=\s*m\s*a\b|\bsum\s*f\b|\bnet\s+force\b|\baccelerat",
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
    """Build the homework-oriented first turn for Newton quiz reflection."""
    lower = (problem or "").lower()
    has_quiz_details = _has_quiz_submission_details(lower)

    sections = [
        "Quiz 4 Reflection with Physics AI Tutor",
        (
            "Let's work through your Quiz 4 reflection one quiz question at a time. "
            "I will ask questions and give hints before revealing an answer."
        ),
    ]

    if has_quiz_details:
        sections.append(
            f"First question: {_first_reflection_question(lower)}"
        )
        sections.append(
            "Reply with that one answer first. After that, I will ask the next question and help you check your reasoning."
        )
    else:
        sections.append(
            "First question: Which missed Quiz 4 question do you want to review first?"
        )
        sections.append(
            "Paste that one quiz question with your original answer and your original reasoning. "
            "Keep your original reasoning visible even if it was incorrect."
        )

    return "\n\n".join(sections)


def _has_quiz_submission_details(lower: str) -> bool:
    detail_markers = (
        "quiz question:",
        "question:",
        "q1",
        "q2",
        "q3",
        "q4",
        "my original answer:",
        "my answer:",
        "my reasoning:",
        "i answered",
        "i chose",
        "i picked",
        "i said",
        "because",
        "(5 pts)",
    )
    return any(marker in lower for marker in detail_markers)


def _first_reflection_question(lower: str) -> str:
    if "puck" in lower or "hockey" in lower:
        return "After the puck loses contact with the stick, what horizontal forces are still acting on the puck?"
    if "crate" in lower:
        return "For the crate, what forces act on it while it remains at rest?"
    if "three forces" in lower or "component" in lower:
        return "For equilibrium, what must be true about the sum of the x-components and the sum of the y-components?"
    if "speed" in lower and "net force" in lower:
        return "If the net force points in the direction of motion, what happens to the object's speed while that net force is present?"
    return "What object or system is the quiz question asking about?"


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
