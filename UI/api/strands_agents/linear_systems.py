"""Conversation parsing and LaTeX presentation for Math MCP system results."""

import re

def is_system_request(text):
    return bool(re.search(
        r"(?:two|2)[ -]+equations?\b.*\b(?:two|2)[ -]+(?:unknowns?|variables?)\b"
        r"|\b(?:simultaneous equations?|systems? of (?:linear )?equations?)\b",
        text, re.I | re.S,
    ))


def _plain_math(text):
    """Normalize the notation emitted to the chat back to calculator symbols."""
    text = text.replace("$", "")
    text = re.sub(r"([A-Za-z][A-Za-z0-9]*)_\{([A-Za-z0-9]+)\}", r"\1_\2", text)
    return text.replace(r"\cdot", "*").replace(r"\times", "*")


def latex_math(expression):
    expression = re.sub(r"([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9]+)", r"\1_{\2}", expression)
    return expression.replace("*", r" \cdot ")


def extract_equations(text):
    text = _plain_math(text)
    text = text.replace("−", "-").replace("×", "*")
    equations = []
    for part in re.split(r"[\n;,]|\band\b", text):
        if "=" not in part:
            continue
        part = part.rsplit(":", 1)[-1].strip()
        part = re.sub(r"^\d+[.)]\s+", "", part)
        part = re.sub(r"^solve\s+(?:(?:the\s+)?system\s+)?", "", part, flags=re.I)
        part = re.sub(r"\s+for\s+.*$", "", part, flags=re.I).rstrip(" .?!")
        if part.count("=") != 1:
            return []
        equations.append(part)
    return equations


def practice_systems(worksheet):
    """Read equations and quantity definitions from the actual saved worksheet."""
    worksheet = _plain_math(worksheet)
    systems = []
    for block in re.split(r"^\d+\. ", worksheet.split("Answer key:")[0], flags=re.M)[1:]:
        equations = extract_equations(block)
        if len(equations) != 2:
            continue
        quantities = {
            variable: (label, unit or "")
            for variable, label, unit in re.findall(
                r"^[ \t]*([A-Za-z][A-Za-z0-9_]*): (.+?)(?: \[([^\]]+)\])?[ \t]*$", block, re.M
            )
        }
        systems.append((equations, quantities))
    return systems


TOPICS = {
    "momentum": ("Momentum", r"\b(momentum|collisions?|impulse)\b"),
    "forces": ("Forces", r"\b(forces?|newton|pulley|tension)\b"),
    "kinematics": ("Kinematics", r"\b(kinematics|motion|velocity|acceleration)\b"),
    "energy": ("Energy and power", r"\b(energy|power|work|heaters?)\b"),
    "circuits": ("Circuits", r"\b(circuits?|currents?|resistors?|electricity)\b"),
    "torque": ("Torque", r"\b(torque|rotation|beams?)\b"),
    "thermal": ("Thermal physics", r"\b(thermal|thermodynamics|heat|temperature)\b"),
}

# These describe physical quantities, not equation templates.
QUANTITIES = {
    "velocity": ("Velocity", {"v", "u"}),
    "acceleration": ("Acceleration", {"a"}),
    "mass": ("Mass", {"m"}),
    "force": ("Force", {"F"}),
    "tension": ("Tension", {"T"}),
    "power": ("Power", {"P"}),
    "current": ("Current", {"I"}),
    "voltage": ("Voltage", {"V"}),
    "momentum": ("Momentum", {"p"}),
    "energy": ("Energy", {"E", "K", "U"}),
    "work": ("Work", {"W"}),
    "heat": ("Heat", {"Q"}),
    "time": ("Time", {"t"}),
    "distance": ("Distance", {"d", "s"}),
    "resistance": ("Resistance", {"R"}),
    "temperature": ("Temperature", {"T"}),
    "frequency": ("Frequency", {"f"}),
}



def requested_topics(request):
    topics = [topic for topic, (_, pattern) in TOPICS.items() if re.search(pattern, request, re.I)]
    if not topics or re.search(r"\b(mixed|variety|different topics|not limited)\b", request, re.I):
        return list(TOPICS)
    return topics


def render_solution(calculation):
    """Present structured Math MCP results; perform no numerical calculations."""
    u, v = calculation["variables"]
    step = calculation["elimination"]
    a,b,c,d,e,f = (step[key] for key in ("a","b","c","d","e","f"))
    lines = ["Solve the system:", ""]
    for equation in calculation["equations"]:
        lines += ["$$", latex_math(equation), "$$", ""]
    lines += ["Elimination:",
              f"Multiply equation 1 by {e} and equation 2 by {b}, then subtract.",
              f"({a} × {e} - {b} × {d}){u} = {c} × {e} - {b} × {f}"]
    if calculation["status"] != "unique":
        lines.append("No solution: the equations are inconsistent." if calculation["status"] == "inconsistent"
                     else "Infinitely many solutions: the equations are dependent.")
        return "\n".join(lines)
    first, second = calculation["solution"][u], calculation["solution"][v]
    p,q,r = (calculation["substitution"][key] for key in ("p","q","r"))
    lines += [f"{step['determinant']}{u} = {step['numerator']}, so {u} = {first}.",
              f"Substitute {u} = {first}: ({p})({first}) + ({q}){v} = {r}.",
              f"{v} = ({r} - ({p})({first})) / ({q}) = {second}.",
              "", f"Answer: {u} = {first}, {v} = {second}", "", "Check by substitution:"]
    for check in calculation["checks"]:
        lines.append(f"{check['equation']}: {check['lhs']} = {check['rhs']} ✓")
    quantities = calculation.get("quantities") or {}
    if quantities:
        lines += ["", "Physical answer:"]
        for symbol, (label, unit) in quantities.items():
            lines.append(f"{label}: {symbol} = {calculation['solution'][symbol]} {unit}".rstrip())
    return "\n".join(lines)


def render_practice(problems, include_answers):
    """Render only Math MCP-approved problems and calculator-provided solutions."""
    lines = ["Two Equations, Two Unknowns Practice", "", "Solve each pair for the two quantities listed.", ""]
    for index, problem in enumerate(problems, 1):
        indent = " " * (len(str(index)) + 2)
        lines += [f"{index}. {problem['title']}  "]
        lines += [f"{indent}${latex_math(variable)}$: {label}  "
                  for variable, (label, _) in problem["quantities"].items()]
        for equation in problem["equations"]:
            lines += ["", f"{indent}$$", f"{indent}{latex_math(equation)}", f"{indent}$$"]
        lines.append("")
    if include_answers:
        lines.append("Answer key:")
        for index, problem in enumerate(problems, 1):
            lines += [f"Problem {index}", render_solution(problem["calculation"]), ""]
    else:
        lines.append("Send your answers for checking, or ask me to solve a problem by number.")
    return "\n".join(lines)
