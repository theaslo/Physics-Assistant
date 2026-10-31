"""Reusable exact linear-system solving and physics validation for Math MCP."""

import ast
import hashlib
import json
import math
import re
from fractions import Fraction

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



def _plain_math(text):
    """Normalize the notation emitted to the chat back to calculator symbols."""
    text = text.replace("$", "")
    text = re.sub(r"([A-Za-z][A-Za-z0-9]*)_\{([A-Za-z0-9]+)\}", r"\1_\2", text)
    return text.replace(r"\cdot", "*").replace(r"\times", "*")


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
        try:
            for side in part.split("="):
                _expression(side)
        except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
            return []
        equations.append(part)
    return equations


def _expression(text):
    """Read only linear arithmetic; never execute input expressions."""
    text = re.sub(r"(?<=[\d)])\s*(?=[A-Za-z])", "*", text.strip())

    def read(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return {"": Fraction(str(node.value))}
        if isinstance(node, ast.Name) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node.id):
            return {node.id: Fraction(1)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return {key: sign * value for key, value in read(node.operand).items()}
        if isinstance(node, ast.BinOp):
            left, right = read(node.left), read(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                sign = -1 if isinstance(node.op, ast.Sub) else 1
                return {key: left.get(key, 0) + sign * right.get(key, 0)
                        for key in left.keys() | right.keys()}
            if isinstance(node.op, ast.Mult):
                if set(left) <= {""}:
                    return {key: left.get("", 0) * value for key, value in right.items()}
                if set(right) <= {""}:
                    return {key: right.get("", 0) * value for key, value in left.items()}
            if isinstance(node.op, ast.Div) and set(right) <= {""}:
                return {key: value / right[""] for key, value in left.items()}
        raise ValueError("Expected a linear expression")

    return read(ast.parse(text, mode="eval").body)


def equation_fingerprint(equations):
    sides = [tuple(map(_expression, equation.split("="))) for equation in equations]
    variables = sorted(set().union(*(set(left) | set(right) for left, right in sides)) - {""})
    rows = []
    for left, right in sides:
        row = [left.get(symbol, 0) - right.get(symbol, 0) for symbol in variables]
        row.append(right.get("", 0) - left.get("", 0))
        pivot = next((value for value in row if value), None)
        if pivot is None:
            raise ValueError("A zero equation does not constrain the unknowns")
        rows.append(tuple(Fraction(value) / pivot for value in row))
    canonical = json.dumps([[str(value) for value in row] for row in sorted(rows)])
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_generated_problem(candidate, allowed_topics):
    """Validate model data and compute an exact, independently checked solution."""
    if not isinstance(candidate, dict) or candidate.get("topic") not in allowed_topics:
        raise ValueError("Use one of the requested physics topics")
    equations = candidate.get("equations")
    kinds = candidate.get("quantities")
    if not isinstance(equations, list) or len(equations) != 2:
        raise ValueError("Provide exactly two equations")
    if not isinstance(kinds, dict) or len(kinds) != 2:
        raise ValueError("Declare exactly two physical unknowns")
    quantities = {}
    normalized_kinds = {}
    for symbol, kind in kinds.items():
        kind = re.sub(r"[\s-]+", "_", kind.strip().lower()) if isinstance(kind, str) else kind
        aliases = {"kinetic_energy": "energy", "potential_energy": "energy",
                   "initial_velocity": "velocity", "final_velocity": "velocity",
                   "electric_current": "current", "electrical_current": "current",
                   "electric_potential": "voltage", "linear_momentum": "momentum"}
        original_kind = kind
        kind = aliases.get(kind, kind) if isinstance(kind, str) else kind
        if (not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z](?:_[A-Za-z0-9]{1,3})?", symbol)
                or not isinstance(kind, str) or kind not in QUANTITIES
                or symbol.split("_")[0] not in QUANTITIES[kind][1]):
            raise ValueError("Use physical symbols matched to their quantity types, not x and y")
        label = original_kind.replace("_", " ").capitalize() if original_kind in aliases else QUANTITIES[kind][0]
        quantities[symbol] = (label, "")
        normalized_kinds[symbol] = kind
    variables = sorted(quantities)
    normalized = []
    rows = []
    for equation in equations:
        if isinstance(equation, dict):
            coefficients = equation.get("coefficients")
            rhs = equation.get("rhs")
            if not isinstance(coefficients, dict) or set(coefficients) != set(quantities):
                raise ValueError("Each coefficient object must contain exactly the two declared symbols")
            if any(type(value) not in (int, float) or abs(value) > 100000 or not math.isfinite(value)
                   for value in [*coefficients.values(), rhs]):
                raise ValueError("All coefficients and right-hand sides must be finite numbers")
            terms = []
            for symbol in quantities:
                coefficient = coefficients[symbol]
                sign = "-" if coefficient < 0 else ("+" if terms else "")
                magnitude = abs(coefficient)
                term = symbol if magnitude == 1 else f"{magnitude}*{symbol}"
                terms.append(f"{sign} {term}".strip())
            equation = f"{' '.join(terms)} = {rhs}"
        if (not isinstance(equation, str) or len(equation) > 160 or equation.count("=") != 1
                or not re.fullmatch(r"[A-Za-z0-9_+*/().= \-]+", equation)
                or re.search(r"\d{7}", equation)):
            raise ValueError("Use short plain linear equations without units, prose, or LaTeX")
        parsed = extract_equations(equation)
        if len(parsed) != 1:
            raise ValueError("Each equation must be linear and parseable")
        equation = parsed[0]
        left, right = map(_expression, equation.split("="))
        if (set(left) | set(right)) - {""} != set(variables):
            raise ValueError("Both equations must use the same two declared unknowns")
        row = [left.get(symbol, 0) - right.get(symbol, 0) for symbol in variables]
        if any(coefficient == 0 for coefficient in row):
            raise ValueError("Each equation must constrain both unknowns")
        row.append(right.get("", 0) - left.get("", 0))
        rows.append(row)
        normalized.append(equation)
    calculation = solve_system(normalized, quantities)
    if calculation["status"] != "unique":
        raise ValueError("The two equations must have one unique solution")
    for symbol, raw_value in calculation["solution"].items():
        value = Fraction(raw_value)
        if abs(value) > 100000 or value.denominator > 1000:
            raise ValueError("Choose manageable solutions for students")
        if normalized_kinds[symbol] in {"mass", "time", "resistance", "frequency"} and value <= 0:
            raise ValueError("Mass, time, resistance, and frequency must be positive")
    # Ignore row order and scalar multiples when identifying repeated systems.
    fingerprint = equation_fingerprint(normalized)
    return {"topic": candidate["topic"], "title": TOPICS[candidate["topic"]][0],
            "equations": normalized, "quantities": quantities, "calculation": calculation}, fingerprint


def solve_system(equations, quantities=None):
    """Return exact solutions, elimination data and substitution checks as JSON-safe data."""
    if not isinstance(equations, list) or len(equations) != 2:
        raise ValueError("Expected exactly two equations")
    normalized, rows, sides = [], [], []
    for equation in equations:
        if not isinstance(equation, str) or len(equation) > 240:
            raise ValueError("Expected short equation strings")
        parsed = extract_equations(equation)
        if len(parsed) != 1:
            raise ValueError("Expected a linear equation")
        equation = parsed[0]
        left, right = map(_expression, equation.split("="))
        sides.append((left, right))
        normalized.append(equation)
        rows.append({key: left.get(key, 0) - right.get(key, 0) for key in left.keys() | right.keys()})
    variables = sorted(set().union(*(row.keys() for row in rows)) - {""})
    if len(variables) != 2:
        raise ValueError("Expected exactly two unknowns")
    if quantities is not None:
        if not isinstance(quantities, dict) or not set(quantities) <= set(variables):
            raise ValueError("Quantity labels must refer to the solved variables")
        if any(not isinstance(label, (list, tuple)) or len(label) != 2
               or not all(isinstance(part, str) for part in label) for label in quantities.values()):
            raise ValueError("Each quantity label must contain a name and a unit string")
    if quantities and set(quantities) == set(variables):
        variables = list(quantities)
    u, v = variables
    a, b, c = rows[0].get(u, 0), rows[0].get(v, 0), -rows[0].get("", 0)
    d, e, f = rows[1].get(u, 0), rows[1].get(v, 0), -rows[1].get("", 0)
    determinant = a*e-b*d
    result = {"equations": normalized, "variables": variables, "quantities": quantities or {},
              "elimination": {key: str(value) for key, value in
                              dict(a=a,b=b,c=c,d=d,e=e,f=f,determinant=determinant,numerator=c*e-b*f).items()}}
    if determinant == 0:
        inconsistent = (a*f != c*d or b*f != c*e or (a == b == 0 and c != 0) or (d == e == 0 and f != 0))
        result.update(status="inconsistent" if inconsistent else "dependent", solution={}, checks=[])
        return result
    values = {u: Fraction(c*e-b*f)/determinant, v: Fraction(a*f-c*d)/determinant}
    p, q, r = (a,b,c) if b else (d,e,f)
    result["substitution"] = {"p": str(p), "q": str(q), "r": str(r)}
    checks = []
    for equation, (left, right) in zip(normalized, sides):
        lhs = sum(value*values.get(key, 1) for key, value in left.items())
        rhs = sum(value*values.get(key, 1) for key, value in right.items())
        if lhs != rhs:
            raise ValueError("Computed solution does not satisfy both original equations")
        checks.append({"equation": equation, "lhs": str(lhs), "rhs": str(rhs), "satisfied": True})
    result.update(status="unique", solution={symbol: str(value) for symbol,value in values.items()}, checks=checks,
                  fingerprint=equation_fingerprint(normalized))
    return result


def solve_linear_systems(systems):
    """Solve a bounded batch, retaining per-item errors and singular classifications."""
    if not isinstance(systems, list) or not 1 <= len(systems) <= 20:
        raise ValueError("Provide between one and twenty systems")
    results = []
    for index, system in enumerate(systems):
        try:
            if not isinstance(system, dict):
                raise ValueError("Expected a system object")
            result = solve_system(system.get("equations"), system.get("quantities"))
            results.append({"index": index, "ok": True, "calculation": result})
        except (ValueError, SyntaxError, ZeroDivisionError, RecursionError, TypeError) as exc:
            results.append({"index": index, "ok": False, "error": str(exc)})
    return {"schema_version": 1, "results": results}


def validate_physics_equations(problems, allowed_topics, exclude_fingerprints=None, previous_systems=None):
    """Validate and solve generated candidates, rejecting repeated mathematical systems."""
    if not isinstance(problems, list) or len(problems) > 10:
        raise ValueError("Provide at most ten candidate problems")
    if not isinstance(allowed_topics, list) or not allowed_topics or any(topic not in TOPICS for topic in allowed_topics):
        raise ValueError("Provide valid allowed physics topics")
    seen = set((exclude_fingerprints or [])[-100:])
    for equations in (previous_systems or [])[-30:]:
        try:
            seen.add(equation_fingerprint(equations))
        except (ValueError, SyntaxError, ZeroDivisionError, RecursionError, TypeError):
            continue
    results = []
    for index, candidate in enumerate(problems):
        try:
            problem, fingerprint = validate_generated_problem(candidate, allowed_topics)
            if fingerprint in seen:
                raise ValueError("Generate a new system rather than repeating an earlier one")
            seen.add(fingerprint)
            results.append({"index": index, "accepted": True, "problem": problem, "fingerprint": fingerprint})
        except (ValueError, SyntaxError, ZeroDivisionError, RecursionError, TypeError) as exc:
            results.append({"index": index, "accepted": False, "error": str(exc)})
    return {"schema_version": 1, "results": results}
