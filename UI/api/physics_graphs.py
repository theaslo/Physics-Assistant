"""
Reusable graph payload builders for student-facing physics visualizations.

The functions in this module intentionally return JSON-serializable data only.
That keeps graph generation independent from any particular frontend charting
library and makes it simple to test.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPEED_OF_LIGHT = 299_792_458.0
DEFAULT_SAMPLE_COUNT = 61

KINEMATICS_EXAMPLE_PROBLEMS: List[Dict[str, Any]] = [
    {
        "name": "car_accelerates_from_rest",
        "problem": "Graph a car that accelerates from rest at 3 m/s^2 for 5 seconds.",
        "expected_motion_type": "constant_acceleration",
    },
    {
        "name": "structured_cart_motion",
        "problem": "Create position, velocity, and acceleration graphs for this cart.",
        "structured": {"x0": 0, "v0": 2, "a": 1.5, "t": 4},
        "expected_motion_type": "constant_acceleration",
    },
    {
        "name": "runner_constant_velocity",
        "problem": "Graph a runner moving at constant velocity 6 m/s for 8 s.",
        "expected_motion_type": "uniform_motion",
    },
    {
        "name": "train_slows_to_rest",
        "problem": "A train slows from 20 m/s to rest in 10 s. Make the motion graphs.",
        "expected_motion_type": "constant_acceleration",
    },
    {
        "name": "bike_offset_start",
        "problem": "A bike starts 10 m from the origin with initial velocity 4 m/s and acceleration 0.5 m/s^2 for 6 s.",
        "expected_motion_type": "constant_acceleration",
    },
    {
        "name": "constant_velocity_from_displacement",
        "problem": "An object moves 30 m in 5 s at constant velocity. Show the motion graphs.",
        "expected_motion_type": "uniform_motion",
    },
    {
        "name": "cart_reaches_final_velocity",
        "problem": "A cart has initial velocity 2 m/s and acceleration 3 m/s^2 until it reaches 20 m/s. Graph it.",
        "expected_motion_type": "constant_acceleration",
    },
    {
        "name": "car_accelerates_for_distance",
        "problem": "A car accelerates from rest at 2 m/s^2 until it travels 100 m. Generate kinematics graphs.",
        "expected_motion_type": "constant_acceleration",
    },
]


def build_kinematics_graph_response(
    problem: str = "",
    structured: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build graph payloads for 1D introductory kinematics.

    Args:
        problem: Natural-language student question.
        structured: Optional structured inputs. Supported aliases include x0,
            initial_position, x, displacement, v0, initial_velocity, v,
            final_velocity, a, acceleration, t, duration, and time_range.

    Returns:
        A JSON-serializable response with graphs, warnings, errors, and parsed
        input values.
    """
    warnings: List[str] = []
    errors: List[str] = []

    parsed = _parse_kinematics_text(problem, warnings)
    structured_values = _normalize_structured_inputs(structured or {}, warnings)
    values = {**parsed, **structured_values}

    if not values:
        return {
            "graphs": [],
            "warnings": [],
            "errors": ["No kinematics graph inputs were found."],
            "parsed_inputs": {},
        }

    solution = _complete_kinematics_values(values, warnings, errors)
    if errors:
        return {
            "graphs": [],
            "warnings": _unique(warnings),
            "errors": _unique(errors),
            "parsed_inputs": _public_inputs(values),
        }

    payload = _build_kinematics_payload(problem, solution, warnings)
    return {
        "graphs": [payload],
        "warnings": _unique(warnings),
        "errors": [],
        "parsed_inputs": _public_inputs(values),
    }


def should_attempt_kinematics_graph(problem: str, structured: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when a request likely wants or can support kinematics graphs."""
    if structured:
        candidate = _extract_graph_context(structured)
        if candidate:
            return True

    text = problem.lower()
    graph_words = ("graph", "plot", "chart", "diagram", "visualize")
    motion_words = (
        "kinematic",
        "motion",
        "position",
        "velocity",
        "speed",
        "acceleration",
        "accelerates",
        "accelerate",
        "decelerates",
        "slows",
        "from rest",
        "m/s",
        "m/s^2",
    )
    return any(word in text for word in graph_words) or any(word in text for word in motion_words)


def build_kinematics_graphs_from_context(problem: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build graphs from an API problem/context pair."""
    structured = _extract_graph_context(context or {})
    return build_kinematics_graph_response(problem=problem, structured=structured)


def _extract_graph_context(context: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("graph", "graphs", "kinematics", "inputs"):
        candidate = context.get(key)
        if isinstance(candidate, dict):
            return candidate
    if any(key in context for key in _STRUCTURED_ALIASES):
        return context
    return {}


def _parse_kinematics_text(problem: str, warnings: List[str]) -> Dict[str, float]:
    text = _normalize_text(problem)
    values: Dict[str, float] = {}

    if "from rest" in text or "starts from rest" in text or "starting from rest" in text:
        values["v0"] = 0.0

    from_to_rest = re.search(
        r"from\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?\s+to\s+rest",
        text,
    )
    if from_to_rest:
        values["v0"] = float(from_to_rest.group(1))
        values["v"] = 0.0
        _warn_if_missing_unit(warnings, from_to_rest.group(2), "initial velocity", "m/s")

    x0_match = _first_match(
        text,
        [
            r"(?:x0|x_0|initial\s+position|starting\s+position)\s*(?:=|is|of|at)?\s*(-?\d+(?:\.\d+)?)\s*(m|meters?)?",
            r"starts?\s+(-?\d+(?:\.\d+)?)\s*(m|meters?)\s+from\s+(?:the\s+)?origin",
        ],
    )
    if x0_match:
        values["x0"] = float(x0_match.group(1))
        _warn_if_missing_unit(warnings, x0_match.group(2), "initial position", "m")

    v0_match = _first_match(
        text,
        [
            r"(?:v0|v_0|initial\s+velocity|initial\s+speed)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"(?:starts?|starting)\s+(?:at|with)\s+(?:an?\s+)?(?:initial\s+)?(?:velocity|speed)\s+(?:of\s+)?(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"with\s+initial\s+velocity\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
        ],
    )
    if v0_match:
        values["v0"] = float(v0_match.group(1))
        _warn_if_missing_unit(warnings, v0_match.group(2), "initial velocity", "m/s")

    final_velocity_match = _first_match(
        text,
        [
            r"(?:final\s+velocity|final\s+speed|vf|v_f)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"(?:reaches|until\s+it\s+reaches|ending\s+at)\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)",
        ],
    )
    if final_velocity_match:
        values["v"] = float(final_velocity_match.group(1))
        _warn_if_missing_unit(warnings, final_velocity_match.group(2), "final velocity", "m/s")

    constant_velocity_match = _first_match(
        text,
        [
            r"(?:constant\s+velocity|constant\s+speed|moves?\s+at|moving\s+at)\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"(?:velocity|speed)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)",
        ],
    )
    if constant_velocity_match and "v" not in values and "v0" not in values:
        velocity = float(constant_velocity_match.group(1))
        values["v0"] = velocity
        values["v"] = velocity
        _warn_if_missing_unit(warnings, constant_velocity_match.group(2), "velocity", "m/s")

    acceleration_match = _first_match(
        text,
        [
            r"(?:a|acceleration)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m/s\^?2|m/s2|meters?\s+per\s+second\s+squared)?",
            r"(?:accelerates?|accelerating)\s+(?:from\s+rest\s+)?(?:at|with|by)?\s*(-?\d+(?:\.\d+)?)\s*(m/s\^?2|m/s2|meters?\s+per\s+second\s+squared)?",
            r"(?:decelerates?|slows|brakes)\s+(?:at|with|by)?\s*(-?\d+(?:\.\d+)?)\s*(m/s\^?2|m/s2|meters?\s+per\s+second\s+squared)?",
        ],
    )
    if acceleration_match:
        acceleration = float(acceleration_match.group(1))
        if re.search(r"decelerates?|slows|brakes", acceleration_match.group(0)) and acceleration > 0:
            acceleration = -acceleration
        values["a"] = acceleration
        _warn_if_missing_unit(warnings, acceleration_match.group(2), "acceleration", "m/s^2")

    time_match = _first_match(
        text,
        [
            r"(?:for|in|after|during|over)\s+(-?\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)?",
            r"(?:t|time|duration)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)?",
        ],
    )
    if time_match:
        values["t"] = float(time_match.group(1))
        _warn_if_missing_unit(warnings, time_match.group(2), "time", "s")

    displacement_match = _first_match(
        text,
        [
            r"(?:travels?|moves?|covers?)\s+(-?\d+(?:\.\d+)?)\s*(m|meters?)\b",
            r"(?:distance|displacement)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m|meters?)?",
        ],
    )
    if displacement_match:
        values["displacement"] = float(displacement_match.group(1))
        _warn_if_missing_unit(warnings, displacement_match.group(2), "displacement", "m")

    final_position_match = _first_match(
        text,
        [
            r"(?:final\s+position|ending\s+position)\s*(?:=|is|of|at)?\s*(-?\d+(?:\.\d+)?)\s*(m|meters?)?",
            r"\bx\s*=\s*(-?\d+(?:\.\d+)?)\s*(m|meters?)?",
        ],
    )
    if final_position_match:
        values["x"] = float(final_position_match.group(1))
        _warn_if_missing_unit(warnings, final_position_match.group(2), "final position", "m")

    return values


_STRUCTURED_ALIASES = {
    "x0": "x0",
    "x_0": "x0",
    "initial_position": "x0",
    "start_position": "x0",
    "x": "x",
    "final_position": "x",
    "position": "x",
    "displacement": "displacement",
    "distance": "displacement",
    "v0": "v0",
    "v_0": "v0",
    "initial_velocity": "v0",
    "initial_speed": "v0",
    "u": "v0",
    "v": "v",
    "vf": "v",
    "v_f": "v",
    "final_velocity": "v",
    "final_speed": "v",
    "a": "a",
    "acceleration": "a",
    "t": "t",
    "time": "t",
    "duration": "t",
}


def _normalize_structured_inputs(inputs: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for key, value in inputs.items():
        if key == "time_range":
            values["time_range"] = value
            continue
        normalized_key = _STRUCTURED_ALIASES.get(key)
        if not normalized_key:
            continue
        numeric = _to_float(value)
        if numeric is None:
            warnings.append(f"Ignored non-numeric value for {key}.")
            continue
        values[normalized_key] = numeric
    return values


def _complete_kinematics_values(
    values: Dict[str, Any],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    x0 = _to_float(values.get("x0"))
    if x0 is None:
        x0 = 0.0
        warnings.append("Initial position was not provided, so x0 = 0 m was used.")

    x = _to_float(values.get("x"))
    displacement = _to_float(values.get("displacement"))
    if x is None and displacement is not None:
        x = x0 + displacement

    v0 = _to_float(values.get("v0"))
    v = _to_float(values.get("v"))
    a = _to_float(values.get("a"))
    t = _to_float(values.get("t"))

    if t is not None and t <= 0:
        errors.append("Time must be greater than 0 seconds to make a motion graph.")
        return {}

    if a is None and v0 is not None and v is not None and t is not None:
        a = (v - v0) / t

    if a is None and v0 is not None and x is not None and t is not None:
        a = 2 * (x - x0 - v0 * t) / (t * t)

    if a is None:
        if v0 is None and v is not None:
            v0 = v
        if v is None and v0 is not None:
            v = v0
        if v0 is None and v is None and x is not None and t is not None:
            v0 = (x - x0) / t
            v = v0
        if t is None and x is not None and v0 is not None:
            if abs(v0) < 1e-12:
                errors.append("A zero-velocity object cannot cover a nonzero displacement.")
                return {}
            t = (x - x0) / v0
        if x is None and v0 is not None and t is not None:
            x = x0 + v0 * t
        a = 0.0
    else:
        if v0 is None and v is not None and t is not None:
            v0 = v - a * t
        if v0 is None and x is not None and t is not None:
            v0 = (x - x0 - 0.5 * a * t * t) / t
        if v is None and v0 is not None and t is not None:
            v = v0 + a * t
        if t is None and v0 is not None and v is not None:
            if abs(a) < 1e-12:
                if abs(v - v0) > 1e-9:
                    errors.append("Final velocity cannot change when acceleration is zero.")
                    return {}
            else:
                t = (v - v0) / a
        if t is None and x is not None and v0 is not None:
            t = _solve_time_from_position(x0, x, v0, a)
        if x is None and v0 is not None and t is not None:
            x = x0 + v0 * t + 0.5 * a * t * t
        if v is None and v0 is not None and t is not None:
            v = v0 + a * t

    if t is None or v0 is None or a is None:
        errors.append(
            "Not enough information to graph 1D motion. Provide at least time plus velocity/acceleration, or enough values to solve for time."
        )
        return {}

    if t <= 0:
        errors.append("The solved time is not positive, so the graph would not be physically meaningful.")
        return {}

    if x is None:
        x = x0 + v0 * t + 0.5 * a * t * t
    if v is None:
        v = v0 + a * t

    if any(abs(candidate) > SPEED_OF_LIGHT for candidate in (v0, v)):
        errors.append("Velocity exceeds the speed of light, so this introductory kinematics graph is not physically valid.")
        return {}

    if abs(a) > 1e6:
        warnings.append("Acceleration is extremely large for an introductory physics example; please check the units.")

    return {
        "x0": x0,
        "x": x,
        "v0": v0,
        "v": v,
        "a": a,
        "t": t,
        "time_range": values.get("time_range"),
    }


def _build_kinematics_payload(problem: str, solution: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    time_values = _build_time_values(solution["t"], solution.get("time_range"), warnings)

    x0 = solution["x0"]
    v0 = solution["v0"]
    a = solution["a"]
    final_time = solution["t"]
    final_position = solution["x"]
    final_velocity = solution["v"]

    position_points = []
    velocity_points = []
    acceleration_points = []
    for current_time in time_values:
        position = x0 + v0 * current_time + 0.5 * a * current_time * current_time
        velocity = v0 + a * current_time
        position_points.append({"x": _round(current_time), "y": _round(position)})
        velocity_points.append({"x": _round(current_time), "y": _round(velocity)})
        acceleration_points.append({"x": _round(current_time), "y": _round(a)})

    motion_type = "uniform_motion" if abs(a) < 1e-12 else "constant_acceleration"
    parameter_summary = [
        {"symbol": "x0", "label": "Initial position", "value": _round(x0), "unit": "m"},
        {"symbol": "v0", "label": "Initial velocity", "value": _round(v0), "unit": "m/s"},
        {"symbol": "a", "label": "Acceleration", "value": _round(a), "unit": "m/s^2"},
        {"symbol": "t", "label": "Duration", "value": _round(final_time), "unit": "s"},
        {"symbol": "v", "label": "Final velocity", "value": _round(final_velocity), "unit": "m/s"},
        {"symbol": "dx", "label": "Displacement", "value": _round(final_position - x0), "unit": "m"},
    ]

    return {
        "type": "kinematics_1d",
        "title": "1D Kinematics Motion Graphs",
        "subtitle": "Position, velocity, and acceleration as functions of time",
        "motionType": motion_type,
        "parameters": parameter_summary,
        "warnings": _unique(warnings),
        "graphs": [
            {
                "id": "position-time",
                "title": "Position vs. Time",
                "xAxis": {"label": "Time", "unit": "s"},
                "yAxis": {"label": "Position", "unit": "m"},
                "series": [
                    {
                        "label": "x(t)",
                        "unit": "m",
                        "color": "#1976d2",
                        "points": position_points,
                    }
                ],
            },
            {
                "id": "velocity-time",
                "title": "Velocity vs. Time",
                "xAxis": {"label": "Time", "unit": "s"},
                "yAxis": {"label": "Velocity", "unit": "m/s"},
                "series": [
                    {
                        "label": "v(t)",
                        "unit": "m/s",
                        "color": "#f57c00",
                        "points": velocity_points,
                    }
                ],
            },
            {
                "id": "acceleration-time",
                "title": "Acceleration vs. Time",
                "xAxis": {"label": "Time", "unit": "s"},
                "yAxis": {"label": "Acceleration", "unit": "m/s^2"},
                "series": [
                    {
                        "label": "a(t)",
                        "unit": "m/s^2",
                        "color": "#388e3c",
                        "points": acceleration_points,
                    }
                ],
            },
        ],
        "source": {
            "problem": problem,
            "equations": [
                "x = x0 + v0*t + 0.5*a*t^2",
                "v = v0 + a*t",
                "a = constant",
            ],
        },
    }


def _build_time_values(total_time: float, time_range: Any, warnings: List[str]) -> List[float]:
    start = 0.0
    end = total_time
    step: Optional[float] = None

    if time_range is not None:
        parsed = _parse_time_range(time_range)
        if parsed is None:
            warnings.append("Time range was not understood, so the solved duration was used.")
        else:
            start, end, step = parsed

    if end <= start:
        warnings.append("Time range must increase, so the graph uses 0 to the solved duration.")
        start, end = 0.0, total_time
        step = None

    if step is not None and step > 0:
        count = int(math.floor((end - start) / step)) + 1
        if count > 140:
            warnings.append("Time step created too many points, so the graph was resampled.")
            count = DEFAULT_SAMPLE_COUNT
            return _linspace(start, end, count)
        values = [start + index * step for index in range(count)]
        if values[-1] < end:
            values.append(end)
        return [_round(value) for value in values]

    return _linspace(start, end, DEFAULT_SAMPLE_COUNT)


def _parse_time_range(time_range: Any) -> Optional[Tuple[float, float, Optional[float]]]:
    if isinstance(time_range, str):
        parts = [part.strip() for part in time_range.split(",")]
        if len(parts) not in (2, 3):
            return None
        numbers = [_to_float(part) for part in parts]
        if any(number is None for number in numbers):
            return None
        return numbers[0], numbers[1], numbers[2] if len(numbers) == 3 else None

    if isinstance(time_range, dict):
        start = _to_float(time_range.get("start", 0))
        end = _to_float(time_range.get("end"))
        step = _to_float(time_range.get("step")) if "step" in time_range else None
        if start is None or end is None:
            return None
        return start, end, step

    if isinstance(time_range, (list, tuple)) and len(time_range) in (2, 3):
        numbers = [_to_float(part) for part in time_range]
        if any(number is None for number in numbers):
            return None
        return numbers[0], numbers[1], numbers[2] if len(numbers) == 3 else None

    return None


def _solve_time_from_position(x0: float, x: float, v0: float, a: float) -> Optional[float]:
    if abs(a) < 1e-12:
        if abs(v0) < 1e-12:
            return 0.0 if abs(x - x0) < 1e-12 else None
        return (x - x0) / v0

    coefficient_a = 0.5 * a
    coefficient_b = v0
    coefficient_c = x0 - x
    discriminant = coefficient_b * coefficient_b - 4 * coefficient_a * coefficient_c
    if discriminant < 0:
        return None
    root = math.sqrt(discriminant)
    candidates = [
        (-coefficient_b + root) / (2 * coefficient_a),
        (-coefficient_b - root) / (2 * coefficient_a),
    ]
    positive_candidates = [candidate for candidate in candidates if candidate > 0]
    return min(positive_candidates) if positive_candidates else None


def _first_match(text: str, patterns: Iterable[str]) -> Optional[re.Match[str]]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match
    return None


def _normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("²", "^2")
        .replace("−", "-")
        .replace("meters/second", "m/s")
        .replace("meter/second", "m/s")
    )


def _warn_if_missing_unit(warnings: List[str], unit: Optional[str], quantity: str, assumed_unit: str) -> None:
    if not unit:
        warnings.append(f"No unit was found for {quantity}; assuming {assumed_unit}.")


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _linspace(start: float, end: float, count: int) -> List[float]:
    if count <= 1:
        return [_round(start)]
    step = (end - start) / (count - 1)
    return [_round(start + index * step) for index in range(count)]


def _round(value: float) -> float:
    return round(float(value), 6)


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _public_inputs(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _round(value) if isinstance(value, (int, float)) else value
        for key, value in values.items()
        if key in {"x0", "x", "displacement", "v0", "v", "a", "t", "time_range"}
    }
