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
ELEVATED_LAUNCH_WORDS = (
    "hill",
    "cliff",
    "ledge",
    "building",
    "platform",
    "tower",
    "roof",
    "bridge",
    "balcony",
    "window",
    "mountain",
    "table",
)

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

PROJECTILE_EXAMPLE_PROBLEMS: List[Dict[str, Any]] = [
    {
        "name": "ground_launch_45_degrees",
        "problem": "Graph the trajectory of a ball launched at 30 m/s at 45 degrees from ground level.",
        "expected_motion_type": "projectile_motion",
    },
    {
        "name": "height_vs_time_steep_launch",
        "problem": "Show height vs time for a projectile launched at 20 m/s at 60 degrees.",
        "expected_motion_type": "projectile_motion",
    },
    {
        "name": "velocity_components_low_launch",
        "problem": "Graph velocity components vs time for a ball thrown at 25 m/s at 30 degrees.",
        "expected_motion_type": "projectile_motion",
    },
    {
        "name": "horizontal_cliff_launch",
        "problem": "A ball is thrown horizontally from a 20 m cliff at 10 m/s. Graph its path.",
        "expected_motion_type": "projectile_motion",
    },
    {
        "name": "elevated_launch",
        "problem": "A projectile is launched from 10 m high at 40 m/s and 35 degrees. Show trajectory, height, and velocity components.",
        "expected_motion_type": "projectile_motion",
    },
    {
        "name": "structured_projectile",
        "problem": "Create projectile graphs for this launch.",
        "structured": {"motion_type": "projectile", "v0": 18, "angle": 50, "h0": 2},
        "expected_motion_type": "projectile_motion",
    },
    {
        "name": "dropped_object",
        "problem": "Object dropped from 50 m. Graph height, velocity, and acceleration.",
        "expected_motion_type": "projectile_motion",
    },
    {
        "name": "vertical_throw",
        "problem": "A ball is thrown upward at 15 m/s from 1.5 m high. Show the kinematics graphs.",
        "expected_motion_type": "projectile_motion",
    },
]


def build_kinematics_graph_response(
    problem: str = "",
    structured: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build graph payloads for introductory kinematics.

    Args:
        problem: Natural-language student question.
        structured: Optional structured inputs. Supported aliases include x0,
            initial_position, x, displacement, v0, initial_velocity, v,
            final_velocity, a, acceleration, t, duration, time_range, plus
            projectile values such as launch_speed, angle, h0, vx, vy, and g.

    Returns:
        A JSON-serializable response with graphs, warnings, errors, and parsed
        input values.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if _looks_like_projectile_request(problem, structured or {}):
        return _build_projectile_graph_response(problem, structured or {})

    if _looks_like_piecewise_request(problem):
        return _build_piecewise_kinematics_graph_response(problem)

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
        "projectile",
        "trajectory",
        "launched",
        "launch",
        "kicked",
        "kick",
        "thrown",
        "dropped",
        "cliff",
        "hill",
        "ledge",
        "building",
        "platform",
        "tower",
        "height",
        "angle",
        "degrees",
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
    if any(key in context for key in _STRUCTURED_ALIASES) or any(key in context for key in _PROJECTILE_ALIASES):
        return context
    return {}


def _parse_kinematics_text(problem: str, warnings: List[str]) -> Dict[str, float]:
    text = _normalize_text(problem)
    values: Dict[str, float] = {}

    if "from rest" in text or "starts from rest" in text or "starting from rest" in text:
        values["v0"] = 0.0
    if any(phrase in text for phrase in ("to rest", "to a stop", "comes to rest", "comes to a stop", "until stopping", "until it stops")):
        values["v"] = 0.0

    from_to_rest = re.search(
        r"from\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?\s+to\s+rest",
        text,
    )
    if from_to_rest:
        values["v0"] = float(from_to_rest.group(1))
        values["v"] = 0.0
        _warn_if_missing_unit(warnings, from_to_rest.group(2), "initial velocity", "m/s")

    velocity_change_match = _first_match(
        text,
        [
            r"(?:from|between)\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?\s+(?:to|and)\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"(?:speeds?\s+up|slows?\s+down|accelerates?|decelerates?|brakes?)\s+from\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?\s+to\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
        ],
    )
    if velocity_change_match and "v0" not in values:
        values["v0"] = float(velocity_change_match.group(1))
        values["v"] = float(velocity_change_match.group(3))
        _warn_if_missing_unit(warnings, velocity_change_match.group(2), "initial velocity", "m/s")
        _warn_if_missing_unit(warnings, velocity_change_match.group(4), "final velocity", "m/s")

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
            r"(?:moving|traveling|travelling|going)\s+(?:at\s+)?(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)",
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
            r"(?:\bt\b|time|duration)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)?",
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


def _looks_like_piecewise_request(problem: str) -> bool:
    text = _normalize_text(problem)
    if not any(separator in text for separator in (" then ", ";", " followed by ", " and then ")):
        return False
    return any(word in text for word in ("graph", "plot", "motion", "position", "velocity", "acceleration"))


def _build_piecewise_kinematics_graph_response(problem: str) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    segments = _split_motion_segments(problem)

    if len(segments) < 2:
        return build_kinematics_graph_response(problem)

    state = {"x0": 0.0, "v0": 0.0, "t_offset": 0.0}
    all_segments = []
    for index, segment_text in enumerate(segments):
        segment_warnings: List[str] = []
        parsed = _parse_kinematics_text(segment_text, segment_warnings)
        warnings.extend(segment_warnings)

        if index == 0 and "v0" not in parsed:
            parsed["v0"] = state["v0"]
            warnings.append("Initial velocity for the first stage was not provided, so v0 = 0 m/s was used.")
        elif index > 0:
            parsed["v0"] = state["v0"]

        parsed["x0"] = state["x0"]

        if _segment_is_rest_interval(segment_text):
            parsed["v0"] = 0.0
            parsed["v"] = 0.0
            parsed["a"] = 0.0

        if _segment_is_constant_speed(segment_text) and "a" not in parsed:
            parsed["a"] = 0.0
            parsed.setdefault("v", parsed["v0"])

        if any(phrase in _normalize_text(segment_text) for phrase in ("until stopping", "until it stops", "to rest", "to a stop")):
            parsed["v"] = 0.0

        local_errors: List[str] = []
        solution = _complete_kinematics_values(parsed, warnings, local_errors)
        if local_errors:
            errors.append(f"Stage {index + 1}: {' '.join(local_errors)}")
            continue

        duration = solution["t"]
        if duration <= 0:
            errors.append(f"Stage {index + 1}: duration must be positive.")
            continue

        all_segments.append(
            {
                "text": segment_text.strip(),
                "x0": solution["x0"],
                "v0": solution["v0"],
                "a": solution["a"],
                "t": duration,
                "start_time": state["t_offset"],
                "end_time": state["t_offset"] + duration,
                "x": solution["x"],
                "v": solution["v"],
            }
        )
        state["x0"] = solution["x"]
        state["v0"] = solution["v"]
        state["t_offset"] += duration

    if errors:
        return {
            "graphs": [],
            "warnings": _unique(warnings),
            "errors": _unique(errors),
            "parsed_inputs": {"segments": segments},
        }

    if not all_segments:
        return {
            "graphs": [],
            "warnings": _unique(warnings),
            "errors": ["No valid motion stages were found."],
            "parsed_inputs": {"segments": segments},
        }

    payload = _build_piecewise_payload(problem, all_segments, warnings)
    return {
        "graphs": [payload],
        "warnings": _unique(warnings),
        "errors": [],
        "parsed_inputs": {"segments": segments},
    }


def _split_motion_segments(problem: str) -> List[str]:
    normalized = re.sub(r"\band then\b", " then ", problem, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfollowed by\b", " then ", normalized, flags=re.IGNORECASE)
    parts = re.split(r"\s+then\s+|;", normalized, flags=re.IGNORECASE)
    return [part.strip(" .") for part in parts if part.strip(" .")]


def _segment_is_constant_speed(segment_text: str) -> bool:
    text = _normalize_text(segment_text)
    return any(
        phrase in text
        for phrase in (
            "constant speed",
            "constant velocity",
            "coasts",
            "coast",
            "continues at that speed",
            "same speed",
            "is stopped",
            "stops for",
            "rests for",
            "at rest for",
        )
    )


def _segment_is_rest_interval(segment_text: str) -> bool:
    text = _normalize_text(segment_text)
    return bool(
        re.search(r"\b(?:stops?|is stopped|rests?|is at rest|remains at rest|waits)\s+for\b", text)
        or "at rest for" in text
    )


def _build_piecewise_payload(problem: str, segments: List[Dict[str, Any]], warnings: List[str]) -> Dict[str, Any]:
    total_time = segments[-1]["end_time"]
    time_values = _linspace(0.0, total_time, DEFAULT_SAMPLE_COUNT)
    position_points = []
    velocity_points = []
    acceleration_points = []

    for global_time in time_values:
        segment = _segment_for_time(global_time, segments)
        local_time = min(max(global_time - segment["start_time"], 0.0), segment["t"])
        position = segment["x0"] + segment["v0"] * local_time + 0.5 * segment["a"] * local_time * local_time
        velocity = segment["v0"] + segment["a"] * local_time
        position_points.append({"x": _round(global_time), "y": _round(position)})
        velocity_points.append({"x": _round(global_time), "y": _round(velocity)})
        acceleration_points.append({"x": _round(global_time), "y": _round(segment["a"])})

    parameter_summary = [
        {"symbol": "stages", "label": "Stages", "value": len(segments), "unit": ""},
        {"symbol": "T", "label": "Total time", "value": _round(total_time), "unit": "s"},
        {"symbol": "x final", "label": "Final position", "value": _round(segments[-1]["x"]), "unit": "m"},
        {"symbol": "v final", "label": "Final velocity", "value": _round(segments[-1]["v"]), "unit": "m/s"},
    ]

    return {
        "type": "kinematics_piecewise",
        "title": "Piecewise Kinematics Motion Graphs",
        "subtitle": "Position, velocity, and acceleration across multiple motion stages",
        "motionType": "piecewise_kinematics",
        "parameters": parameter_summary,
        "warnings": _unique(warnings),
        "graphs": [
            {
                "id": "piecewise-position-time",
                "title": "Position vs. Time",
                "xAxis": {"label": "Time", "unit": "s"},
                "yAxis": {"label": "Position", "unit": "m"},
                "series": [{"label": "x(t)", "unit": "m", "color": "#1976d2", "points": position_points}],
            },
            {
                "id": "piecewise-velocity-time",
                "title": "Velocity vs. Time",
                "xAxis": {"label": "Time", "unit": "s"},
                "yAxis": {"label": "Velocity", "unit": "m/s"},
                "series": [{"label": "v(t)", "unit": "m/s", "color": "#f57c00", "points": velocity_points}],
            },
            {
                "id": "piecewise-acceleration-time",
                "title": "Acceleration vs. Time",
                "xAxis": {"label": "Time", "unit": "s"},
                "yAxis": {"label": "Acceleration", "unit": "m/s^2"},
                "series": [{"label": "a(t)", "unit": "m/s^2", "color": "#388e3c", "points": acceleration_points}],
            },
        ],
        "source": {
            "problem": problem,
            "equations": [
                "Each stage uses x = x0 + v0*t + 0.5*a*t^2",
                "Each stage uses v = v0 + a*t",
                "Final state of one stage becomes the initial state of the next stage",
            ],
            "segments": [
                {
                    "description": segment["text"],
                    "start_time": _round(segment["start_time"]),
                    "end_time": _round(segment["end_time"]),
                    "duration": _round(segment["t"]),
                    "initial_position": _round(segment["x0"]),
                    "final_position": _round(segment["x"]),
                    "initial_velocity": _round(segment["v0"]),
                    "final_velocity": _round(segment["v"]),
                    "acceleration": _round(segment["a"]),
                }
                for segment in segments
            ],
        },
    }


def _segment_for_time(global_time: float, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    for segment in segments:
        if global_time <= segment["end_time"] + 1e-9:
            return segment
    return segments[-1]


_PROJECTILE_ALIASES = {
    "motion_type": "motion_type",
    "type": "motion_type",
    "v0": "v0",
    "speed": "v0",
    "initial_speed": "v0",
    "launch_speed": "v0",
    "initial_velocity": "v0",
    "launch_velocity": "v0",
    "angle": "angle",
    "theta": "angle",
    "launch_angle": "angle",
    "angle_degrees": "angle",
    "h0": "h0",
    "y0": "h0",
    "height": "h0",
    "initial_height": "h0",
    "launch_height": "h0",
    "start_height": "h0",
    "x0": "x0",
    "initial_x": "x0",
    "vx": "v0x",
    "v0x": "v0x",
    "initial_vx": "v0x",
    "horizontal_velocity": "v0x",
    "vy": "v0y",
    "v0y": "v0y",
    "initial_vy": "v0y",
    "vertical_velocity": "v0y",
    "g": "gravity",
    "gravity": "gravity",
    "t": "t",
    "time": "t",
    "duration": "t",
    "time_range": "time_range",
}


def _looks_like_projectile_request(problem: str, structured: Dict[str, Any]) -> bool:
    normalized_structured = _normalize_projectile_structured_inputs(structured, [])
    motion_type = str(normalized_structured.get("motion_type", "")).lower()
    if motion_type in {"projectile", "projectile_motion", "free_fall", "vertical"}:
        return True
    if any(key in normalized_structured for key in ("angle", "h0", "v0x", "v0y", "gravity")):
        return True

    text = _normalize_text(problem)
    projectile_words = (
        "projectile",
        "trajectory",
        "launched",
        "launch",
        "kicked",
        "kick",
        "thrown horizontally",
        "thrown upward",
        "thrown downward",
        "velocity components",
        "component",
        "cliff",
        "hill",
        "ledge",
        "building",
        "platform",
        "tower",
        "cannon",
        "range",
    )
    free_fall_words = ("dropped", "free fall", "falls from", "falling from")
    has_angle = re.search(r"-?\d+(?:\.\d+)?\s*(degrees?|deg|°)", text) is not None
    has_graph_word = any(word in text for word in ("graph", "plot", "show", "visualize", "draw"))

    return (
        any(word in text for word in projectile_words)
        or any(word in text for word in free_fall_words)
        or (has_angle and any(word in text for word in ("ball", "projectile", "launched", "thrown", "kicked")))
        or (has_graph_word and "height" in text and any(word in text for word in ("ball", "projectile", "thrown", "launched", "kicked")))
    )


def _build_projectile_graph_response(problem: str, structured: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    parsed = _parse_projectile_text(problem, warnings)
    structured_values = _normalize_projectile_structured_inputs(structured, warnings)
    values = {**parsed, **structured_values}

    _guard_projectile_height_assumptions(problem, values, errors)
    if errors:
        return {
            "graphs": [],
            "warnings": _unique(warnings),
            "errors": _unique(errors),
            "parsed_inputs": _public_projectile_inputs(values),
        }

    solution = _complete_projectile_values(values, warnings, errors)
    if errors:
        return {
            "graphs": [],
            "warnings": _unique(warnings),
            "errors": _unique(errors),
            "parsed_inputs": _public_projectile_inputs(values),
        }

    payload = _build_projectile_payload(problem, solution, warnings)
    return {
        "graphs": [payload],
        "warnings": _unique(warnings),
        "errors": [],
        "parsed_inputs": _public_projectile_inputs(values),
    }


def _normalize_projectile_structured_inputs(inputs: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for key, value in inputs.items():
        normalized_key = _PROJECTILE_ALIASES.get(key)
        if not normalized_key:
            continue
        if normalized_key == "motion_type":
            values[normalized_key] = str(value)
            continue
        if normalized_key == "time_range":
            values[normalized_key] = value
            continue
        numeric = _to_float(value)
        if numeric is None:
            warnings.append(f"Ignored non-numeric value for {key}.")
            continue
        values[normalized_key] = numeric
    return values


def _parse_projectile_text(problem: str, warnings: List[str]) -> Dict[str, Any]:
    text = _normalize_text(problem)
    values: Dict[str, Any] = {"motion_type": "projectile"}

    if "ground level" in text or "from the ground" in text or "from ground" in text:
        values["h0"] = 0.0

    if "dropped" in text or "free fall" in text:
        values["v0"] = 0.0
        values["angle"] = 0.0

    if "thrown horizontally" in text or "launched horizontally" in text or "fired horizontally" in text or "kicked horizontally" in text:
        values["angle"] = 0.0

    if "thrown upward" in text or "straight up" in text or "vertically upward" in text:
        values.setdefault("angle", 90.0)
    if "thrown downward" in text or "straight down" in text or "vertically downward" in text:
        values.setdefault("angle", -90.0)

    speed_match = _first_match(
        text,
        [
            r"(?:launched|launches|thrown|throws|fired|shot|projected|kicked|kicks)\s+(?:upward|downward|vertically\s+upward|vertically\s+downward|straight\s+up|straight\s+down)\s+(?:at|with)?\s*(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"(?:launched|launches|thrown|throws|fired|shot|projected|kicked|kicks)\s+(?:from\s+[-\d.]+\s*m\s+(?:high|hill|cliff|ledge|building|platform|tower)\s+)?(?:at|with)?\s*(?:an?\s+)?(?:initial\s+)?(?:speed|velocity)?\s*(?:of\s+)?(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)\s+(?:at|and)\s+-?\d+(?:\.\d+)?\s*(?:degrees?|deg|°)",
            r"(?:speed|velocity)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
        ],
    )
    if speed_match and "v0" not in values:
        values["v0"] = float(speed_match.group(1))
        _warn_if_missing_unit(warnings, speed_match.group(2), "launch speed", "m/s")

    horizontal_velocity_match = _first_match(
        text,
        [
            r"(?:horizontal\s+velocity|vx|v0x)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"(?:thrown|launched|fired|kicked)\s+horizontally\s+(?:at|with)\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
            r"(?:thrown|launched|fired|kicked)\s+horizontally\s+from\s+(?:a\s+)?[-\d.]+\s*m\s+(?:cliff|hill|height|ledge|building|platform|tower)?\s+at\s+(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
        ],
    )
    if horizontal_velocity_match:
        values["v0x"] = float(horizontal_velocity_match.group(1))
        values.setdefault("v0y", 0.0)
        _warn_if_missing_unit(warnings, horizontal_velocity_match.group(2), "horizontal velocity", "m/s")

    vertical_velocity_match = _first_match(
        text,
        [
            r"(?:vertical\s+velocity|vy|v0y)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m/s|meters?\s+per\s+second)?",
        ],
    )
    if vertical_velocity_match:
        values["v0y"] = float(vertical_velocity_match.group(1))
        _warn_if_missing_unit(warnings, vertical_velocity_match.group(2), "vertical velocity", "m/s")

    angle_match = _first_match(
        text,
        [
            r"(?:angle|launch\s+angle)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)",
            r"(?:at|and)\s+(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)",
            r"(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)",
        ],
    )
    if angle_match:
        values["angle"] = float(angle_match.group(1))

    inferred_height = _infer_projectile_initial_height(text, warnings)
    if inferred_height is not None:
        values["h0"] = inferred_height

    height_match = _first_match(
        text,
        [
            r"(?:dropped|falls|falling)\s+from\s+(-?\d+(?:\.\d+)?)\s*(m|meters?)",
            r"(?:from|off)\s+(?:a\s+|an\s+)?(-?\d+(?:\.\d+)?)\s*(m|meters?)\s*(?:high|height|cliff|hill|ledge|building|platform|tower|roof|bridge|balcony|window|mountain|table)",
            r"(-?\d+(?:\.\d+)?)\s*(m|meters?)\s*(?:high|height|cliff|hill|ledge|building|platform|tower|roof|bridge|balcony|window|mountain|table)",
            r"(?:from|off)\s+(?:a\s+|an\s+)?(?:hill|cliff|ledge|building|platform|tower|roof|bridge|balcony|window|mountain|table)\s+(-?\d+(?:\.\d+)?)\s*(m|meters?)\s*(?:high)?",
            r"(?:initial\s+height|launch\s+height|h0|y0)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(m|meters?)?",
            r"from\s+(-?\d+(?:\.\d+)?)\s*(m|meters?)\s+high",
        ],
    )
    if height_match and "h0" not in values:
        values["h0"] = float(height_match.group(1))
        _warn_if_missing_unit(warnings, height_match.group(2), "initial height", "m")

    gravity_match = _first_match(
        text,
        [
            r"(?:gravity|g)\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)\s*(m/s\^?2|m/s2|meters?\s+per\s+second\s+squared)?",
        ],
    )
    if gravity_match:
        values["gravity"] = float(gravity_match.group(1))
        _warn_if_missing_unit(warnings, gravity_match.group(2), "gravity", "m/s^2")

    time_match = _first_match(
        text,
        [
            r"(?:for|during|over|after)\s+(-?\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)?",
            r"(?:\bt\b|time|duration)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)?",
        ],
    )
    if time_match:
        values["t"] = float(time_match.group(1))
        _warn_if_missing_unit(warnings, time_match.group(2), "time", "s")

    return values


def _infer_projectile_initial_height(text: str, warnings: List[str]) -> Optional[float]:
    elevation_sources = "|".join(ELEVATED_LAUNCH_WORDS)
    height_patterns = [
        rf"(?:from|off|on|atop|on\s+top\s+of)\s+(?:a\s+|an\s+|the\s+)?(?P<value>-?\d+(?:\.\d+)?)\s*-?\s*(?P<unit>m|meter|meters|ft|foot|feet)\s*(?:-?\s*(?:high|tall))?\s*(?:{elevation_sources})",
        rf"(?P<value>-?\d+(?:\.\d+)?)\s*-?\s*(?P<unit>m|meter|meters|ft|foot|feet)\s*(?:-?\s*(?:high|tall))?\s*(?:{elevation_sources})",
        rf"(?:from|off|on|atop|on\s+top\s+of)\s+(?:a\s+|an\s+|the\s+)?(?:{elevation_sources})\s+(?P<value>-?\d+(?:\.\d+)?)\s*-?\s*(?P<unit>m|meter|meters|ft|foot|feet)\s*(?:high|tall)?",
        r"(?:initial\s+height|launch\s+height|starting\s+height|height|h0|y0)\s*(?:=|is|of|:)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*-?\s*(?P<unit>m|meter|meters|ft|foot|feet)?",
        r"(?P<value>-?\d+(?:\.\d+)?)\s*-?\s*(?P<unit>m|meter|meters|ft|foot|feet)\s*(?:above\s+(?:the\s+)?ground)",
    ]

    for pattern in height_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = float(match.group("value"))
        unit = match.groupdict().get("unit") or "m"
        if unit in {"ft", "foot", "feet"}:
            warnings.append("Launch height was given in feet, so it was converted to meters.")
            return value * 0.3048
        return value

    return None


def _guard_projectile_height_assumptions(problem: str, values: Dict[str, Any], errors: List[str]) -> None:
    if "h0" in values:
        return

    text = _normalize_text(problem)
    mentions_elevated_surface = any(word in text for word in ELEVATED_LAUNCH_WORDS)
    mentions_height_without_value = any(phrase in text for phrase in ("above ground", "above the ground", "from height", "from a height"))
    has_distance_measurement = re.search(
        r"-?\d+(?:\.\d+)?\s*-?\s*(?:meter|meters|ft|foot|feet|m(?!/s))\b",
        text,
    ) is not None

    if mentions_elevated_surface and has_distance_measurement:
        errors.append(
            "I found an elevated launch surface and a distance value, but could not confidently classify the launch height. Please write it like 'from a 20 m hill' or provide h0."
        )
    elif mentions_elevated_surface or mentions_height_without_value:
        errors.append(
            "The projectile appears to start above ground, but the launch height is missing. Please provide h0 in meters."
        )


def _complete_projectile_values(
    values: Dict[str, Any],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    x0 = _to_float(values.get("x0")) or 0.0
    h0 = _to_float(values.get("h0"))
    if h0 is None:
        h0 = 0.0
        warnings.append("Initial height was not provided, so h0 = 0 m was used.")

    gravity = _to_float(values.get("gravity"))
    if gravity is None:
        gravity = 9.81
    gravity = abs(gravity)

    if gravity <= 0:
        errors.append("Gravity must be greater than 0 m/s^2 for projectile motion.")
        return {}
    if h0 < 0:
        errors.append("Initial height cannot be below ground for this introductory projectile graph.")
        return {}

    v0x = _to_float(values.get("v0x"))
    v0y = _to_float(values.get("v0y"))
    v0 = _to_float(values.get("v0"))
    angle = _to_float(values.get("angle"))

    if v0x is not None or v0y is not None:
        v0x = v0x or 0.0
        v0y = v0y or 0.0
        v0 = math.hypot(v0x, v0y)
        angle = math.degrees(math.atan2(v0y, v0x)) if v0 > 0 else 0.0
    elif v0 is not None:
        if v0 < 0:
            errors.append("Launch speed must be nonnegative.")
            return {}
        if angle is None:
            if v0 == 0:
                angle = 0.0
            else:
                errors.append("Projectile graphs need a launch angle, or horizontal and vertical velocity components.")
                return {}
        if angle < -90 or angle > 90:
            errors.append("Launch angle should be between -90 and 90 degrees for this introductory graph.")
            return {}
        angle_rad = math.radians(angle)
        v0x = v0 * math.cos(angle_rad)
        v0y = v0 * math.sin(angle_rad)
    else:
        errors.append("Projectile graphs need launch speed and angle, or horizontal and vertical velocity components.")
        return {}

    if any(abs(candidate) > SPEED_OF_LIGHT for candidate in (v0x, v0y, v0 or 0)):
        errors.append("Launch velocity exceeds the speed of light, so this graph is not physically valid.")
        return {}

    duration = _to_float(values.get("t"))
    if duration is not None and duration <= 0:
        errors.append("Time must be greater than 0 seconds to make a projectile graph.")
        return {}

    flight_time = _solve_projectile_flight_time(h0, v0y, gravity)
    if duration is None:
        duration = flight_time
    elif flight_time is not None and duration > flight_time:
        warnings.append("The requested duration extends past ground impact, so the graph stops at impact.")
        duration = flight_time

    if duration is None or duration <= 0:
        errors.append("This projectile has no positive flight time. Add a launch height or a graph duration.")
        return {}

    range_x = v0x * duration
    impact_vy = v0y - gravity * duration
    impact_speed = math.hypot(v0x, impact_vy)
    time_to_max = max(v0y / gravity, 0.0) if v0y > 0 else 0.0
    max_height = h0 + v0y * time_to_max - 0.5 * gravity * time_to_max * time_to_max

    if abs(range_x) < 1e-9 and abs(v0x) < 1e-9:
        warnings.append("Horizontal velocity is zero, so the trajectory graph is a vertical line.")

    return {
        "x0": x0,
        "h0": h0,
        "v0": v0,
        "angle": angle,
        "v0x": v0x,
        "v0y": v0y,
        "gravity": gravity,
        "t": duration,
        "flight_time": flight_time,
        "range": range_x,
        "max_height": max_height,
        "time_to_max": time_to_max,
        "impact_speed": impact_speed,
        "impact_vy": impact_vy,
        "time_range": values.get("time_range"),
    }


def _solve_projectile_flight_time(h0: float, v0y: float, gravity: float) -> Optional[float]:
    # y(t) = h0 + v0y*t - 0.5*g*t^2. Solve y = 0.
    coefficient_a = -0.5 * gravity
    coefficient_b = v0y
    coefficient_c = h0
    discriminant = coefficient_b * coefficient_b - 4 * coefficient_a * coefficient_c
    if discriminant < 0:
        return None
    root = math.sqrt(discriminant)
    candidates = [
        (-coefficient_b + root) / (2 * coefficient_a),
        (-coefficient_b - root) / (2 * coefficient_a),
    ]
    positive_candidates = [candidate for candidate in candidates if candidate > 1e-9]
    return max(positive_candidates) if positive_candidates else None


def _build_projectile_payload(problem: str, solution: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    time_values = _build_time_values(solution["t"], solution.get("time_range"), warnings)
    x0 = solution["x0"]
    h0 = solution["h0"]
    v0x = solution["v0x"]
    v0y = solution["v0y"]
    gravity = solution["gravity"]

    trajectory_points = []
    height_points = []
    velocity_x_points = []
    velocity_y_points = []
    speed_points = []

    for current_time in time_values:
        x = x0 + v0x * current_time
        y = h0 + v0y * current_time - 0.5 * gravity * current_time * current_time
        vx = v0x
        vy = v0y - gravity * current_time
        trajectory_points.append({"x": _round(x), "y": _round(max(y, 0.0))})
        height_points.append({"x": _round(current_time), "y": _round(max(y, 0.0))})
        velocity_x_points.append({"x": _round(current_time), "y": _round(vx)})
        velocity_y_points.append({"x": _round(current_time), "y": _round(vy)})
        speed_points.append({"x": _round(current_time), "y": _round(math.hypot(vx, vy))})

    parameter_summary = [
        {"symbol": "v0", "label": "Launch speed", "value": _round(solution["v0"]), "unit": "m/s"},
        {"symbol": "theta", "label": "Launch angle", "value": _round(solution["angle"]), "unit": "deg"},
        {"symbol": "h0", "label": "Initial height", "value": _round(h0), "unit": "m"},
        {"symbol": "g", "label": "Gravity", "value": _round(gravity), "unit": "m/s^2"},
        {"symbol": "T", "label": "Flight time", "value": _round(solution["t"]), "unit": "s"},
        {"symbol": "R", "label": "Range", "value": _round(solution["range"]), "unit": "m"},
        {"symbol": "hmax", "label": "Max height", "value": _round(solution["max_height"]), "unit": "m"},
    ]

    return {
        "type": "projectile_motion",
        "title": "Projectile Motion Graphs",
        "subtitle": "Trajectory, height, and velocity components for the launch",
        "motionType": "projectile_motion",
        "parameters": parameter_summary,
        "warnings": _unique(warnings),
        "graphs": [
            {
                "id": "trajectory",
                "title": "Trajectory",
                "xAxis": {"label": "Horizontal position", "unit": "m"},
                "yAxis": {"label": "Height", "unit": "m"},
                "series": [
                    {
                        "label": "path",
                        "unit": "m",
                        "color": "#1976d2",
                        "points": trajectory_points,
                    }
                ],
            },
            {
                "id": "height-time",
                "title": "Height vs. Time",
                "xAxis": {"label": "Time", "unit": "s"},
                "yAxis": {"label": "Height", "unit": "m"},
                "series": [
                    {
                        "label": "y(t)",
                        "unit": "m",
                        "color": "#388e3c",
                        "points": height_points,
                    }
                ],
            },
            {
                "id": "velocity-components-time",
                "title": "Velocity Components vs. Time",
                "xAxis": {"label": "Time", "unit": "s"},
                "yAxis": {"label": "Velocity", "unit": "m/s"},
                "series": [
                    {
                        "label": "vx(t)",
                        "unit": "m/s",
                        "color": "#7b1fa2",
                        "points": velocity_x_points,
                    },
                    {
                        "label": "vy(t)",
                        "unit": "m/s",
                        "color": "#f57c00",
                        "points": velocity_y_points,
                    },
                    {
                        "label": "speed",
                        "unit": "m/s",
                        "color": "#455a64",
                        "points": speed_points,
                    },
                ],
            },
        ],
        "source": {
            "problem": problem,
            "equations": [
                "x = x0 + v0x*t",
                "y = h0 + v0y*t - 0.5*g*t^2",
                "vx = v0x",
                "vy = v0y - g*t",
            ],
        },
    }


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


def _public_projectile_inputs(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _round(value) if isinstance(value, (int, float)) else value
        for key, value in values.items()
        if key in {"motion_type", "x0", "h0", "v0", "angle", "v0x", "v0y", "gravity", "t", "time_range"}
    }
