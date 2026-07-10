"""
Guided tutoring workflow for physics problem-solving requests.

This module sits in front of the calculation agents. It keeps the default
student experience Socratic: diagnose the setup first, respond to attempts,
and only let a full worked solution through once the student has either
engaged with the reasoning or explicitly asked for a worked example.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


GUIDED_RESPONSE_MARKER = "Tutor checkpoint"


GUIDED_TUTORING_SYSTEM_POLICY = """
Guided tutoring policy:
- For student problem-solving requests, do not immediately give a full worked
  solution by default. First ask diagnostic questions that check the student's
  understanding of the concept, variables, assumptions, equation choice,
  diagram/setup, and units.
- Use the student's responses to adapt. If they are close, give a focused hint
  or ask for the next setup step. If they show a misconception, correct that
  specific idea and ask a targeted follow-up question.
- Give complete worked solutions only when the request includes
  [GUIDED_TUTORING_FULL_SOLUTION_APPROVED], when the student explicitly asks
  for a worked example, or after a guided exchange where the student has
  attempted the reasoning.
- For conceptual physics questions, ask the student for their current thinking
  first, then guide with short hints or misconception checks rather than giving
  a full explanation immediately.
- This policy overrides any topic instruction that says to always show complete
  solutions.
"""


FULL_SOLUTION_APPROVED_PREFIX = """[GUIDED_TUTORING_FULL_SOLUTION_APPROVED]
The student has either completed guided setup work or explicitly needs a worked
example. Provide a complete, student-facing worked solution now. Include the
diagram/setup description, knowns and unknowns, equation choice, substitutions,
units, final answer, and a quick reasonableness check.
"""


@dataclass(frozen=True)
class GuidedTutoringDecision:
    """Decision returned by the guided tutoring gate."""

    intercept: bool
    response: Optional[str]
    allow_full_solution: bool
    stage: str
    metadata: Dict[str, Any]


TOPIC_CLASSIFIER_RULES: List[Dict[str, Any]] = [
    {
        "topic": "constant_velocity",
        "agent_id": "kinematics_agent",
        "course": "PHYS 1201",
        "patterns": (r"\bconstant velocity\b", r"\bconstant speed\b", r"\bno acceleration\b"),
    },
    {
        "topic": "piecewise_motion",
        "agent_id": "kinematics_agent",
        "course": "PHYS 1201",
        "patterns": (r"\bthen\b.*\bfor\b.*\bs\b", r"\bmultiple intervals?\b", r"\bpiecewise\b", r"\bstops?\s+for\b", r"\bcoasts?\s+for\b"),
    },
    {
        "topic": "projectile_motion",
        "agent_id": "kinematics_agent",
        "course": "PHYS 1201",
        "patterns": (r"\bprojectile\b", r"\blaunched\b", r"\bthrown\b", r"\bkicked\b", r"\bat an angle\b"),
    },
    {
        "topic": "constant_acceleration",
        "agent_id": "kinematics_agent",
        "course": "PHYS 1201",
        "patterns": (r"\baccelerat", r"\bfrom rest\b", r"\bbrak", r"\bspeed(?:s|ing)? up\b"),
    },
    {
        "topic": "newtons_second_law",
        "agent_id": "forces_agent",
        "course": "PHYS 1201",
        "patterns": (r"\bnewton", r"\bsum f\b", r"\bf\s*=\s*m\s*a\b", r"\bnet force\b"),
    },
    {
        "topic": "friction",
        "agent_id": "forces_agent",
        "course": "PHYS 1201",
        "patterns": (r"\bfriction\b", r"\bmu_k\b", r"\bmu_s\b", r"\bcoefficient\b"),
    },
    {
        "topic": "work_energy",
        "agent_id": "energy_agent",
        "course": "PHYS 1201",
        "patterns": (r"\bwork[- ]energy\b", r"\bconservation of energy\b", r"\bkinetic energy\b", r"\bpotential energy\b"),
    },
    {
        "topic": "momentum_collision",
        "agent_id": "momentum_agent",
        "course": "PHYS 1201",
        "patterns": (r"\bcollision\b", r"\bstick", r"\belastic\b", r"\binelastic\b", r"\bmomentum\b"),
    },
    {
        "topic": "torque_equilibrium",
        "agent_id": "angular_motion_agent",
        "course": "PHYS 1201",
        "patterns": (r"\btorque\b", r"\bstatic equilibrium\b", r"\brotational equilibrium\b", r"\bbalance\b"),
    },
    {
        "topic": "oscillations",
        "agent_id": "waves_agent",
        "course": "PHYS 1202",
        "patterns": (r"\boscillat", r"\bsimple harmonic\b", r"\bshm\b", r"\bpendulum\b", r"\bspring-mass\b"),
    },
    {
        "topic": "waves",
        "agent_id": "waves_agent",
        "course": "PHYS 1202",
        "patterns": (r"\bwave\b", r"\bfrequency\b", r"\bwavelength\b", r"\bharmonic\b", r"\bstanding\b"),
    },
    {
        "topic": "electric_field_force",
        "agent_id": "electromagnetism_agent",
        "course": "PHYS 1202",
        "patterns": (r"\belectric force\b", r"\belectric field\b", r"\bpoint charge", r"\bcoulomb"),
    },
    {
        "topic": "electric_potential",
        "agent_id": "electromagnetism_agent",
        "course": "PHYS 1202",
        "patterns": (r"\belectric potential\b", r"\bpotential difference\b", r"\bvoltage due to\b", r"\bvoltage at\b"),
    },
    {
        "topic": "simple_circuit",
        "agent_id": "electromagnetism_agent",
        "course": "PHYS 1202",
        "patterns": (r"\bcircuit\b", r"\bresistor\b", r"\bohm", r"\bseries\b", r"\bparallel\b", r"\bcurrent\b"),
    },
    {
        "topic": "magnetism",
        "agent_id": "electromagnetism_agent",
        "course": "PHYS 1202",
        "patterns": (r"\bmagnet", r"\bmagnetic field\b", r"\blorentz\b", r"\bwire\b.*\bfield\b"),
    },
    {
        "topic": "optics",
        "agent_id": "optics_agent",
        "course": "PHYS 1202",
        "patterns": (r"\blens\b", r"\bmirror\b", r"\bray diagram\b", r"\brefraction\b", r"\bdiffraction\b"),
    },
]


AGENT_TUTORING_CONFIG: Dict[str, Dict[str, Any]] = {
    "kinematics_agent": {
        "course": "PHYS 1201",
        "domain": "kinematics",
        "diagram": "motion diagram or coordinate axis",
        "diagnostics": [
            "What type of motion is this: constant velocity, constant acceleration, free fall, or projectile motion?",
            "Which quantities are given, with units, and which quantity are we trying to find?",
            "What sign convention and coordinate direction will you use before choosing an equation?",
        ],
        "equation_hint": "Pick the kinematics equation that contains the knowns and the target unknown without adding extra unknowns.",
    },
    "forces_agent": {
        "course": "PHYS 1201",
        "domain": "forces",
        "diagram": "free-body diagram",
        "diagnostics": [
            "What object or system are we isolating?",
            "Which forces belong on the free-body diagram, and in what directions?",
            "What axes make the force components easiest to write before applying sum F = ma?",
        ],
        "equation_hint": "Start from sum F_x = ma_x and sum F_y = ma_y after the free-body diagram is clear.",
    },
    "energy_agent": {
        "course": "PHYS 1201",
        "domain": "energy",
        "diagram": "initial/final energy sketch",
        "diagnostics": [
            "What system are we choosing, and what are the initial and final states?",
            "Which forms of energy are present: kinetic, gravitational, spring, thermal, or work by an external force?",
            "Are non-conservative forces doing work, or can mechanical energy be conserved?",
        ],
        "equation_hint": "Compare E_initial + W_nonconservative with E_final before substituting numbers.",
    },
    "momentum_agent": {
        "course": "PHYS 1201",
        "domain": "momentum",
        "diagram": "before/after momentum sketch",
        "diagnostics": [
            "What system are we treating as isolated during the interaction?",
            "What are the masses and velocities before and after, including signs or directions?",
            "Is the collision elastic, inelastic, perfectly inelastic, or is this an impulse problem?",
        ],
        "equation_hint": "Write total momentum before equals total momentum after only after checking the system is isolated.",
    },
    "angular_motion_agent": {
        "course": "PHYS 1201",
        "domain": "rotational motion",
        "diagram": "rotation axis and torque diagram",
        "diagnostics": [
            "What is the rotation axis, and which quantities are angular versus linear?",
            "Are we using torque, rotational kinematics, angular momentum, or rotational energy?",
            "Do any angles need to be converted to radians before using the equation?",
        ],
        "equation_hint": "Choose the rotational analogue that matches the linear idea: torque for force, I for mass, omega for velocity.",
    },
    "thermodynamics_agent": {
        "course": "PHYS 1202",
        "domain": "thermodynamics",
        "diagram": "state/process diagram",
        "diagnostics": [
            "What is the thermodynamic system and process: isobaric, isochoric, isothermal, adiabatic, or unknown?",
            "Which state variables or heat/work quantities are given, with units?",
            "What sign convention are you using for heat added to the system and work done by the system?",
        ],
        "equation_hint": "Decide whether the ideal gas law, Q = mc delta T, or the first law delta U = Q - W matches the process.",
    },
    "waves_agent": {
        "course": "PHYS 1202",
        "domain": "waves",
        "diagram": "wave or boundary-condition sketch",
        "diagnostics": [
            "What kind of wave situation is this: wave speed, Doppler effect, sound intensity, standing wave, or interference?",
            "Which quantities are known: frequency, wavelength, speed, harmonic number, length, or intensity?",
            "What boundary condition or relative motion assumption matters before selecting the equation?",
        ],
        "equation_hint": "For waves or oscillations, first decide whether v = f lambda, a standing-wave relation, or a simple-harmonic-motion relation applies.",
    },
    "electromagnetism_agent": {
        "course": "PHYS 1202",
        "domain": "electricity and magnetism",
        "diagram": "charge, field, or circuit diagram",
        "diagnostics": [
            "What sources are present: charges, currents, fields, capacitors, or circuit elements?",
            "What geometry or circuit topology should we draw before writing equations?",
            "Are direction, sign, superposition, or series/parallel assumptions important here?",
        ],
        "equation_hint": "Draw the source geometry or circuit first, then choose Coulomb's law, field/potential relations, Ohm's law, or induction as appropriate.",
    },
    "optics_agent": {
        "course": "PHYS 1202",
        "domain": "optics",
        "diagram": "ray diagram",
        "diagnostics": [
            "What optical element or phenomenon is involved: refraction, lens, mirror, diffraction, or thin-film interference?",
            "What distances, focal length, indices, angles, wavelength, or order are given?",
            "Which sign convention or constructive/destructive condition should we apply?",
        ],
        "equation_hint": "For lenses and mirrors, settle the sign convention before using 1/f = 1/do + 1/di.",
    },
    "modern_physics_agent": {
        "course": "PHYS 1202",
        "domain": "modern physics",
        "diagram": "interaction or energy-level sketch",
        "diagnostics": [
            "What model is relevant: relativity, photoelectric effect, de Broglie wavelength, Bohr levels, or radioactive decay?",
            "Which constants and units are needed, and do any energies need conversion between joules and eV?",
            "What assumption tells us the equation is appropriate, such as v being relativistic or a photon-electron interaction?",
        ],
        "equation_hint": "Identify the model first, then match the variables to the corresponding relation before substituting constants.",
    },
    "math_agent": {
        "course": "PHYS 1201/1202",
        "domain": "physics math",
        "diagram": "variable map",
        "diagnostics": [
            "What physics quantity does each variable represent?",
            "What equation are we rearranging or evaluating, and what is the target variable?",
            "What units should the final expression or number have?",
        ],
        "equation_hint": "Keep the target variable isolated symbolically before substituting numbers.",
    },
}


REPRESENTATIVE_TUTORING_PROBLEMS: List[Dict[str, Any]] = [
    {
        "template": "constant_velocity",
        "agent_id": "kinematics_agent",
        "problem": "A cyclist moves at a constant velocity of 6 m/s for 12 s. How far does the cyclist travel?",
        "expected_terms": ["constant", "6 m/s", "distance"],
    },
    {
        "template": "constant_acceleration",
        "agent_id": "kinematics_agent",
        "problem": "A car accelerates from rest at 3 m/s^2 for 5 s. How far does it travel?",
        "expected_terms": ["1d constant-acceleration", "v0 = 0 m/s", "sign"],
    },
    {
        "template": "projectile_motion",
        "agent_id": "kinematics_agent",
        "problem": "A ball is launched at 18 m/s at 35 degrees. Find its flight time and range.",
        "expected_terms": ["projectile", "18 m/s", "35 deg"],
    },
    {
        "template": "newtons_second_law",
        "agent_id": "forces_agent",
        "problem": "A 4 kg box has a net horizontal force of 20 N. Find its acceleration.",
        "expected_terms": ["newton", "20 n", "sum f_x"],
    },
    {
        "template": "friction",
        "agent_id": "forces_agent",
        "problem": "A 5 kg block slides down a 30 degree incline with friction. Find its acceleration.",
        "expected_terms": ["free-body", "forces", "axes"],
    },
    {
        "template": "work_energy",
        "agent_id": "energy_agent",
        "problem": "A 2 kg cart starts from rest on a 4 m high track. Find its speed at the bottom.",
        "expected_terms": ["system", "initial", "non-conservative"],
    },
    {
        "template": "momentum_collision",
        "agent_id": "momentum_agent",
        "problem": "A 2 kg cart moving at 4 m/s sticks to a 3 kg cart at rest. Find the final speed.",
        "expected_terms": ["isolated", "before", "collision"],
    },
    {
        "template": "torque_equilibrium",
        "agent_id": "angular_motion_agent",
        "problem": "A 3 m beam is in static equilibrium with a 50 N weight at one end. Find the support force.",
        "expected_terms": ["torque", "rotation axis", "equilibrium"],
    },
    {
        "template": "electric_field_force",
        "agent_id": "electromagnetism_agent",
        "problem": "Two point charges are 0.2 m apart. Find the electric force between them.",
        "expected_terms": ["sources", "geometry", "sign"],
    },
    {
        "template": "simple_circuit",
        "agent_id": "electromagnetism_agent",
        "problem": "A 12 V battery is connected to a 6 ohm resistor. Find the current in the circuit.",
        "expected_terms": ["circuit", "12 v", "v = i*r"],
    },
]


MISCONCEPTION_PATTERNS: List[Dict[str, str]] = [
    {
        "pattern": r"\bheavier\b.*\bfall\b.*\bfaster\b|\bmore mass\b.*\bfall\b.*\bfaster\b",
        "correction": "In ideal free fall, mass does not change the gravitational acceleration. Air resistance is the usual reason heavier and lighter objects behave differently in everyday situations.",
        "question": "What acceleration should both objects have if we ignore air resistance?",
    },
    {
        "pattern": r"\bnormal\b.*\balways\b.*\bmg\b|\bN\s*=\s*mg\b.*\balways\b",
        "correction": "The normal force equals mg only in special cases, such as a level surface with no vertical acceleration and no extra vertical forces.",
        "question": "What does the force balance perpendicular to the surface look like for this situation?",
    },
    {
        "pattern": r"\bstatic friction\b.*\b(mu|u|coefficient)\b.*\bN\b|\bfriction\b.*\balways\b.*\b(mu|u)\b.*\bN\b",
        "correction": "Kinetic friction is usually mu_k N, but static friction adjusts up to a maximum of mu_s N.",
        "question": "Is the object already sliding, or are we checking whether static friction can hold?",
    },
    {
        "pattern": r"\b(acceleration|a)\b.*\bzero\b.*\b(top|highest)\b|\btop\b.*\bacceleration\b.*\bzero\b",
        "correction": "At the top of projectile motion, the vertical velocity is momentarily zero, but acceleration is still downward with magnitude g.",
        "question": "Which variable becomes zero at the top: velocity, acceleration, or both?",
    },
    {
        "pattern": r"\bcentripetal\b.*\boutward\b|\bcentrifugal\b.*\bforce\b.*\boutward\b",
        "correction": "For circular motion, the net centripetal force points inward toward the center of the circle.",
        "question": "What real force or component points inward to provide the centripetal acceleration?",
    },
    {
        "pattern": r"\bmomentum\b.*\bconserved\b.*\bexternal\b|\bconserve momentum\b.*\bexternal force\b",
        "correction": "Momentum is conserved for an isolated system, or when the external impulse is negligible during the interaction.",
        "question": "What system can we choose so external forces are negligible during the collision or explosion?",
    },
    {
        "pattern": r"\bvoltage\b.*\bsame\b.*\bseries\b|\bseries\b.*\bvoltage\b.*\bsame\b",
        "correction": "In a series circuit, current is the same through each element; voltage generally divides across the elements.",
        "question": "Which quantity must be equal through all series resistors before applying Ohm's law?",
    },
    {
        "pattern": r"\bcurrent\b.*\bused up\b|\bcharge\b.*\bused up\b",
        "correction": "Current is not used up by circuit elements. Energy is transferred, while charge flow is continuous in a closed circuit.",
        "question": "Where does the energy go in the circuit element if the current entering and leaving is the same?",
    },
    {
        "pattern": r"\bheat\b.*\bsame\b.*\btemperature\b|\btemperature\b.*\bsame\b.*\bheat\b",
        "correction": "Heat is energy transferred because of a temperature difference; temperature describes the thermal state of the material.",
        "question": "Are we solving for energy transfer Q, a temperature change, or an equilibrium temperature?",
    },
    {
        "pattern": r"\blonger wavelength\b.*\bhigher frequency\b|\bhigher frequency\b.*\blonger wavelength\b",
        "correction": "For a fixed wave speed, frequency and wavelength are inversely related by v = f lambda.",
        "question": "If the wave speed stays the same and wavelength increases, what must happen to frequency?",
    },
    {
        "pattern": r"\bimage distance\b.*\balways\b.*\bpositive\b|\bdi\b.*\balways\b.*\bpositive\b",
        "correction": "Image distance depends on the optical sign convention. Virtual images often have negative image distance.",
        "question": "Does the ray diagram predict a real image or a virtual image?",
    },
    {
        "pattern": r"\brelativity\b.*\b(any|all)\b.*\bspeed\b|\brelativistic\b.*\bslow\b",
        "correction": "Relativistic corrections exist in principle, but they are usually negligible unless the speed is a significant fraction of c.",
        "question": "What is the object's speed as a fraction of c?",
    },
]


FULL_SOLUTION_PATTERNS = (
    r"\bshow\b.*\bfull solution\b",
    r"\bgive\b.*\b(full|complete)\b.*\bsolution\b",
    r"\bjust\b.*\banswer\b",
    r"\bsolve\b.*\bfor me\b",
    r"\bwhat'?s the answer\b",
)


WORKED_EXAMPLE_PATTERNS = (
    r"\bworked example\b",
    r"\bexample solution\b",
    r"\bwalk me through\b.*\bexample\b",
    r"\bi need\b.*\bworked\b.*\bexample\b",
)


STUCK_PATTERNS = (
    r"\bi('?m| am)? stuck\b",
    r"\bi don'?t know\b",
    r"\bno idea\b",
    r"\bnot sure\b",
    r"\bconfused\b",
)


PHYSICS_UNITS = (
    "m/s", "m/s^2", "m/s2", "kg", "newton", " n", "joule", " j", "watt",
    "hz", "ohm", "volt", "amp", "coulomb", "tesla", "rad/s", "ev", "nm",
    "cm", "meter", "second", "degrees", "degree", "pa", "kpa", "atm",
    "liter", "liters", " l ",
)


KINEMATICS_VARIANT_TOPICS = {
    "piecewise_motion": "piecewise_motion",
    "vertical_motion": "vertical_motion",
    "free_fall": "free_fall",
    "horizontal_launch": "horizontal_launch",
    "projectile_motion": "projectile_motion",
}


def _classify_kinematics_motion_variant(normalized: str) -> Optional[str]:
    if _looks_like_piecewise_kinematics(normalized):
        return "piecewise_motion"
    if not _has_kinematics_launch_or_gravity_cue(normalized):
        return None
    if _looks_like_horizontal_launch(normalized):
        return "horizontal_launch"
    if _looks_like_general_projectile_motion(normalized):
        return "projectile_motion"
    if _looks_like_free_fall_motion(normalized):
        return "free_fall"
    if _looks_like_vertical_motion(normalized):
        return "vertical_motion"
    return None


def _has_kinematics_launch_or_gravity_cue(normalized: str) -> bool:
    cues = (
        "projectile",
        "launched",
        "launch",
        "thrown",
        "throw",
        "kicked",
        "dropped",
        "drop",
        "released",
        "falls",
        "falling",
        "free fall",
        "upward",
        "downward",
        "straight up",
        "straight down",
        "highest point",
        "maximum height",
        "how high",
        "rise",
    )
    return any(cue in normalized for cue in cues)


def _looks_like_horizontal_launch(normalized: str) -> bool:
    return bool(
        re.search(r"\b(?:launched|thrown|kicked|projected)\s+horizontally\b", normalized)
        or re.search(r"\b(?:rolls?|rolled|slides?|slid)\s+off\b", normalized)
        or "horizontal launch" in normalized
        or "horizontal velocity" in normalized
        or "v0x" in normalized
    )


def _looks_like_general_projectile_motion(normalized: str) -> bool:
    has_launch_cue = any(cue in normalized for cue in ("projectile", "launched", "launch", "thrown", "kicked"))
    has_angle = bool(
        re.search(r"\bat\s+(?:an\s+)?angle\b", normalized)
        or "above the horizontal" in normalized
        or "below the horizontal" in normalized
        or (has_launch_cue and re.search(r"\b\d+(?:\.\d+)?\s*(?:degrees?|deg)\b", normalized))
    )
    has_components = any(
        cue in normalized
        for cue in (
            "v0x",
            "v0y",
            "x-component",
            "y-component",
            "x component",
            "y component",
            "horizontal component",
            "vertical component",
            "horizontal and vertical",
        )
    )
    has_horizontal_distance = any(
        cue in normalized
        for cue in ("horizontal distance", "horizontal displacement", "range")
    )
    return has_angle or has_components or (has_launch_cue and has_horizontal_distance)


def _looks_like_free_fall_motion(normalized: str) -> bool:
    return bool(
        "free fall" in normalized
        or re.search(r"\b(?:dropped|drops?|released|falls?)\s+(?:from\s+)?rest\b", normalized)
        or re.search(r"\b(?:dropped|drops?|released)\b", normalized)
    )


def _looks_like_vertical_motion(normalized: str) -> bool:
    if _looks_like_horizontal_launch(normalized) or _looks_like_general_projectile_motion(normalized):
        return False
    vertical_cues = (
        "upward",
        "downward",
        "straight up",
        "straight down",
        "thrown up",
        "thrown upward",
        "thrown down",
        "thrown downward",
        "maximum height",
        "highest point",
        "how high",
        "rise",
        "vertical",
    )
    if any(cue in normalized for cue in vertical_cues):
        return True
    launch_words = ("launched", "launch", "thrown", "throw", "kicked", "projectile")
    has_launch_word = any(word in normalized for word in launch_words)
    has_speed = _first_number_before_units(normalized, ("m/s", "meter/s", "meters/s")) is not None
    return has_launch_word and has_speed


def append_guided_tutoring_policy(system_prompt: str) -> str:
    """Append the shared guided tutoring policy to an agent prompt."""

    return f"{system_prompt.rstrip()}\n\n{GUIDED_TUTORING_SYSTEM_POLICY.strip()}"


def apply_full_solution_instruction(problem: str, context: Optional[Dict[str, Any]]) -> str:
    """Prefix the problem when the API gate has approved a full solution."""

    tutoring_context = (context or {}).get("guided_tutoring") or {}
    if tutoring_context.get("full_solution_allowed"):
        reason = tutoring_context.get("reason", "guided_workflow_complete")
        return f"{FULL_SOLUTION_APPROVED_PREFIX}\nReason: {reason}\n\n{problem}"

    return problem


def merge_tutoring_context(
    context: Optional[Dict[str, Any]],
    decision: GuidedTutoringDecision,
) -> Dict[str, Any]:
    """Attach tutoring decision metadata to context passed into an agent."""

    merged = dict(context or {})
    merged["guided_tutoring"] = {
        **decision.metadata,
        "stage": decision.stage,
        "full_solution_allowed": decision.allow_full_solution,
    }
    return merged


def active_problem_from_context(
    agent_id: str,
    current_message: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the physics problem currently being tutored, if context has one."""

    explicit_problem = (context or {}).get("active_problem")
    if isinstance(explicit_problem, str) and explicit_problem.strip():
        return explicit_problem.strip()

    context_messages = _context_messages(context)
    prior_messages = _prior_messages(context_messages, current_message)
    active_problem = _active_problem_text(prior_messages, current_message, agent_id)

    if _matches_any(_normalize(active_problem), FULL_SOLUTION_PATTERNS):
        intervals = _piecewise_intervals_from_history(prior_messages)
        if len(intervals) >= 2:
            return _piecewise_problem_from_intervals(intervals)

    return active_problem


def _piecewise_problem_from_intervals(intervals: List[Dict[str, float]]) -> str:
    segments = []
    for index, interval in enumerate(intervals):
        duration = interval.get("duration")
        if duration is None:
            duration = interval["end"] - interval["start"]
        velocity = interval["velocity"]

        if abs(velocity) < 1e-9:
            phrase = f"stops for {_format_number(duration)} s"
        else:
            phrase = f"moves at {_format_number(velocity)} m/s for {_format_number(duration)} s"

        if index == 0:
            segments.append(f"A runner {phrase}")
        else:
            segments.append(f"then {phrase}")

    return " ".join(segments) + ". Find the total distance traveled and draw the velocity-time graph."


def classify_physics_topic(message: str, agent_id: str = "") -> Dict[str, Any]:
    """Classify a PHYS 1201/1202 prompt into a tutoring topic."""

    normalized = _normalize_physics_text(message)
    kinematics_variant = _classify_kinematics_motion_variant(normalized)
    if kinematics_variant:
        config = _get_config("kinematics_agent")
        return {
            "topic": KINEMATICS_VARIANT_TOPICS[kinematics_variant],
            "agent_id": "kinematics_agent",
            "course": "PHYS 1201",
            "domain": config["domain"],
            "model": _infer_model_label(config, message),
        }

    for rule in TOPIC_CLASSIFIER_RULES:
        if any(re.search(pattern, normalized) for pattern in rule["patterns"]):
            config = _get_config(rule["agent_id"])
            return {
                "topic": rule["topic"],
                "agent_id": rule["agent_id"],
                "course": rule["course"],
                "domain": config["domain"],
                "model": _infer_model_label(config, message),
            }

    config = _get_config(agent_id)
    return {
        "topic": config["domain"].replace(" ", "_"),
        "agent_id": agent_id or "math_agent",
        "course": config["course"],
        "domain": config["domain"],
        "model": _infer_model_label(config, message),
    }


def evaluate_guided_tutoring(
    agent_id: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
) -> GuidedTutoringDecision:
    """Decide whether to intercept a request with guided tutoring."""

    normalized = _normalize(message)
    config = _get_config(agent_id)
    context_messages = _context_messages(context)
    prior_messages = _prior_messages(context_messages, message)
    guided_history = _has_guided_history(prior_messages)
    student_turns_after_guidance = _student_turns_after_guidance(prior_messages)
    previous_attempts = _count_student_attempts(prior_messages)
    current_attempt = _has_student_attempt(message)
    current_reasoning_attempt = _has_explicit_reasoning_attempt(message)
    problem_solving = _looks_like_problem_solving(normalized, agent_id)
    active_problem = _active_problem_text(prior_messages, message, agent_id)
    classification = classify_physics_topic(active_problem, agent_id)
    physics_question = problem_solving or _looks_like_physics_question(normalized, agent_id)
    conceptual_question = _looks_like_conceptual_question(normalized, agent_id)
    requests_full_solution = _matches_any(normalized, FULL_SOLUTION_PATTERNS)
    needs_worked_example = _matches_any(normalized, WORKED_EXAMPLE_PATTERNS)
    misconception = _detect_misconception(normalized)

    if guided_history:
        problem_solving = True

    base_metadata = {
        "intercepted": False,
        "course": config["course"],
        "domain": config["domain"],
        "guided_history": guided_history,
        "student_turns_after_guidance": student_turns_after_guidance,
        "student_attempt_detected": current_attempt,
        "previous_attempts": previous_attempts,
        "topic_classification": classification,
    }

    if needs_worked_example:
        return GuidedTutoringDecision(
            intercept=False,
            response=None,
            allow_full_solution=True,
            stage="worked_example_requested",
            metadata={
                **base_metadata,
                "reason": "explicit_worked_example_request",
            },
        )

    if requests_full_solution and (guided_history or previous_attempts > 0 or current_reasoning_attempt):
        return GuidedTutoringDecision(
            intercept=False,
            response=None,
            allow_full_solution=True,
            stage="full_solution_after_guidance",
            metadata={
                **base_metadata,
                "reason": "student_requested_full_solution_after_guidance",
            },
        )

    if requests_full_solution and not guided_history:
        return _intercept(
            stage="full_solution_deferred",
            config=config,
            response=_initial_response(config, active_problem, full_solution_deferred=True),
            metadata={
                **base_metadata,
                "reason": "full_solution_requested_before_guidance",
            },
        )

    if misconception and physics_question:
        return _intercept(
            stage="misconception",
            config=config,
            response=_misconception_response(misconception, config),
            metadata={
                **base_metadata,
                "reason": "misconception_detected",
                "misconception": misconception["correction"],
            },
        )

    if conceptual_question and not guided_history:
        return _intercept(
            stage="conceptual_checkpoint",
            config=config,
            response=_conceptual_initial_response(config, active_problem, classification),
            metadata={
                **base_metadata,
                "reason": "first_conceptual_turn",
            },
        )

    if not problem_solving:
        return GuidedTutoringDecision(
            intercept=False,
            response=None,
            allow_full_solution=False,
            stage="not_problem_solving",
            metadata={
                **base_metadata,
                "reason": "conceptual_or_non_problem_request",
            },
        )

    if not guided_history:
        return _intercept(
            stage="initial_diagnostic",
            config=config,
            response=_initial_response(config, active_problem),
            metadata={
                **base_metadata,
                "reason": "first_problem_solving_turn",
            },
        )

    if conceptual_question or _looks_like_conceptual_question(_normalize(active_problem), agent_id):
        return _intercept(
            stage="conceptual_guidance",
            config=config,
            response=_conceptual_attempt_response(config, message, active_problem, classification, current_attempt),
            metadata={
                **base_metadata,
                "reason": "conceptual_followup",
            },
        )

    if _matches_any(normalized, STUCK_PATTERNS):
        return _intercept(
            stage="stuck_hint",
            config=config,
            response=_stuck_response(config, current_attempt),
            metadata={
                **base_metadata,
                "reason": "student_stuck_after_guidance",
            },
        )

    specialized_response = _specialized_attempt_feedback_response(
        config,
        message,
        active_problem,
        prior_messages,
    )
    if specialized_response:
        return _intercept(
            stage="specialized_guidance",
            config=config,
            response=specialized_response,
            metadata={
                **base_metadata,
                "reason": "active_specialized_workflow",
            },
        )

    if not current_attempt:
        return _intercept(
            stage="prompt_for_attempt",
            config=config,
            response=_prompt_for_attempt_response(config),
            metadata={
                **base_metadata,
                "reason": "no_student_attempt_detected",
            },
        )

    return _intercept(
        stage=_attempt_stage(message),
        config=config,
        response=_attempt_feedback_response(
            config,
            message,
            student_turns_after_guidance,
            active_problem,
            prior_messages,
        ),
        metadata={
            **base_metadata,
            "reason": "student_attempt_needs_next_step",
        },
    )


def _intercept(
    stage: str,
    config: Dict[str, Any],
    response: str,
    metadata: Dict[str, Any],
) -> GuidedTutoringDecision:
    return GuidedTutoringDecision(
        intercept=True,
        response=response,
        allow_full_solution=False,
        stage=stage,
        metadata={
            **metadata,
            "intercepted": True,
            "course": config["course"],
            "domain": config["domain"],
        },
    )


def _initial_response(
    config: Dict[str, Any],
    problem: str = "",
    full_solution_deferred: bool = False,
) -> str:
    graph_response = _graph_initial_response(config, problem, full_solution_deferred)
    if graph_response:
        return graph_response

    piecewise_response = _piecewise_kinematics_initial_response(config, problem, full_solution_deferred)
    if piecewise_response:
        return piecewise_response

    vertical_response = _vertical_kinematics_initial_response(config, problem, full_solution_deferred)
    if vertical_response:
        return vertical_response

    horizontal_launch_response = _horizontal_launch_initial_response(config, problem, full_solution_deferred)
    if horizontal_launch_response:
        return horizontal_launch_response

    concrete_response = _kinematics_initial_response(config, problem, full_solution_deferred)
    if concrete_response:
        return concrete_response

    profiled_response = _profiled_initial_response(config, problem, full_solution_deferred)
    if profiled_response:
        return profiled_response

    lead = (
        "I can give a full worked solution, but first let's do a quick setup check so the answer is not just a black box."
        if full_solution_deferred
        else "Before we calculate, let's make sure the setup is solid."
    )
    diagnostics = "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(config["diagnostics"], start=1)
    )
    worked_example_line = (
        '\n\nIf this is for a worked example rather than a problem you are trying, say "I need a worked example" and I will write it out fully.'
        if full_solution_deferred
        else '\n\nAfter you try the setup, you can ask "show me the full solution" and I will write it out.'
    )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"{lead}\n\n"
        f"{diagnostics}\n\n"
        f"Start by sending your {config['diagram']} plus a short knowns/unknowns list."
        f"{worked_example_line}"
    )


def _misconception_response(misconception: Dict[str, str], config: Dict[str, Any]) -> str:
    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"One idea to fix first: {misconception['correction']}\n\n"
        f"Try this next: {misconception['question']}\n\n"
        f"Then update your {config['diagram']} or equation choice so we can keep going."
    )


def _conceptual_initial_response(
    config: Dict[str, Any],
    problem: str,
    classification: Dict[str, Any],
) -> str:
    model = classification.get("model") or _infer_model_label(config, problem)
    focus = _conceptual_focus_question(config, problem)

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        "Let's reason this out together instead of jumping straight to the explanation.\n\n"
        f"Topic to test: {model}.\n\n"
        f"Checkpoint: {focus}\n\n"
        "Reply with one sentence about your current thinking. I will use that to give the next hint."
    )


def _conceptual_attempt_response(
    config: Dict[str, Any],
    message: str,
    problem: str,
    classification: Dict[str, Any],
    current_attempt: bool,
) -> str:
    if not current_attempt:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Before I explain it, I need one bit of your thinking so I can guide the right part.\n\n"
            f"{_conceptual_focus_question(config, problem)}"
        )

    model = classification.get("model") or _infer_model_label(config, problem)
    hint = _conceptual_hint(config, problem, message)

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"Good, that gives us something to test. For this {config['domain']} idea, the model is {model}.\n\n"
        f"Hint: {hint}\n\n"
        "Now answer the checkpoint in one short sentence. After that, you can ask \"show me the full solution\" if you want the complete explanation."
    )


def _conceptual_focus_question(config: Dict[str, Any], problem: str) -> str:
    text = _normalize(problem)
    domain = config.get("domain", "")

    if "top" in text and ("projectile" in text or "acceleration" in text):
        return "At the top of the motion, which quantity becomes zero: velocity, acceleration, or only one component of velocity?"
    if domain == "forces":
        return "What interaction or force do you think causes the motion or balance in this situation?"
    if domain == "energy":
        return "Which quantity do you think is conserved or transferred: energy, work, or both?"
    if domain == "momentum":
        return "What system would you choose, and do you think external impulse is negligible?"
    if domain == "rotational motion":
        return "What point would you choose as the rotation axis, and which torque direction is positive?"
    if domain == "waves":
        return "Which feature controls the behavior here: frequency, wavelength, wave speed, amplitude, or boundary condition?"
    if domain == "electricity and magnetism":
        if any(word in text for word in ("circuit", "current", "voltage", "resistor")):
            return "What do you think stays the same in the circuit, and what gets shared or divided?"
        if "magnet" in text:
            return "Which direction rule or moving charge/current interaction do you think matters?"
        return "What source creates the field or potential, and what test object feels it?"
    if domain == "optics":
        return "Does your ray diagram predict a real image or a virtual image, and why?"
    return "What principle do you think controls the situation, and what part feels uncertain?"


def _conceptual_hint(config: Dict[str, Any], problem: str, message: str) -> str:
    text = _normalize(problem)
    attempt = _normalize(message)
    domain = config.get("domain", "")

    if "top" in text and ("projectile" in text or "acceleration" in text):
        return "Separate velocity from acceleration. Gravity still acts downward even when vertical velocity is momentarily zero."
    if domain == "forces":
        return "Start from the free-body diagram. A change in motion comes from net force, while balance means the force components sum to zero."
    if domain == "energy":
        return "Compare the initial and final states. Conservative forces store energy; non-conservative work changes mechanical energy."
    if domain == "momentum":
        return "Momentum conservation depends on the system choice. Check whether external impulse is negligible during the interaction."
    if domain == "rotational motion":
        return "Torque depends on force, lever arm, and rotation direction. In equilibrium, clockwise and counterclockwise torques balance."
    if domain == "waves":
        if "amplitude" in attempt and "speed" in text:
            return "For many waves in a fixed medium, speed is set by the medium, while amplitude changes energy, not wave speed."
        return "First identify what is fixed by the medium or boundary condition, then use that to reason about frequency and wavelength."
    if domain == "electricity and magnetism":
        if any(word in text for word in ("series", "circuit")):
            return "In a single series path, charge cannot pile up, so current is the same through each element; voltage can divide."
        if "potential" in text or "voltage" in text:
            return "Electric potential is energy per charge. Field points in the direction a positive test charge would accelerate."
        if "magnet" in text:
            return "Use the right-hand rule after identifying velocity/current direction and magnetic-field direction."
        return "Use source and test object language: charges create fields/potentials, and other charges respond to them."
    if domain == "optics":
        return "Use the principal rays first. The sign of image distance should match whether the image is real or virtual."
    return "Name the conserved quantity, interaction, or model first; then test whether your explanation matches that principle."


def _stuck_response(config: Dict[str, Any], current_attempt: bool) -> str:
    if current_attempt:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, you have a starting point. The next useful move is narrower:\n\n"
            f"{config['equation_hint']}\n\n"
            "Send the equation with the known values substituted, and include the units on each quantity."
        )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        "No worries. Let's shrink the problem to the first checkpoint:\n\n"
        f"Draw or describe the {config['diagram']}, then list only the givens and the unknown. "
        "Do not calculate yet. Once that is visible, the equation choice gets much easier."
    )


def _prompt_for_attempt_response(config: Dict[str, Any]) -> str:
    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        "I need one piece of reasoning from you before I take over.\n\n"
        f"Use your {config['diagram']} to choose a starting equation. "
        f"Hint: {config['equation_hint']}\n\n"
        "Send that equation and tell me what each symbol represents."
    )


def _attempt_feedback_response(
    config: Dict[str, Any],
    message: str,
    student_turns_after_guidance: int,
    problem: str = "",
    prior_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    attempt_text = _student_attempt_text_after_guidance(prior_messages or [], message)

    graph_response = _graph_attempt_response(config, message, problem, attempt_text, prior_messages or [])
    if graph_response:
        return graph_response

    concrete_response = _kinematics_attempt_response(
        config,
        attempt_text,
        problem,
        student_turns_after_guidance,
    )
    if concrete_response:
        return concrete_response

    adaptive_response = _adaptive_attempt_response(
        config,
        message,
        attempt_text,
        problem,
        student_turns_after_guidance,
    )
    if adaptive_response:
        return adaptive_response

    equations = _equation_candidates(config, problem)
    has_equation = _has_model_equation(attempt_text, equations)
    has_units = _has_units(attempt_text)

    if not has_equation:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Your knowns/unknowns are a good start. Now choose the relationship that connects them.\n\n"
            f"Hint: {config['equation_hint']}\n\n"
            "What equation would you write before substituting numbers?"
        )

    if not has_units:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"That {config['domain']} equation choice is the right kind of move. Add units to the givens before calculating; units are a quick error check.\n\n"
            "Now substitute the values and simplify one line. What units should the final answer have?"
        )

    if student_turns_after_guidance >= 2:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "You have done enough setup work for me to give more direct help now.\n\n"
            "Try the arithmetic once, then ask \"show me the full solution\" if you want me to write the complete worked version and compare it to your attempt."
        )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"Nice, you have a {config['domain']} equation and units in play. The next check is substitution and reasonableness.\n\n"
        "Plug in the known values, carry the units through the calculation, and tell me whether the sign and size of the result make physical sense."
    )


def _specialized_attempt_feedback_response(
    config: Dict[str, Any],
    message: str,
    problem: str,
    prior_messages: List[Dict[str, str]],
) -> Optional[str]:
    if not prior_messages:
        return None

    if config.get("domain") == "kinematics":
        parsed = _parse_kinematics_graph_inputs(problem)
        active_piecewise = (
            parsed.get("graph_type") == "piecewise_motion"
            or _active_piecewise_graph_workflow(problem, prior_messages)
        )
        if active_piecewise:
            if parsed.get("graph_type") != "piecewise_motion":
                parsed = {
                    **parsed,
                    "graph_type": "piecewise_motion",
                    "intervals": _piecewise_intervals_from_history(prior_messages),
                }
            attempt_text = _student_attempt_text_after_guidance(prior_messages, message)
            piecewise_response = _piecewise_graph_attempt_response(
                message,
                attempt_text or message,
                parsed,
                prior_messages,
            )
            if piecewise_response:
                return piecewise_response

            return _piecewise_graph_fallback_response(
                parsed.get("intervals", []),
                _last_guided_response(prior_messages),
            )

    return None


def _adaptive_attempt_response(
    config: Dict[str, Any],
    message: str,
    attempt_text: str,
    problem: str,
    student_turns_after_guidance: int,
) -> Optional[str]:
    if not problem.strip():
        return None

    quantities = _extract_quantities(problem)
    model = _infer_model_label(config, problem)
    target = _infer_target_unknown(problem, config)
    equations = _equation_candidates(config, problem)
    equation_focus = _best_equation_candidate(equations, attempt_text, target)
    quantity_lines = _format_quantity_bullets(quantities)
    expected_unit = _expected_unit_hint(target, config, problem)
    has_equation = _has_model_equation(attempt_text, equations)
    has_units = _has_units(attempt_text)
    has_result = _has_numeric_result(message)

    if has_result and has_equation and has_units:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, you have reached a numerical answer with an equation and units visible.\n\n"
            f"Final check before I call it complete: the target is {target}, so the expected unit is {expected_unit}. "
            "Compare your result against that unit and give one reason the sign or size is physically reasonable.\n\n"
            "After that, ask \"show me the full solution\" if you want the polished worked version to compare with your attempt."
        )

    if not has_equation:
        equation_lines = "\n".join(f"- {equation}" for equation in equations[:3])
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"Good start. For this exact prompt, the model to test is {model}, and the target is {target}.\n\n"
            f"Known quantities from the problem:\n{quantity_lines}\n\n"
            "Next checkpoint: choose the relation that connects those knowns to the target. Useful candidates are:\n"
            f"{equation_lines}\n\n"
            "Reply with the one equation you want to use and map each symbol to a value from the prompt."
        )

    if not has_units:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"That equation direction fits the {model} setup. Now make it checkable with units.\n\n"
            f"Use the prompt values:\n{quantity_lines}\n\n"
            f"Expected unit for {target}: {expected_unit}.\n\n"
            f"Next checkpoint: substitute into {equation_focus} with units attached to every number. "
            "Do not skip the unit cancellation."
        )

    if has_equation and has_units and not has_result:
        if student_turns_after_guidance >= 2:
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                "You have enough setup on the page for a more direct comparison now.\n\n"
                f"Use {equation_focus}, simplify the units to {expected_unit}, and compute the number once. "
                "Then ask \"show me the full solution\" if you want me to write the full worked version next to your attempt."
            )

        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"Good, the setup has a solid base now: {model}, target {target}, and units are visible.\n\n"
            f"Next checkpoint: do one substitution line using {equation_focus}. "
            f"The final unit should reduce to {expected_unit}.\n\n"
            "Send your numerical result plus one reasonableness check: sign, direction, or approximate size."
        )

    return None


def _attempt_stage(message: str) -> str:
    if _has_equation(message):
        return "equation_attempt"
    if _has_knowns_language(message):
        return "knowns_attempt"
    return "student_attempt"


def _profiled_initial_response(
    config: Dict[str, Any],
    problem: str,
    full_solution_deferred: bool,
) -> Optional[str]:
    if not problem.strip():
        return None

    quantities = _extract_quantities(problem)
    model = _infer_model_label(config, problem)
    equations = _equation_candidates(config, problem)
    target = _infer_target_unknown(problem, config)
    diagnostics = "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(config["diagnostics"], start=1)
    )
    quantity_lines = (
        "\n".join(f"- {quantity}" for quantity in quantities[:8])
        if quantities
        else "- I do not see enough numerical givens yet; list the values and units from the problem statement."
    )
    equation_lines = "\n".join(f"- {equation}" for equation in equations[:4])
    lead = (
        "I can give the full worked solution, but first let's build a reliable setup from this exact prompt."
        if full_solution_deferred
        else "Let's build the setup from this exact prompt before doing the calculation."
    )
    worked_example_line = (
        '\n\nIf this is meant to be a worked example, say "I need a worked example" and I will write the full solution.'
        if full_solution_deferred
        else '\n\nAfter you try the setup, you can ask "show me the full solution" and I will write it out.'
    )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"{lead}\n\n"
        f"Likely model to test: {model}.\n\n"
        f"Knowns/unknowns, pulled from your wording:\n{quantity_lines}\n"
        f"- Target unknown: {target}.\n\n"
        f"Diagram/setup checkpoint: use a {config['diagram']} and mark directions, signs, and units before substituting numbers.\n\n"
        f"Equation candidates to choose from:\n{equation_lines}\n\n"
        f"Diagnostic questions:\n{diagnostics}\n\n"
        "Your next reply should choose the model/equation and map each symbol to one value from the prompt. "
        "Do not calculate the final number yet."
        f"{worked_example_line}"
    )


def _graph_initial_response(
    config: Dict[str, Any],
    problem: str,
    full_solution_deferred: bool,
) -> Optional[str]:
    if not _is_graph_request(problem):
        return None

    if config.get("domain") == "kinematics":
        return _kinematics_graph_initial_response(problem, full_solution_deferred)

    quantities = _extract_quantities(problem)
    quantity_lines = (
        "\n".join(f"- {quantity}" for quantity in quantities[:8])
        if quantities
        else "- No numerical scale is clear yet; identify the values that set the graph's axes."
    )
    equations = "\n".join(f"- {equation}" for equation in _equation_candidates(config, problem)[:4])
    lead = (
        "I can make this a worked graph example, but first let's define the graph correctly."
        if full_solution_deferred
        else "Graph questions still need a setup checkpoint before any answer is plotted."
    )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"{lead}\n\n"
        f"Likely graph model: {_infer_model_label(config, problem)}.\n\n"
        f"Known quantities or scales I can read:\n{quantity_lines}\n\n"
        "Graph setup checkpoint:\n"
        "- What goes on the horizontal axis, and what are its units?\n"
        "- What goes on the vertical axis, and what are its units?\n"
        "- Should the graph show a relationship, a time history, or a spatial pattern?\n"
        "- What shape should we expect before plotting: line, curve, inverse curve, sinusoid, or discrete points?\n\n"
        f"Equation candidates:\n{equations}\n\n"
        "Your next reply should define the axes and choose the equation that generates the graph."
    )


def _kinematics_graph_initial_response(
    problem: str,
    full_solution_deferred: bool,
) -> str:
    normalized = _normalize_physics_text(problem)
    parsed = _parse_kinematics_graph_inputs(problem)
    known_lines = _format_known_lines(parsed.get("knowns", []))
    graph_type = parsed.get("graph_type", "kinematics graph")
    missing_lines = parsed.get("missing_lines", [])
    missing_text = ""
    if missing_lines:
        missing_text = "\n\nMissing or assumption check:\n" + "\n".join(
            f"- {line}" for line in missing_lines
        )

    if graph_type == "piecewise_motion":
        intervals = parsed.get("intervals", [])
        interval_lines = _format_piecewise_interval_lines(intervals)
        lead = (
            "I can give the finished graph values, but first let's separate the intervals."
            if full_solution_deferred
            else "This is a piecewise motion graph, so we should not treat it as one constant-acceleration interval."
        )
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"{lead}\n\n"
            f"Piecewise intervals I can read:\n{interval_lines}\n\n"
            "For a velocity-time graph, each constant-velocity interval is a horizontal line. "
            "Distance comes from the area under the velocity-time graph.\n\n"
            f"Confirm these {len(intervals)} intervals first. Then we will calculate the distance."
        )

    if graph_type == "projectile_motion":
        scaffold = (
            "Graph setup checkpoint:\n"
            "- For trajectory: horizontal axis x in meters, vertical axis y in meters.\n"
            "- For time graphs: horizontal axis t in seconds; vertical axes can be x, y, vx, or vy.\n"
            "- Resolve launch velocity first: v0x = v0 cos(theta), v0y = v0 sin(theta).\n"
            "- Use x(t) = x0 + v0x*t and y(t) = h0 + v0y*t - (1/2)g*t^2.\n"
            "- The flight ends when y(t) = 0 if the object lands on the ground."
        )
        next_step = "Your next reply: identify which graph(s) you want and compute v0x and v0y symbolically or numerically."
    elif graph_type == "horizontal_launch":
        scaffold = (
            "Graph setup checkpoint:\n"
            "- For trajectory: horizontal axis x in meters, vertical axis y in meters.\n"
            "- Horizontal velocity is constant: x(t) = x0 + v0x*t.\n"
            "- Vertical motion starts with v0y = 0 and uses y(t) = h0 - (1/2)g*t^2.\n"
            "- The graph ends when y(t) = 0."
        )
        next_step = "Your next reply: confirm the axes and the landing condition before calculating graph points."
    elif _is_constant_acceleration_graph_setup(parsed):
        return _constant_acceleration_graph_initial_response(parsed, problem, full_solution_deferred)
    else:
        motion_variant = _classify_kinematics_motion_variant(normalized)
        if motion_variant == "vertical_motion":
            scaffold = (
                "Graph setup checkpoint:\n"
                "- Use a vertical y-axis with upward positive unless the prompt says otherwise.\n"
                "- For a velocity-time graph, the slope is -g because gravity points downward.\n"
                "- For a height-time graph, the curve is concave down.\n"
                "- At the highest point, vertical velocity is 0 m/s."
            )
            next_step = "Your next reply: identify the requested graph type and mark the vertical direction/sign convention."
        elif motion_variant == "free_fall":
            scaffold = (
                "Graph setup checkpoint:\n"
                "- Use a vertical y-axis and choose a positive direction.\n"
                "- If the object is dropped from rest, v0 = 0 m/s.\n"
                "- The velocity-time graph changes linearly because acceleration is constant at g downward.\n"
                "- The position-time graph curves in the direction of acceleration."
            )
            next_step = "Your next reply: identify the graph axes and the sign of gravitational acceleration."
        elif _looks_like_constant_velocity_graph(normalized, parsed):
            scaffold = (
                "Graph setup checkpoint:\n"
                "- Velocity-time graph: horizontal line at the constant velocity.\n"
                "- Position-time graph: straight line with slope equal to velocity.\n"
                "- Acceleration-time graph: horizontal line at 0 m/s^2.\n"
                "- Distance comes from area under the velocity-time graph."
            )
            next_step = "Your next reply: identify the graph axes and the constant velocity value."
        else:
            scaffold = (
                "Graph setup checkpoint:\n"
                "- First identify the motion subtype: constant velocity, constant acceleration, vertical/free fall, or piecewise motion.\n"
                "- Then choose the graph rule for that subtype.\n"
                "- Do not use constant-acceleration equations unless the motion actually has constant acceleration."
            )
            next_step = "Your next reply: name the motion subtype and the graph axes before choosing any equation."

    if _is_constant_acceleration_graph_setup(parsed):
        return _constant_acceleration_graph_initial_response(parsed, problem, full_solution_deferred)

    if graph_type not in ("projectile_motion", "horizontal_launch") and _is_constant_acceleration_graph_setup(parsed):
        return _constant_acceleration_graph_initial_response(parsed, problem, full_solution_deferred)

    lead = (
        "I can give the finished graph values, but first let's define the graph correctly."
        if full_solution_deferred
        else "Let's set up the graph from the actual prompt before calculating graph points."
    )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"{lead}\n\n"
        f"Likely graph type: {graph_type.replace('_', ' ')}.\n\n"
        f"Knowns I can read from the prompt:\n{known_lines}"
        f"{missing_text}\n\n"
        f"{scaffold}\n\n"
        f"{next_step}"
    )

def _graph_attempt_response(
    config: Dict[str, Any],
    message: str,
    problem: str,
    attempt_text: str = "",
    prior_messages: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    if not _is_graph_request(problem):
        return None

    if config.get("domain") == "kinematics":
        parsed = _parse_kinematics_graph_inputs(problem)
        active_piecewise = (
            parsed.get("graph_type") == "piecewise_motion"
            or (
                prior_messages is not None
                and _active_piecewise_graph_workflow(problem, prior_messages)
            )
        )
        if active_piecewise:
            if parsed.get("graph_type") != "piecewise_motion":
                parsed = {
                    **parsed,
                    "graph_type": "piecewise_motion",
                    "intervals": _piecewise_intervals_from_history(prior_messages or []),
                }
            piecewise_response = _piecewise_graph_attempt_response(
                message,
                attempt_text or message,
                parsed,
                prior_messages or [],
            )
            if piecewise_response:
                return piecewise_response

            return _piecewise_graph_fallback_response(
                parsed.get("intervals", []),
                _last_guided_response(prior_messages or []),
            )

        if parsed.get("graph_type") == "projectile_motion":
            projectile_response = _projectile_graph_attempt_response(attempt_text or message, parsed)
            if projectile_response:
                return projectile_response

            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                "Good. For a projectile graph, the solid next step is component setup, not plotting yet.\n\n"
                "Check these before graphing:\n"
                "- v0x = v0 cos(theta), which stays constant if air resistance is ignored.\n"
                "- v0y = v0 sin(theta), which changes because gravity acts vertically.\n"
                "- x(t) is linear in time; y(t) is a downward-opening parabola.\n"
                "- The trajectory y versus x is also a downward-opening parabola.\n\n"
                "Send v0x, v0y, and the landing condition you will use for the end time."
            )

        endpoint_response = _kinematics_graph_endpoint_response(attempt_text or message, parsed)
        if endpoint_response:
            return endpoint_response

        progress_response = _kinematics_graph_progress_response(attempt_text or message, parsed, problem)
        if progress_response:
            return progress_response

        return _kinematics_graph_subtype_attempt_response(parsed, problem)

    general_graph_response = _general_graph_attempt_response(config, attempt_text or message, problem)
    if general_graph_response:
        return general_graph_response

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        "Good. Before plotting, make the graph definition precise:\n\n"
        "- State the horizontal-axis variable and units.\n"
        "- State the vertical-axis variable and units.\n"
        "- Write the equation or proportionality that links them.\n"
        "- Predict the shape qualitatively before using numbers.\n\n"
        f"For this topic, the most likely starting relation is: {_equation_candidates(config, problem)[0]}."
    )


def _is_constant_acceleration_graph_setup(parsed: Dict[str, Any]) -> bool:
    return (
        parsed.get("graph_type") == "kinematics_1d"
        and _known_value(parsed, "a") is not None
        and _known_value(parsed, "t") is not None
    )


def _looks_like_constant_velocity_graph(normalized: str, parsed: Dict[str, Any]) -> bool:
    if any(cue in normalized for cue in ("constant velocity", "constant speed", "at rest", "stopped")):
        return True
    return (
        parsed.get("graph_type") == "kinematics_1d"
        and _known_value(parsed, "v0") is not None
        and _known_value(parsed, "a") is None
        and _known_value(parsed, "t") is not None
    )


def _kinematics_graph_subtype_attempt_response(
    parsed: Dict[str, Any],
    problem: str,
) -> str:
    normalized = _normalize_physics_text(problem)
    motion_variant = _classify_kinematics_motion_variant(normalized)

    if _looks_like_constant_velocity_graph(normalized, parsed):
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Keep this as a constant-velocity graph.\n\n"
            "For a velocity-time graph, use a horizontal line at the constant velocity. "
            "For a position-time graph, use a straight line whose slope is that velocity.\n\n"
            "Next checkpoint: state the graph axes and the two endpoint coordinates for the segment."
        )

    if motion_variant == "vertical_motion":
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Keep this as vertical 1D motion under gravity.\n\n"
            "For graphing, choose upward or downward as positive first. Then use gravity's sign to decide the slope of the velocity-time graph and the curvature of the position-time graph.\n\n"
            "Next checkpoint: state the axes and your sign convention."
        )

    if motion_variant == "free_fall":
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Keep this as free fall, not a generic horizontal-motion graph.\n\n"
            "The acceleration is gravitational and constant downward. The velocity-time graph is linear; the position-time graph curves in the acceleration direction.\n\n"
            "Next checkpoint: state the axes and whether downward or upward is positive."
        )

    if _is_constant_acceleration_graph_setup(parsed):
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "This is the constant-acceleration graph case.\n\n"
            "Before writing functions, confirm the acceleration sign and the graph axes. Then we will use the one equation needed for that graph."
        )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        "Before graphing, identify the motion subtype first: constant velocity, constant acceleration, piecewise motion, vertical/free fall, or projectile motion.\n\n"
        "Once the subtype is clear, we will use the graph rule for that subtype. Do not use constant-acceleration formulas unless the motion actually has constant acceleration."
    )


def _constant_acceleration_graph_initial_response(
    parsed: Dict[str, Any],
    problem: str,
    full_solution_deferred: bool,
) -> str:
    v0 = _known_value(parsed, "v0")
    acceleration = _known_value(parsed, "a")
    time = _known_value(parsed, "t")
    if v0 is None:
        v0 = 0.0
    v0_reason = " because the car starts from rest" if abs(v0) < 1e-9 and "rest" in _normalize(problem) else ""
    requested = _requested_kinematics_graph(problem)
    graph_text = (
        "velocity-time graph"
        if requested == "velocity"
        else f"{requested}-time graph"
    )
    lead = (
        "I can give the finished graph values, but first let's confirm the setup."
        if full_solution_deferred
        else "This is a constant-acceleration problem."
    )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"{lead}\n\n"
        "Knowns:\n"
        f"- v0 = {_format_number(v0)} m/s{v0_reason}\n"
        f"- a = {_format_signed_number(acceleration)} m/s^2\n"
        f"- t = {_format_number(time)} s\n\n"
        "Before calculating, confirm:\n"
        "1. Is the acceleration positive, negative, or zero?\n"
        f"2. Since the prompt asks for a {graph_text}, what should go on each axis?"
    )


def _projectile_graph_attempt_response(
    attempt_text: str,
    parsed: Dict[str, Any],
) -> Optional[str]:
    normalized = _normalize_physics_text(attempt_text)
    has_components = (
        ("v0x" in normalized and "v0y" in normalized)
        or ("v_0x" in normalized and "v_0y" in normalized)
        or ("cos" in normalized and "sin" in normalized)
    )
    has_landing_condition = any(
        cue in normalized
        for cue in ("y(t) = 0", "y=0", "lands", "landing", "ground", "flight time", "end time")
    )

    if has_components and has_landing_condition:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, the component setup and graph endpoint condition are now in place.\n\n"
            "Next checkpoint for the graph:\n"
            "- Use x(t) = x0 + v0x*t for horizontal position.\n"
            "- Use y(t) = h0 + v0y*t - (1/2)g*t^2 for height.\n"
            "- Stop the plotted curve at the positive time where y(t) = 0.\n\n"
            "Now solve or estimate that positive end time, then use it to get the final x-value for the trajectory graph."
        )

    if has_components:
        height = _known_value(parsed, "h0")
        height_text = (
            f"h0 = {_format_number(height)} m"
            if height is not None
            else "the starting height from the prompt"
        )
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, the velocity components are the right next step for a projectile graph.\n\n"
            f"Now set the endpoint condition. With {height_text}, the vertical equation is:\n"
            "y(t) = h0 + v0y*t - (1/2)g*t^2\n\n"
            "For a ground landing, the graph should stop at the positive time where y(t) = 0. "
            "Write that landing equation next; do not plot beyond that time."
        )

    return None


def _general_graph_attempt_response(
    config: Dict[str, Any],
    attempt_text: str,
    problem: str,
) -> Optional[str]:
    normalized = _normalize_physics_text(attempt_text)
    has_axes = _mentions_graph_axes(normalized)
    has_relation = _has_equation(normalized) or _mentions_graph_relation(normalized)
    has_shape = _mentions_graph_shape(normalized)
    relation = _equation_candidates(config, problem)[0]

    if has_axes and has_relation and has_shape:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, the graph definition is no longer the blocker.\n\n"
            f"Use the relation {relation} to create plotting anchors. "
            "Pick two or three horizontal-axis values, compute the matching vertical-axis values with units, "
            "and mark any intercept, asymptote, or maximum/minimum that applies."
        )

    if has_axes and has_relation:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, you have the axes and the governing relation. Before calculating points, predict the shape.\n\n"
            f"Using {relation}, should the graph be linear, inverse, quadratic/parabolic, sinusoidal, or horizontal? "
            "State the shape and one feature such as slope, intercept, or asymptote."
        )

    if has_axes:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "The axes are set. Now choose the equation or proportionality that generates the graph.\n\n"
            f"For this topic, test this relation first: {relation}. "
            "Reply with the relation and say how each variable matches your axes."
        )

    if has_relation:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "The relation is a useful start. Now make the graph itself precise.\n\n"
            "State the horizontal axis, vertical axis, and units for both. "
            "Then predict the graph shape before computing points."
        )

    return None


def _kinematics_graph_endpoint_response(
    message: str,
    parsed: Dict[str, Any],
) -> Optional[str]:
    v_result = _extract_velocity_result(message)
    if v_result is None:
        return None

    v0 = _known_value(parsed, "v0")
    acceleration = _known_value(parsed, "a")
    time = _known_value(parsed, "t")
    if acceleration is None or time is None:
        return None
    if v0 is None:
        v0 = 0.0

    expected = v0 + acceleration * time
    if abs(v_result - expected) <= max(0.05, abs(expected) * 0.01):
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"Correct. The endpoint velocity is v({_format_number(time)} s) = {_format_number(expected)} m/s.\n\n"
            "That means the velocity-time graph is a straight line:\n"
            f"- start point: (0 s, {_format_number(v0)} m/s)\n"
            f"- end point: ({_format_number(time)} s, {_format_number(expected)} m/s)\n"
            f"- slope: {_format_signed_number(acceleration)} m/s^2, which is the acceleration\n\n"
            "Final graph check: label the vertical axis v in m/s and the horizontal axis t in s. "
            "If you also need the displacement, the area under this v-t line gives it."
        )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"Your graph setup is on track, but v({_format_number(time)} s) should not be {_format_number(v_result)} m/s for these givens.\n\n"
        "Use the velocity function for a constant-acceleration graph:\n"
        "v(t) = v0 + a*t\n\n"
        f"Substitute: v({_format_number(time)}) = {_format_number(v0)} + "
        f"({_format_number(acceleration)})({_format_number(time)}). "
        "What endpoint velocity does that give?"
    )


def _kinematics_graph_progress_response(
    message: str,
    parsed: Dict[str, Any],
    problem: str,
) -> Optional[str]:
    normalized = _normalize_physics_text(message)
    has_interval = _has_graph_time_interval(normalized)
    acceleration_sign = _graph_acceleration_sign(normalized)
    has_graph_equation = _has_graph_equation(normalized)
    has_axes = _mentions_graph_axes(normalized)

    if not (has_interval or acceleration_sign or has_graph_equation or has_axes):
        return None

    v0 = _known_value(parsed, "v0")
    acceleration = _known_value(parsed, "a")
    time = _known_value(parsed, "t")
    if time is None:
        time = _extract_graph_time_end(normalized)
    if v0 is None:
        v0 = 0.0

    if acceleration_sign and has_axes and acceleration is not None and time is not None:
        requested = _requested_kinematics_graph(problem)
        if requested == "velocity":
            slope_text = _format_signed_number(acceleration)
            end_time = _format_number(time)
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                "Great, the setup is confirmed.\n\n"
                "For this velocity-time graph, use one equation now:\n"
                "v(t) = v0 + a*t\n\n"
                f"With v0 = {_format_number(v0)} m/s and a = {slope_text} m/s^2, "
                f"the graph starts at (0 s, {_format_number(v0)} m/s) and has slope {slope_text} m/s^2.\n\n"
                f"Next checkpoint: compute v({end_time}) = {_format_number(v0)} + "
                f"({_format_number(acceleration)})({end_time}). What endpoint velocity do you get?"
            )

    if has_interval and acceleration_sign and acceleration is not None and time is not None:
        slope_text = _format_signed_number(acceleration)
        end_time = _format_number(time)
        requested = _requested_kinematics_graph(problem)

        if requested == "position":
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                "Good, that finishes the graph setup checkpoint.\n\n"
                "For the position-time graph, use:\n"
                "x(t) = x0 + v0*t + (1/2)*a*t^2\n\n"
                f"From the prompt: v0 = {_format_number(v0)} m/s, a = {slope_text} m/s^2, "
                f"and the interval is 0 <= t <= {end_time} s. If no starting position is given, take x0 = 0 m.\n\n"
                f"Next checkpoint: compute x({end_time}) = 0 + ({_format_number(v0)})({end_time}) "
                f"+ (1/2)({_format_number(acceleration)})({end_time})^2. "
                "That endpoint anchors the curve."
            )

        if requested == "acceleration":
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                "Good, that finishes the graph setup checkpoint.\n\n"
                "For the acceleration-time graph, the function is:\n"
                "a(t) = a\n\n"
                f"From the prompt: a = {slope_text} m/s^2 on 0 <= t <= {end_time} s. "
                "So the graph is a horizontal line at that value, not a sloped line.\n\n"
                "Next checkpoint: what are the two endpoints of that horizontal segment?"
            )

        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, that finishes the graph setup checkpoint.\n\n"
            "For the velocity-time graph, use:\n"
            "v(t) = v0 + a*t\n\n"
            f"From the prompt: v0 = {_format_number(v0)} m/s, a = {slope_text} m/s^2, "
            f"and the interval is 0 <= t <= {end_time} s.\n\n"
            f"So the line starts at (0 s, {_format_number(v0)} m/s) and has slope {slope_text} m/s^2. "
            f"Next checkpoint: compute v({end_time}) = {_format_number(v0)} + "
            f"({_format_number(acceleration)})({end_time}). What is the endpoint velocity?"
        )

    missing_parts = []
    if not has_interval:
        missing_parts.append("the time interval, such as 0 <= t <= 6 s")
    if not acceleration_sign:
        missing_parts.append("whether acceleration is positive, negative, or zero")

    if missing_parts:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "You are moving in the right direction. One setup piece is still missing before we draw points:\n\n"
            f"Please add {', and '.join(missing_parts)}. "
            "Then we can turn the function into graph endpoints."
        )

    return None


def _piecewise_kinematics_initial_response(
    config: Dict[str, Any],
    problem: str,
    full_solution_deferred: bool,
) -> Optional[str]:
    if config.get("domain") != "kinematics":
        return None

    intervals = _parse_piecewise_kinematics_intervals(problem)
    if len(intervals) < 2:
        return None

    lead = (
        "I can give the full worked solution, but first let's separate the motion into intervals."
        if full_solution_deferred
        else "This is a piecewise motion problem, so let's split it into intervals before calculating."
    )
    worked_example_line = (
        '\n\nIf this is meant to be a worked example, say "I need a worked example" and I will write the full solution.'
        if full_solution_deferred
        else '\n\nAfter you confirm the intervals, you can ask "show me the full solution" and I will write it out.'
    )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"{lead}\n\n"
        f"Intervals I can read:\n{_format_piecewise_interval_lines(intervals)}\n\n"
        "Use constant-acceleration equations only for intervals that actually have acceleration. "
        "Here, the useful idea is distance from each interval: area under the velocity-time graph.\n\n"
        "Checkpoint: confirm these velocity intervals first. Then we can find total distance by adding each segment's area."
        f"{worked_example_line}"
    )


def _piecewise_graph_attempt_response(
    current_message: str,
    attempt_text: str,
    parsed: Dict[str, Any],
    prior_messages: List[Dict[str, str]],
) -> Optional[str]:
    normalized_current = _normalize_physics_text(current_message)
    normalized = _normalize_physics_text(attempt_text)
    last_checkpoint = _normalize_physics_text(_last_guided_response(prior_messages))
    intervals = parsed.get("intervals", [])
    if not intervals:
        return None

    state_response = _piecewise_graph_state_response(normalized_current, last_checkpoint, intervals)
    if state_response:
        return state_response

    confirmed = _piecewise_intervals_are_confirmed(normalized, intervals)
    mentions_area = "area" in normalized or "v*t" in normalized or "v * t" in normalized

    if confirmed and mentions_area:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, you have the piecewise graph setup and the distance idea.\n\n"
            "Next checkpoint: compute each rectangular area under the velocity-time graph separately, then add them. "
            "Use distance = v*delta t for each horizontal segment."
        )

    if confirmed:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good, those are the right velocity intervals.\n\n"
            "For the velocity-time graph, draw each interval as a horizontal segment:\n"
            f"{_format_piecewise_interval_lines(intervals)}\n\n"
            "Next checkpoint: what does the area under each horizontal segment represent?"
        )

    if "acceleration" in normalized or "1/2" in normalized or "v0" in normalized:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Careful: this prompt is not one single constant-acceleration interval.\n\n"
            "Treat each segment separately. For the constant-velocity parts, the velocity-time graph is horizontal and distance is area = v*delta t.\n\n"
            f"Start by confirming:\n{_format_piecewise_interval_lines(intervals)}"
        )

    return None


def _piecewise_graph_fallback_response(
    intervals: List[Dict[str, float]],
    last_guided_response: str = "",
) -> str:
    checkpoint = _piecewise_checkpoint_kind(_normalize_physics_text(last_guided_response))
    interval_lines = _format_piecewise_interval_lines(intervals)

    if checkpoint == "confirm_intervals":
        next_step = "Confirm these intervals first. Then we will use area under the velocity-time graph."
    elif checkpoint == "area_meaning":
        next_step = "Answer this checkpoint: what physical quantity does area under a velocity-time graph represent?"
    elif checkpoint == "first_rectangle":
        next_step = "Use area = velocity*time for Interval 1 and send that distance in meters."
    elif checkpoint == "second_rectangle":
        next_step = "Use area = velocity*time for Interval 2 and send that distance in meters."
    elif checkpoint == "total_distance":
        next_step = "Add the interval distances and send the total distance."
    elif checkpoint == "graph_description":
        next_step = "Describe the two horizontal velocity-time segments with their time ranges and velocities."
    else:
        next_step = (
            "Continue one interval at a time: confirm the intervals if needed, then use rectangular area "
            "under each velocity-time segment for distance."
        )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        "Stay in the piecewise motion workflow. There is no single constant-acceleration function for the whole motion.\n\n"
        f"{interval_lines}\n\n"
        "For the velocity-time graph, each constant-velocity interval is a horizontal segment. "
        "Distance comes from the area under each segment.\n\n"
        f"{next_step}"
    )


def _active_piecewise_graph_workflow(
    problem: str,
    prior_messages: List[Dict[str, str]],
) -> bool:
    if _parse_piecewise_kinematics_intervals(problem):
        return True

    history_text = _normalize_physics_text(" ".join(message["content"] for message in prior_messages))
    if "piecewise motion graph" in history_text or "piecewise motion problem" in history_text:
        return True
    if "velocity-time graph" in history_text and len(_piecewise_intervals_from_history(prior_messages)) >= 2:
        return True
    return False


def _piecewise_intervals_from_history(
    prior_messages: List[Dict[str, str]],
) -> List[Dict[str, float]]:
    intervals: List[Dict[str, float]] = []
    history_text = _normalize_physics_text(" ".join(message["content"] for message in prior_messages))

    for match in re.finditer(
        r"interval\s+\d+\s*:\s*([-+]?\d+(?:\.\d+)?)\s*(?:-|to)\s*([-+]?\d+(?:\.\d+)?)\s*s\s*,?\s*v\s*=\s*([-+]?\d+(?:\.\d+)?)\s*m/s",
        history_text,
    ):
        start = float(match.group(1))
        end = float(match.group(2))
        velocity = float(match.group(3))
        if end <= start:
            continue
        intervals.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "velocity": velocity,
        })

    if intervals:
        return intervals

    for message in reversed(prior_messages):
        if message["role"] != "user":
            continue
        intervals = _parse_piecewise_kinematics_intervals(message["content"])
        if intervals:
            return intervals

    return []


def _piecewise_graph_state_response(
    normalized_current: str,
    last_checkpoint: str,
    intervals: List[Dict[str, float]],
) -> Optional[str]:
    checkpoint = _piecewise_checkpoint_kind(last_checkpoint)

    if checkpoint == "confirm_intervals":
        if (
            _piecewise_confirmation_is_valid(normalized_current)
            or _piecewise_intervals_are_confirmed(normalized_current, intervals)
        ):
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                "Great, the intervals are confirmed.\n\n"
                "What does the area under a velocity-time graph represent?"
            )

        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Before calculating, confirm the two velocity intervals first:\n"
            f"{_format_piecewise_interval_lines(intervals)}"
        )

    if checkpoint == "area_meaning":
        if _piecewise_area_meaning_is_correct(normalized_current):
            first = intervals[0]
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                "Correct: on a velocity-time graph, area represents distance traveled for each interval.\n\n"
                "Now calculate the area of the first rectangle:\n"
                f"- base = {_format_number(first['duration'])} s\n"
                f"- height = {_format_number(first['velocity'])} m/s\n\n"
                "What distance does Interval 1 give?"
            )

        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Close, but connect the graph to the units: velocity times time gives meters.\n\n"
            "For a velocity-time graph, what physical quantity has units of meters?"
        )

    if checkpoint == "first_rectangle":
        return _piecewise_rectangle_response(
            normalized_current,
            intervals,
            interval_index=0,
            next_step="second_rectangle",
        )

    if checkpoint == "second_rectangle":
        return _piecewise_rectangle_response(
            normalized_current,
            intervals,
            interval_index=1,
            next_step="add_distances",
        )

    if checkpoint == "total_distance":
        result = _extract_displacement_result(normalized_current)
        expected = sum(abs(interval["velocity"]) * interval["duration"] for interval in intervals)
        if result is not None and _numbers_close(result, expected):
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                f"Yes. The total distance is {_format_number(expected)} m.\n\n"
                "Final graph checkpoint: describe the velocity-time graph in words before we call it complete.\n\n"
                "What are the two horizontal segments, including their time ranges and velocities?"
            )

        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Add the interval distances only after each rectangle area is clear.\n\n"
            f"Use: total distance = {_format_number(abs(intervals[0]['velocity']) * intervals[0]['duration'])} m "
            f"+ {_format_number(abs(intervals[1]['velocity']) * intervals[1]['duration'])} m. "
            "What total distance do you get?"
        )

    if checkpoint == "graph_description":
        if _piecewise_graph_description_is_correct(normalized_current, intervals):
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                "Correct. The velocity-time graph is made of these horizontal segments:\n"
                f"{_format_piecewise_interval_lines(intervals)}\n\n"
                "You have completed the guided setup: intervals, graph meaning, each rectangle area, total distance, and graph description. "
                "If you want the polished write-up, ask \"show me the full solution\"."
            )

        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Almost. Keep the velocity-time graph piecewise and horizontal.\n\n"
            f"Describe these two segments:\n{_format_piecewise_interval_lines(intervals)}"
        )

    return None


def _piecewise_checkpoint_kind(last_checkpoint: str) -> Optional[str]:
    if not last_checkpoint:
        return None

    if (
        "confirm these" in last_checkpoint
        or "confirm the intervals" in last_checkpoint
        or "confirm these velocity intervals" in last_checkpoint
        or "confirm these 2 intervals" in last_checkpoint
        or "confirm the two velocity intervals" in last_checkpoint
    ):
        return "confirm_intervals"

    if (
        "what does the area under each horizontal segment represent" in last_checkpoint
        or "what does the area under a velocity-time graph represent" in last_checkpoint
        or "what physical quantity has units of meters" in last_checkpoint
    ):
        return "area_meaning"

    if (
        "calculate the area of the first rectangle" in last_checkpoint
        or ("interval 1" in last_checkpoint and "what distance" in last_checkpoint)
        or ("interval 1" in last_checkpoint and "velocity" in last_checkpoint and "time" in last_checkpoint)
        or ("interval 1" in last_checkpoint and "rectangle area" in last_checkpoint)
    ):
        return "first_rectangle"

    if (
        "calculate the area of the second rectangle" in last_checkpoint
        or ("interval 2" in last_checkpoint and "what distance" in last_checkpoint)
        or ("interval 2" in last_checkpoint and "velocity" in last_checkpoint and "time" in last_checkpoint)
        or ("interval 2" in last_checkpoint and "rectangle area" in last_checkpoint)
    ):
        return "second_rectangle"

    if (
        "add the two distances" in last_checkpoint
        or "what is the total distance" in last_checkpoint
        or "what total distance" in last_checkpoint
    ):
        return "total_distance"

    if (
        "final graph checkpoint" in last_checkpoint
        or ("describe" in last_checkpoint and "velocity-time graph" in last_checkpoint)
        or ("two horizontal segments" in last_checkpoint and "time ranges" in last_checkpoint)
    ):
        return "graph_description"

    return None


def _piecewise_rectangle_response(
    normalized_current: str,
    intervals: List[Dict[str, float]],
    interval_index: int,
    next_step: str,
) -> Optional[str]:
    if interval_index >= len(intervals):
        return None

    result = _extract_displacement_result(normalized_current)
    interval = intervals[interval_index]
    expected = abs(interval["velocity"]) * interval["duration"]
    interval_number = interval_index + 1
    if result is None:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"Use area = velocity*time for Interval {interval_number}.\n\n"
            f"Substitute: ({_format_number(abs(interval['velocity']))} m/s)({_format_number(interval['duration'])} s). "
            "What distance does that give in meters?"
        )

    if not _numbers_close(result, expected):
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"Check Interval {interval_number} again. Its rectangle area should use "
            f"{_format_number(abs(interval['velocity']))} m/s times {_format_number(interval['duration'])} s.\n\n"
            "What distance does that give?"
        )

    if next_step == "second_rectangle" and len(intervals) > 1:
        second = intervals[1]
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"Correct. Interval 1 gives {_format_number(expected)} m.\n\n"
            "Now calculate the area of the second rectangle:\n"
            f"- base = {_format_number(second['duration'])} s\n"
            f"- height = {_format_number(second['velocity'])} m/s\n\n"
            "What is the distance during Interval 2?"
        )

    first_distance = abs(intervals[0]["velocity"]) * intervals[0]["duration"]
    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"Correct. Interval {interval_number} gives {_format_number(expected)} m.\n\n"
        "Now add the two distances:\n"
        f"- Interval 1: {_format_number(first_distance)} m\n"
        f"- Interval 2: {_format_number(expected)} m\n\n"
        "What is the total distance traveled?"
    )


def _piecewise_area_meaning_is_correct(normalized_current: str) -> bool:
    return any(
        cue in normalized_current
        for cue in (
            "distance",
            "displacement",
            "change in position",
            "meters",
            "m traveled",
        )
    )


def _piecewise_confirmation_is_valid(normalized_current: str) -> bool:
    stripped = normalized_current.strip(" .,!?:;")
    if stripped in {"yes", "y", "yeah", "yep", "correct", "confirm", "confirmed", "yes confirm", "looks right", "right"}:
        return True
    return (
        any(cue in normalized_current for cue in ("yes", "correct", "confirm", "confirmed", "looks right"))
        and not any(cue in normalized_current for cue in ("no", "wrong", "incorrect"))
    )


def _piecewise_intervals_are_confirmed(
    normalized_text: str,
    intervals: List[Dict[str, float]],
) -> bool:
    return (
        "interval" in normalized_text
        and all(_format_number(interval["velocity"]) in normalized_text for interval in intervals)
        and all(_format_number(interval["end"]) in normalized_text for interval in intervals)
    )


def _piecewise_graph_description_is_correct(
    normalized_current: str,
    intervals: List[Dict[str, float]],
) -> bool:
    if "horizontal" not in normalized_current:
        return False
    for interval in intervals:
        if _format_number(interval["velocity"]) not in normalized_current:
            return False
        compact_range = f"{_format_number(interval['start'])}-{_format_number(interval['end'])}"
        word_range = f"{_format_number(interval['start'])} to {_format_number(interval['end'])}"
        if compact_range not in normalized_current and word_range not in normalized_current:
            return False
    return (
        "v" in normalized_current
        or "velocity" in normalized_current
        or "m/s" in normalized_current
    )


def _numbers_close(value: float, expected: float) -> bool:
    return abs(value - expected) <= max(0.05, abs(expected) * 0.01)


def _vertical_kinematics_initial_response(
    config: Dict[str, Any],
    problem: str,
    full_solution_deferred: bool,
) -> Optional[str]:
    if config.get("domain") != "kinematics":
        return None

    normalized = _normalize_physics_text(problem)
    variant = _classify_kinematics_motion_variant(normalized)
    if variant not in ("vertical_motion", "free_fall"):
        return None

    setup = _parse_vertical_motion_setup(problem, variant)
    known_lines = _format_vertical_motion_knowns(setup, variant)
    lead = (
        "This is a free-fall problem, which is one-dimensional vertical motion under gravity."
        if variant == "free_fall"
        else "This is a one-dimensional vertical motion problem."
    )
    worked_example_line = (
        '\n\nIf you need this as a worked example after the setup check, say "I need a worked example" and I will write it out fully.'
        if full_solution_deferred
        else ""
    )

    if setup.get("highest_point"):
        checkpoint = (
            "Before calculating, identify:\n"
            "- Which quantity is the unknown?\n"
            "- Which kinematics equation relates v, v0, a, and displacement without introducing time?"
        )
    elif setup.get("target") == "time":
        checkpoint = (
            "Before calculating, identify:\n"
            "- Which quantity is the unknown?\n"
            "- Which kinematics equation relates displacement, v0, a, and time without needing final velocity?"
        )
    else:
        checkpoint = (
            "Before calculating, identify:\n"
            "- Which quantity is the unknown?\n"
            "- With upward as positive, what signs should velocity, acceleration, and displacement have?"
        )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"{lead}\n\n"
        f"Knowns:\n{known_lines}\n\n"
        f"{checkpoint}"
        f"{worked_example_line}"
    )


def _horizontal_launch_initial_response(
    config: Dict[str, Any],
    problem: str,
    full_solution_deferred: bool,
) -> Optional[str]:
    if config.get("domain") != "kinematics":
        return None

    normalized = _normalize_physics_text(problem)
    if _classify_kinematics_motion_variant(normalized) != "horizontal_launch":
        return None

    speed = _first_number_before_units(normalized, ("m/s", "meter/s", "meters/s"))
    height = _extract_height_value(normalized)
    known_lines = []
    if speed is not None:
        known_lines.append(f"- Horizontal velocity: v0x = {_format_number(speed)} m/s")
    else:
        known_lines.append("- Horizontal velocity: identify v0x from the prompt")
    known_lines.append("- Initial vertical velocity: v0y = 0 m/s")
    if height is not None:
        known_lines.append(f"- Initial height: {_format_number(height)} m")
    known_lines.append("- Vertical acceleration: -9.8 m/s^2")
    worked_example_line = (
        '\n\nIf you need this as a worked example after the setup check, say "I need a worked example" and I will write it out fully.'
        if full_solution_deferred
        else ""
    )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        "This is a horizontal-launch problem.\n\n"
        f"Knowns:\n{chr(10).join(known_lines)}\n\n"
        "Before calculating, identify:\n"
        "- Which vertical-motion equation gives the time to hit the ground?\n"
        "- Once you have that time, which horizontal equation gives the distance?"
        f"{worked_example_line}"
    )


def _parse_vertical_motion_setup(problem: str, variant: str) -> Dict[str, Any]:
    normalized = _normalize_physics_text(problem)
    speed = _extract_initial_velocity(normalized)
    if speed is None:
        speed = _first_number_before_units(normalized, ("m/s", "meter/s", "meters/s"))

    direction = ""
    if any(cue in normalized for cue in ("upward", "straight up", "thrown up", "rise", "highest point", "how high")):
        direction = "upward"
    elif any(cue in normalized for cue in ("downward", "straight down", "thrown down", "fall", "drop")):
        direction = "downward"

    if variant == "free_fall" and ("from rest" in normalized or "rest" in normalized or speed is None):
        speed = 0.0
        direction = "from rest"

    return {
        "speed": speed,
        "direction": direction,
        "height": _extract_height_value(normalized),
        "highest_point": _is_highest_point_question(normalized),
        "target": _vertical_motion_target(normalized),
    }


def _format_vertical_motion_knowns(setup: Dict[str, Any], variant: str) -> str:
    lines = []
    speed = setup.get("speed")
    direction = setup.get("direction")
    if speed is not None:
        if variant == "free_fall" and abs(speed) < 1e-9:
            lines.append("- Initial velocity: 0 m/s because it is dropped or released from rest")
        elif direction in ("upward", "downward"):
            lines.append(f"- Initial velocity: {_format_number(speed)} m/s {direction}")
        else:
            lines.append(f"- Initial velocity: {_format_number(speed)} m/s")
    else:
        lines.append("- Initial velocity: identify the value and direction from the prompt")

    if setup.get("highest_point"):
        lines.append("- Final velocity at the highest point: 0 m/s")

    height = setup.get("height")
    if height is not None:
        lines.append(f"- Initial height or vertical displacement scale: {_format_number(height)} m")

    lines.append("- Acceleration: -9.8 m/s^2")
    return "\n".join(lines)


def _vertical_motion_target(normalized: str) -> str:
    if any(cue in normalized for cue in ("how long", "time", "when")):
        return "time"
    if any(cue in normalized for cue in ("how fast", "speed", "velocity")):
        return "velocity"
    if any(cue in normalized for cue in ("how high", "height", "rise", "highest point")):
        return "height"
    return "vertical kinematics quantity"


def _is_highest_point_question(normalized: str) -> bool:
    return any(cue in normalized for cue in ("how high", "highest point", "maximum height", "rise"))


def _kinematics_initial_response(
    config: Dict[str, Any],
    problem: str,
    full_solution_deferred: bool,
) -> Optional[str]:
    if config.get("domain") != "kinematics":
        return None

    setup = _parse_constant_acceleration_setup(problem)
    if not setup:
        return None

    lead = (
        "I can write the full solution, but first let's lock down the physics model so the answer has a real base."
        if full_solution_deferred
        else "Let's anchor the physics before calculating."
    )
    worked_example_line = (
        '\n\nIf you need this as a worked example instead, say "I need a worked example" and I will write it out fully.'
        if full_solution_deferred
        else '\n\nOnce you try this setup, you can ask "show me the full solution" and I will write the complete version.'
    )

    return (
        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
        f"{lead}\n\n"
        "Model: 1D constant-acceleration motion, because the car has a constant acceleration for a fixed time.\n\n"
        "Sign convention / coordinate choice: take the car's forward direction as +x, so the acceleration is positive.\n\n"
        "Knowns/unknowns, from the given information:\n"
        f"- Initial velocity: v0 = {_format_number(setup['v0'])} m/s, because \"from rest\" means v0 = 0.\n"
        f"- Acceleration: a = +{_format_number(setup['a'])} m/s^2.\n"
        f"- Time: t = {_format_number(setup['t'])} s.\n"
        "- Unknown: displacement, delta x.\n\n"
        "Your checkpoint: choose the kinematics equation that uses delta x, v0, a, and t without needing final velocity. "
        "Write that equation first, then substitute the values with units."
        f"{worked_example_line}"
    )


def _kinematics_attempt_response(
    config: Dict[str, Any],
    message: str,
    problem: str,
    student_turns_after_guidance: int,
) -> Optional[str]:
    if config.get("domain") != "kinematics":
        return None

    setup = _parse_constant_acceleration_setup(problem)
    if not setup:
        return None

    expected = setup["v0"] * setup["t"] + 0.5 * setup["a"] * setup["t"] ** 2
    result = _extract_displacement_result(message)
    has_equation = _has_equation(message)

    if result is not None:
        if abs(result - expected) <= max(0.05, abs(expected) * 0.01):
            return (
                f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                f"Yes. Your displacement value is correct: delta x = {_format_number(expected)} m.\n\n"
                "Why it is solid:\n"
                "- Equation: delta x = v0*t + (1/2)*a*t^2.\n"
                f"- Substitution: delta x = ({_format_number(setup['v0'])} m/s)({_format_number(setup['t'])} s) "
                f"+ (1/2)({_format_number(setup['a'])} m/s^2)({_format_number(setup['t'])} s)^2.\n"
                "- Unit check: (m/s)*s gives m, and (m/s^2)*s^2 gives m.\n"
                "- Sign check: the answer is positive because +x was chosen in the car's direction of motion.\n\n"
                "Final checkpoint for you: say in one sentence why the distance is reasonable for a car speeding up for 5 s. "
                "Then ask \"show me the full solution\" if you want the polished worked version."
            )

        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            f"Your setup is close, but the arithmetic result should not be {result:g} m for these numbers.\n\n"
            "Use the same equation:\n"
            "delta x = v0*t + (1/2)*a*t^2\n\n"
            f"Substitute carefully: delta x = ({_format_number(setup['v0'])})({_format_number(setup['t'])}) "
            f"+ (1/2)({_format_number(setup['a'])})({_format_number(setup['t'])})^2.\n\n"
            f"The first term is 0. Recalculate the second term: (1/2)*{_format_number(setup['a'])}*{_format_number(setup['t'] ** 2)}. "
            "What do you get, with meters as the unit?"
        )

    if has_equation:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Good: your equation choice is the right base for this problem.\n\n"
            "Make it precise:\n"
            "- Equation: delta x = v0*t + (1/2)*a*t^2.\n"
            f"- Substitution: delta x = ({_format_number(setup['v0'])} m/s)({_format_number(setup['t'])} s) "
            f"+ (1/2)({_format_number(setup['a'])} m/s^2)({_format_number(setup['t'])} s)^2.\n"
            "- Unit check before arithmetic: both terms reduce to meters.\n\n"
            f"Your turn: compute (1/2)*{_format_number(setup['a'])}*({_format_number(setup['t'])})^2 and report delta x in meters."
        )

    if _has_knowns_language(message) or student_turns_after_guidance >= 1:
        return (
            f"**{GUIDED_RESPONSE_MARKER}**\n\n"
            "Your knowns are the right starting point. Now connect them with the equation.\n\n"
            "Because the unknown is displacement and we know v0, a, and t, use:\n\n"
            "delta x = v0*t + (1/2)*a*t^2\n\n"
            f"Substitute but do not skip units: delta x = ({_format_number(setup['v0'])} m/s)({_format_number(setup['t'])} s) "
            f"+ (1/2)({_format_number(setup['a'])} m/s^2)({_format_number(setup['t'])} s)^2.\n\n"
            "What value do you get for delta x?"
        )

    return None


def _parse_constant_acceleration_setup(problem: str) -> Optional[Dict[str, float]]:
    normalized = _normalize_physics_text(problem)
    if _looks_like_piecewise_kinematics(normalized):
        return None

    if "accelerat" not in normalized and "from rest" not in normalized:
        return None

    acceleration_match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:m/s\^?2|m/s2|meters?\s*/\s*s(?:ec(?:ond)?s?)?\^?2)",
        normalized,
    )
    time_match = re.search(
        r"(?:for|time|t\s*=?)\s*([-+]?\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)\b",
        normalized,
    )

    if not acceleration_match or not time_match:
        return None

    v0 = 0.0 if "from rest" in normalized or re.search(r"\bv0?\s*=\s*0", normalized) else 0.0
    acceleration = float(acceleration_match.group(1))
    time = float(time_match.group(1))

    if time <= 0:
        return None

    return {"v0": v0, "a": acceleration, "t": time}


def _looks_like_piecewise_kinematics(normalized: str) -> bool:
    if "piecewise" in normalized or "multiple interval" in normalized:
        return True
    if re.search(r"\bthen\b", normalized) and len(re.findall(r"\bfor\s+[-+]?\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)\b", normalized)) >= 2:
        return True
    if re.search(r"\bstops?\s+for\s+[-+]?\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)\b", normalized):
        return True
    return False


def _parse_piecewise_kinematics_intervals(problem: str) -> List[Dict[str, float]]:
    normalized = _normalize_physics_text(problem)
    if not _looks_like_piecewise_kinematics(normalized):
        return []

    intervals: List[Dict[str, float]] = []
    elapsed = 0.0

    patterns = (
        r"(?:moves?|runs?|travels?|walks?|drives?)\s+at\s+([-+]?\d+(?:\.\d+)?)\s*m/s\s+for\s+([-+]?\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)",
        r"(?:continues?|coasts?)\s+at\s+([-+]?\d+(?:\.\d+)?)\s*m/s\s+for\s+([-+]?\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)",
    )
    matches: List[Dict[str, Any]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            matches.append({
                "start_index": match.start(),
                "velocity": float(match.group(1)),
                "duration": float(match.group(2)),
            })

    for match in re.finditer(r"(?:then\s+)?(?:stops?|is stopped|rests?|is at rest)\s+for\s+([-+]?\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)", normalized):
        matches.append({
            "start_index": match.start(),
            "velocity": 0.0,
            "duration": float(match.group(1)),
        })

    matches.sort(key=lambda item: item["start_index"])
    for item in matches:
        duration = item["duration"]
        if duration <= 0:
            continue
        start = elapsed
        end = elapsed + duration
        intervals.append({
            "start": start,
            "end": end,
            "duration": duration,
            "velocity": item["velocity"],
        })
        elapsed = end

    return intervals


def _format_piecewise_interval_lines(intervals: List[Dict[str, float]]) -> str:
    if not intervals:
        return "- I can tell there are multiple intervals, but I need the velocity and duration for each one."

    lines = []
    for index, interval in enumerate(intervals, start=1):
        lines.append(
            f"- Interval {index}: {_format_number(interval['start'])}-{_format_number(interval['end'])} s, "
            f"v = {_format_number(interval['velocity'])} m/s"
        )
    return "\n".join(lines)


def _extract_displacement_result(message: str) -> Optional[float]:
    normalized = _normalize_physics_text(message)
    patterns = (
        r"(?:result|answer|delta x|deltax|displacement|distance|dx|x|s)\s*(?:=|is|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b(?!\s*/)",
        r"([-+]?\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b(?!\s*/)",
    )

    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        if matches:
            return float(matches[-1])

    return None


def _extract_velocity_result(message: str) -> Optional[float]:
    normalized = _normalize_physics_text(message)
    patterns = (
        r"(?:v(?:\([^)]*\)|_?f)?|final velocity|endpoint velocity|velocity)\s*(?:=|is|:)?\s*([-+]?\d+(?:\.\d+)?)\s*m/s\b(?!\s*(?:\^?2|2))",
        r"([-+]?\d+(?:\.\d+)?)\s*m/s\b(?!\s*(?:\^?2|2))",
        r"\bv(?:\([^)]*\)|_?f|final)?\s*(?:=|is|:)\s*([-+]?\d+(?:\.\d+)?)\b",
    )

    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[0]
            return float(value)

    return None


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_signed_number(value: float) -> str:
    if value > 0:
        return f"+{_format_number(value)}"
    return _format_number(value)


def _parse_kinematics_graph_inputs(problem: str) -> Dict[str, Any]:
    normalized = _normalize_physics_text(problem)
    knowns = []
    missing_lines = []
    graph_type = "kinematics_1d"
    intervals = _parse_piecewise_kinematics_intervals(problem)

    if len(intervals) >= 2:
        graph_type = "piecewise_motion"
    else:
        motion_variant = _classify_kinematics_motion_variant(normalized)
        if motion_variant == "projectile_motion":
            graph_type = "projectile_motion"
        elif motion_variant == "horizontal_launch":
            graph_type = "horizontal_launch"

    speed = _first_number_before_units(normalized, ("m/s", "meter/s", "meters/s"))
    angle = _first_number_before_units(normalized, ("degree", "degrees", "deg"))
    height = _extract_height_value(normalized)
    v0 = _extract_initial_velocity(normalized)
    vf = _extract_final_velocity(normalized)
    acceleration = _first_number_before_units(normalized, ("m/s^2", "m/s2"))
    time = _extract_time_value(normalized)

    if graph_type == "piecewise_motion":
        for index, interval in enumerate(intervals, start=1):
            knowns.append((
                f"interval {index}",
                "v",
                interval["velocity"],
                f"m/s from {_format_number(interval['start'])} s to {_format_number(interval['end'])} s",
            ))
    elif graph_type == "projectile_motion":
        if speed is not None:
            knowns.append(("launch speed", "v0", speed, "m/s"))
        else:
            missing_lines.append("Launch speed is needed to compute graph scales.")
        if angle is not None:
            knowns.append(("launch angle", "theta", angle, "deg"))
        else:
            missing_lines.append("Launch angle is needed to split velocity into x and y components.")
        if height is not None:
            knowns.append(("initial height", "h0", height, "m"))
        elif any(word in normalized for word in ("hill", "bridge", "roof", "balcony", "cliff", "above")):
            missing_lines.append("Initial height is implied but not numerically specified.")
        else:
            knowns.append(("initial height", "h0", 0.0, "m assumed"))
        knowns.append(("gravity", "g", 9.81, "m/s^2 assumed downward"))
    elif graph_type == "horizontal_launch":
        if speed is not None:
            knowns.append(("horizontal velocity", "v0x", speed, "m/s"))
        else:
            missing_lines.append("Horizontal launch speed is needed to compute graph scales.")
        knowns.append(("initial vertical velocity", "v0y", 0.0, "m/s"))
        if height is not None:
            knowns.append(("initial height", "h0", height, "m"))
        else:
            missing_lines.append("Launch height is needed to determine the flight time.")
        knowns.append(("gravity", "g", 9.81, "m/s^2 assumed downward"))
    else:
        motion_variant = _classify_kinematics_motion_variant(normalized)
        if motion_variant in ("vertical_motion", "free_fall") and acceleration is None:
            acceleration = -9.81
        if motion_variant in ("vertical_motion", "free_fall") and v0 is None and speed is not None:
            v0 = speed
        if motion_variant == "free_fall" and v0 is None:
            v0 = 0.0
        if motion_variant in ("vertical_motion", "free_fall") and vf is None and _is_highest_point_question(normalized):
            vf = 0.0
        if v0 is not None:
            knowns.append(("initial velocity", "v0", v0, "m/s"))
        elif "from rest" in normalized:
            knowns.append(("initial velocity", "v0", 0.0, "m/s"))
        if vf is not None:
            knowns.append(("final velocity", "v", vf, "m/s"))
        if acceleration is not None:
            knowns.append(("acceleration", "a", acceleration, "m/s^2"))
        if time is not None:
            knowns.append(("time interval", "t", time, "s"))
        if not knowns:
            for quantity in _extract_quantities(problem):
                knowns.append(("quantity", "", quantity, ""))
        if time is None:
            missing_lines.append("A time interval or x-axis range is needed for a complete time graph.")

    return {
        "graph_type": graph_type,
        "knowns": knowns,
        "missing_lines": missing_lines,
        "intervals": intervals,
    }


def _known_value(parsed: Dict[str, Any], symbol: str) -> Optional[float]:
    for known in parsed.get("knowns", []):
        if not (isinstance(known, tuple) and len(known) == 4):
            continue
        _, known_symbol, value, _ = known
        if known_symbol == symbol and isinstance(value, (int, float)):
            return float(value)
    return None


def _has_graph_time_interval(normalized: str) -> bool:
    interval_patterns = (
        r"\btime interval\b",
        r"0\s*(?:<=|<|≤)\s*t\s*(?:<=|<|≤)\s*[-+]?\d+(?:\.\d+)?",
        r"\bt\s*(?:from|=)?\s*0\s*(?:to|-)\s*[-+]?\d+(?:\.\d+)?",
        r"\bfor\s+[-+]?\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)\b",
    )
    return any(re.search(pattern, normalized) for pattern in interval_patterns)


def _extract_graph_time_end(normalized: str) -> Optional[float]:
    patterns = (
        r"0\s*(?:<=|<|≤)\s*t\s*(?:<=|<|≤)\s*([-+]?\d+(?:\.\d+)?)",
        r"\bt\s*(?:from|=)?\s*0\s*(?:to|-)\s*([-+]?\d+(?:\.\d+)?)",
        r"\bfor\s+([-+]?\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)\b",
        r"\btime interval\b.*?([-+]?\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return None


def _graph_acceleration_sign(normalized: str) -> Optional[str]:
    if "acceleration" not in normalized and not re.search(r"\ba\s*=", normalized):
        return None
    if re.search(r"\bpositive\b|\b\+a\b|\ba\s*=\s*\+", normalized):
        return "positive"
    if re.search(r"\bnegative\b|\b-a\b|\ba\s*=\s*-", normalized):
        return "negative"
    if re.search(r"\bzero\b|\ba\s*=\s*0\b", normalized):
        return "zero"
    match = re.search(r"\ba\s*=\s*([-+]?\d+(?:\.\d+)?)", normalized)
    if match:
        value = float(match.group(1))
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "zero"
    return None


def _has_graph_equation(normalized: str) -> bool:
    return any(
        cue in normalized
        for cue in ("v(t)", "x(t)", "a(t)", "v = v0", "x = x0", "velocity-time", "position-time")
    )


def _requested_kinematics_graph(problem: str) -> str:
    normalized = _normalize(problem)
    if any(cue in normalized for cue in ("position-time", "position time", "x-t", "x(t)")):
        return "position"
    if any(cue in normalized for cue in ("acceleration-time", "acceleration time", "a-t", "a(t)")):
        return "acceleration"
    return "velocity"


def _format_known_lines(knowns: List[Any]) -> str:
    if not knowns:
        return "- I cannot read enough numerical givens yet; list the values and units that set the graph scale."

    lines = []
    for known in knowns[:8]:
        if isinstance(known, tuple) and len(known) == 4:
            label, symbol, value, unit = known
            symbol_text = f" ({symbol})" if symbol else ""
            if isinstance(value, (int, float)):
                value_text = _format_number(float(value))
            else:
                value_text = str(value)
            lines.append(f"- {label}{symbol_text}: {value_text} {unit}".strip())
        else:
            lines.append(f"- {known}")
    return "\n".join(lines)


def _format_quantity_bullets(quantities: List[str]) -> str:
    if not quantities:
        return "- No clear numerical givens were detected; list the values and units from the prompt."
    return "\n".join(f"- {quantity}" for quantity in quantities[:8])


def _best_equation_candidate(
    equations: List[str],
    attempt_text: str,
    target: str,
) -> str:
    normalized_attempt = _normalize_physics_text(attempt_text)
    for equation in equations:
        left_side = equation.split("=", 1)[0].strip().lower()
        if left_side and left_side in normalized_attempt:
            return equation

    normalized_target = _normalize(target)
    target_cues = {
        "acceleration": ("sum f", "a", "alpha"),
        "displacement": ("delta x", "x(t)"),
        "distance": ("delta x", "x(t)", "range"),
        "height": ("delta y", "v^2"),
        "velocity": ("v =", "omega"),
        "speed": ("v =", "omega"),
        "force": ("f =", "sum f"),
        "work": ("w =", "w_net"),
        "energy": ("k =", "u_", "e_photon"),
        "momentum": ("p =", "sum p"),
        "current": ("v = i", "i"),
        "voltage": ("v = i", "v"),
        "frequency": ("v = f", "f_n"),
        "wavelength": ("lambda",),
        "image distance": ("1/f",),
    }
    for cue, equation_cues in target_cues.items():
        if cue not in normalized_target:
            continue
        for equation in equations:
            equation_lower = equation.lower()
            if any(equation_cue in equation_lower for equation_cue in equation_cues):
                return equation

    return equations[0] if equations else "the relation connecting the knowns to the target"


def _expected_unit_hint(target: str, config: Dict[str, Any], problem: str) -> str:
    normalized_target = _normalize(target)
    normalized_problem = _normalize_physics_text(problem)
    domain = config.get("domain", "")

    if "graph" in normalized_target:
        return "the units named on the graph axes"
    if "acceleration" in normalized_target:
        return "m/s^2"
    if any(word in normalized_target for word in ("velocity", "speed")):
        return "m/s"
    if any(word in normalized_target for word in ("distance", "displacement", "height", "image distance", "wavelength")):
        return "m"
    if "force" in normalized_target:
        return "N"
    if any(word in normalized_target for word in ("work", "energy")):
        return "J"
    if "power" in normalized_target:
        return "W"
    if "momentum" in normalized_target:
        return "kg*m/s"
    if "torque" in normalized_target:
        return "N*m"
    if "frequency" in normalized_target:
        return "Hz"
    if "period" in normalized_target:
        return "s"
    if "amplitude" in normalized_target:
        return "m, rad, or the displacement unit used in the prompt"
    if "current" in normalized_target:
        return "A"
    if "voltage" in normalized_target or "electric potential" in normalized_target:
        return "V"
    if "charge" in normalized_target:
        return "C"
    if "temperature" in normalized_target:
        return "K or degrees C, depending on the prompt"
    if domain == "thermodynamics" and "pressure" in normalized_problem and "volume" in normalized_problem:
        return "J for work/energy, since Pa*m^3 = J"
    return "the SI unit for the requested quantity"


def _has_numeric_result(message: str) -> bool:
    normalized = _normalize_physics_text(message)
    if "known" in normalized or "given" in normalized:
        return False
    value_with_unit = r"[-+]?\d+(?:\.\d+)?\s*(?:m/s\^?2|m/s2|m/s|kg|n|j|w|hz|v|a|c|pa|kpa|ohm|tesla|rad/s|ev|nm|cm|m|s)\b"
    if re.search(rf"\b(result|answer)\b.*{value_with_unit}", normalized):
        return True
    if re.search(rf"\b(final|therefore|so)\b.*(?:=|is)\s*{value_with_unit}", normalized):
        return True
    if re.fullmatch(rf"\s*(?:result|answer)?\s*(?:=|is|:)?\s*{value_with_unit}\s*", normalized):
        return True
    return False


def _mentions_graph_axes(normalized: str) -> bool:
    horizontal = any(cue in normalized for cue in ("horizontal", "x-axis", "x axis", "xaxis"))
    vertical = any(cue in normalized for cue in ("vertical", "y-axis", "y axis", "yaxis"))
    axis_words = "axis" in normalized and any(unit in normalized for unit in ("m/s", "meter", "m ", "s ", "hz", "pa", "j", "n"))
    return (horizontal and vertical) or axis_words


def _mentions_graph_relation(normalized: str) -> bool:
    return any(
        cue in normalized
        for cue in (
            "proportional",
            "inverse",
            "versus",
            " vs ",
            "lambda",
            "sin",
            "cos",
            "slope",
            "area under",
        )
    )


def _mentions_graph_shape(normalized: str) -> bool:
    return any(
        cue in normalized
        for cue in (
            "line",
            "linear",
            "straight",
            "curve",
            "inverse",
            "parabola",
            "parabolic",
            "sinusoidal",
            "sine",
            "cosine",
            "horizontal",
            "vertical",
            "slope",
            "increasing",
            "decreasing",
            "concave",
            "asymptote",
        )
    )


def _first_number_before_units(normalized: str, unit_options: Iterable[str]) -> Optional[float]:
    for unit in unit_options:
        pattern = rf"([-+]?\d+(?:\.\d+)?)\s*{re.escape(unit)}\b"
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return None


def _extract_time_value(normalized: str) -> Optional[float]:
    match = re.search(r"(?:for|in|over|time|t\s*=?)\s*([-+]?\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)\b", normalized)
    if match:
        return float(match.group(1))
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)\b", normalized)
    if match:
        return float(match.group(1))
    return None


def _extract_initial_velocity(normalized: str) -> Optional[float]:
    if "from rest" in normalized:
        return 0.0
    patterns = (
        r"(?:from|initial(?: velocity| speed)?|v0\s*=?)\s*(?:of|is|=|:)?\s*([-+]?\d+(?:\.\d+)?)\s*m/s\b",
        r"with\s+(?:an?\s+)?initial(?: velocity| speed)?\s*(?:of|is|=|:)?\s*([-+]?\d+(?:\.\d+)?)\s*m/s\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return None


def _extract_final_velocity(normalized: str) -> Optional[float]:
    if "to rest" in normalized or "to a stop" in normalized:
        return 0.0
    match = re.search(r"(?:to|final(?: velocity)?|v\s*=?)\s*([-+]?\d+(?:\.\d+)?)\s*m/s\b", normalized)
    if match:
        return float(match.group(1))
    return None


def _extract_height_value(normalized: str) -> Optional[float]:
    patterns = (
        r"(?:from|on|atop|above)\s+(?:a\s+)?([-+]?\d+(?:\.\d+)?)\s*(?:m|meter|meters)\s+(?:hill|bridge|roof|balcony|cliff|height)",
        r"([-+]?\d+(?:\.\d+)?)\s*(?:m|meter|meters)\s+(?:high|above|hill|bridge|roof|balcony|cliff)",
        r"(?:height|h0)\s*=?\s*([-+]?\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return None


def _is_graph_request(message: str) -> bool:
    normalized = _normalize(message)
    return any(
        word in normalized
        for word in ("graph", "plot", "sketch", "draw the graph", "versus", " vs ")
    )


def _extract_quantities(problem: str) -> List[str]:
    normalized = _normalize_physics_text(problem)
    quantity_pattern = (
        r"[-+]?\d+(?:\.\d+)?\s*"
        r"(?:m/s\^?2|m/s2|m/s|rad/s|kg|kpa|pa|atm|ohm|tesla|ev|nm|cm|mm|km|liter|liters|degrees?|deg|hz|n|j|w|v|a|c|t|l|m|s|sec|seconds?)"
    )
    quantities = [match.group(0).strip() for match in re.finditer(quantity_pattern, normalized)]

    if "from rest" in normalized and not any("0 m/s" in quantity for quantity in quantities):
        quantities.insert(0, "0 m/s (from rest)")

    coefficient_pattern = r"(?:mu_k|mu_s|coefficient(?: of)?(?: kinetic| static)? friction|coefficient|μ_k|μ_s)\s*(?:is|=|:)?\s*([-+]?\d+(?:\.\d+)?)"
    for match in re.finditer(coefficient_pattern, normalized):
        quantities.append(f"friction coefficient = {match.group(1)}")

    return _unique_preserving_order(quantities)


def _infer_model_label(config: Dict[str, Any], problem: str) -> str:
    text = _normalize(problem)
    domain = config.get("domain", "")

    if domain == "kinematics":
        motion_variant = _classify_kinematics_motion_variant(_normalize_physics_text(problem))
        if _looks_like_piecewise_kinematics(_normalize_physics_text(problem)):
            return "piecewise 1D motion with separate time intervals"
        if motion_variant == "vertical_motion":
            return "1D vertical motion under gravity"
        if motion_variant == "free_fall":
            return "free fall with acceleration due to gravity"
        if motion_variant == "horizontal_launch":
            return "horizontal launch with independent horizontal and vertical motion"
        if motion_variant == "projectile_motion":
            return "projectile motion with independent horizontal and vertical components"
        if any(word in text for word in ("constant velocity", "constant speed", "no acceleration")):
            return "1D constant-velocity motion"
        if any(word in text for word in ("accelerat", "speeding up", "slowing", "brak", "from rest")):
            return "1D constant-acceleration motion"
        if any(word in text for word in ("drop", "fall", "free fall")):
            return "free fall with acceleration due to gravity"
        return "1D motion; decide whether velocity or acceleration is constant"

    if domain == "forces":
        if "incline" in text or "ramp" in text:
            return "Newton's second law on an inclined plane"
        if "friction" in text or "coefficient" in text:
            return "Newton's second law with friction"
        if "spring" in text:
            return "Hooke's law plus force balance"
        if "equilibrium" in text:
            return "static equilibrium, sum of forces equals zero"
        return "Newton's second law with a free-body diagram"

    if domain == "energy":
        if "spring" in text:
            return "spring energy and conservation/work-energy"
        if "power" in text:
            return "power from work or force and velocity"
        if "friction" in text or "nonconservative" in text:
            return "work-energy with non-conservative work"
        return "mechanical energy or work-energy theorem"

    if domain == "momentum":
        if "stick" in text or "perfectly inelastic" in text:
            return "perfectly inelastic collision"
        if "elastic" in text:
            return "elastic collision"
        if "impulse" in text or "force" in text and "time" in text:
            return "impulse-momentum theorem"
        return "conservation of momentum for an isolated system"

    if domain == "rotational motion":
        if "equilibrium" in text or "balance" in text:
            return "torque equilibrium with sum of torques equal to zero"
        if "torque" in text:
            return "torque and rotational dynamics"
        if "angular momentum" in text:
            return "conservation of angular momentum"
        if "roll" in text:
            return "rolling motion with rotational energy or torque"
        return "rotational kinematics or rotational energy"

    if domain == "thermodynamics":
        if "ideal gas" in text or "pressure" in text or "volume" in text:
            return "ideal gas process plus thermodynamic work/first law"
        if "heat" in text or "temperature" in text:
            return "heat transfer or calorimetry"
        return "first law of thermodynamics"

    if domain == "waves":
        if any(word in text for word in ("oscillat", "simple harmonic", "shm", "pendulum", "spring-mass")):
            return "simple harmonic motion or oscillations"
        if "doppler" in text:
            return "Doppler effect"
        if "standing" in text or "harmonic" in text or "string" in text or "pipe" in text:
            return "standing wave boundary conditions"
        if "intensity" in text or "decibel" in text:
            return "sound intensity and decibel scale"
        return "wave equation v = f lambda"

    if domain == "electricity and magnetism":
        if "circuit" in text or "resistor" in text or "current" in text or "voltage" in text:
            return "circuit analysis with Ohm's law and series/parallel rules"
        if "potential" in text or "voltage due to" in text or "voltage at" in text:
            return "electric potential from charges or fields"
        if "charge" in text or "electric field" in text:
            return "electric force/field using superposition"
        if "magnetic" in text or "wire" in text:
            return "magnetic force or magnetic field geometry"
        return "electric/magnetic field model; identify sources and geometry"

    if domain == "optics":
        if "lens" in text or "mirror" in text:
            return "thin lens/mirror equation with sign conventions"
        if "snell" in text or "refraction" in text:
            return "Snell's law"
        if "diffraction" in text or "slit" in text or "grating" in text:
            return "diffraction/interference condition"
        return "ray or wave optics setup"

    if domain == "modern physics":
        if "photoelectric" in text or "electron" in text and "light" in text:
            return "photoelectric effect energy balance"
        if "relativ" in text or "speed of light" in text:
            return "special relativity"
        if "decay" in text or "half-life" in text:
            return "radioactive decay"
        return "modern physics model; identify energy, momentum, or decay relation"

    return f"{domain} setup"


def _equation_candidates(config: Dict[str, Any], problem: str) -> List[str]:
    text = _normalize(problem)
    domain = config.get("domain", "")

    if domain == "kinematics":
        motion_variant = _classify_kinematics_motion_variant(_normalize_physics_text(problem))
        if _looks_like_piecewise_kinematics(_normalize_physics_text(problem)):
            return [
                "distance for each constant-velocity interval = v*delta t",
                "total distance = area under the velocity-time graph",
                "draw each velocity-time segment separately",
                "use constant-acceleration equations only for intervals with acceleration",
            ]
        if motion_variant in ("vertical_motion", "free_fall"):
            return [
                "v = v0 + a*t",
                "delta y = v0*t + (1/2)*a*t^2",
                "v^2 = v0^2 + 2*a*delta y",
                "a = -g if upward is positive",
            ]
        if motion_variant == "horizontal_launch":
            return [
                "x = v0x*t",
                "y = y0 + v0y*t - (1/2)*g*t^2",
                "v0y = 0 for a horizontal launch",
                "the landing time comes from the vertical equation",
            ]
        if motion_variant == "projectile_motion":
            return [
                "v0x = v0 cos(theta), v0y = v0 sin(theta)",
                "x(t) = x0 + v0x*t",
                "y(t) = y0 + v0y*t - (1/2)g*t^2",
                "vy(t) = v0y - g*t",
            ]
        if any(word in text for word in ("constant velocity", "constant speed", "no acceleration")):
            return [
                "delta x = v*delta t",
                "v = delta x / delta t",
                "x(t) = x0 + v*t",
                "a = 0",
            ]
        return [
            "v = v0 + a*t",
            "delta x = v0*t + (1/2)*a*t^2",
            "v^2 = v0^2 + 2*a*delta x",
            "average velocity = delta x / delta t",
        ]

    if domain == "forces":
        return [
            "sum F_x = m*a_x",
            "sum F_y = m*a_y",
            "f_k = mu_k*N or f_s <= mu_s*N",
            "weight = m*g",
        ]

    if domain == "energy":
        return [
            "K_i + U_i + W_nonconservative = K_f + U_f",
            "W_net = delta K",
            "K = (1/2)*m*v^2",
            "U_g = m*g*h or U_s = (1/2)*k*x^2",
        ]

    if domain == "momentum":
        return [
            "sum p_before = sum p_after for an isolated system",
            "p = m*v",
            "J = F*delta t = delta p",
            "elastic collisions also conserve kinetic energy",
        ]

    if domain == "rotational motion":
        if "equilibrium" in text or "balance" in text:
            return [
                "sum tau = 0",
                "tau = r*F*sin(theta)",
                "sum F_x = 0",
                "sum F_y = 0",
            ]
        return [
            "theta = theta0 + omega0*t + (1/2)*alpha*t^2",
            "omega = omega0 + alpha*t",
            "sum tau = I*alpha",
            "K_rot = (1/2)*I*omega^2",
        ]

    if domain == "thermodynamics":
        return [
            "P*V = n*R*T",
            "W = P*delta V for constant pressure",
            "delta U = Q - W",
            "Q = m*c*delta T",
        ]

    if domain == "waves":
        if any(word in text for word in ("oscillat", "simple harmonic", "shm", "pendulum", "spring-mass")):
            return [
                "x(t) = A*cos(omega*t + phi)",
                "omega = sqrt(k/m) for a mass-spring oscillator",
                "T = 2*pi*sqrt(m/k) or T = 2*pi*sqrt(L/g)",
                "f = 1/T",
            ]
        return [
            "v = f*lambda",
            "standing string: lambda_n = 2L/n",
            "standing open pipe: f_n = n*v/(2L)",
            "Doppler relation depends on source/observer motion",
        ]

    if domain == "electricity and magnetism":
        if "circuit" in text or "resistor" in text or "current" in text or "battery" in text:
            return [
                "V = I*R",
                "series: R_eq = R1 + R2 + ...",
                "parallel: 1/R_eq = 1/R1 + 1/R2 + ...",
                "P = I*V = I^2*R = V^2/R",
            ]
        if "potential" in text or "voltage due to" in text or "voltage at" in text:
            return [
                "V = k*q/r for a point charge",
                "Delta V = -W_field/q",
                "Delta U = q*Delta V",
                "E = -Delta V / Delta s for a uniform field",
            ]
        if "magnetic" in text or "magnet" in text or "wire" in text:
            return [
                "F = q*v*B*sin(theta)",
                "F = I*L*B*sin(theta)",
                "B = mu0*I/(2*pi*r) for a long straight wire",
                "Use the right-hand rule for direction",
            ]
        return [
            "F = k*q1*q2/r^2",
            "E = k*q/r^2",
            "V = I*R",
            "P = I*V = I^2*R = V^2/R",
        ]

    if domain == "optics":
        return [
            "n1*sin(theta1) = n2*sin(theta2)",
            "1/f = 1/do + 1/di",
            "m = -di/do",
            "d*sin(theta) = m*lambda",
        ]

    if domain == "modern physics":
        return [
            "E_photon = h*f = h*c/lambda",
            "K_max = h*f - work function",
            "gamma = 1/sqrt(1 - v^2/c^2)",
            "N = N0*e^(-lambda*t)",
        ]

    return [config.get("equation_hint", "Choose the equation that connects the knowns to the unknown.")]


def _infer_target_unknown(problem: str, config: Dict[str, Any]) -> str:
    text = _normalize(problem)
    targets = (
        ("acceleration", "acceleration"),
        ("how high", "maximum height or vertical displacement"),
        ("maximum height", "maximum height"),
        ("height", "height"),
        ("rise", "vertical displacement"),
        ("how far", "displacement or distance"),
        ("distance", "distance"),
        ("displacement", "displacement"),
        ("velocity", "velocity"),
        ("speed", "speed"),
        ("time", "time"),
        ("force", "force"),
        ("work", "work"),
        ("energy", "energy"),
        ("power", "power"),
        ("momentum", "momentum"),
        ("current", "current"),
        ("voltage", "voltage"),
        ("electric potential", "electric potential"),
        ("potential", "electric potential"),
        ("image distance", "image distance"),
        ("frequency", "frequency"),
        ("wavelength", "wavelength"),
        ("period", "period"),
        ("amplitude", "amplitude"),
        ("torque", "torque"),
    )
    for cue, target in targets:
        if cue in text:
            return target
    if _is_graph_request(problem):
        return "a correctly scaled graph and the equations behind it"
    return f"the requested {config.get('domain', 'physics')} quantity"


def _unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _get_config(agent_id: str) -> Dict[str, Any]:
    return AGENT_TUTORING_CONFIG.get(agent_id, AGENT_TUTORING_CONFIG["math_agent"])


def _context_messages(context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not context:
        return []

    raw_messages = context.get("recent_conversation") or context.get("messages") or []
    messages = []
    for message in raw_messages:
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "")).strip()
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def _prior_messages(context_messages: List[Dict[str, str]], current_message: str) -> List[Dict[str, str]]:
    if not context_messages:
        return []

    current = _normalize(current_message)
    last = context_messages[-1]
    if last["role"] == "user" and _normalize(last["content"]) == current:
        return context_messages[:-1]
    return context_messages


def _has_guided_history(messages: Iterable[Dict[str, str]]) -> bool:
    return any(
        message["role"] == "assistant"
        and GUIDED_RESPONSE_MARKER.lower() in message["content"].lower()
        for message in messages
    )


def _last_guided_response(messages: Iterable[Dict[str, str]]) -> str:
    for message in reversed(list(messages)):
        if (
            message["role"] == "assistant"
            and GUIDED_RESPONSE_MARKER.lower() in message["content"].lower()
        ):
            return message["content"]
    return ""


def _student_turns_after_guidance(messages: List[Dict[str, str]]) -> int:
    last_guided_index = -1
    for index, message in enumerate(messages):
        if (
            message["role"] == "assistant"
            and GUIDED_RESPONSE_MARKER.lower() in message["content"].lower()
        ):
            last_guided_index = index

    if last_guided_index < 0:
        return 0

    return sum(
        1
        for message in messages[last_guided_index + 1 :]
        if message["role"] == "user"
    )


def _student_attempt_text_after_guidance(
    messages: List[Dict[str, str]],
    current_message: str,
) -> str:
    first_guided_index = -1
    for index, message in enumerate(messages):
        if (
            message["role"] == "assistant"
            and GUIDED_RESPONSE_MARKER.lower() in message["content"].lower()
        ):
            first_guided_index = index
            break

    attempt_parts = [
        message["content"]
        for message in messages[first_guided_index + 1 :]
        if message["role"] == "user"
    ]
    if not attempt_parts or _normalize(attempt_parts[-1]) != _normalize(current_message):
        attempt_parts.append(current_message)

    return " ".join(attempt_parts).strip()


def _count_student_attempts(messages: Iterable[Dict[str, str]]) -> int:
    return sum(
        1
        for message in messages
        if message["role"] == "user" and _has_student_attempt(message["content"])
    )


def _active_problem_text(
    prior_messages: List[Dict[str, str]],
    current_message: str,
    agent_id: str,
) -> str:
    normalized_current = _normalize(current_message)
    prior_problem = _latest_problem_text_from_prior(prior_messages, agent_id)
    if (
        prior_problem
        and _has_guided_history(prior_messages)
        and not _looks_like_new_problem_request(normalized_current, agent_id)
    ):
        return prior_problem

    if (
        _looks_like_problem_solving(normalized_current, agent_id)
        or _looks_like_conceptual_question(normalized_current, agent_id)
    ):
        return current_message

    for message in reversed(prior_messages):
        if message["role"] != "user":
            continue
        normalized = _normalize(message["content"])
        if (
            _looks_like_problem_solving(normalized, agent_id)
            or _looks_like_conceptual_question(normalized, agent_id)
        ):
            return message["content"]

    return current_message


def _latest_problem_text_from_prior(
    prior_messages: List[Dict[str, str]],
    agent_id: str,
) -> str:
    for message in reversed(prior_messages):
        if message["role"] != "user":
            continue
        normalized = _normalize(message["content"])
        if _looks_like_new_problem_request(normalized, agent_id):
            return message["content"]
        if (
            _looks_like_problem_solving(normalized, agent_id)
            or _looks_like_conceptual_question(normalized, agent_id)
        ):
            return message["content"]
    return ""


def _looks_like_new_problem_request(normalized_message: str, agent_id: str) -> bool:
    if any(cue in normalized_message for cue in ("new problem", "another problem", "next problem")):
        return True

    problem_action = any(
        cue in normalized_message
        for cue in (
            "find",
            "calculate",
            "determine",
            "compute",
            "how far",
            "how long",
            "how much",
            "how fast",
            "draw",
            "graph",
        )
    )
    starts_like_word_problem = bool(
        re.search(
            r"^(a|an|the)\s+.{0,120}\b(car|runner|ball|block|crate|cart|charge|gas|lens|wave|object|stone|projectile)\b",
            normalized_message,
        )
    )
    has_problem_punctuation = "?" in normalized_message

    if starts_like_word_problem and problem_action:
        return True
    if has_problem_punctuation and problem_action and _looks_like_physics_question(normalized_message, agent_id):
        return True
    return False


def _detect_misconception(normalized_message: str) -> Optional[Dict[str, str]]:
    if "?" in normalized_message and not _has_explicit_reasoning_attempt(normalized_message):
        return None

    for misconception in MISCONCEPTION_PATTERNS:
        if re.search(misconception["pattern"], normalized_message):
            return misconception
    return None


def _looks_like_problem_solving(normalized_message: str, agent_id: str) -> bool:
    if _matches_any(normalized_message, FULL_SOLUTION_PATTERNS):
        return True

    problem_verbs = (
        "calculate", "find", "solve", "determine", "compute", "what is",
        "how far", "how long", "how much", "how fast", "graph", "draw",
    )
    strong_problem_verbs = (
        "calculate", "find", "solve", "determine", "compute",
        "how far", "how long", "how much", "how fast", "graph", "draw",
    )
    has_problem_verb = any(verb in normalized_message for verb in problem_verbs)
    has_strong_problem_verb = any(verb in normalized_message for verb in strong_problem_verbs)
    has_number_or_unit = bool(re.search(r"\d", normalized_message)) or _has_units(normalized_message)
    starts_like_word_problem = bool(re.search(r"^(a|an|the)\s+.{0,80}\b(\d|kg|car|block|ball|charge|gas|lens|wave|cart)\b", normalized_message))

    domain = _get_config(agent_id)["domain"]
    domain_word_present = any(part in normalized_message for part in domain.split())

    return (
        (has_problem_verb and (has_number_or_unit or domain_word_present))
        or (has_strong_problem_verb and _looks_like_physics_question(normalized_message, agent_id))
        or (starts_like_word_problem and has_number_or_unit)
        or (_has_equation(normalized_message) and has_problem_verb)
    )


def _looks_like_conceptual_question(normalized_message: str, agent_id: str) -> bool:
    conceptual_cues = (
        "why",
        "explain",
        "conceptually",
        "what happens",
        "what would happen",
        "does",
        "do ",
        "is it true",
        "can",
        "how does",
        "what is the difference",
        "why doesn't",
        "why does",
    )
    asks_like_question = "?" in normalized_message or any(cue in normalized_message for cue in conceptual_cues)
    if not asks_like_question:
        return False
    if _looks_like_problem_solving(normalized_message, agent_id):
        return False
    return _looks_like_physics_question(normalized_message, agent_id)


def _looks_like_physics_question(normalized_message: str, agent_id: str) -> bool:
    physics_words = {
        "force", "velocity", "acceleration", "energy", "momentum", "torque",
        "heat", "temperature", "wave", "frequency", "charge", "current",
        "voltage", "field", "lens", "mirror", "photon", "electron",
        "relativity", "friction", "normal", "projectile", "fall", "gravity",
        "mass", "gas", "pressure", "volume", "work", "oscillation", "oscillate",
        "simple harmonic", "spring", "pendulum", "electric potential", "potential",
        "circuit", "resistor", "magnetic", "magnetism", "optics", "ray",
    }
    domain = _get_config(agent_id)["domain"]
    return any(word in normalized_message for word in physics_words) or any(
        part in normalized_message for part in domain.split()
    )


def _has_student_attempt(message: str) -> bool:
    normalized = _normalize(message)
    attempt_words = (
        "i think", "i tried", "my equation", "known", "unknown", "given",
        "fbd", "free body", "diagram", "use", "using", "equals", "because",
        "so", "therefore", "substitute", "plug", "axis", "horizontal",
        "vertical", "positive", "negative", "distance", "displacement",
        "area", "rectangle", "interval",
    )
    return (
        any(word in normalized for word in attempt_words)
        or _has_equation(normalized)
        or (_has_units(normalized) and bool(re.search(r"\d", normalized)))
    )


def _has_explicit_reasoning_attempt(message: str) -> bool:
    normalized = _normalize(message)
    reasoning_words = (
        "i think", "i tried", "my equation", "known", "unknown", "given",
        "fbd", "free body", "diagram", "because", "therefore", "substitute",
        "plug", "i used", "i use", "i would use",
    )
    return any(word in normalized for word in reasoning_words) or _has_equation(normalized)


def _has_knowns_language(message: str) -> bool:
    normalized = _normalize(message)
    return any(word in normalized for word in ("known", "unknown", "given", "find", "looking for"))


def _has_equation(message: str) -> bool:
    normalized = _normalize(message)
    if "=" in normalized or "sum f" in normalized or "sigma f" in normalized:
        return True
    equation_patterns = (
        r"\bf\s*=\s*m\s*a\b",
        r"\bv\s*=\s*v0",
        r"\bp\s*=\s*m\s*v\b",
        r"\bke\b.*\b1/2\b",
        r"\bq\s*=\s*m\s*c",
        r"\bv\s*=\s*f",
        r"\bv\s*=\s*i\s*r\b",
        r"\b1/f\b",
    )
    return any(re.search(pattern, normalized) for pattern in equation_patterns)


def _has_model_equation(message: str, equations: Iterable[str]) -> bool:
    normalized = _normalize_physics_text(message)
    relation_cues = (
        "sum f",
        "sigma f",
        "delta",
        "1/2",
        "mu",
        "lambda",
        "sin",
        "cos",
        "tan",
        "v^2",
        "v0^2",
        "p_before",
        "p_after",
        "before =",
        "after",
        "m*a",
        "ma",
        "m v",
        "mv",
        "q = m*c",
        "v = f",
        "v=f",
        "f lambda",
        "1/f",
        "p*delta v",
        "pv",
        "n*r*t",
        "h*c",
        "gamma",
        "e^",
    )
    if any(cue in normalized for cue in relation_cues):
        return True

    for equation in equations:
        equation_lower = _normalize_physics_text(equation)
        left_side = equation_lower.split("=", 1)[0].strip()
        if left_side and len(left_side) > 1 and left_side in normalized:
            return True

    assignment_only = re.fullmatch(
        r"(?:knowns?:\s*)?(?:[a-z][a-z0-9_]*\s*=\s*[-+]?\d+(?:\.\d+)?\s*(?:m/s\^?2|m/s2|m/s|kg|n|j|w|hz|v|a|c|pa|kpa|ohm|rad/s|cm|m|s)?\s*,?\s*)+",
        normalized,
    )
    if assignment_only:
        return False

    return _has_equation(normalized) and any(
        symbol in normalized
        for symbol in ("+", "*", "^", "(", ")")
    )


def _has_units(message: str) -> bool:
    normalized = _normalize_physics_text(message)
    padded = f" {normalized} "
    compact_unit_pattern = (
        r"\d\s*(?:m|cm|kg|n|j|w|hz|v|a|c|t|ev|nm|s)\b"
        r"|\d\s*m/s(?:\^?2|2)?\b"
    )
    return any(unit in padded for unit in PHYSICS_UNITS) or bool(
        re.search(compact_unit_pattern, normalized)
    )


def _matches_any(normalized_message: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, normalized_message) for pattern in patterns)


def _normalize(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


def _normalize_physics_text(message: str) -> str:
    normalized = _normalize(message).replace("²", "^2")
    normalized = re.sub(r"m\s*/\s*s\s*(?:\^\s*)?2\b", "m/s^2", normalized)
    normalized = re.sub(r"m\s*/\s*s\b", "m/s", normalized)
    return normalized
