# Guided Tutoring Handoff

## Purpose

The assistant now uses a human-in-the-loop tutoring workflow for physics problem-solving questions. The default behavior is to guide the student through setup and reasoning before any complete worked solution is given.

This applies across PHYS 1201 and PHYS 1202 topics:

- PHYS 1201: kinematics, Newton's laws, forces/free-body diagrams, work-energy, momentum, rotation, torque, and equilibrium.
- PHYS 1202: oscillations, waves, electric force/field, electric potential, circuits, magnetism, optics, thermodynamics, and modern physics.

## Runtime Flow

The `/agent/{agent_id}/solve` endpoint calls `evaluate_guided_tutoring()` before creating or invoking a Strands agent.

The controller is modular:

- `classify_physics_topic()` identifies the likely PHYS 1201/1202 topic from the actual prompt text.
- `evaluate_guided_tutoring()` acts as the tutoring controller/state machine.
- Topic-specific checkpoint generators build initial diagnostics, graph checkpoints, conceptual prompts, attempt feedback, and misconception hints.
- `merge_tutoring_context()` and `apply_full_solution_instruction()` handle the full-solution fallback path.

If the student is asking a new problem-solving question, the API returns a guided response immediately with:

- diagnostic questions about the concept and assumptions,
- a prompt to draw or describe the relevant diagram,
- a knowns/unknowns checkpoint,
- an equation-selection or units checkpoint.

This response uses `tools_used=["guided_tutoring"]` and stores tutoring metadata under `metadata.guided_tutoring`.
The metadata includes `topic_classification` with topic, course, domain, model, and selected agent.

## Adaptive Behavior

The tutor gate uses recent conversation context from the UI. It adapts as follows:

- First problem-solving turn: ask diagnostic/setup questions instead of solving.
- Random prompt profiling: extract quantities and units from the student's wording, infer the likely physics model, list equation candidates, and identify the target unknown without relying on pre-seeded problem text.
- Conceptual questions: ask the student for their current thinking first, then give a small targeted hint or misconception check.
- Multi-turn attempt memory: read the student's full attempt since guided tutoring started for the active problem, so a short follow-up like "meters" or "m/s^2" does not cause the tutor to ask for an equation that was already supplied.
- Graph requests: define the graph type, axes, units, governing functions, qualitative shape, and missing assumptions before plotting or solving. Kinematics graph prompts get special handling for constant velocity, constant acceleration, piecewise motion, free fall, and projectile motion; general graph prompts advance from axes to relation to shape to plotting anchors instead of repeating the same setup question.
- Piecewise kinematics: split motion into separate intervals, ask the student to confirm each velocity/time segment, draw each velocity-time graph segment separately, and use area under the velocity-time graph for distance. Constant-acceleration equations are only used for intervals that actually accelerate.
- Piecewise graph state: after intervals are confirmed, the next checkpoints advance through area meaning, first rectangle, second rectangle, total distance, and velocity-time graph description. The handler classifies the most recent tutor checkpoint by meaning before using accumulated student attempts, which prevents repeating a completed checkpoint after correct short answers like "distance traveled" or "20 m."
- Active specialized workflow routing: if a guided piecewise graph workflow is active, short confirmations such as "yes", "confirm", "correct", and "yes confirm" are routed to the piecewise state handler before the generic attempt/equation fallback. Generic equation-selection prompts should only appear when no specialized workflow applies.
- Active problem lock: during guided history, follow-up answers are interpreted against the earlier problem statement unless the student clearly starts a new problem. This prevents graph-description replies such as "the velocity-time graph is horizontal..." from becoming a new generic graph prompt and switching away from the specialized piecewise workflow.
- Constant-acceleration velocity-time graph prompts: first show only the knowns and ask the student to confirm acceleration sign and graph axes. The equation `v(t) = v0 + a*t` is revealed after that setup is confirmed.
- Vertical/projectile distinction: vertical throws and free fall stay in 1D vertical kinematics unless the prompt explicitly gives a launch angle, horizontal distance, horizontal velocity, or x/y components. Horizontal-launch prompts get their own setup with `v0x`, `v0y = 0`, height, and gravity; angled projectiles are the only path that asks for sine/cosine component decomposition.
- Student provides knowns or a diagram: ask for the equation choice and symbol meanings.
- Student provides an equation: ask for substitution, units, and reasonableness checks.
- Student says they are stuck: give a narrower hint based on the topic.
- Student shows a misconception: correct that specific idea and ask a targeted follow-up.

Common misconception patterns are defined in `guided_tutoring.py`, including free fall and mass, normal force, static friction, projectile acceleration at the top, centripetal force direction, momentum conservation with external forces, series circuits, heat versus temperature, wave frequency/wavelength, optics sign conventions, and relativity scale.

## Representative Test Templates

`REPRESENTATIVE_TUTORING_PROBLEMS` covers the required 10 templates:

- Constant velocity
- Constant acceleration
- Projectile motion
- Newton's second law
- Friction
- Work-energy
- Momentum collision
- Torque equilibrium
- Electric field/force
- Simple circuit

Additional tests cover conceptual questions, graph progression, random non-seeded prompts, multi-turn short replies, misconception handling, and full-solution switching.

## When to Switch to a Full Solution

The assistant should switch from guided tutoring to a complete worked solution only when one of these is true:

- The student explicitly asks for a worked example, such as "I need a worked example."
- The student asks for the full solution after a guided exchange has already happened.
- The student has made a visible reasoning attempt and asks for the answer or full solution.

When a full solution is allowed, `merge_tutoring_context()` adds `guided_tutoring.full_solution_allowed=true`. The base Strands agent then prefixes the prompt with `[GUIDED_TUTORING_FULL_SOLUTION_APPROVED]`, which tells the LLM to provide the full setup, substitutions, units, final answer, and reasonableness check.

If a student asks for "the full solution" on the first turn, the assistant defers it and asks one quick setup checkpoint first. It also tells the student they can ask for a worked example if that is their actual need.

## Files

- `UI/api/guided_tutoring.py`: topic classifier, tutor-state gate, topic-specific diagnostics, graph/conceptual checkpoints, misconception detection, full-solution approval metadata, and shared agent prompt policy.
- `UI/api/main.py`: invokes the tutor gate before kinematics graph fallback or Strands agent calls.
- `UI/api/strands_agents/base_physics_agent.py`: appends the shared tutoring policy to every agent prompt and honors full-solution approval.
- `student-ui/src/pages/ChatPage.tsx`: sends recent conversation history so the API can run a multi-turn tutoring workflow. It does not pre-create agents before each message, allowing guided checkpoints to work even when a domain MCP server is not running yet.
- `UI/api/test_guided_tutoring.py`: tests first-turn guidance, the ten required templates, classifier output, conceptual prompts, random non-seeded prompts, graph scaffolding, misconceptions, and full-solution switching.
