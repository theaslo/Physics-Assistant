# Contextual physics tutoring

Momentum, work/energy, angular motion and thermodynamics share the conversational
tutoring flow. Forces retains its existing tutoring implementation.

The tutor chooses a conceptual check when a new calculation involves a useful
physical decision. Simple calculations and direct conceptual questions can skip
the check. A skipped check still requires a complete numerical solution: givens,
principle, equation, numerical substitution, result, and a physical check.

Generated multiple-choice questions are independently reviewed for one correct
answer. Ambiguous options are rewritten and reviewed once more. A wrong choice
starts guided feedback; uncertainty asks for a smaller hint. Correct choices and
explicit full-solution requests release the worked solution. Server-held receipts
bind the problem, answer and follow-up history to the user, session and agent.

Student messages do not show MCP names or internal verification details. Tool
metadata remains available to application diagnostics. Each domain has a
conceptual-reference tool so symbolic explanations do not require fabricated
numerical inputs.

## Domain conditions

- Energy: distinguish net work from work by one force, mechanical energy from
  total energy, and a potential-energy reference from a physical height change.
- Rotation: choose an axis and signed direction; constant-acceleration formulas
  need constant angular acceleration. Conserved angular momentum does not imply
  conserved rotational kinetic energy when inertia changes.
- Thermodynamics: use `Delta U = Q - W_by`, with heat into the system positive.
  Use kelvin for absolute gas/Carnot temperatures and signed temperature changes
  for sensible heating. Constant pressure, isothermal and adiabatic conditions
  have different meanings.

Updated calculator paths reject missing physical givens, nonfinite inputs and
inconsistent or underdetermined states. Zero heat, zero elapsed time when other
data suffice, cooling, negative work and signed angular velocities are supported.

## Configuration

`HITL_CONTEXTUAL_AGENTS` defaults to
`momentum_agent,energy_agent,angular_motion_agent,thermodynamics_agent`.
Set it to a comma-separated subset to limit check-ins, or an empty string to
disable these check-ins. This does not disable conversational follow-ups or the
worked-solution requirements. Existing forces/kinematics pilot settings remain
separate.

The three additional tutors independently check that every requested numerical
quantity was calculated. Genuinely missing givens trigger one clarification
question. Repeated requests for an already-completed calculation reuse its
evidence instead of running the same calculation until the turn limit expires.

## Validation

From `UI/api`, using the installed API environment:

```sh
python -B -m unittest test_extended_tutoring test_mechanics_tutoring \
  test_momentum_release_scope test_forces_fastpaths test_hitl_regressions \
  test_math_linear_systems test_kinematics_fastpaths test_rag_client_contract
```

From `mcp_tools`, using the installed MCP environment:

```sh
python -B -m unittest test_extended_tutoring_tools test_momentum_hitl_tools \
  test_concept_principles test_linear_systems
```

`test_extended_tutoring_live` is opt-in. It uses synthetic physics questions with
real model/MCP calls, with agent database logging and RAG disabled. Point the
three `MCP_*_HOST` variables at isolated test services and direct their telemetry
away from student records, then set `RUN_EXTENDED_TUTORING_LIVE_TESTS=1`.

Build the API, MCP and student UI images and run the UI formatting, rendering,
intent and agent-history tests before release.

## Server updates

Preserve server-only authentication code, deployment settings, environment files
and data when reconciling the release with an existing server checkout. Keep
environment files outside Git and use sanitized example templates only. A
development checkout is not a replacement for a server's private configuration.

Deploy reviewed images with the existing Compose project, mount paths and
database volumes. Retain rollback image tags. Wait for application health checks
and the UI-to-API proxy to succeed, then verify login, a worked calculation, a
guided check-in and existing math/example features. No database migration is
required for this tutoring change.
