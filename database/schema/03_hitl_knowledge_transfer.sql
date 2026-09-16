-- HITL Knowledge Transfer schema for guided MCQ gating
-- Pilot scope: forces + kinematics

CREATE TABLE IF NOT EXISTS knowledge_transfer_guidance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_type agent_type NOT NULL,
    concept_tag VARCHAR(120) NOT NULL,
    question_text TEXT NOT NULL,
    options JSONB NOT NULL, -- [{"id":"A","text":"...","is_correct":true,"feedback":"..."}]
    explanation_correct TEXT NOT NULL,
    explanation_incorrect TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    difficulty SMALLINT NOT NULL DEFAULT 1 CHECK (difficulty BETWEEN 1 AND 5),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_kt_guidance_unique_active
    ON knowledge_transfer_guidance(agent_type, concept_tag, question_text);

CREATE INDEX IF NOT EXISTS idx_kt_guidance_agent
    ON knowledge_transfer_guidance(agent_type);

CREATE INDEX IF NOT EXISTS idx_kt_guidance_concept
    ON knowledge_transfer_guidance(concept_tag);

CREATE INDEX IF NOT EXISTS idx_kt_guidance_active
    ON knowledge_transfer_guidance(active);

CREATE TABLE IF NOT EXISTS knowledge_transfer_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    check_id VARCHAR(64) NOT NULL,
    user_identifier VARCHAR(255) NOT NULL,
    session_identifier VARCHAR(255),
    class_identifier VARCHAR(255),
    agent_type agent_type NOT NULL,
    concept_tag VARCHAR(120) NOT NULL,
    question_id UUID REFERENCES knowledge_transfer_guidance(id) ON DELETE SET NULL,
    confidence_score DECIMAL(5,4),
    threshold_score DECIMAL(5,4),
    selected_option_id VARCHAR(32) NOT NULL,
    was_correct BOOLEAN NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_kt_attempts_user
    ON knowledge_transfer_attempts(user_identifier);

CREATE INDEX IF NOT EXISTS idx_kt_attempts_class
    ON knowledge_transfer_attempts(class_identifier);

CREATE INDEX IF NOT EXISTS idx_kt_attempts_agent
    ON knowledge_transfer_attempts(agent_type);

CREATE INDEX IF NOT EXISTS idx_kt_attempts_concept
    ON knowledge_transfer_attempts(concept_tag);

CREATE INDEX IF NOT EXISTS idx_kt_attempts_created_at
    ON knowledge_transfer_attempts(created_at);

CREATE INDEX IF NOT EXISTS idx_kt_attempts_correct
    ON knowledge_transfer_attempts(was_correct);

COMMENT ON TABLE knowledge_transfer_guidance IS 'Curated MCQ knowledge checks used for HITL pedagogy gates.';
COMMENT ON TABLE knowledge_transfer_attempts IS 'Student responses to HITL knowledge checks for success analytics.';

-- Seed pilot guidance rows (idempotent by conflict update)
INSERT INTO knowledge_transfer_guidance (
    agent_type, concept_tag, question_text, options, explanation_correct, explanation_incorrect, difficulty
) VALUES
(
    'forces',
    'newton_first_law',
    'Which statement best describes Newton''s First Law?',
    '[
      {"id":"A","text":"An object changes velocity only when a net external force acts on it.","is_correct":true,"feedback":"Correct: inertia means no change in motion without net force."},
      {"id":"B","text":"A moving object always needs a force to keep moving.","is_correct":false,"feedback":"Constant velocity needs zero net force."},
      {"id":"C","text":"Heavier objects always resist motion changes less.","is_correct":false,"feedback":"Larger mass means greater inertia."}
    ]'::jsonb,
    'Newton''s First Law is the inertia law: no net force implies constant velocity.',
    'Motion changes only when net external force is nonzero.',
    1
),
(
    'forces',
    'newton_second_law',
    'For Newton''s Second Law, what equation links net force, mass, and acceleration?',
    '[
      {"id":"A","text":"F = m a","is_correct":true,"feedback":"Correct: net force equals mass times acceleration."},
      {"id":"B","text":"F = m / a","is_correct":false,"feedback":"Acceleration multiplies mass, it does not divide it."},
      {"id":"C","text":"F = a / m","is_correct":false,"feedback":"This reverses the relationship."}
    ]'::jsonb,
    'Use vector form: ΣF = m a.',
    'Use ΣF = m a. Acceleration scales with net force and inverse mass.',
    1
),
(
    'forces',
    'newton_third_law',
    'Which pair is a Newton''s Third Law action-reaction pair?',
    '[
      {"id":"A","text":"Earth pulls on ball and ball pulls on Earth with equal magnitude opposite direction.","is_correct":true,"feedback":"Correct: equal and opposite on different bodies."},
      {"id":"B","text":"Weight and normal force on the same block.","is_correct":false,"feedback":"Not a third-law pair; these act on one object."},
      {"id":"C","text":"Friction and acceleration of one object.","is_correct":false,"feedback":"Acceleration is not a force."}
    ]'::jsonb,
    'Third-law forces are equal-opposite and act on different interacting bodies.',
    'Third-law pairs must act on different objects.',
    1
),
(
    'forces',
    'hookes_law',
    'For an ideal spring near equilibrium, which relation is Hooke''s law (magnitude form)?',
    '[
      {"id":"A","text":"F = k x","is_correct":true,"feedback":"Correct for magnitudes; vector form is F = -k x."},
      {"id":"B","text":"F = x / k","is_correct":false,"feedback":"This inverts spring-constant dependence."},
      {"id":"C","text":"F = k / x","is_correct":false,"feedback":"Nonphysical near x=0 for this context."}
    ]'::jsonb,
    'Magnitude relation is |F| = k|x| and restoring vector form is F = -k x.',
    'Hooke''s law is linear in displacement near equilibrium.',
    1
),
(
    'forces',
    'incline_components',
    'On an incline at angle θ, which is the component of weight parallel to plane (down slope)?',
    '[
      {"id":"A","text":"W sin(θ)","is_correct":true,"feedback":"Correct: W_parallel = mg sin θ."},
      {"id":"B","text":"W cos(θ)","is_correct":false,"feedback":"W cos θ is perpendicular component."},
      {"id":"C","text":"W tan(θ)","is_correct":false,"feedback":"Tan is not direct component magnitude."}
    ]'::jsonb,
    'Resolve weight into incline axes: parallel is mg sin θ, perpendicular is mg cos θ.',
    'In incline-aligned axes, parallel component is mg sin θ.',
    2
),
(
    'kinematics',
    'projectile_components',
    'For projectile launch speed v0 at angle θ, what is the initial vertical component?',
    '[
      {"id":"A","text":"v0 sin(θ)","is_correct":true,"feedback":"Correct: vertical component uses sine."},
      {"id":"B","text":"v0 cos(θ)","is_correct":false,"feedback":"Cosine gives horizontal component."},
      {"id":"C","text":"v0 tan(θ)","is_correct":false,"feedback":"Tan is a ratio, not a component projection."}
    ]'::jsonb,
    'Use decomposition: v0x = v0 cos θ, v0y = v0 sin θ.',
    'Vertical and horizontal components use sine/cosine respectively.',
    1
),
(
    'kinematics',
    'projectile_peak',
    'At maximum height of a projectile (neglecting air resistance), which is true?',
    '[
      {"id":"A","text":"Vertical velocity vy = 0 while horizontal velocity remains nonzero.","is_correct":true,"feedback":"Correct: vy crosses zero at peak, vx remains constant."},
      {"id":"B","text":"Both vx and vy are zero.","is_correct":false,"feedback":"Horizontal speed persists without horizontal acceleration."},
      {"id":"C","text":"Acceleration is zero at peak.","is_correct":false,"feedback":"Acceleration remains downward at g."}
    ]'::jsonb,
    'At peak vy=0 but acceleration is still -g and vx remains constant.',
    'Only vy becomes zero at the top; acceleration remains downward.',
    2
),
(
    'kinematics',
    'equation_selection',
    'Which constant-acceleration equation directly avoids time t?',
    '[
      {"id":"A","text":"v_f^2 = v_0^2 + 2 a Δx","is_correct":true,"feedback":"Correct: this relation eliminates time."},
      {"id":"B","text":"v_f = v_0 + a t","is_correct":false,"feedback":"This equation includes time explicitly."},
      {"id":"C","text":"x = x_0 + v_0 t + (1/2) a t^2","is_correct":false,"feedback":"This also includes time."}
    ]'::jsonb,
    'Use v_f^2 = v_0^2 + 2aΔx when you want to avoid time.',
    'Only the squared-velocity kinematics relation removes t directly.',
    2
),
(
    'kinematics',
    'sign_convention',
    'If +y is upward in projectile motion near Earth, what is acceleration ay?',
    '[
      {"id":"A","text":"-g","is_correct":true,"feedback":"Correct: gravity points downward."},
      {"id":"B","text":"+g","is_correct":false,"feedback":"Not correct under +y upward convention."},
      {"id":"C","text":"0","is_correct":false,"feedback":"Gravity is present throughout flight."}
    ]'::jsonb,
    'With +y upward, ay = -g by sign convention.',
    'Keep sign convention consistent: ay is negative in +y-up coordinates.',
    1
),
(
    'kinematics',
    'range_formula',
    'For launch and landing at same height, projectile range is:',
    '[
      {"id":"A","text":"R = v0^2 sin(2θ) / g","is_correct":true,"feedback":"Correct for same launch/landing height."},
      {"id":"B","text":"R = v0 sin(θ) / g","is_correct":false,"feedback":"Not dimensionally correct for range distance."},
      {"id":"C","text":"R = 2 v0 cos(θ) / g","is_correct":false,"feedback":"Missing required speed-angle structure."}
    ]'::jsonb,
    'For level-ground projectile motion: R = (v0^2 sin 2θ)/g.',
    'Use the level-ground range formula with sin(2θ).',
    2
)
ON CONFLICT (agent_type, concept_tag, question_text)
DO UPDATE SET
    options = EXCLUDED.options,
    explanation_correct = EXCLUDED.explanation_correct,
    explanation_incorrect = EXCLUDED.explanation_incorrect,
    active = TRUE,
    updated_at = CURRENT_TIMESTAMP;
