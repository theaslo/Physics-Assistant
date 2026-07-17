-- Physics Assistant Database Schema
-- Core tables for user management, sessions, and interaction logging
-- Created: 2025-08-14

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create enum types idempotently. PostgreSQL does not support
-- CREATE TYPE IF NOT EXISTS for enum types on all supported versions.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('student', 'instructor', 'admin');
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'session_status') THEN
        CREATE TYPE session_status AS ENUM ('active', 'expired', 'terminated');
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'interaction_type') THEN
        CREATE TYPE interaction_type AS ENUM ('chat', 'mcp_tool', 'agent_call', 'file_upload', 'calculation');
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'message_type') THEN
        CREATE TYPE message_type AS ENUM ('user', 'assistant', 'system', 'error');
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'agent_type') THEN
        CREATE TYPE agent_type AS ENUM (
            'kinematics',
            'forces',
            'energy',
            'momentum',
            'angular_motion',
            'math_helper',
            'math',
            'thermodynamics',
            'waves',
            'electromagnetism',
            'optics',
            'modern_physics'
        );
    END IF;
END;
$$;

-- Add enum values required by current Strands agents to existing databases.
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'math';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'thermodynamics';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'waves';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'electromagnetism';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'optics';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'modern_physics';

-- Users table - Core user management
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role user_role NOT NULL DEFAULT 'student',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    verification_token VARCHAR(255),
    reset_token VARCHAR(255),
    reset_token_expires_at TIMESTAMP WITH TIME ZONE,
    last_login TIMESTAMP WITH TIME ZONE,
    login_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- User sessions table - Track active sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    status session_status NOT NULL DEFAULT 'active',
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Interactions table - Log all user interactions with the system
CREATE TABLE IF NOT EXISTS interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES user_sessions(id) ON DELETE SET NULL,
    type interaction_type NOT NULL,
    agent_type agent_type,
    request_data JSONB,
    response_data JSONB,
    execution_time_ms INTEGER,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Messages table - Store chat messages and system communications
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type message_type NOT NULL,
    content TEXT NOT NULL,
    content_latex TEXT, -- For LaTeX equations
    attachments JSONB DEFAULT '[]'::jsonb, -- File attachments metadata
    tokens_used INTEGER,
    model_name VARCHAR(100),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent calls table - Track specific MCP tool and agent invocations
CREATE TABLE IF NOT EXISTS agent_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_type agent_type NOT NULL,
    tool_name VARCHAR(100),
    function_name VARCHAR(100),
    input_parameters JSONB,
    output_result JSONB,
    execution_time_ms INTEGER,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_details TEXT,
    model_used VARCHAR(100),
    tokens_consumed INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User preferences table - Store user settings and preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    theme VARCHAR(20) DEFAULT 'light',
    language VARCHAR(10) DEFAULT 'en',
    notifications_enabled BOOLEAN DEFAULT TRUE,
    latex_rendering_enabled BOOLEAN DEFAULT TRUE,
    preferred_units VARCHAR(20) DEFAULT 'metric',
    difficulty_level VARCHAR(20) DEFAULT 'beginner',
    preferences JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User progress table - Track learning progress and achievements
CREATE TABLE IF NOT EXISTS user_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic VARCHAR(100) NOT NULL, -- physics topic (kinematics, forces, etc.)
    problems_attempted INTEGER DEFAULT 0,
    problems_solved INTEGER DEFAULT 0,
    total_interaction_time INTEGER DEFAULT 0, -- in minutes
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    proficiency_score DECIMAL(5,2) DEFAULT 0.0, -- 0-100 scale
    achievements JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, topic)
);

-- File uploads table - Track uploaded files and diagrams
CREATE TABLE IF NOT EXISTS file_uploads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    interaction_id UUID REFERENCES interactions(id) ON DELETE SET NULL,
    original_filename VARCHAR(255) NOT NULL,
    stored_filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    file_hash VARCHAR(64) NOT NULL, -- SHA-256 hash
    is_processed BOOLEAN DEFAULT FALSE,
    processing_results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HITL knowledge transfer questions shown before full agent solving.
CREATE TABLE IF NOT EXISTS hitl_questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question_key VARCHAR(150) UNIQUE NOT NULL,
    agent_type agent_type NOT NULL,
    topic VARCHAR(100) NOT NULL,
    leg INTEGER NOT NULL DEFAULT 1,
    question_type VARCHAR(50) NOT NULL DEFAULT 'multiple_choice',
    question_text TEXT NOT NULL,
    choices JSONB NOT NULL DEFAULT '[]'::jsonb,
    correct_choice_id VARCHAR(100),
    correct_answer TEXT,
    explanation TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HITL attempts intentionally keep the incoming user/session IDs as text so
-- UI sessions can be evaluated before they are linked to durable accounts.
CREATE TABLE IF NOT EXISTS hitl_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question_id UUID NOT NULL REFERENCES hitl_questions(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    agent_type agent_type NOT NULL,
    selected_choice_id VARCHAR(100),
    answer_text TEXT,
    is_correct BOOLEAN NOT NULL,
    feedback TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Additive repair for existing databases created by earlier bootstrap scripts.
-- CREATE TABLE IF NOT EXISTS leaves old tables untouched, so keep runtime
-- columns idempotently aligned before indexes, triggers, and seed data run.
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_token VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_token VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS reset_token_expires_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

ALTER TABLE interactions ADD COLUMN IF NOT EXISTS type interaction_type NOT NULL DEFAULT 'chat';
ALTER TABLE interactions ADD COLUMN IF NOT EXISTS agent_type agent_type;
ALTER TABLE interactions ADD COLUMN IF NOT EXISTS request_data JSONB;
ALTER TABLE interactions ADD COLUMN IF NOT EXISTS response_data JSONB;
ALTER TABLE interactions ADD COLUMN IF NOT EXISTS execution_time_ms INTEGER;
ALTER TABLE interactions ADD COLUMN IF NOT EXISTS success BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE interactions ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE interactions ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
ALTER TABLE interactions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE messages ADD COLUMN IF NOT EXISTS content_latex TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS attachments JSONB DEFAULT '[]'::jsonb;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

ALTER TABLE agent_calls ADD COLUMN IF NOT EXISTS model_used VARCHAR(100);
ALTER TABLE agent_calls ADD COLUMN IF NOT EXISTS tokens_consumed INTEGER;

ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS latex_rendering_enabled BOOLEAN DEFAULT TRUE;
ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS preferred_units VARCHAR(20) DEFAULT 'metric';
ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS difficulty_level VARCHAR(20) DEFAULT 'beginner';
ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}'::jsonb;

ALTER TABLE user_progress ADD COLUMN IF NOT EXISTS total_interaction_time INTEGER DEFAULT 0;
ALTER TABLE user_progress ADD COLUMN IF NOT EXISTS proficiency_score DECIMAL(5,2) DEFAULT 0.0;
ALTER TABLE user_progress ADD COLUMN IF NOT EXISTS achievements JSONB DEFAULT '[]'::jsonb;

ALTER TABLE file_uploads ADD COLUMN IF NOT EXISTS processing_results JSONB;

ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS question_key VARCHAR(150);
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS agent_type agent_type;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS topic VARCHAR(100);
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS leg INTEGER NOT NULL DEFAULT 1;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS question_type VARCHAR(50) NOT NULL DEFAULT 'multiple_choice';
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS question_text TEXT;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS choices JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS correct_choice_id VARCHAR(100);
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS correct_answer TEXT;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS explanation TEXT;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS question_id UUID;
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS session_id VARCHAR(255);
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS agent_type agent_type;
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS selected_choice_id VARCHAR(100);
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS answer_text TEXT;
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS is_correct BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS feedback TEXT;
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_status ON user_sessions(status);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(type);
CREATE INDEX IF NOT EXISTS idx_interactions_agent_type ON interactions(agent_type);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_success ON interactions(success);

CREATE INDEX IF NOT EXISTS idx_messages_interaction_id ON messages(interaction_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(type);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

CREATE INDEX IF NOT EXISTS idx_agent_calls_interaction_id ON agent_calls(interaction_id);
CREATE INDEX IF NOT EXISTS idx_agent_calls_user_id ON agent_calls(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_calls_agent_type ON agent_calls(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_calls_tool_name ON agent_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_agent_calls_created_at ON agent_calls(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_calls_success ON agent_calls(success);

CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_user_progress_topic ON user_progress(topic);
CREATE INDEX IF NOT EXISTS idx_user_progress_updated_at ON user_progress(updated_at);

CREATE INDEX IF NOT EXISTS idx_file_uploads_user_id ON file_uploads(user_id);
CREATE INDEX IF NOT EXISTS idx_file_uploads_interaction_id ON file_uploads(interaction_id);
CREATE INDEX IF NOT EXISTS idx_file_uploads_created_at ON file_uploads(created_at);
CREATE INDEX IF NOT EXISTS idx_file_uploads_file_hash ON file_uploads(file_hash);

CREATE UNIQUE INDEX IF NOT EXISTS idx_hitl_questions_question_key ON hitl_questions(question_key);
CREATE INDEX IF NOT EXISTS idx_hitl_questions_agent_leg ON hitl_questions(agent_type, leg);
CREATE INDEX IF NOT EXISTS idx_hitl_questions_active ON hitl_questions(is_active);
CREATE INDEX IF NOT EXISTS idx_hitl_attempts_user_id ON hitl_attempts(user_id);
CREATE INDEX IF NOT EXISTS idx_hitl_attempts_agent_type ON hitl_attempts(agent_type);
CREATE INDEX IF NOT EXISTS idx_hitl_attempts_question_id ON hitl_attempts(question_id);
CREATE INDEX IF NOT EXISTS idx_hitl_attempts_created_at ON hitl_attempts(created_at);

-- Create function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_users_updated_at') THEN
        CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_user_preferences_updated_at') THEN
        CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_user_progress_updated_at') THEN
        CREATE TRIGGER update_user_progress_updated_at BEFORE UPDATE ON user_progress
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_hitl_questions_updated_at') THEN
        CREATE TRIGGER update_hitl_questions_updated_at BEFORE UPDATE ON hitl_questions
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;

-- Seed baseline HITL questions for first-leg and follow-up checks.
INSERT INTO hitl_questions (
    question_key, agent_type, topic, leg, question_text, choices,
    correct_choice_id, correct_answer, explanation, metadata
) VALUES
(
    'kinematics_constant_acceleration_leg_1',
    'kinematics',
    'constant_acceleration',
    1,
    'A car accelerates uniformly from rest and you need displacement after a known time. Which equation is the direct starting point?',
    '[
        {"id": "dx_time", "text": "dx = v0*t + 1/2*a*t^2"},
        {"id": "velocity_displacement", "text": "v^2 = v0^2 + 2*a*dx"}
    ]'::jsonb,
    'dx_time',
    'dx = v0*t + 1/2*a*t^2',
    'Use the displacement-time equation because time, initial velocity, and acceleration are the known setup quantities.',
    '{"next_leg_on_wrong": 2}'::jsonb
),
(
    'kinematics_constant_acceleration_leg_2',
    'kinematics',
    'constant_acceleration',
    2,
    'Before substituting into dx = v0*t + 1/2*a*t^2, what setup must be clear?',
    '[
        {"id": "knowns_signs", "text": "Known v0, a, t, and the sign convention"},
        {"id": "mass_force", "text": "Mass and net force"}
    ]'::jsonb,
    'knowns_signs',
    'Known v0, a, t, and the sign convention',
    'The equation only works after the kinematic knowns and direction convention are defined.',
    '{}'::jsonb
),
(
    'forces_free_body_leg_1',
    'forces',
    'newtons_second_law',
    1,
    'For a forces problem, what should you set up before choosing component equations?',
    '[
        {"id": "free_body_axes", "text": "Draw a free-body diagram and choose useful axes"},
        {"id": "kinematics_first", "text": "Start with a constant-acceleration displacement equation"}
    ]'::jsonb,
    'free_body_axes',
    'Draw a free-body diagram and choose useful axes',
    'Forces problems start by isolating the object, drawing forces, and choosing axes before applying Newtons second law.',
    '{"next_leg_on_wrong": 2}'::jsonb
),
(
    'forces_newton_second_leg_2',
    'forces',
    'newtons_second_law',
    2,
    'Once the x-axis is chosen, which equation represents Newtons second law along that axis?',
    '[
        {"id": "sum_fx", "text": "sum F_x = m*a_x"},
        {"id": "vf_squared", "text": "v_f^2 = v_i^2 + 2*a*dx"}
    ]'::jsonb,
    'sum_fx',
    'sum F_x = m*a_x',
    'After the free-body diagram, write the net force along each axis as mass times acceleration along that axis.',
    '{}'::jsonb
)
ON CONFLICT (question_key) DO UPDATE SET
    agent_type = EXCLUDED.agent_type,
    topic = EXCLUDED.topic,
    leg = EXCLUDED.leg,
    question_type = EXCLUDED.question_type,
    question_text = EXCLUDED.question_text,
    choices = EXCLUDED.choices,
    correct_choice_id = EXCLUDED.correct_choice_id,
    correct_answer = EXCLUDED.correct_answer,
    explanation = EXCLUDED.explanation,
    metadata = EXCLUDED.metadata,
    is_active = TRUE,
    updated_at = CURRENT_TIMESTAMP;

-- Add comments for documentation
COMMENT ON TABLE users IS 'Core user accounts for the Physics Assistant platform';
COMMENT ON TABLE user_sessions IS 'Active user sessions with expiration tracking';
COMMENT ON TABLE interactions IS 'Log of all user interactions with physics agents and tools';
COMMENT ON TABLE messages IS 'Individual messages within interactions, including chat and system messages';
COMMENT ON TABLE agent_calls IS 'Specific calls to MCP tools and physics agents';
COMMENT ON TABLE user_preferences IS 'User-specific settings and preferences';
COMMENT ON TABLE user_progress IS 'Learning progress tracking for different physics topics';
COMMENT ON TABLE file_uploads IS 'Metadata for uploaded physics diagrams and problem files';
COMMENT ON TABLE hitl_questions IS 'Human-in-the-loop diagnostic questions shown before full agent solving';
COMMENT ON TABLE hitl_attempts IS 'Student responses to human-in-the-loop diagnostic questions';

COMMENT ON COLUMN users.metadata IS 'Additional user data as JSON (profile info, settings, etc.)';
COMMENT ON COLUMN interactions.metadata IS 'Additional interaction context and debug information';
COMMENT ON COLUMN messages.attachments IS 'Array of file attachment metadata';
COMMENT ON COLUMN user_progress.achievements IS 'Array of earned achievements and milestones';
COMMENT ON COLUMN hitl_questions.choices IS 'Multiple choice options as an array of objects with id and text';
COMMENT ON COLUMN hitl_attempts.user_id IS 'Incoming UI or account user identifier';
