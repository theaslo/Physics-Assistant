-- Sample data for Physics Assistant Database
-- This file contains sample data for testing and development
-- Created: 2025-08-14

-- Insert sample users
INSERT INTO users (email, username, password_hash, first_name, last_name, role, is_verified) VALUES
    ('admin@physics-assistant.com', 'admin', crypt('admin123', gen_salt('bf')), 'System', 'Administrator', 'admin', true),
    ('instructor@physics-assistant.com', 'instructor', crypt('instructor123', gen_salt('bf')), 'Jane', 'Smith', 'instructor', true),
    ('student1@university.edu', 'student1', crypt('student123', gen_salt('bf')), 'John', 'Doe', 'student', true),
    ('student2@university.edu', 'student2', crypt('student123', gen_salt('bf')), 'Alice', 'Johnson', 'student', true),
    ('student3@university.edu', 'student3', crypt('student123', gen_salt('bf')), 'Bob', 'Wilson', 'student', false);

-- Insert user preferences for sample users
INSERT INTO user_preferences (user_id, theme, language, preferred_units, difficulty_level) 
SELECT 
    id, 
    CASE 
        WHEN username = 'admin' THEN 'dark'
        WHEN username = 'instructor' THEN 'light'
        ELSE 'light'
    END as theme,
    'en',
    'metric',
    CASE 
        WHEN role = 'student' THEN 'beginner'
        ELSE 'advanced'
    END as difficulty_level
FROM users;

-- Insert sample user progress for physics topics
INSERT INTO user_progress (user_id, topic, problems_attempted, problems_solved, proficiency_score)
SELECT 
    u.id,
    topic.name,
    CASE 
        WHEN u.username = 'student1' THEN floor(random() * 20 + 5)::int
        WHEN u.username = 'student2' THEN floor(random() * 15 + 3)::int
        ELSE 0
    END as problems_attempted,
    CASE 
        WHEN u.username = 'student1' THEN floor(random() * 15 + 2)::int
        WHEN u.username = 'student2' THEN floor(random() * 10 + 1)::int
        ELSE 0
    END as problems_solved,
    CASE 
        WHEN u.username = 'student1' THEN round((random() * 40 + 60)::numeric, 2)
        WHEN u.username = 'student2' THEN round((random() * 30 + 50)::numeric, 2)
        ELSE 0.0
    END as proficiency_score
FROM users u
CROSS JOIN (VALUES 
    ('kinematics'),
    ('forces'),
    ('energy'),
    ('momentum'),
    ('angular_motion')
) AS topic(name)
WHERE u.role = 'student';

-- Insert sample active sessions (only for verified users)
INSERT INTO user_sessions (user_id, session_token, ip_address, user_agent, expires_at)
SELECT 
    id,
    encode(gen_random_bytes(32), 'hex'),
    '127.0.0.1'::inet,
    'Mozilla/5.0 (compatible; Physics-Assistant-Test)',
    CURRENT_TIMESTAMP + INTERVAL '7 days'
FROM users 
WHERE is_verified = true AND role != 'admin';

-- Insert sample interactions
WITH sample_sessions AS (
    SELECT user_id, id as session_id 
    FROM user_sessions 
    LIMIT 3
)
INSERT INTO interactions (user_id, session_id, type, agent_type, request_data, response_data, execution_time_ms, success)
SELECT 
    s.user_id,
    s.session_id,
    'chat'::interaction_type,
    'kinematics'::agent_type,
    '{"question": "What is the velocity of an object in free fall after 3 seconds?", "context": "introductory physics"}'::jsonb,
    '{"answer": "v = gt = 9.8 * 3 = 29.4 m/s downward", "latex": "v = gt = 9.8 \\times 3 = 29.4 \\text{ m/s}"}'::jsonb,
    floor(random() * 2000 + 500)::int,
    true
FROM sample_sessions s
UNION ALL
SELECT 
    s.user_id,
    s.session_id,
    'mcp_tool'::interaction_type,
    'forces'::agent_type,
    '{"force_type": "spring", "displacement": 0.1, "spring_constant": 200}'::jsonb,
    '{"force_magnitude": 20, "direction": "restoring", "unit": "N"}'::jsonb,
    floor(random() * 1500 + 300)::int,
    true
FROM sample_sessions s
UNION ALL
SELECT 
    s.user_id,
    s.session_id,
    'calculation'::interaction_type,
    'energy'::agent_type,
    '{"problem_type": "kinetic_energy", "mass": 2.5, "velocity": 15}'::jsonb,
    '{"kinetic_energy": 281.25, "unit": "J", "formula": "KE = (1/2)mv²"}'::jsonb,
    floor(random() * 800 + 200)::int,
    true
FROM sample_sessions s;

-- Insert sample messages for the interactions
INSERT INTO messages (interaction_id, user_id, type, content, content_latex, tokens_used, model_name)
SELECT 
    i.id,
    i.user_id,
    'user'::message_type,
    'What is the velocity of an object in free fall after 3 seconds?',
    NULL,
    15,
    NULL
FROM interactions i
WHERE i.type = 'chat'
UNION ALL
SELECT 
    i.id,
    i.user_id,
    'assistant'::message_type,
    'For an object in free fall, we use the equation v = gt, where g is acceleration due to gravity (9.8 m/s²) and t is time.',
    'v = gt = 9.8 \times 3 = 29.4 \text{ m/s}',
    45,
    'ollama/physics-tutor'
FROM interactions i
WHERE i.type = 'chat';

-- Insert sample agent calls
INSERT INTO agent_calls (interaction_id, user_id, agent_type, tool_name, function_name, input_parameters, output_result, execution_time_ms, success, model_used, tokens_consumed)
SELECT 
    i.id,
    i.user_id,
    'forces'::agent_type,
    'spring_force_calculator',
    'calculate_spring_force',
    '{"displacement": 0.1, "spring_constant": 200}'::jsonb,
    '{"force": 20, "direction": "restoring", "unit": "N"}'::jsonb,
    floor(random() * 100 + 50)::int,
    true,
    'ollama/physics-forces',
    25
FROM interactions i
WHERE i.type = 'mcp_tool';

-- Add some sample file uploads
INSERT INTO file_uploads (user_id, original_filename, stored_filename, file_path, file_size, mime_type, file_hash, is_processed)
SELECT 
    u.id,
    'physics_diagram_' || u.username || '.png',
    uuid_generate_v4()::text || '.png',
    '/uploads/diagrams/' || uuid_generate_v4()::text || '.png',
    floor(random() * 500000 + 10000)::bigint,
    'image/png',
    encode(sha256(random()::text::bytea), 'hex'),
    true
FROM users u
WHERE u.role = 'student' AND u.is_verified = true;

-- Update some statistics
UPDATE user_progress 
SET 
    total_interaction_time = floor(random() * 120 + 30)::int,
    last_activity = CURRENT_TIMESTAMP - (random() * INTERVAL '7 days'),
    achievements = '[{"type": "first_problem", "earned_at": "2024-01-15"}, {"type": "quick_solver", "earned_at": "2024-01-20"}]'::jsonb
WHERE proficiency_score > 0;

-- Add some metadata examples
UPDATE users 
SET metadata = jsonb_build_object(
    'registration_ip', '192.168.1.100',
    'timezone', 'America/New_York',
    'academic_year', '2024-2025',
    'course_codes', ARRAY['PHYS101', 'PHYS102']
)
WHERE role = 'student';

UPDATE interactions 
SET metadata = jsonb_build_object(
    'browser_info', 'Chrome/119.0',
    'screen_resolution', '1920x1080',
    'response_time', '2.3s'
)
WHERE type = 'chat';