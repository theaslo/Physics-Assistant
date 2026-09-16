-- Add newer agent types to the PostgreSQL enum used by interaction logging.
-- Safe to run repeatedly on existing databases.

ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'thermodynamics';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'waves';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'electromagnetism';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'optics';
ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'modern_physics';
