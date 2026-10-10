-- Service telemetry is not a student interaction and must not require a login account.
CREATE TABLE IF NOT EXISTS mcp_tool_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    response TEXT NOT NULL,
    execution_time_ms INTEGER NOT NULL CHECK (execution_time_ms >= 0),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_events_service_time
    ON mcp_tool_events (service_name, created_at DESC);
