#!/bin/bash
set -e

# Initialize Physics Assistant Database
echo "Initializing Physics Assistant database..."

# Create application database and user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create application user
    CREATE USER physics_user WITH PASSWORD '$PHYSICS_DB_PASSWORD';
    
    -- Create application database
    CREATE DATABASE physics_assistant OWNER physics_user;
    
    -- Grant privileges
    GRANT ALL PRIVILEGES ON DATABASE physics_assistant TO physics_user;
    
    -- Connect to the application database
    \c physics_assistant
    
    -- Grant schema privileges
    GRANT ALL ON SCHEMA public TO physics_user;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO physics_user;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO physics_user;
    
    -- Create extensions
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
    CREATE EXTENSION IF NOT EXISTS "pg_trgm";
    
    -- Create monitoring user (read-only)
    CREATE USER monitor_user WITH PASSWORD '$MONITOR_DB_PASSWORD';
    GRANT CONNECT ON DATABASE physics_assistant TO monitor_user;
    GRANT USAGE ON SCHEMA public TO monitor_user;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitor_user;
    GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO monitor_user;
    
    -- Set default privileges for future tables
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO monitor_user;
EOSQL

echo "Physics Assistant database initialization completed."