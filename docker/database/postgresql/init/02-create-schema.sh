#!/bin/bash
set -e

echo "Creating Physics Assistant schema..."

# Execute schema creation scripts
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "physics_assistant" <<-EOSQL
    -- Switch to application user
    SET ROLE physics_user;
    
    -- Execute schema creation
    \i /opt/physics-assistant/schema/01_core_tables.sql
    \i /opt/physics-assistant/schema/02_sample_data.sql
EOSQL

echo "Schema creation completed."