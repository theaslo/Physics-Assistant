#!/bin/bash
set -e

echo "Starting Physics Agents API Server..."

# Wait for Database API to be ready
echo "Waiting for Database API to be ready..."
while ! curl -f -s http://${DATABASE_API_HOST:-database-api}:${DATABASE_API_PORT:-8001}/health > /dev/null; do
    echo "Database API is not ready yet..."
    sleep 5
done
echo "Database API is ready!"

# Wait for MCP servers to be ready (check a few critical ones)
echo "Waiting for MCP servers to be ready..."
for service in forces kinematics math energy momentum; do
    echo "Checking MCP $service server..."
    while ! nc -z mcp-${service} 10100; do
        echo "MCP $service server is not ready yet..."
        sleep 2
    done
    echo "MCP $service server is ready!"
done

# Start the Physics Agents API server
echo "Starting Physics Agents API server on port 8000..."
cd /app/api
exec uv run python main.py