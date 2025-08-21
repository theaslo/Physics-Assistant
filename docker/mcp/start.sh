#!/bin/bash
set -e

echo "Starting MCP Physics Tools Server..."

# Determine which service to start based on environment variable
MCP_SERVICE=${MCP_SERVICE:-forces}
MCP_PORT=${MCP_PORT:-10100}
MCP_HOST=${MCP_HOST:-0.0.0.0}
MCP_TRANSPORT=${MCP_TRANSPORT:-streamable_http}

echo "Starting MCP $MCP_SERVICE server on $MCP_HOST:$MCP_PORT with transport $MCP_TRANSPORT"

# Health check endpoint (basic HTTP server for health checks)
(
    while true; do
        echo -e "HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\nHealthy" | nc -l -p 8080
    done
) &

# Start the appropriate MCP server
case $MCP_SERVICE in
    "forces")
        exec uv run physics-mcp --run forces-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "kinematics")
        exec uv run physics-mcp --run kinematics-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "math")
        exec uv run physics-mcp --run math-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "energy")
        exec uv run physics-mcp --run energy-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "momentum")
        exec uv run physics-mcp --run momentum-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "angular-motion")
        exec uv run physics-mcp --run angular-motion-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "circuit")
        exec uv run physics-mcp --run circuit-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    *)
        echo "Unknown MCP service: $MCP_SERVICE"
        echo "Available services: forces, kinematics, math, energy, momentum, angular-motion, circuit"
        exit 1
        ;;
esac