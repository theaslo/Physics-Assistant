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
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run forces-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "kinematics")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run kinematics-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "math")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run math-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "energy")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run energy-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "momentum")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run momentum-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "angular-motion")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run angular-motion-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "circuit")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run circuit-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "thermodynamics")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run thermodynamics-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "waves")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run waves-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "electromagnetism")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run electromagnetism-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "optics")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run optics-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "modern-physics")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run modern-physics-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    "knowledge-transfer")
        cd /app/mcp_tools && exec python -m physics_mcp_tools --run knowledge-transfer-server --host $MCP_HOST --port $MCP_PORT --transport $MCP_TRANSPORT
        ;;
    *)
        echo "Unknown MCP service: $MCP_SERVICE"
        echo "Available services: forces, kinematics, math, energy, momentum, angular-motion, circuit, thermodynamics, waves, electromagnetism, optics, modern-physics, knowledge-transfer"
        exit 1
        ;;
esac
