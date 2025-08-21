#!/bin/bash
set -e

echo "Starting Streamlit UI..."

# Wait for Physics Agents API to be ready
echo "Waiting for Physics Agents API to be ready..."
while ! curl -f -s http://${PHYSICS_API_HOST:-physics-agents-api}:${PHYSICS_API_PORT:-8000}/health > /dev/null; do
    echo "Physics Agents API is not ready yet..."
    sleep 5
done
echo "Physics Agents API is ready!"

# Wait for Database API to be ready
echo "Waiting for Database API to be ready..."
while ! curl -f -s http://${DATABASE_API_HOST:-database-api}:${DATABASE_API_PORT:-8001}/health > /dev/null; do
    echo "Database API is not ready yet..."
    sleep 5
done
echo "Database API is ready!"

# Start the Streamlit application
echo "Starting Streamlit UI on port 8501..."
cd /app/frontend
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true