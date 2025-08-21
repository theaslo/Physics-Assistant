#!/bin/bash
set -e

echo "Starting Dashboard API Server..."

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! nc -z ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432}; do
    echo "PostgreSQL is not ready yet..."
    sleep 2
done
echo "PostgreSQL is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
while ! nc -z ${REDIS_HOST:-redis} ${REDIS_PORT:-6379}; do
    echo "Redis is not ready yet..."
    sleep 2
done
echo "Redis is ready!"

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to be ready..."
while ! curl -f -s http://${NEO4J_HOST:-neo4j}:${NEO4J_HTTP_PORT:-7474}/db/data/ > /dev/null; do
    echo "Neo4j is not ready yet..."
    sleep 5
done
echo "Neo4j is ready!"

# Wait for Database API to be ready
echo "Waiting for Database API to be ready..."
while ! curl -f -s http://${DATABASE_API_HOST:-database-api}:${DATABASE_API_PORT:-8001}/health > /dev/null; do
    echo "Database API is not ready yet..."
    sleep 5
done
echo "Database API is ready!"

# Start the Dashboard API server
echo "Starting Dashboard API server on port 8002..."
exec python dashboard_api_server.py