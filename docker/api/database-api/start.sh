#!/bin/bash
set -e

echo "Starting Database API Server..."

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

# Run database migrations if needed
echo "Running database setup..."
cd /app/database
python setup_schema.py

# Start the API server
echo "Starting Database API server on port 8001..."
exec python api_server.py