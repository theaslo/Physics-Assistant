#!/bin/bash
set -e

echo "Starting Flower Task Monitor..."

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
while ! nc -z ${REDIS_HOST:-redis} ${REDIS_PORT:-6379}; do
    echo "Redis is not ready yet..."
    sleep 2
done
echo "Redis is ready!"

# Start Flower monitoring (use celery flower command)
echo "Starting Flower on port 5555..."
exec celery --broker=${CELERY_BROKER_URL} flower \
    --port=5555 \
    --address=0.0.0.0 \
    --basic_auth=admin:physics_flower_2024