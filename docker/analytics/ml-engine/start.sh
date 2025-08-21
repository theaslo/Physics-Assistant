#!/bin/bash
set -e

echo "Starting ML Analytics Engine..."

# Wait for Database API to be ready
echo "Waiting for Database API to be ready..."
while ! curl -f -s http://${DATABASE_API_HOST:-database-api}:${DATABASE_API_PORT:-8001}/health > /dev/null; do
    echo "Database API is not ready yet..."
    sleep 5
done
echo "Database API is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
while ! nc -z ${REDIS_HOST:-redis} ${REDIS_PORT:-6379}; do
    echo "Redis is not ready yet..."
    sleep 2
done
echo "Redis is ready!"

# Initialize ML models if needed
echo "Initializing ML models..."
cd /app/analytics
python -c "
import os
from learning_analytics import LearningAnalytics
from predictive_analytics import PredictiveAnalytics

print('Initializing analytics engines...')
learning_analytics = LearningAnalytics()
predictive_analytics = PredictiveAnalytics()
print('Analytics engines initialized successfully!')
"

# Start the ML analytics service
echo "Starting ML Analytics Engine on port 8003..."
exec python -m uvicorn analytics.api:app --host 0.0.0.0 --port 8003 --workers 2