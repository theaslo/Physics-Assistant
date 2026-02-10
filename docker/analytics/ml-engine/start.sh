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
print('Checking analytics modules...')
try:
    from learning_analytics import LearningAnalyticsEngine
    print('  - LearningAnalyticsEngine loaded')
except ImportError as e:
    print(f'  - LearningAnalyticsEngine not available: {e}')

try:
    from predictive_analytics import Phase63PredictiveAnalyticsEngine
    print('  - Phase63PredictiveAnalyticsEngine loaded')
except ImportError as e:
    print(f'  - Phase63PredictiveAnalyticsEngine not available: {e}')

print('Analytics modules check complete!')
"

# Start the ML analytics service
echo "Starting ML Analytics Engine on port 8003..."
exec python -m uvicorn analytics.api:app --host 0.0.0.0 --port 8003 --workers 2