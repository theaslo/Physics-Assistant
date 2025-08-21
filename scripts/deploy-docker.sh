#!/bin/bash
set -e

# Physics Assistant Docker Deployment Script
echo "=== Physics Assistant Docker Deployment ==="

# Configuration
ENVIRONMENT=${1:-production}
COMPOSE_FILE=""
ENV_FILE=""

case $ENVIRONMENT in
    "production")
        COMPOSE_FILE="docker-compose.production.yml"
        ENV_FILE=".env.production"
        ;;
    "development")
        COMPOSE_FILE="docker-compose.development.yml"
        ENV_FILE=".env.development"
        ;;
    *)
        echo "Usage: $0 [production|development]"
        echo "Default: production"
        exit 1
        ;;
esac

echo "Deploying in $ENVIRONMENT environment..."
echo "Using compose file: $COMPOSE_FILE"
echo "Using environment file: $ENV_FILE"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file $ENV_FILE not found!"
    echo "Please copy and configure the environment file:"
    echo "cp $ENV_FILE.template $ENV_FILE"
    exit 1
fi

# Source environment variables
source $ENV_FILE

# Validate required environment variables
REQUIRED_VARS=(
    "POSTGRES_PASSWORD"
    "PHYSICS_DB_PASSWORD"
    "NEO4J_PASSWORD"
    "REDIS_PASSWORD"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set!"
        exit 1
    fi
done

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p database/logs
mkdir -p database/backups
mkdir -p database/neo4j/logs
mkdir -p UI/logs
mkdir -p UI/uploads
mkdir -p UI/temp
mkdir -p analytics/models
mkdir -p analytics/logs
mkdir -p analytics/exports

# Build and start services
echo "Building and starting Docker services..."
docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d --build

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 30

# Check service health
echo "Checking service health..."
SERVICES=(
    "postgres:5432"
    "redis:6379"
    "neo4j:7474"
    "database-api:8001"
    "dashboard-api:8002"
    "physics-agents-api:8000"
)

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name port <<< "$service"
    echo "Checking $name on port $port..."
    
    # Wait up to 60 seconds for service to be ready
    timeout 60 bash -c "until nc -z localhost $port; do sleep 1; done" || {
        echo "Error: $name service failed to start on port $port"
        docker-compose -f $COMPOSE_FILE logs $name
        exit 1
    }
    echo "$name is ready!"
done

# Show deployment status
echo ""
echo "=== Deployment Status ==="
docker-compose -f $COMPOSE_FILE ps

echo ""
echo "=== Access Information ==="
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Main Application: http://localhost"
    echo "Analytics Dashboard: http://localhost/dashboard"
    echo "Grafana Monitoring: http://localhost:3000"
    echo "Prometheus Metrics: http://localhost:9090"
    echo "Flower Task Monitor: http://localhost:5555"
else
    echo "Main Application: http://localhost:8501"
    echo "Analytics Dashboard: http://localhost:5173"
    echo "Database API: http://localhost:8001"
    echo "Dashboard API: http://localhost:8002"
    echo "PostgreSQL: localhost:5432"
    echo "Neo4j: http://localhost:7474"
    echo "Redis: localhost:6379"
    echo "Grafana: http://localhost:3000"
    echo "Prometheus: http://localhost:9090"
fi

echo ""
echo "=== Deployment Complete ==="
echo "All services are running successfully!"

# Show logs for critical services
if [ "$ENVIRONMENT" = "development" ]; then
    echo ""
    echo "To view logs, use:"
    echo "docker-compose -f $COMPOSE_FILE logs -f [service-name]"
    echo ""
    echo "To stop all services:"
    echo "docker-compose -f $COMPOSE_FILE down"
fi