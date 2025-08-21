#!/bin/bash
set -e

echo "🚀 Starting Physics Assistant Monitoring Stack"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create network if it doesn't exist
echo "📡 Creating network..."
docker network create physics_assistant_network --driver bridge || true

# Start main database services if not already running
echo "🗄️ Checking main database services..."
if ! docker compose -f docker-compose.yml ps | grep -q "Up"; then
    echo "Starting main database services first..."
    docker compose -f docker-compose.yml up -d
    
    echo "⏳ Waiting for databases to be ready..."
    sleep 30
fi

# Start monitoring stack
echo "📊 Starting monitoring services..."
cd monitoring
docker compose -f docker-compose-monitoring.yml up -d

echo "⏳ Waiting for Prometheus and Grafana to start..."
sleep 20

# Check service health
echo "🔍 Checking service health..."
services=(
    "prometheus:9090"
    "grafana:3000"
    "node-exporter:9100"
    "postgres-exporter:9187"
    "redis-exporter:9121"
    "alertmanager:9093"
)

for service in "${services[@]}"; do
    host_port=${service#*:}
    service_name=${service%:*}
    
    if curl -s -f "http://localhost:${host_port}" > /dev/null; then
        echo "✅ ${service_name} is healthy"
    else
        echo "⚠️ ${service_name} might not be ready yet"
    fi
done

echo ""
echo "🎉 Monitoring stack started successfully!"
echo ""
echo "📊 Grafana Dashboard: http://localhost:3000"
echo "   Username: admin"
echo "   Password: physics_dashboard_2024"
echo ""
echo "🔍 Prometheus: http://localhost:9090"
echo "📈 Alertmanager: http://localhost:9093"
echo ""
echo "📋 Available dashboards:"
echo "   • Physics Assistant Overview"
echo "   • Database Health"
echo "   • System Metrics"
echo ""
echo "To stop monitoring: ./stop_monitoring.sh"