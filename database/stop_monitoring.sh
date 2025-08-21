#!/bin/bash
set -e

echo "🛑 Stopping Physics Assistant Monitoring Stack"

cd monitoring

# Stop monitoring services
echo "📊 Stopping monitoring services..."
docker compose -f docker-compose-monitoring.yml down

# Clean up unused volumes (optional - comment out if you want to keep data)
echo "🧹 Cleaning up unused resources..."
docker system prune -f --volumes

echo ""
echo "✅ Monitoring stack stopped successfully!"
echo ""
echo "Note: Database services are still running."
echo "To stop everything: docker compose -f ../docker-compose.yml down"