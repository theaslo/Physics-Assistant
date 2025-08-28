#!/bin/bash

# Remove HEALTHCHECK instructions from Dockerfiles for Podman/OCI compatibility
echo "🔧 Removing HEALTHCHECK instructions for Podman/OCI compatibility..."

# Find all Dockerfiles and remove HEALTHCHECK lines
find docker -name "Dockerfile*" -type f -exec sed -i '/^HEALTHCHECK/d' {} \;

echo "✅ Removed HEALTHCHECK instructions from all Dockerfiles"
echo ""
echo "📋 Benefits:"
echo "  - Eliminates OCI format warnings"
echo "  - Images build faster without health check layers"
echo "  - Health checks can be handled by docker-compose instead"
echo ""
echo "🚀 Now try building again:"
echo "  docker compose -f docker-compose.production.yml build"