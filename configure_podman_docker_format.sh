#!/bin/bash

# Configure Podman to use Docker format by default
echo "🔧 Configuring Podman to use Docker format..."

# Create containers directory if it doesn't exist
mkdir -p ~/.config/containers

# Configure containers.conf for Docker format
cat > ~/.config/containers/containers.conf << 'EOF'
[engine]
image_default_format = "docker"

[containers]
default_sysctls = []
EOF

echo "✅ Podman configured to use Docker format"
echo "📋 Configuration applied:"
echo "  - Default image format: docker"
echo "  - HEALTHCHECK instructions will be supported"
echo ""
echo "🔄 Restart your shell or run:"
echo "  source ~/.bashrc"