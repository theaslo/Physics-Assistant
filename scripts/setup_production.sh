#!/bin/bash
# Physics Assistant Production Setup Script

set -e  # Exit on any error

echo "🚀 Physics Assistant Production Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check Docker/Podman
    if command -v docker &> /dev/null; then
        print_status "✅ Docker found: $(docker --version)"
    elif command -v podman &> /dev/null; then
        print_status "✅ Podman found: $(podman --version)"
    else
        print_error "❌ Neither Docker nor Podman found. Please install one of them."
        exit 1
    fi

    # Check Python
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version)
        print_status "✅ Python found: $python_version"
    else
        print_error "❌ Python3 not found. Please install Python 3.13+."
        exit 1
    fi

    # Check UV (optional)
    if command -v uv &> /dev/null; then
        print_status "✅ UV found: $(uv --version)"
    else
        print_warning "⚠️  UV not found. Consider installing for faster Python package management."
    fi
}

# Setup environment configuration
setup_environment() {
    print_status "Setting up environment configuration..."

    if [ ! -f .env.production ]; then
        if [ -f .env.production.template ]; then
            cp .env.production.template .env.production
            print_status "✅ Created .env.production from template"
            print_warning "⚠️  Please edit .env.production and update all CHANGE_ME values!"
        else
            print_error "❌ .env.production.template not found"
            exit 1
        fi
    else
        print_status "✅ .env.production already exists"
    fi
}

# Create required directories
create_directories() {
    print_status "Creating required directories..."

    directories=(
        "logs"
        "backups"
        "data"
        "exports"
        "uploads"
    )

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "✅ Created directory: $dir"
        else
            print_status "✅ Directory exists: $dir"
        fi
    done
}

# Build and start MCP servers
start_mcp_servers() {
    print_status "Building and starting MCP servers..."

    if [ -f docker-compose.mcp-fixed.yaml ]; then
        # Build containers
        print_status "Building MCP server containers..."
        docker compose -f docker-compose.mcp-fixed.yaml build

        # Start containers
        print_status "Starting MCP server containers..."
        docker compose -f docker-compose.mcp-fixed.yaml up -d

        # Wait for containers to start
        sleep 10

        # Check container status
        print_status "Checking container status..."
        docker compose -f docker-compose.mcp-fixed.yaml ps

    else
        print_error "❌ docker-compose.mcp-fixed.yaml not found"
        exit 1
    fi
}

# Run health checks
run_health_checks() {
    print_status "Running health checks..."

    # Check if test script exists
    if [ -f database/fixes/test_physics_workflow.py ]; then
        print_status "Running end-to-end system test..."
        python database/fixes/test_physics_workflow.py

        if [ $? -eq 0 ]; then
            print_status "✅ Health checks passed!"
        else
            print_error "❌ Health checks failed!"
            return 1
        fi
    else
        print_warning "⚠️  Health check script not found. Running manual checks..."

        # Manual health checks
        servers=(10100 10101 10103 10104 10105 10106)
        server_names=("forces" "kinematics" "math" "momentum" "energy" "angular-motion")

        for i in "${!servers[@]}"; do
            port=${servers[$i]}
            name=${server_names[$i]}

            if curl -s -m 2 http://localhost:$port/ >/dev/null; then
                print_status "✅ $name server (port $port): Online"
            else
                print_error "❌ $name server (port $port): Offline"
            fi
        done
    fi
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."

    # Create monitoring script
    cat > scripts/monitor_system.sh << 'EOF'
#!/bin/bash
# System monitoring script for Physics Assistant

echo "Physics Assistant System Status"
echo "==============================="
echo "Timestamp: $(date)"
echo ""

# Check container status
echo "MCP Server Status:"
docker compose -f docker-compose.mcp-fixed.yaml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Check resource usage
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""

# Check server connectivity
echo "Server Connectivity:"
for port in 10100 10101 10103 10104 10105 10106; do
    if curl -s -m 2 http://localhost:$port/ >/dev/null; then
        echo "Port $port: ✅ Online"
    else
        echo "Port $port: ❌ Offline"
    fi
done
EOF

    chmod +x scripts/monitor_system.sh
    print_status "✅ Created monitoring script: scripts/monitor_system.sh"
}

# Setup backup procedures
setup_backup() {
    print_status "Setting up backup procedures..."

    # Create backup script
    cat > scripts/backup_system.sh << 'EOF'
#!/bin/bash
# Backup script for Physics Assistant

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup in: $BACKUP_DIR"

# Backup configuration files
cp docker-compose.mcp-fixed.yaml "$BACKUP_DIR/"
cp .env.production "$BACKUP_DIR/"
cp -r mcp_tools/ "$BACKUP_DIR/"

# Backup logs (if they exist)
if [ -d logs ]; then
    cp -r logs/ "$BACKUP_DIR/"
fi

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"

# Clean old backups (keep last 30 days)
find backups/ -name "*.tar.gz" -type f -mtime +30 -delete
EOF

    chmod +x scripts/backup_system.sh
    print_status "✅ Created backup script: scripts/backup_system.sh"
}

# Create production start script
create_start_script() {
    print_status "Creating production start script..."

    cat > start_production.sh << 'EOF'
#!/bin/bash
# Start Physics Assistant in production mode

set -e

echo "🚀 Starting Physics Assistant Production Environment"
echo "=================================================="

# Load environment variables
if [ -f .env.production ]; then
    source .env.production
    echo "✅ Loaded production environment variables"
else
    echo "❌ .env.production not found. Run setup_production.sh first."
    exit 1
fi

# Start MCP servers
echo "Starting MCP servers..."
docker compose -f docker-compose.mcp-fixed.yaml up -d

# Wait for servers to start
echo "Waiting for servers to start..."
sleep 15

# Run health checks
echo "Running health checks..."
if [ -f database/fixes/test_physics_workflow.py ]; then
    python database/fixes/test_physics_workflow.py
else
    echo "Manual health check - verifying server connectivity..."
    for port in 10100 10101 10103 10104 10105 10106; do
        if curl -s -m 2 http://localhost:$port/ >/dev/null; then
            echo "Port $port: ✅"
        else
            echo "Port $port: ❌"
        fi
    done
fi

echo ""
echo "🎉 Physics Assistant is running in production mode!"
echo "================================================="
echo "MCP Servers:"
echo "  Forces:        http://localhost:10100"
echo "  Kinematics:    http://localhost:10101"
echo "  Math:          http://localhost:10103"
echo "  Momentum:      http://localhost:10104"
echo "  Energy:        http://localhost:10105"
echo "  Angular Motion: http://localhost:10106"
echo ""
echo "To monitor: ./scripts/monitor_system.sh"
echo "To backup:  ./scripts/backup_system.sh"
echo "To stop:    docker compose -f docker-compose.mcp-fixed.yaml down"
EOF

    chmod +x start_production.sh
    print_status "✅ Created production start script: start_production.sh"
}

# Main setup function
main() {
    echo ""
    print_status "Starting Physics Assistant production setup..."
    echo ""

    # Create scripts directory if it doesn't exist
    mkdir -p scripts

    # Run setup steps
    check_prerequisites
    setup_environment
    create_directories
    setup_monitoring
    setup_backup
    create_start_script

    # Ask if user wants to start services now
    echo ""
    read -p "Do you want to start the MCP servers now? (y/N): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_mcp_servers
        run_health_checks

        echo ""
        print_status "🎉 Physics Assistant production setup complete!"
        print_status "System is running and ready for physics tutoring."
        echo ""
        print_status "Next steps:"
        print_status "1. Edit .env.production and update CHANGE_ME values"
        print_status "2. Start UI: streamlit run UI/frontend/app.py"
        print_status "3. Monitor system: ./scripts/monitor_system.sh"
        print_status "4. Create backups: ./scripts/backup_system.sh"
    else
        echo ""
        print_status "Setup complete! To start the system later:"
        print_status "1. Edit .env.production and update CHANGE_ME values"
        print_status "2. Run: ./start_production.sh"
    fi
}

# Run main function
main "$@"