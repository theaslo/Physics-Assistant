#!/bin/bash
# Complete Physics Assistant System Startup Script
# Includes Database, UI, and all MCP servers for data collection

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}   Physics Assistant - Complete System Startup${NC}"
    echo -e "${BLUE}   Database + UI + MCP Servers${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check for Docker or Podman
    if ! command -v docker >/dev/null 2>&1 && ! command -v podman >/dev/null 2>&1; then
        print_error "Neither Docker nor Podman found. Please install one of them."
        exit 1
    fi

    # Check for docker-compose or podman-compose
    if command -v docker >/dev/null 2>&1; then
        if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
            print_error "docker-compose not found. Please install docker-compose."
            exit 1
        fi
        COMPOSE_CMD="docker-compose"
        if docker compose version >/dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
        fi
    else
        if ! command -v podman-compose >/dev/null 2>&1; then
            print_error "podman-compose not found. Please install podman-compose."
            exit 1
        fi
        COMPOSE_CMD="podman-compose"
    fi

    print_status "✅ Container runtime available: $(command -v docker >/dev/null 2>&1 && echo "Docker" || echo "Podman")"
    print_status "✅ Compose command: $COMPOSE_CMD"
}

# Check environment configuration
check_environment() {
    print_status "Checking environment configuration..."

    if [ ! -f ".env.production" ]; then
        if [ -f ".env.production.template" ]; then
            print_warning "Creating .env.production from template..."
            cp .env.production.template .env.production
            print_warning "⚠️  IMPORTANT: Edit .env.production and update all CHANGE_ME values!"
            print_warning "   Required: PHYSICS_DB_PASSWORD, NEO4J_PASSWORD, POSTGRES_PASSWORD"
            return 1
        else
            print_error ".env.production.template not found. Cannot proceed."
            exit 1
        fi
    fi

    # Check for critical environment variables
    if grep -q "CHANGE_ME" .env.production 2>/dev/null; then
        print_warning "⚠️  Found CHANGE_ME values in .env.production"
        print_warning "   Please update all CHANGE_ME values before starting the system"
        echo ""
        echo "Critical variables to update:"
        grep "CHANGE_ME" .env.production | head -5
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    print_status "✅ Environment configuration ready"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."

    local dirs=(
        "logs"
        "database/logs"
        "database/backups"
        "database/neo4j/logs"
        "database/neo4j/backups"
        "UI/logs"
        "UI/uploads"
        "UI/temp"
        "analytics/models"
        "analytics/logs"
        "analytics/exports"
        "analytics/temp"
        "analytics/results"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "✅ Created directory: $dir"
        fi
    done
}

# Create network if it doesn't exist
create_network() {
    print_status "Setting up container network..."

    local network_name="database_physics_network"

    if command -v docker >/dev/null 2>&1; then
        if ! docker network ls | grep -q "$network_name"; then
            docker network create "$network_name" --subnet=172.20.0.0/16
            print_status "✅ Created Docker network: $network_name"
        else
            print_status "✅ Docker network already exists: $network_name"
        fi
    elif command -v podman >/dev/null 2>&1; then
        if ! podman network ls | grep -q "$network_name"; then
            podman network create "$network_name" --subnet=172.20.0.0/16
            print_status "✅ Created Podman network: $network_name"
        else
            print_status "✅ Podman network already exists: $network_name"
        fi
    fi
}

# Start database services first
start_databases() {
    print_status "Starting database services..."

    # Start only database services first to ensure they're ready
    local db_services="postgres neo4j redis"

    print_status "Starting: PostgreSQL, Neo4j, Redis..."
    $COMPOSE_CMD -f docker-compose.production.yml up -d $db_services

    print_status "Waiting for databases to initialize (60 seconds)..."
    for i in {1..60}; do
        echo -n "."
        sleep 1
    done
    echo ""

    print_status "✅ Database services started"
}

# Start API services
start_apis() {
    print_status "Starting API services..."

    local api_services="database-api dashboard-api physics-agents-api"

    print_status "Starting: Database API, Dashboard API, Physics Agents API..."
    $COMPOSE_CMD -f docker-compose.production.yml up -d $api_services

    print_status "Waiting for APIs to initialize (30 seconds)..."
    for i in {1..30}; do
        echo -n "."
        sleep 1
    done
    echo ""

    print_status "✅ API services started"
}

# Start MCP servers
start_mcp_servers() {
    print_status "Starting MCP physics servers..."

    local mcp_services="mcp-forces mcp-kinematics mcp-math mcp-energy mcp-momentum mcp-angular-motion"

    print_status "Starting: Forces, Kinematics, Math, Energy, Momentum, Angular Motion..."
    $COMPOSE_CMD -f docker-compose.production.yml up -d $mcp_services

    print_status "Waiting for MCP servers to initialize (30 seconds)..."
    for i in {1..30}; do
        echo -n "."
        sleep 1
    done
    echo ""

    print_status "✅ MCP servers started"
}

# Start frontend services
start_frontend() {
    print_status "Starting frontend services..."

    local frontend_services="streamlit-ui react-dashboard nginx-gateway"

    print_status "Starting: Streamlit UI, React Dashboard, Nginx Gateway..."
    $COMPOSE_CMD -f docker-compose.production.yml up -d $frontend_services

    print_status "Waiting for frontend to initialize (20 seconds)..."
    for i in {1..20}; do
        echo -n "."
        sleep 1
    done
    echo ""

    print_status "✅ Frontend services started"
}

# Start optional services
start_optional_services() {
    print_status "Starting optional services (analytics & monitoring)..."

    local optional_services="ml-engine task-processor flower-monitor prometheus grafana alertmanager node-exporter cadvisor"

    print_status "Starting analytics and monitoring services..."
    $COMPOSE_CMD -f docker-compose.production.yml up -d $optional_services 2>/dev/null || {
        print_warning "⚠️  Some optional services failed to start (this is normal if images aren't built)"
    }

    print_status "✅ Optional services startup attempted"
}

# Verify system health
verify_system() {
    print_status "Verifying system health..."

    echo ""
    echo "🔍 Container Status:"
    if command -v docker >/dev/null 2>&1; then
        docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}" | grep physics || echo "No physics containers found"
    else
        podman ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}" | grep physics || echo "No physics containers found"
    fi

    echo ""
    echo "🔍 MCP Server Health:"

    # Test MCP servers
    local servers=(
        "Forces:10100"
        "Kinematics:10101"
        "Math:10103"
        "Momentum:10104"
        "Energy:10105"
        "Angular-Motion:10106"
    )

    local healthy_count=0
    for server_info in "${servers[@]}"; do
        IFS=':' read -r name port <<< "$server_info"
        if curl -s -m 2 http://localhost:$port/ >/dev/null 2>&1; then
            echo "  ✅ $name (port $port): Online"
            ((healthy_count++))
        else
            echo "  ❌ $name (port $port): Offline"
        fi
    done

    echo ""
    echo "📊 Summary: $healthy_count/6 MCP servers healthy"

    # Test database connectivity
    echo ""
    echo "🔍 Database Health:"

    # PostgreSQL
    if command -v docker >/dev/null 2>&1; then
        if docker exec physics-postgres pg_isready -U physics_user >/dev/null 2>&1; then
            echo "  ✅ PostgreSQL: Connected"
        else
            echo "  ❌ PostgreSQL: Not ready"
        fi

        # Redis
        if docker exec physics-redis redis-cli ping >/dev/null 2>&1; then
            echo "  ✅ Redis: Connected"
        else
            echo "  ❌ Redis: Not ready"
        fi
    else
        echo "  ⚠️  Database health check requires Docker (skipping with Podman)"
    fi
}

# Show access information
show_access_info() {
    echo ""
    echo -e "${BLUE}🎉 Physics Assistant System Started Successfully!${NC}"
    echo ""
    echo "📋 Access Points:"
    echo "  🎓 Student Interface:     http://localhost (or port 80)"
    echo "  📊 Admin Dashboard:       http://localhost/dashboard"
    echo "  📈 Monitoring (Grafana):  http://localhost:3000"
    echo "  🌸 Task Monitor (Flower): http://localhost:5555"
    echo "  🔍 Metrics (Prometheus):  http://localhost:9090"
    echo ""
    echo "🗄️ Database Access:"
    echo "  📊 PostgreSQL:  localhost:5432 (physics_assistant/physics_user)"
    echo "  🔗 Neo4j:       localhost:7474 (neo4j/<your_password>)"
    echo "  💾 Redis:       localhost:6379"
    echo ""
    echo "🔧 Management Commands:"
    echo "  Health Check:  ./scripts/simple_health_monitor.sh"
    echo "  View Logs:     docker logs physics-streamlit-ui -f"
    echo "  Stop System:   $COMPOSE_CMD -f docker-compose.production.yml down"
    echo "  Full Cleanup:  $COMPOSE_CMD -f docker-compose.production.yml down -v"
    echo ""
    echo "📈 Data Collection:"
    echo "  ✅ Database integration active for fine-tuning data collection"
    echo "  ✅ Student interactions will be logged to PostgreSQL"
    echo "  ✅ Tool usage analytics stored in database"
    echo "  ✅ Learning paths tracked in Neo4j graph database"
    echo ""
    echo -e "${GREEN}System ready for physics tutoring and data collection!${NC}"
}

# Main execution
main() {
    print_header

    check_prerequisites
    check_environment || {
        print_error "Environment setup required. Please edit .env.production and run again."
        exit 1
    }

    create_directories
    create_network

    print_status "Starting complete Physics Assistant system..."
    print_status "This will start: Database + UI + MCP Servers + Analytics"
    echo ""

    # Start services in dependency order
    start_databases
    start_apis
    start_mcp_servers
    start_frontend
    start_optional_services

    # Verify everything is working
    verify_system

    # Show access information
    show_access_info
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n🛑 Startup interrupted"; exit 1' INT

# Run main function
main "$@"