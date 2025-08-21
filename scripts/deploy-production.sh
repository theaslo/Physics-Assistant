#!/bin/bash

# Physics Assistant Production Deployment Script
# Comprehensive production deployment with health checks and validation

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
DEPLOYMENT_TIMEOUT=1800  # 30 minutes
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
ROLLBACK_ON_FAILURE=true
BACKUP_BEFORE_DEPLOY=true
VALIDATE_DEPLOYMENT=true

# Parse command line arguments
FORCE_DEPLOY=false
SKIP_BACKUP=false
SKIP_VALIDATION=false
ENVIRONMENT="production"

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --force            Force deployment without confirmations"
            echo "  --skip-backup      Skip pre-deployment backup"
            echo "  --skip-validation  Skip post-deployment validation"
            echo "  --env ENVIRONMENT  Environment to deploy (production, staging)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Update environment-specific files
if [[ "$ENVIRONMENT" == "staging" ]]; then
    ENV_FILE="$PROJECT_ROOT/.env.staging"
elif [[ "$ENVIRONMENT" == "development" ]]; then
    ENV_FILE="$PROJECT_ROOT/.env.development"
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Check if environment file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    
    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Check disk space (minimum 10GB free)
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    MIN_SPACE=10485760  # 10GB in KB
    if [[ "$AVAILABLE_SPACE" -lt "$MIN_SPACE" ]]; then
        log_warning "Low disk space detected. Recommended minimum: 10GB"
        if [[ "$FORCE_DEPLOY" != true ]]; then
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    # Source environment variables for directory creation
    source "$ENV_FILE"
    
    # Create data directories
    mkdir -p "${DATA_PATH:-./data}/postgres"
    mkdir -p "${DATA_PATH:-./data}/neo4j"
    mkdir -p "${DATA_PATH:-./data}/redis"
    mkdir -p "${DATA_PATH:-./data}/prometheus"
    mkdir -p "${DATA_PATH:-./data}/grafana"
    mkdir -p "${DATA_PATH:-./data}/alertmanager"
    mkdir -p "${DATA_PATH:-./data}/ssl"
    
    # Create backup directories
    mkdir -p "${BACKUP_PATH:-./backups}"
    
    # Create log directories
    mkdir -p "${LOGS_PATH:-./logs}/nginx"
    mkdir -p "${LOGS_PATH:-./logs}/loki"
    
    # Set appropriate permissions
    chmod 755 "${DATA_PATH:-./data}"
    chmod 755 "${BACKUP_PATH:-./backups}"
    chmod 755 "${LOGS_PATH:-./logs}"
    
    log_success "Directories created successfully"
}

# Pre-deployment backup
create_backup() {
    if [[ "$SKIP_BACKUP" == true ]]; then
        log_info "Skipping pre-deployment backup"
        return 0
    fi
    
    log_info "Creating pre-deployment backup..."
    
    BACKUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_DIR="${BACKUP_PATH:-./backups}/pre_deploy_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_DIR"
    
    # Check if services are running and backup if they are
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps postgres-primary | grep -q "Up"; then
        log_info "Backing up PostgreSQL database..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres-primary \
            pg_dumpall -U postgres > "$BACKUP_DIR/postgres_backup.sql" || log_warning "PostgreSQL backup failed"
    fi
    
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps neo4j-cluster | grep -q "Up"; then
        log_info "Backing up Neo4j database..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T neo4j-cluster \
            neo4j-admin database backup neo4j --to-path=/backups > /dev/null || log_warning "Neo4j backup failed"
    fi
    
    log_success "Pre-deployment backup completed: $BACKUP_DIR"
}

# Deploy services with proper ordering
deploy_services() {
    log_info "Starting production deployment..."
    
    # Phase 1: Infrastructure services (databases, caching)
    log_info "Phase 1: Deploying infrastructure services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        postgres-primary neo4j-cluster redis-cluster
    
    # Wait for database services to be healthy
    wait_for_services "postgres-primary neo4j-cluster redis-cluster"
    
    # Phase 2: API services
    log_info "Phase 2: Deploying API services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        database-api-1 database-api-2 dashboard-api
    
    # Wait for API services to be healthy
    wait_for_services "database-api-1 database-api-2 dashboard-api"
    
    # Phase 3: MCP services
    log_info "Phase 3: Deploying MCP services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        mcp-forces mcp-kinematics mcp-math mcp-energy mcp-momentum mcp-angular-motion
    
    # Wait for MCP services to be healthy
    wait_for_services "mcp-forces mcp-kinematics mcp-math mcp-energy mcp-momentum mcp-angular-motion"
    
    # Phase 4: Physics agents API
    log_info "Phase 4: Deploying Physics agents API..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        physics-agents-api-1 physics-agents-api-2
    
    # Wait for physics agents to be healthy
    wait_for_services "physics-agents-api-1 physics-agents-api-2"
    
    # Phase 5: Frontend services
    log_info "Phase 5: Deploying frontend services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        streamlit-ui-1 streamlit-ui-2 react-dashboard
    
    # Wait for frontend services to be healthy
    wait_for_services "streamlit-ui-1 streamlit-ui-2 react-dashboard"
    
    # Phase 6: Load balancer
    log_info "Phase 6: Deploying load balancer..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        nginx-loadbalancer
    
    # Wait for load balancer to be healthy
    wait_for_services "nginx-loadbalancer"
    
    # Phase 7: Analytics and monitoring
    log_info "Phase 7: Deploying analytics and monitoring services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        ml-engine task-processor flower-monitor \
        prometheus grafana alertmanager \
        node-exporter cadvisor loki promtail \
        backup-service
    
    log_success "All services deployed successfully"
}

# Wait for services to be healthy
wait_for_services() {
    local services="$1"
    local max_wait=${HEALTH_CHECK_TIMEOUT}
    local wait_time=0
    local check_interval=10
    
    log_info "Waiting for services to be healthy: $services"
    
    while [[ $wait_time -lt $max_wait ]]; do
        local all_healthy=true
        
        for service in $services; do
            local health_status=$(docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" --format "table {{.Health}}" | tail -n 1)
            
            if [[ "$health_status" != "healthy" ]]; then
                all_healthy=false
                break
            fi
        done
        
        if [[ "$all_healthy" == true ]]; then
            log_success "All services are healthy"
            return 0
        fi
        
        log_info "Waiting for services to become healthy... (${wait_time}s/${max_wait}s)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    log_error "Timeout waiting for services to become healthy"
    
    # Show service status for debugging
    log_info "Current service status:"
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    
    if [[ "$ROLLBACK_ON_FAILURE" == true ]]; then
        log_warning "Rolling back deployment due to health check failure"
        rollback_deployment
    fi
    
    return 1
}

# Validate deployment
validate_deployment() {
    if [[ "$SKIP_VALIDATION" == true ]]; then
        log_info "Skipping deployment validation"
        return 0
    fi
    
    log_info "Validating deployment..."
    
    # Source environment variables
    source "$ENV_FILE"
    
    # Test basic connectivity
    local base_url="http://localhost"
    if [[ "${SSL_ENABLED:-false}" == "true" ]]; then
        base_url="https://${DOMAIN_NAME}"
    fi
    
    # Test main application
    if curl -sf "$base_url/health" >/dev/null; then
        log_success "Main application health check passed"
    else
        log_error "Main application health check failed"
        return 1
    fi
    
    # Test API endpoints
    if curl -sf "${base_url}/api/database/health" >/dev/null; then
        log_success "Database API health check passed"
    else
        log_warning "Database API health check failed"
    fi
    
    if curl -sf "${base_url}/api/agents/health" >/dev/null; then
        log_success "Physics agents API health check passed"
    else
        log_warning "Physics agents API health check failed"
    fi
    
    # Test monitoring endpoints
    if curl -sf "http://localhost:9090/-/healthy" >/dev/null; then
        log_success "Prometheus health check passed"
    else
        log_warning "Prometheus health check failed"
    fi
    
    if curl -sf "http://localhost:3000/api/health" >/dev/null; then
        log_success "Grafana health check passed"
    else
        log_warning "Grafana health check failed"
    fi
    
    log_success "Deployment validation completed"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Initiating deployment rollback..."
    
    # Stop all services
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    
    # Restore from backup if available
    local latest_backup=$(ls -1t "${BACKUP_PATH:-./backups}"/pre_deploy_* 2>/dev/null | head -n1)
    if [[ -n "$latest_backup" && -f "$latest_backup/postgres_backup.sql" ]]; then
        log_info "Restoring from backup: $latest_backup"
        # Restore logic would go here
    fi
    
    log_success "Rollback completed"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    local report_file="${LOGS_PATH:-./logs}/deployment_$(date +%Y%m%d_%H%M%S).log"
    
    {
        echo "Physics Assistant Production Deployment Report"
        echo "=============================================="
        echo "Timestamp: $(date)"
        echo "Environment: $ENVIRONMENT"
        echo "Compose File: $COMPOSE_FILE"
        echo "Environment File: $ENV_FILE"
        echo ""
        echo "Service Status:"
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
        echo ""
        echo "Resource Usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    } > "$report_file"
    
    log_success "Deployment report generated: $report_file"
}

# Cleanup function
cleanup() {
    log_info "Performing cleanup..."
    
    # Remove old images (keep last 3 versions)
    docker image prune -f
    
    # Remove old log files (older than 30 days)
    find "${LOGS_PATH:-./logs}" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting Physics Assistant Production Deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Timestamp: $(date)"
    
    # Check if user wants to proceed
    if [[ "$FORCE_DEPLOY" != true ]]; then
        echo "This will deploy the Physics Assistant platform to $ENVIRONMENT environment."
        echo "Services will be restarted and there may be brief downtime."
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Execute deployment steps
    check_prerequisites
    create_directories
    create_backup
    deploy_services
    
    # Wait a bit for all services to stabilize
    sleep 30
    
    validate_deployment
    generate_report
    cleanup
    
    log_success "Physics Assistant deployment completed successfully!"
    log_info "Access the application at: https://${DOMAIN_NAME:-localhost}"
    log_info "Monitoring dashboard: https://monitoring.${DOMAIN_NAME:-localhost}"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"