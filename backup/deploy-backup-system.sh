#!/bin/bash
set -e

# Physics Assistant Backup System Deployment Script
# This script deploys the comprehensive backup and disaster recovery system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_BASE_DIR="/opt/physics-assistant/backups"
LOGS_BASE_DIR="/opt/physics-assistant/logs"
SECRETS_DIR="/opt/physics-assistant/secrets"

# Deployment mode
DEPLOYMENT_MODE=${1:-"development"}  # development, production, test

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites for backup system deployment"
    
    # Check if Docker is installed and running
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker version >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker service."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if main application is running
    if ! docker ps --format "{{.Names}}" | grep -q "physics-"; then
        log_warn "Main Physics Assistant application doesn't appear to be running"
        log_warn "Some backup services may not function properly without the main application"
    fi
    
    # Check available disk space
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -lt 10 ]]; then
        log_error "Insufficient disk space. At least 10GB required, found ${available_gb}GB"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Create directory structure
create_directories() {
    log_info "Creating backup system directory structure"
    
    # Create backup directories
    sudo mkdir -p "$BACKUP_BASE_DIR"/{postgres,neo4j,redis,application}/{daily,weekly,monthly,yearly}
    sudo mkdir -p "$BACKUP_BASE_DIR"/{postgres,neo4j,redis,application}/incremental
    
    # Create log directories
    sudo mkdir -p "$LOGS_BASE_DIR"/{backup,restore,monitoring,validation,volume-health,scheduler,retention}
    
    # Create secrets directory
    sudo mkdir -p "$SECRETS_DIR"
    sudo chmod 700 "$SECRETS_DIR"
    
    # Set proper ownership
    sudo chown -R 1000:1000 "$BACKUP_BASE_DIR" "$LOGS_BASE_DIR"
    sudo chown -R 1000:1000 "$SECRETS_DIR"
    
    log_info "Directory structure created successfully"
}

# Generate configuration files
generate_configuration() {
    log_info "Generating backup system configuration"
    
    # Create environment file if it doesn't exist
    local env_file="$SCRIPT_DIR/.env"
    
    if [[ ! -f "$env_file" ]]; then
        log_info "Creating environment configuration file"
        
        cat > "$env_file" << 'EOF'
# Physics Assistant Backup System Configuration

# Database Passwords (CHANGE THESE!)
POSTGRES_PASSWORD=physics_secure_password_2024
NEO4J_PASSWORD=physics_graph_password_2024
PHYSICS_DB_PASSWORD=physics_secure_password_2024

# Backup Configuration
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
BACKUP_ENCRYPTION=true
BACKUP_VALIDATION_ENABLED=true

# Scheduling
POSTGRES_BACKUP_SCHEDULE=0 2 * * *
NEO4J_BACKUP_SCHEDULE=0 3 * * *
REDIS_BACKUP_SCHEDULE=0 4 * * *
APPLICATION_BACKUP_SCHEDULE=0 5 * * *

# S3 Configuration (Optional - Enable for cloud storage)
BACKUP_S3_ENABLED=false
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_S3_BUCKET=physics-assistant-backups
AWS_S3_REGION=us-west-2

# Monitoring and Alerting
BACKUP_CHECK_INTERVAL=300
BACKUP_MAX_AGE_HOURS=25
BACKUP_MIN_SIZE_MB=1

# Email Notifications (Optional)
BACKUP_EMAIL_ENABLED=false
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
BACKUP_EMAIL_TO=admin@yourorganization.com

# Webhook Notifications (Optional - Slack, Discord, etc.)
WEBHOOK_URL=

# Disaster Recovery
RECOVERY_RTO_MINUTES=60
RECOVERY_RPO_HOURS=4
PARALLEL_RECOVERY=true
VALIDATION_ENABLED=true
RESTORE_VALIDATION_ENABLED=true
RESTORE_TEST_MODE=false

# Volume Management
VOLUME_CHECK_INTERVAL=3600
VOLUME_USAGE_THRESHOLD=80
VOLUME_CRITICAL_THRESHOLD=95
CLEANUP_ENABLED=true

# Performance Tuning
COMPRESSION_LEVEL=6
PARALLEL_JOBS=2
BACKUP_TIMEOUT=7200
VALIDATION_TIMEOUT_SECONDS=3600
MAX_VALIDATION_FILE_SIZE_MB=1000
EOF
        
        log_warn "Configuration file created: $env_file"
        log_warn "Please review and update the configuration before continuing"
        log_warn "Pay special attention to database passwords and notification settings"
    else
        log_info "Using existing configuration file: $env_file"
    fi
    
    # Create backup schedule configuration
    local schedule_config="$SCRIPT_DIR/scripts/config/backup-schedule.conf"
    mkdir -p "$(dirname "$schedule_config")"
    
    if [[ ! -f "$schedule_config" ]]; then
        cp "$SCRIPT_DIR/scripts/scheduler.sh" /tmp/scheduler_template.sh
        cat > "$schedule_config" << 'EOF'
# Backup Schedule Configuration
# Format: SERVICE_TYPE_SCHEDULE="CRON_EXPRESSION"

# PostgreSQL Backups
POSTGRES_DAILY_SCHEDULE="0 2 * * *"
POSTGRES_WEEKLY_SCHEDULE="0 1 * * 0"
POSTGRES_MONTHLY_SCHEDULE="0 0 1 * *"
POSTGRES_YEARLY_SCHEDULE="0 0 1 1 *"

# Neo4j Backups
NEO4J_DAILY_SCHEDULE="0 3 * * *"
NEO4J_WEEKLY_SCHEDULE="0 2 * * 0"
NEO4J_MONTHLY_SCHEDULE="0 1 1 * *"
NEO4J_YEARLY_SCHEDULE="0 1 1 1 *"

# Redis Backups
REDIS_DAILY_SCHEDULE="0 4 * * *"
REDIS_WEEKLY_SCHEDULE="0 3 * * 0"
REDIS_MONTHLY_SCHEDULE="0 2 1 * *"
REDIS_YEARLY_SCHEDULE="0 2 1 1 *"

# Application Data Backups
APPLICATION_DAILY_SCHEDULE="0 5 * * *"
APPLICATION_WEEKLY_SCHEDULE="0 4 * * 0"
APPLICATION_MONTHLY_SCHEDULE="0 3 1 * *"
APPLICATION_YEARLY_SCHEDULE="0 3 1 1 *"

# Retention Policies (in days)
POSTGRES_DAILY_RETENTION=7
POSTGRES_WEEKLY_RETENTION=28
POSTGRES_MONTHLY_RETENTION=365
POSTGRES_YEARLY_RETENTION=2555

NEO4J_DAILY_RETENTION=7
NEO4J_WEEKLY_RETENTION=28
NEO4J_MONTHLY_RETENTION=365
NEO4J_YEARLY_RETENTION=2555

REDIS_DAILY_RETENTION=7
REDIS_WEEKLY_RETENTION=28
REDIS_MONTHLY_RETENTION=365
REDIS_YEARLY_RETENTION=2555

APPLICATION_DAILY_RETENTION=7
APPLICATION_WEEKLY_RETENTION=28
APPLICATION_MONTHLY_RETENTION=365
APPLICATION_YEARLY_RETENTION=2555

# Backup Features
COMPRESSION_ENABLED=true
ENCRYPTION_ENABLED=true
S3_UPLOAD_ENABLED=false
BACKUP_VALIDATION_ENABLED=true

# Cleanup Settings
AUTO_CLEANUP_ENABLED=true
CLEANUP_SCHEDULE="0 6 * * *"
EOF
        
        log_info "Backup schedule configuration created: $schedule_config"
    fi
}

# Make scripts executable
setup_permissions() {
    log_info "Setting up script permissions"
    
    # Make all shell scripts executable
    find "$SCRIPT_DIR" -name "*.sh" -type f -exec chmod +x {} \;
    
    # Make Python scripts executable
    find "$SCRIPT_DIR" -name "*.py" -type f -exec chmod +x {} \;
    
    log_info "Script permissions configured"
}

# Deploy backup services
deploy_services() {
    log_info "Deploying backup services for $DEPLOYMENT_MODE environment"
    
    cd "$SCRIPT_DIR"
    
    # Use appropriate compose file based on deployment mode
    local compose_file="docker-compose.backup.yml"
    
    case "$DEPLOYMENT_MODE" in
        "production")
            if [[ -f "docker-compose.backup.production.yml" ]]; then
                compose_file="docker-compose.backup.production.yml"
            fi
            ;;
        "development")
            if [[ -f "docker-compose.backup.development.yml" ]]; then
                compose_file="docker-compose.backup.development.yml"
            fi
            ;;
        "test")
            if [[ -f "docker-compose.backup.test.yml" ]]; then
                compose_file="docker-compose.backup.test.yml"
            fi
            ;;
    esac
    
    log_info "Using Docker Compose file: $compose_file"
    
    # Build and deploy services
    log_info "Building backup service containers..."
    docker-compose -f "$compose_file" build --no-cache
    
    log_info "Starting backup services..."
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to start
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service status
    local failed_services=0
    
    log_info "Checking service status:"
    docker-compose -f "$compose_file" ps --format "table {{.Name}}\t{{.State}}\t{{.Status}}"
    
    # Count failed services
    if docker-compose -f "$compose_file" ps | grep -q "Exit"; then
        failed_services=1
        log_error "Some services failed to start"
    fi
    
    if [[ $failed_services -eq 0 ]]; then
        log_info "All backup services started successfully"
    else
        log_error "Some backup services failed to start. Check logs for details:"
        docker-compose -f "$compose_file" logs --tail=50
        return 1
    fi
}

# Verify deployment
verify_deployment() {
    log_info "Verifying backup system deployment"
    
    local verification_passed=0
    
    # Check if backup directories are accessible
    if [[ -d "$BACKUP_BASE_DIR" ]] && [[ -w "$BACKUP_BASE_DIR" ]]; then
        log_info "✓ Backup directories accessible"
        ((verification_passed++))
    else
        log_error "✗ Backup directories not accessible"
    fi
    
    # Check if monitoring endpoint is responding
    if curl -s "http://localhost:8084/metrics" >/dev/null 2>&1; then
        log_info "✓ Monitoring endpoint responding"
        ((verification_passed++))
    else
        log_warn "✗ Monitoring endpoint not responding (may take a few minutes to start)"
    fi
    
    # Test backup scheduler
    if [[ -x "$SCRIPT_DIR/scripts/scheduler.sh" ]]; then
        log_info "✓ Backup scheduler accessible"
        ((verification_passed++))
    else
        log_error "✗ Backup scheduler not accessible"
    fi
    
    # Test volume manager
    if [[ -x "$SCRIPT_DIR/scripts/volume_manager.sh" ]]; then
        log_info "✓ Volume manager accessible"
        ((verification_passed++))
    else
        log_error "✗ Volume manager not accessible"
    fi
    
    # Test disaster recovery
    if [[ -x "$SCRIPT_DIR/scripts/restore/disaster_recovery.sh" ]]; then
        log_info "✓ Disaster recovery scripts accessible"
        ((verification_passed++))
    else
        log_error "✗ Disaster recovery scripts not accessible"
    fi
    
    log_info "Verification completed: $verification_passed/5 checks passed"
    
    if [[ $verification_passed -ge 4 ]]; then
        return 0
    else
        return 1
    fi
}

# Run initial backup test
run_initial_test() {
    log_info "Running initial backup system test"
    
    # Wait a bit more for services to fully initialize
    sleep 60
    
    # Test backup monitor health
    log_info "Testing backup monitor health..."
    if curl -s "http://localhost:8084/metrics" | grep -q "backup_monitor"; then
        log_info "✓ Backup monitor is operational"
    else
        log_warn "✗ Backup monitor metrics not available yet"
    fi
    
    # Test volume manager
    log_info "Testing volume manager..."
    if "$SCRIPT_DIR/scripts/volume_manager.sh" status >/dev/null 2>&1; then
        log_info "✓ Volume manager is operational"
    else
        log_warn "✗ Volume manager test failed"
    fi
    
    # Test disaster recovery (dry run)
    log_info "Testing disaster recovery (dry run)..."
    if "$SCRIPT_DIR/scripts/restore/disaster_recovery.sh" test latest >/dev/null 2>&1; then
        log_info "✓ Disaster recovery test passed"
    else
        log_warn "✗ Disaster recovery test failed (may be due to no existing backups)"
    fi
    
    log_info "Initial testing completed"
}

# Display next steps
show_next_steps() {
    log_info "Backup system deployment completed!"
    
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}                                    NEXT STEPS                                    ${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "${GREEN}1. Configuration Review:${NC}"
    echo "   Edit: $SCRIPT_DIR/.env"
    echo "   Update database passwords and notification settings"
    echo
    echo -e "${GREEN}2. Manual Backup Test:${NC}"
    echo "   cd $SCRIPT_DIR"
    echo "   ./scripts/scheduler.sh backup postgres daily"
    echo "   ./scripts/scheduler.sh backup neo4j daily"
    echo "   ./scripts/scheduler.sh backup redis daily"
    echo
    echo -e "${GREEN}3. Monitor Services:${NC}"
    echo "   Prometheus metrics: http://localhost:8084/metrics"
    echo "   Service logs: docker-compose -f backup/docker-compose.backup.yml logs"
    echo "   Backup reports: ls $LOGS_BASE_DIR/monitoring/"
    echo
    echo -e "${GREEN}4. Volume Health Check:${NC}"
    echo "   ./scripts/volume_manager.sh check all"
    echo "   ./scripts/volume_manager.sh report"
    echo
    echo -e "${GREEN}5. Disaster Recovery Test:${NC}"
    echo "   ./scripts/restore/disaster_recovery.sh test latest"
    echo "   ./scripts/restore/disaster_recovery.sh plan"
    echo
    echo -e "${GREEN}6. Validation Test:${NC}"
    echo "   python3 ./scripts/validation/backup_validator.py --all --report"
    echo
    echo -e "${GREEN}7. Integration with Monitoring:${NC}"
    echo "   Import Grafana dashboard from: /backup/monitoring/"
    echo "   Configure Prometheus alerts for backup failures"
    echo
    echo -e "${YELLOW}Important Files:${NC}"
    echo "   Configuration: $SCRIPT_DIR/.env"
    echo "   Schedules: $SCRIPT_DIR/scripts/config/backup-schedule.conf"
    echo "   Backups: $BACKUP_BASE_DIR/"
    echo "   Logs: $LOGS_BASE_DIR/"
    echo "   Documentation: $SCRIPT_DIR/backup-system-deployment.md"
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Main deployment function
main() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}                        Physics Assistant Backup System Deployment                        ${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "${GREEN}Deployment Mode:${NC} $DEPLOYMENT_MODE"
    echo -e "${GREEN}Project Root:${NC} $PROJECT_ROOT"
    echo -e "${GREEN}Backup Directory:${NC} $BACKUP_BASE_DIR"
    echo -e "${GREEN}Logs Directory:${NC} $LOGS_BASE_DIR"
    echo
    
    # Run deployment steps
    check_prerequisites
    create_directories
    generate_configuration
    setup_permissions
    
    # Ask for confirmation before deploying services
    echo
    read -p "Continue with service deployment? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_services
        
        if verify_deployment; then
            log_info "Deployment verification passed"
            
            # Ask if user wants to run initial tests
            echo
            read -p "Run initial system tests? (y/N): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                run_initial_test
            fi
            
            show_next_steps
        else
            log_error "Deployment verification failed"
            log_error "Please check the logs and configuration"
            exit 1
        fi
    else
        log_info "Service deployment cancelled"
        log_info "You can run the deployment later with: $0 $DEPLOYMENT_MODE"
        exit 0
    fi
}

# Handle script arguments
case "${1:-}" in
    "production"|"development"|"test")
        main
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [deployment_mode]"
        echo ""
        echo "Deployment modes:"
        echo "  development  - Development environment (default)"
        echo "  production   - Production environment"
        echo "  test         - Test environment"
        echo ""
        echo "Example:"
        echo "  $0 production"
        ;;
    *)
        DEPLOYMENT_MODE="development"
        main
        ;;
esac