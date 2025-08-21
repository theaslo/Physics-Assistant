#!/bin/bash
set -e

# Source common functions
source /scripts/common/utils.sh
source /scripts/common/monitoring.sh
source /scripts/common/encryption.sh

# Initialize logging
init_logging "postgres-backup"

log_info "Starting PostgreSQL backup service initialization"

# Validate environment variables
validate_env_vars() {
    local required_vars=(
        "POSTGRES_HOST"
        "POSTGRES_PORT"
        "POSTGRES_DB"
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
}

# Wait for PostgreSQL to be ready
wait_for_postgres() {
    log_info "Waiting for PostgreSQL to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" >/dev/null 2>&1; then
            log_info "PostgreSQL is ready"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: PostgreSQL not ready, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log_error "PostgreSQL failed to become ready after $max_attempts attempts"
    exit 1
}

# Setup backup directories
setup_directories() {
    log_info "Setting up backup directories"
    
    mkdir -p /backups/{daily,weekly,monthly,incremental}
    mkdir -p /logs/{backup,restore,monitoring}
    mkdir -p /tmp/backup
    
    # Create .pgpass file for authentication
    echo "${POSTGRES_HOST}:${POSTGRES_PORT}:${POSTGRES_DB}:${POSTGRES_USER}:${POSTGRES_PASSWORD}" > ~/.pgpass
    chmod 600 ~/.pgpass
    
    log_info "Backup directories created successfully"
}

# Setup backup schedule
setup_cron() {
    log_info "Setting up backup schedule"
    
    local schedule="${BACKUP_SCHEDULE:-0 2 * * *}"
    local cron_file="/tmp/backup-cron"
    
    # Create cron entry
    cat > "$cron_file" << EOF
# PostgreSQL Backup Schedule
${schedule} /scripts/backup.sh >> /logs/backup/cron.log 2>&1

# Cleanup old backups daily at 6 AM
0 6 * * * /scripts/cleanup.sh >> /logs/backup/cleanup.log 2>&1

# Health check every 15 minutes
*/15 * * * * /scripts/common/health_check.sh postgres >> /logs/monitoring/health.log 2>&1

# Backup validation daily at 7 AM
0 7 * * * /scripts/validate.sh >> /logs/monitoring/validation.log 2>&1
EOF

    # Install cron job
    sudo crontab -u backup "$cron_file"
    rm "$cron_file"
    
    log_info "Backup schedule configured: $schedule"
}

# Start monitoring service
start_monitoring() {
    log_info "Starting backup monitoring service"
    
    # Start Prometheus metrics exporter in background
    python3 /scripts/common/metrics_exporter.py &
    local metrics_pid=$!
    echo $metrics_pid > /tmp/metrics.pid
    
    log_info "Monitoring service started with PID: $metrics_pid"
}

# Main execution
main() {
    validate_env_vars
    wait_for_postgres
    setup_directories
    setup_cron
    start_monitoring
    
    log_info "PostgreSQL backup service initialization completed"
    
    # Start cron daemon
    log_info "Starting cron daemon"
    sudo crond -f -l 2
}

# Trap signals for graceful shutdown
trap 'cleanup_and_exit' TERM INT

cleanup_and_exit() {
    log_info "Received shutdown signal, cleaning up..."
    
    # Stop metrics exporter
    if [[ -f /tmp/metrics.pid ]]; then
        local metrics_pid=$(cat /tmp/metrics.pid)
        if kill -0 "$metrics_pid" 2>/dev/null; then
            kill "$metrics_pid"
            log_info "Stopped metrics exporter"
        fi
    fi
    
    # Stop cron
    sudo pkill crond || true
    
    log_info "Cleanup completed, exiting"
    exit 0
}

# Execute main function
main "$@"