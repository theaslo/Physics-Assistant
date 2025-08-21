#!/bin/bash

# Health check script for backup services

SERVICE_TYPE=${1:-"generic"}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-30}

# Source common utilities
if [[ -f /scripts/common/utils.sh ]]; then
    source /scripts/common/utils.sh
else
    # Fallback logging functions if utils.sh not available
    log_info() { echo "[INFO] $1"; }
    log_warn() { echo "[WARN] $1"; }
    log_error() { echo "[ERROR] $1" >&2; }
fi

# Initialize logging for health checks
init_logging "health-check-$SERVICE_TYPE"

# Generic health check functions
check_disk_space() {
    local path=${1:-"/"}
    local threshold=${2:-90}
    
    local usage=$(df "$path" | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [[ $usage -gt $threshold ]]; then
        log_error "Disk usage too high: ${usage}% (threshold: ${threshold}%)"
        return 1
    fi
    
    log_info "Disk usage OK: ${usage}%"
    return 0
}

check_memory_usage() {
    local threshold=${1:-90}
    
    local usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    
    if [[ $usage -gt $threshold ]]; then
        log_error "Memory usage too high: ${usage}% (threshold: ${threshold}%)"
        return 1
    fi
    
    log_info "Memory usage OK: ${usage}%"
    return 0
}

check_process_running() {
    local process_pattern=$1
    
    if pgrep -f "$process_pattern" >/dev/null; then
        log_info "Process running: $process_pattern"
        return 0
    else
        log_error "Process not running: $process_pattern"
        return 1
    fi
}

check_network_connectivity() {
    local host=$1
    local port=${2:-80}
    local timeout=${3:-10}
    
    if timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        log_info "Network connectivity OK: $host:$port"
        return 0
    else
        log_error "Network connectivity failed: $host:$port"
        return 1
    fi
}

check_file_exists() {
    local file_path=$1
    local max_age_seconds=${2:-86400}  # 24 hours default
    
    if [[ ! -f "$file_path" ]]; then
        log_error "File not found: $file_path"
        return 1
    fi
    
    local file_age=$(($(date +%s) - $(stat -f%m "$file_path" 2>/dev/null || stat -c%Y "$file_path")))
    
    if [[ $file_age -gt $max_age_seconds ]]; then
        log_error "File too old: $file_path (age: ${file_age}s, max: ${max_age_seconds}s)"
        return 1
    fi
    
    log_info "File OK: $file_path (age: ${file_age}s)"
    return 0
}

# Service-specific health checks
postgres_health_check() {
    log_info "Performing PostgreSQL backup service health check"
    
    local checks_passed=0
    local total_checks=5
    
    # Check if backup script exists
    if [[ -f /scripts/backup.sh ]]; then
        ((checks_passed++))
        log_info "Backup script found"
    else
        log_error "Backup script not found: /scripts/backup.sh"
    fi
    
    # Check PostgreSQL connectivity
    if [[ -n "${POSTGRES_HOST:-}" ]] && [[ -n "${POSTGRES_PORT:-}" ]]; then
        if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "${POSTGRES_USER:-postgres}" >/dev/null 2>&1; then
            ((checks_passed++))
            log_info "PostgreSQL connectivity OK"
        else
            log_error "PostgreSQL connectivity failed"
        fi
    else
        log_warn "PostgreSQL connection parameters not set"
    fi
    
    # Check backup directory
    if [[ -d /backups ]]; then
        ((checks_passed++))
        log_info "Backup directory exists"
    else
        log_error "Backup directory not found: /backups"
    fi
    
    # Check recent backup
    if find /backups -name "postgres_*" -type f -mtime -2 | head -1 | grep -q .; then
        ((checks_passed++))
        log_info "Recent backup found"
    else
        log_warn "No recent backups found"
    fi
    
    # Check system resources
    if check_disk_space "/backups" 80 && check_memory_usage 80; then
        ((checks_passed++))
        log_info "System resources OK"
    else
        log_error "System resources insufficient"
    fi
    
    if [[ $checks_passed -ge 3 ]]; then
        log_info "PostgreSQL backup service health check passed ($checks_passed/$total_checks)"
        return 0
    else
        log_error "PostgreSQL backup service health check failed ($checks_passed/$total_checks)"
        return 1
    fi
}

neo4j_health_check() {
    log_info "Performing Neo4j backup service health check"
    
    local checks_passed=0
    local total_checks=5
    
    # Check if backup script exists
    if [[ -f /scripts/backup.sh ]]; then
        ((checks_passed++))
        log_info "Backup script found"
    else
        log_error "Backup script not found: /scripts/backup.sh"
    fi
    
    # Check Neo4j connectivity
    if [[ -n "${NEO4J_HOST:-}" ]] && [[ -n "${NEO4J_PORT:-}" ]]; then
        if curl -s -u "${NEO4J_USER:-neo4j}:${NEO4J_PASSWORD}" \
           "http://${NEO4J_HOST}:7474/db/data/" >/dev/null 2>&1; then
            ((checks_passed++))
            log_info "Neo4j connectivity OK"
        else
            log_error "Neo4j connectivity failed"
        fi
    else
        log_warn "Neo4j connection parameters not set"
    fi
    
    # Check backup directory
    if [[ -d /backups ]]; then
        ((checks_passed++))
        log_info "Backup directory exists"
    else
        log_error "Backup directory not found: /backups"
    fi
    
    # Check recent backup
    if find /backups -name "neo4j_*" -type f -mtime -2 | head -1 | grep -q .; then
        ((checks_passed++))
        log_info "Recent backup found"
    else
        log_warn "No recent backups found"
    fi
    
    # Check system resources
    if check_disk_space "/backups" 80 && check_memory_usage 80; then
        ((checks_passed++))
        log_info "System resources OK"
    else
        log_error "System resources insufficient"
    fi
    
    if [[ $checks_passed -ge 3 ]]; then
        log_info "Neo4j backup service health check passed ($checks_passed/$total_checks)"
        return 0
    else
        log_error "Neo4j backup service health check failed ($checks_passed/$total_checks)"
        return 1
    fi
}

redis_health_check() {
    log_info "Performing Redis backup service health check"
    
    local checks_passed=0
    local total_checks=5
    
    # Check if backup script exists
    if [[ -f /scripts/backup.sh ]]; then
        ((checks_passed++))
        log_info "Backup script found"
    else
        log_error "Backup script not found: /scripts/backup.sh"
    fi
    
    # Check Redis connectivity
    if [[ -n "${REDIS_HOST:-}" ]] && [[ -n "${REDIS_PORT:-}" ]]; then
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ${REDIS_PASSWORD:+-a "$REDIS_PASSWORD"} ping >/dev/null 2>&1; then
            ((checks_passed++))
            log_info "Redis connectivity OK"
        else
            log_error "Redis connectivity failed"
        fi
    else
        log_warn "Redis connection parameters not set"
    fi
    
    # Check backup directory
    if [[ -d /backups ]]; then
        ((checks_passed++))
        log_info "Backup directory exists"
    else
        log_error "Backup directory not found: /backups"
    fi
    
    # Check recent backup
    if find /backups -name "redis_*" -type f -mtime -2 | head -1 | grep -q .; then
        ((checks_passed++))
        log_info "Recent backup found"
    else
        log_warn "No recent backups found"
    fi
    
    # Check system resources
    if check_disk_space "/backups" 80 && check_memory_usage 80; then
        ((checks_passed++))
        log_info "System resources OK"
    else
        log_error "System resources insufficient"
    fi
    
    if [[ $checks_passed -ge 3 ]]; then
        log_info "Redis backup service health check passed ($checks_passed/$total_checks)"
        return 0
    else
        log_error "Redis backup service health check failed ($checks_passed/$total_checks)"
        return 1
    fi
}

restore_health_check() {
    log_info "Performing restore service health check"
    
    local checks_passed=0
    local total_checks=4
    
    # Check if restore scripts exist
    if [[ -f /scripts/disaster_recovery.sh ]]; then
        ((checks_passed++))
        log_info "Disaster recovery script found"
    else
        log_error "Disaster recovery script not found"
    fi
    
    # Check backup directory accessibility
    if [[ -d /backups ]] && [[ -r /backups ]]; then
        ((checks_passed++))
        log_info "Backup directory accessible"
    else
        log_error "Backup directory not accessible: /backups"
    fi
    
    # Check Docker connectivity (for restoration operations)
    if docker version >/dev/null 2>&1; then
        ((checks_passed++))
        log_info "Docker connectivity OK"
    else
        log_error "Docker connectivity failed"
    fi
    
    # Check system resources
    if check_disk_space "/" 70 && check_memory_usage 70; then
        ((checks_passed++))
        log_info "System resources OK"
    else
        log_error "System resources insufficient for restoration"
    fi
    
    if [[ $checks_passed -ge 3 ]]; then
        log_info "Restore service health check passed ($checks_passed/$total_checks)"
        return 0
    else
        log_error "Restore service health check failed ($checks_passed/$total_checks)"
        return 1
    fi
}

monitor_health_check() {
    log_info "Performing backup monitor service health check"
    
    local checks_passed=0
    local total_checks=5
    
    # Check if monitor script is running
    if check_process_running "backup_monitor.py"; then
        ((checks_passed++))
        log_info "Monitor process running"
    else
        log_error "Monitor process not running"
    fi
    
    # Check metrics endpoint
    if curl -s "http://localhost:8084/metrics" >/dev/null 2>&1; then
        ((checks_passed++))
        log_info "Metrics endpoint accessible"
    else
        log_error "Metrics endpoint not accessible"
    fi
    
    # Check backup directory
    if [[ -d /backups ]] && [[ -r /backups ]]; then
        ((checks_passed++))
        log_info "Backup directory accessible"
    else
        log_error "Backup directory not accessible"
    fi
    
    # Check log directory
    if [[ -d /logs ]] && [[ -w /logs ]]; then
        ((checks_passed++))
        log_info "Log directory writable"
    else
        log_error "Log directory not writable"
    fi
    
    # Check system resources
    if check_disk_space "/" 80 && check_memory_usage 80; then
        ((checks_passed++))
        log_info "System resources OK"
    else
        log_error "System resources insufficient"
    fi
    
    if [[ $checks_passed -ge 4 ]]; then
        log_info "Monitor service health check passed ($checks_passed/$total_checks)"
        return 0
    else
        log_error "Monitor service health check failed ($checks_passed/$total_checks)"
        return 1
    fi
}

# Main health check dispatcher
main() {
    log_info "Starting health check for service type: $SERVICE_TYPE"
    
    case "$SERVICE_TYPE" in
        "postgres")
            postgres_health_check
            ;;
        "neo4j")
            neo4j_health_check
            ;;
        "redis")
            redis_health_check
            ;;
        "restore")
            restore_health_check
            ;;
        "monitor")
            monitor_health_check
            ;;
        "generic"|*)
            log_info "Performing generic health check"
            
            # Basic system checks
            local checks_passed=0
            
            if check_disk_space "/" 90; then
                ((checks_passed++))
            fi
            
            if check_memory_usage 90; then
                ((checks_passed++))
            fi
            
            if [[ $checks_passed -eq 2 ]]; then
                log_info "Generic health check passed"
                return 0
            else
                log_error "Generic health check failed"
                return 1
            fi
            ;;
    esac
    
    local result=$?
    
    if [[ $result -eq 0 ]]; then
        log_info "Health check completed successfully"
    else
        log_error "Health check failed"
    fi
    
    return $result
}

# Execute health check with timeout
timeout "$HEALTH_CHECK_TIMEOUT" bash -c "$(declare -f main check_disk_space check_memory_usage check_process_running check_network_connectivity check_file_exists postgres_health_check neo4j_health_check redis_health_check restore_health_check monitor_health_check log_info log_warn log_error); main"

exit $?