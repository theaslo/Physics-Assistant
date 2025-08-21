#!/bin/bash

# Common utility functions for backup operations

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize logging
init_logging() {
    local service_name=$1
    LOG_PREFIX="[$service_name]"
    
    # Create log directory if it doesn't exist
    mkdir -p /logs/{backup,restore,monitoring}
    
    # Set up log rotation
    if [[ ! -f /etc/logrotate.d/backup ]]; then
        cat > /tmp/backup-logrotate << EOF
/logs/backup/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 backup backup
}

/logs/restore/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 backup backup
}

/logs/monitoring/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 backup backup
}
EOF
        sudo mv /tmp/backup-logrotate /etc/logrotate.d/backup
    fi
}

# Logging functions
log_info() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}${LOG_PREFIX} ${timestamp} [INFO]${NC} $message"
    echo "${timestamp} [INFO] $message" >> "${LOG_FILE:-/logs/backup/default.log}"
}

log_warn() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}${LOG_PREFIX} ${timestamp} [WARN]${NC} $message"
    echo "${timestamp} [WARN] $message" >> "${LOG_FILE:-/logs/backup/default.log}"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}${LOG_PREFIX} ${timestamp} [ERROR]${NC} $message" >&2
    echo "${timestamp} [ERROR] $message" >> "${LOG_FILE:-/logs/backup/default.log}"
}

log_debug() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}${LOG_PREFIX} ${timestamp} [DEBUG]${NC} $message"
        echo "${timestamp} [DEBUG] $message" >> "${LOG_FILE:-/logs/backup/default.log}"
    fi
}

# Validate environment variables
validate_env_vars() {
    local required_vars=("$@")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    
    return 0
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for service to be ready
wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local max_attempts=${4:-30}
    local timeout=${5:-10}
    
    log_info "Waiting for $service_name to be ready at $host:$port..."
    
    local attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            log_info "$service_name is ready"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log_error "$service_name failed to become ready after $max_attempts attempts"
    return 1
}

# Calculate file size in human readable format
human_readable_size() {
    local bytes=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit=0
    
    while [[ $bytes -ge 1024 && $unit -lt $((${#units[@]} - 1)) ]]; do
        bytes=$((bytes / 1024))
        ((unit++))
    done
    
    echo "${bytes}${units[$unit]}"
}

# Generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}

# Create secure temporary file
create_temp_file() {
    local prefix=${1:-backup}
    local suffix=${2:-.tmp}
    
    mktemp "/tmp/${prefix}_XXXXXX${suffix}"
}

# Cleanup function for temporary files
cleanup_temp_files() {
    local temp_dir=${1:-/tmp/backup}
    
    if [[ -d "$temp_dir" ]]; then
        log_info "Cleaning up temporary files in $temp_dir"
        rm -rf "$temp_dir"/*
    fi
}

# Check disk space
check_disk_space() {
    local path=$1
    local required_space_gb=$2
    
    local available_space=$(df "$path" | awk 'NR==2 {print $4}')
    local available_space_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_space_gb -lt $required_space_gb ]]; then
        log_error "Insufficient disk space. Required: ${required_space_gb}GB, Available: ${available_space_gb}GB"
        return 1
    fi
    
    log_info "Disk space check passed. Available: ${available_space_gb}GB"
    return 0
}

# Create directory with proper permissions
create_backup_directory() {
    local dir_path=$1
    local owner=${2:-backup:backup}
    local permissions=${3:-755}
    
    if [[ ! -d "$dir_path" ]]; then
        mkdir -p "$dir_path"
        chown "$owner" "$dir_path"
        chmod "$permissions" "$dir_path"
        log_info "Created backup directory: $dir_path"
    fi
}

# Lock file management for preventing concurrent backups
acquire_lock() {
    local lock_file=$1
    local timeout=${2:-300}  # 5 minutes default
    
    local waited=0
    while [[ $waited -lt $timeout ]]; do
        if (set -C; echo $$ > "$lock_file") 2>/dev/null; then
            log_info "Lock acquired: $lock_file"
            return 0
        fi
        
        # Check if process holding the lock is still running
        if [[ -f "$lock_file" ]]; then
            local lock_pid=$(cat "$lock_file" 2>/dev/null)
            if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
                log_warn "Removing stale lock file: $lock_file"
                rm -f "$lock_file"
                continue
            fi
        fi
        
        log_info "Waiting for lock: $lock_file (waited ${waited}s)"
        sleep 10
        waited=$((waited + 10))
    done
    
    log_error "Failed to acquire lock: $lock_file (timeout after ${timeout}s)"
    return 1
}

release_lock() {
    local lock_file=$1
    
    if [[ -f "$lock_file" ]]; then
        rm -f "$lock_file"
        log_info "Lock released: $lock_file"
    fi
}

# Retry function with exponential backoff
retry_with_backoff() {
    local max_attempts=$1
    local delay=$2
    shift 2
    local cmd=("$@")
    
    local attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if "${cmd[@]}"; then
            return 0
        fi
        
        if [[ $attempt -lt $max_attempts ]]; then
            log_warn "Command failed (attempt $attempt/$max_attempts), retrying in ${delay}s..."
            sleep "$delay"
            delay=$((delay * 2))  # Exponential backoff
        fi
        
        ((attempt++))
    done
    
    log_error "Command failed after $max_attempts attempts"
    return 1
}

# Network connectivity check
check_network_connectivity() {
    local host=$1
    local port=${2:-80}
    local timeout=${3:-10}
    
    if timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        log_info "Network connectivity to $host:$port is OK"
        return 0
    else
        log_error "No network connectivity to $host:$port"
        return 1
    fi
}

# File integrity check using checksums
calculate_checksum() {
    local file_path=$1
    local algorithm=${2:-md5}
    
    case "$algorithm" in
        "md5")
            md5sum "$file_path" | cut -d' ' -f1
            ;;
        "sha256")
            sha256sum "$file_path" | cut -d' ' -f1
            ;;
        "sha1")
            sha1sum "$file_path" | cut -d' ' -f1
            ;;
        *)
            log_error "Unsupported checksum algorithm: $algorithm"
            return 1
            ;;
    esac
}

verify_checksum() {
    local file_path=$1
    local expected_checksum=$2
    local algorithm=${3:-md5}
    
    local actual_checksum=$(calculate_checksum "$file_path" "$algorithm")
    
    if [[ "$actual_checksum" == "$expected_checksum" ]]; then
        log_info "Checksum verification passed for $file_path"
        return 0
    else
        log_error "Checksum verification failed for $file_path"
        log_error "Expected: $expected_checksum"
        log_error "Actual: $actual_checksum"
        return 1
    fi
}

# Docker container management helpers
is_container_running() {
    local container_name=$1
    docker ps --format "table {{.Names}}" | grep -q "^${container_name}$"
}

wait_for_container() {
    local container_name=$1
    local max_attempts=${2:-30}
    
    local attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if is_container_running "$container_name"; then
            log_info "Container $container_name is running"
            return 0
        fi
        
        log_info "Waiting for container $container_name to start (attempt $attempt/$max_attempts)..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Container $container_name failed to start after $max_attempts attempts"
    return 1
}

# Resource monitoring
check_system_resources() {
    local memory_threshold_percent=${1:-80}
    local disk_threshold_percent=${2:-90}
    local cpu_threshold_percent=${3:-90}
    
    # Check memory usage
    local memory_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    if [[ $memory_usage -gt $memory_threshold_percent ]]; then
        log_warn "High memory usage: ${memory_usage}% (threshold: ${memory_threshold_percent}%)"
    fi
    
    # Check disk usage
    local disk_usage=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt $disk_threshold_percent ]]; then
        log_warn "High disk usage: ${disk_usage}% (threshold: ${disk_threshold_percent}%)"
    fi
    
    # Check CPU load (1-minute average)
    local cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_cores=$(nproc)
    local cpu_usage=$(echo "$cpu_load $cpu_cores" | awk '{printf "%.0f", ($1/$2)*100}')
    
    if [[ $cpu_usage -gt $cpu_threshold_percent ]]; then
        log_warn "High CPU usage: ${cpu_usage}% (threshold: ${cpu_threshold_percent}%)"
    fi
    
    log_info "System resources - Memory: ${memory_usage}%, Disk: ${disk_usage}%, CPU: ${cpu_usage}%"
}

# Backup rotation management
rotate_backups() {
    local backup_dir=$1
    local keep_daily=${2:-7}
    local keep_weekly=${3:-4}
    local keep_monthly=${4:-12}
    
    log_info "Starting backup rotation in $backup_dir"
    
    # Rotate daily backups
    if [[ -d "${backup_dir}/daily" ]]; then
        local daily_count=$(find "${backup_dir}/daily" -type f -name "*.gz" -o -name "*.dump" -o -name "*.sql" | wc -l)
        if [[ $daily_count -gt $keep_daily ]]; then
            find "${backup_dir}/daily" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.sql" \) -mtime +$keep_daily -delete
            log_info "Rotated daily backups, keeping last $keep_daily days"
        fi
    fi
    
    # Rotate weekly backups
    if [[ -d "${backup_dir}/weekly" ]]; then
        local weekly_count=$(find "${backup_dir}/weekly" -type f -name "*.gz" -o -name "*.dump" -o -name "*.sql" | wc -l)
        if [[ $weekly_count -gt $keep_weekly ]]; then
            find "${backup_dir}/weekly" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.sql" \) -mtime +$((keep_weekly * 7)) -delete
            log_info "Rotated weekly backups, keeping last $keep_weekly weeks"
        fi
    fi
    
    # Rotate monthly backups
    if [[ -d "${backup_dir}/monthly" ]]; then
        local monthly_count=$(find "${backup_dir}/monthly" -type f -name "*.gz" -o -name "*.dump" -o -name "*.sql" | wc -l)
        if [[ $monthly_count -gt $keep_monthly ]]; then
            find "${backup_dir}/monthly" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.sql" \) -mtime +$((keep_monthly * 30)) -delete
            log_info "Rotated monthly backups, keeping last $keep_monthly months"
        fi
    fi
    
    log_info "Backup rotation completed"
}