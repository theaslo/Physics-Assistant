#!/bin/bash
set -e

# Backup Scheduler and Retention Policy Manager
# This script manages backup scheduling, execution, and retention policies

# Source common functions
source /scripts/common/utils.sh
source /scripts/common/monitoring.sh
source /scripts/common/encryption.sh
source /scripts/common/s3_upload.sh

# Initialize logging
init_logging "backup-scheduler"

# Configuration
BACKUP_BASE_DIR="/backups"
LOCK_DIR="/var/lock/backup"
CONFIG_FILE="/scripts/config/backup-schedule.conf"

# Default retention policies (in days)
DEFAULT_DAILY_RETENTION=7
DEFAULT_WEEKLY_RETENTION=28
DEFAULT_MONTHLY_RETENTION=365
DEFAULT_YEARLY_RETENTION=2555  # 7 years

# Initialize scheduler
init_scheduler() {
    log_info "Initializing backup scheduler"
    
    # Create necessary directories
    mkdir -p "$LOCK_DIR" "$BACKUP_BASE_DIR"/{postgres,neo4j,redis,application}/{daily,weekly,monthly,yearly}
    mkdir -p /scripts/config /logs/{scheduler,retention}
    
    # Create default configuration if it doesn't exist
    if [[ ! -f "$CONFIG_FILE" ]]; then
        create_default_config
    fi
    
    # Load configuration
    source "$CONFIG_FILE"
    
    log_info "Backup scheduler initialized"
}

# Create default backup configuration
create_default_config() {
    log_info "Creating default backup configuration"
    
    cat > "$CONFIG_FILE" << 'EOF'
# Backup Schedule Configuration
# Format: CRON_EXPRESSION|SERVICE|BACKUP_TYPE|RETENTION_DAYS

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
    
    log_info "Default backup configuration created: $CONFIG_FILE"
}

# Execute backup for a specific service and type
execute_backup() {
    local service=$1
    local backup_type=$2
    local lock_file="$LOCK_DIR/${service}_${backup_type}.lock"
    
    log_info "Starting $backup_type backup for $service"
    
    # Acquire lock to prevent concurrent backups
    if ! acquire_lock "$lock_file" 3600; then
        log_error "Failed to acquire lock for $service $backup_type backup"
        return 1
    fi
    
    # Set backup environment variables
    export BACKUP_TYPE="$backup_type"
    export BACKUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    
    local backup_script="/scripts/${service}/backup.sh"
    local success=false
    
    if [[ -f "$backup_script" ]]; then
        log_info "Executing backup script: $backup_script"
        
        # Execute backup with timeout
        if timeout 7200 "$backup_script"; then  # 2 hour timeout
            success=true
            log_info "$service $backup_type backup completed successfully"
            
            # Update metrics
            update_backup_metrics "$service" "$backup_type" "success"
            
            # Send notification if configured
            send_backup_notification "$service" "$backup_type" "SUCCESS"
        else
            log_error "$service $backup_type backup failed or timed out"
            update_backup_metrics "$service" "$backup_type" "failure"
            send_backup_notification "$service" "$backup_type" "FAILURE"
        fi
    else
        log_error "Backup script not found: $backup_script"
    fi
    
    # Release lock
    release_lock "$lock_file"
    
    if [[ "$success" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

# Update backup metrics
update_backup_metrics() {
    local service=$1
    local backup_type=$2
    local status=$3
    
    local metrics_file="/tmp/backup_scheduler_metrics.prom"
    local timestamp=$(date +%s)
    
    cat >> "$metrics_file" << EOF
# HELP backup_job_status Backup job status (1=success, 0=failure)
# TYPE backup_job_status gauge
backup_job_status{service="$service",type="$backup_type"} $([ "$status" = "success" ] && echo "1" || echo "0")

# HELP backup_job_timestamp Backup job completion timestamp
# TYPE backup_job_timestamp gauge
backup_job_timestamp{service="$service",type="$backup_type"} $timestamp

EOF
}

# Send backup notification
send_backup_notification() {
    local service=$1
    local backup_type=$2
    local status=$3
    
    if [[ "${WEBHOOK_URL:-}" ]]; then
        local message="Backup $status: $service ($backup_type) at $(date)"
        local color=$([ "$status" = "SUCCESS" ] && echo "good" || echo "danger")
        
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"text\": \"$message\",
                \"color\": \"$color\",
                \"fields\": [
                    {\"title\": \"Service\", \"value\": \"$service\", \"short\": true},
                    {\"title\": \"Type\", \"value\": \"$backup_type\", \"short\": true},
                    {\"title\": \"Status\", \"value\": \"$status\", \"short\": true},
                    {\"title\": \"Timestamp\", \"value\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"short\": true}
                ]
            }" >/dev/null 2>&1 || log_warn "Failed to send webhook notification"
    fi
    
    if [[ "${BACKUP_EMAIL_ENABLED:-false}" == "true" ]] && [[ -n "${BACKUP_EMAIL_TO:-}" ]]; then
        send_email_notification "$service" "$backup_type" "$status"
    fi
}

# Send email notification
send_email_notification() {
    local service=$1
    local backup_type=$2
    local status=$3
    
    if ! command_exists "sendmail" && ! command_exists "mail"; then
        log_warn "No email client available for notifications"
        return 1
    fi
    
    local subject="Physics Assistant Backup $status: $service ($backup_type)"
    local body="
Backup Report
=============

Service: $service
Type: $backup_type
Status: $status
Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Host: $(hostname)

This is an automated notification from the Physics Assistant backup system.
"
    
    if command_exists "mail"; then
        echo "$body" | mail -s "$subject" "$BACKUP_EMAIL_TO"
    elif command_exists "sendmail"; then
        {
            echo "To: $BACKUP_EMAIL_TO"
            echo "Subject: $subject"
            echo "Content-Type: text/plain"
            echo ""
            echo "$body"
        } | sendmail "$BACKUP_EMAIL_TO"
    fi
}

# Apply retention policy
apply_retention_policy() {
    local service=$1
    local backup_type=$2
    
    log_info "Applying retention policy for $service $backup_type backups"
    
    # Get retention days for this service and type
    local retention_var="${service^^}_${backup_type^^}_RETENTION"
    local retention_days=${!retention_var:-30}
    
    local backup_dir="$BACKUP_BASE_DIR/$service/$backup_type"
    
    if [[ ! -d "$backup_dir" ]]; then
        log_warn "Backup directory not found: $backup_dir"
        return 0
    fi
    
    # Find and delete old backups
    local deleted_count=0
    
    while IFS= read -r -d '' file; do
        local file_age_days=$(( ($(date +%s) - $(stat -f%m "$file" 2>/dev/null || stat -c%Y "$file")) / 86400 ))
        
        if [[ $file_age_days -gt $retention_days ]]; then
            log_info "Deleting old backup: $file (age: ${file_age_days} days)"
            
            # Delete backup file and associated metadata
            rm -f "$file" "${file}.metadata.json" "${file}.s3.json"
            ((deleted_count++))
        fi
    done < <(find "$backup_dir" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" -o -name "*.tar.gz" \) -print0)
    
    log_info "Retention policy applied: deleted $deleted_count old backups"
    
    # Clean up S3 if enabled
    if [[ "${S3_UPLOAD_ENABLED:-false}" == "true" ]]; then
        cleanup_s3_backups "$service/$backup_type/" "$retention_days"
    fi
}

# Cleanup orphaned files
cleanup_orphaned_files() {
    log_info "Cleaning up orphaned backup files"
    
    local orphaned_count=0
    
    # Find metadata files without corresponding backup files
    while IFS= read -r -d '' metadata_file; do
        local backup_file="${metadata_file%.metadata.json}"
        
        if [[ ! -f "$backup_file" ]]; then
            log_info "Removing orphaned metadata: $metadata_file"
            rm -f "$metadata_file"
            ((orphaned_count++))
        fi
    done < <(find "$BACKUP_BASE_DIR" -name "*.metadata.json" -print0)
    
    # Find S3 metadata files without corresponding backup files
    while IFS= read -r -d '' s3_file; do
        local backup_file="${s3_file%.s3.json}"
        
        if [[ ! -f "$backup_file" ]]; then
            log_info "Removing orphaned S3 metadata: $s3_file"
            rm -f "$s3_file"
            ((orphaned_count++))
        fi
    done < <(find "$BACKUP_BASE_DIR" -name "*.s3.json" -print0)
    
    # Find and remove empty directories
    find "$BACKUP_BASE_DIR" -type d -empty -delete 2>/dev/null || true
    
    log_info "Cleanup completed: removed $orphaned_count orphaned files"
}

# Generate backup report
generate_backup_report() {
    local report_file="/logs/scheduler/backup_report_$(date +%Y%m%d).txt"
    
    log_info "Generating backup report: $report_file"
    
    cat > "$report_file" << EOF
Physics Assistant Backup Report
Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Host: $(hostname)

BACKUP INVENTORY
================
EOF
    
    # Inventory each service
    for service in postgres neo4j redis application; do
        echo "" >> "$report_file"
        echo "$service Backups:" >> "$report_file"
        echo "$(printf '=%.0s' {1..20})" >> "$report_file"
        
        for backup_type in daily weekly monthly yearly; do
            local backup_dir="$BACKUP_BASE_DIR/$service/$backup_type"
            
            if [[ -d "$backup_dir" ]]; then
                local count=$(find "$backup_dir" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) | wc -l)
                local total_size=$(du -sh "$backup_dir" 2>/dev/null | cut -f1 || echo "0")
                local latest=$(find "$backup_dir" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -exec stat -f%m {} \; 2>/dev/null | sort -n | tail -1)
                
                if [[ -n "$latest" ]]; then
                    local latest_date=$(date -d@"$latest" 2>/dev/null || date -r "$latest" 2>/dev/null || echo "Unknown")
                else
                    local latest_date="No backups"
                fi
                
                echo "  $backup_type: $count files, $total_size, latest: $latest_date" >> "$report_file"
            fi
        done
    done
    
    # Add system information
    cat >> "$report_file" << EOF

SYSTEM STATUS
=============
Disk Usage: $(df -h "$BACKUP_BASE_DIR" | awk 'NR==2 {print $5}') of $(df -h "$BACKUP_BASE_DIR" | awk 'NR==2 {print $2}') used
Memory Usage: $(free -h | awk 'NR==2{printf "%.1f%%\n", $3/$2*100}')
Load Average: $(uptime | awk -F'load average:' '{print $2}')

CONFIGURATION
=============
Encryption: $([ "${ENCRYPTION_ENABLED:-false}" = "true" ] && echo "Enabled" || echo "Disabled")
S3 Upload: $([ "${S3_UPLOAD_ENABLED:-false}" = "true" ] && echo "Enabled" || echo "Disabled")
Validation: $([ "${BACKUP_VALIDATION_ENABLED:-false}" = "true" ] && echo "Enabled" || echo "Disabled")
EOF
    
    log_info "Backup report generated successfully"
    
    # Send report via email if configured
    if [[ "${BACKUP_EMAIL_ENABLED:-false}" == "true" ]] && [[ -n "${BACKUP_EMAIL_TO:-}" ]]; then
        mail -s "Physics Assistant Backup Report - $(date +%Y-%m-%d)" "$BACKUP_EMAIL_TO" < "$report_file" 2>/dev/null || \
            log_warn "Failed to send backup report via email"
    fi
}

# Main scheduler function
main() {
    local action=${1:-"help"}
    
    case "$action" in
        "backup")
            local service=$2
            local backup_type=$3
            
            if [[ -z "$service" ]] || [[ -z "$backup_type" ]]; then
                log_error "Usage: $0 backup <service> <type>"
                exit 1
            fi
            
            init_scheduler
            execute_backup "$service" "$backup_type"
            ;;
        
        "cleanup")
            local service=${2:-"all"}
            local backup_type=${3:-"all"}
            
            init_scheduler
            
            if [[ "$service" == "all" ]]; then
                for svc in postgres neo4j redis application; do
                    if [[ "$backup_type" == "all" ]]; then
                        for type in daily weekly monthly yearly; do
                            apply_retention_policy "$svc" "$type"
                        done
                    else
                        apply_retention_policy "$svc" "$backup_type"
                    fi
                done
            else
                if [[ "$backup_type" == "all" ]]; then
                    for type in daily weekly monthly yearly; do
                        apply_retention_policy "$service" "$type"
                    done
                else
                    apply_retention_policy "$service" "$backup_type"
                fi
            fi
            
            cleanup_orphaned_files
            ;;
        
        "report")
            init_scheduler
            generate_backup_report
            ;;
        
        "status")
            init_scheduler
            
            echo "Backup System Status"
            echo "==================="
            echo "Configuration: $CONFIG_FILE"
            echo "Base Directory: $BACKUP_BASE_DIR"
            echo "Lock Directory: $LOCK_DIR"
            echo ""
            
            # Show active locks
            echo "Active Locks:"
            if ls "$LOCK_DIR"/*.lock >/dev/null 2>&1; then
                for lock_file in "$LOCK_DIR"/*.lock; do
                    local lock_pid=$(cat "$lock_file" 2>/dev/null)
                    if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
                        echo "  $(basename "$lock_file") (PID: $lock_pid)"
                    else
                        echo "  $(basename "$lock_file") (STALE)"
                    fi
                done
            else
                echo "  None"
            fi
            
            echo ""
            echo "Recent Backup Activity:"
            find "$BACKUP_BASE_DIR" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -mtime -1 -exec ls -lh {} \; 2>/dev/null | head -10 || echo "  No recent backups found"
            ;;
        
        "test")
            init_scheduler
            
            log_info "Running backup system tests"
            
            # Test encryption
            if [[ "${ENCRYPTION_ENABLED:-false}" == "true" ]]; then
                if test_encryption; then
                    log_info "Encryption test passed"
                else
                    log_error "Encryption test failed"
                fi
            fi
            
            # Test S3 connectivity
            if [[ "${S3_UPLOAD_ENABLED:-false}" == "true" ]]; then
                if test_s3_connectivity; then
                    log_info "S3 connectivity test passed"
                else
                    log_error "S3 connectivity test failed"
                fi
            fi
            
            log_info "Backup system tests completed"
            ;;
        
        "help"|*)
            echo "Physics Assistant Backup Scheduler"
            echo "Usage: $0 <action> [options]"
            echo ""
            echo "Actions:"
            echo "  backup <service> <type>  - Execute backup for service (postgres|neo4j|redis|application)"
            echo "                            and type (daily|weekly|monthly|yearly)"
            echo "  cleanup [service] [type] - Apply retention policy and cleanup old backups"
            echo "  report                   - Generate backup inventory report"
            echo "  status                   - Show backup system status"
            echo "  test                     - Run backup system tests"
            echo "  help                     - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 backup postgres daily"
            echo "  $0 cleanup all all"
            echo "  $0 report"
            ;;
    esac
}

# Execute main function
main "$@"