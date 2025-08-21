#!/bin/bash
set -e

# Volume Health Monitoring and Management Script
# This script manages Docker volumes, monitors their health, and performs cleanup operations

# Source common functions
source /scripts/common/utils.sh
source /scripts/common/monitoring.sh

# Initialize logging
init_logging "volume-manager"

# Configuration
VOLUME_CHECK_INTERVAL=${VOLUME_CHECK_INTERVAL:-3600}  # 1 hour
VOLUME_USAGE_THRESHOLD=${VOLUME_USAGE_THRESHOLD:-80}  # 80% usage warning
VOLUME_CRITICAL_THRESHOLD=${VOLUME_CRITICAL_THRESHOLD:-95}  # 95% critical
CLEANUP_ENABLED=${CLEANUP_ENABLED:-true}
DRY_RUN=${DRY_RUN:-false}

# Volume definitions for the Physics Assistant platform
declare -A EXPECTED_VOLUMES=(
    ["postgres-data"]="Database data for PostgreSQL"
    ["neo4j-data"]="Graph database data for Neo4j"
    ["redis-data"]="Cache data for Redis"
    ["prometheus-data"]="Monitoring metrics data"
    ["grafana-data"]="Dashboard and alerting data"
    ["alertmanager-data"]="Alert management data"
    ["backup-postgres"]="PostgreSQL backup storage"
    ["backup-neo4j"]="Neo4j backup storage"
    ["backup-redis"]="Redis backup storage"
    ["backup-application"]="Application backup storage"
    ["backup-logs"]="Backup operation logs"
)

# Get volume information
get_volume_info() {
    local volume_name=$1
    
    if docker volume inspect "$volume_name" >/dev/null 2>&1; then
        docker volume inspect "$volume_name" --format '{{json .}}'
    else
        echo "null"
    fi
}

# Get volume usage statistics
get_volume_usage() {
    local volume_name=$1
    
    # Get mount point from volume inspect
    local mount_point=$(docker volume inspect "$volume_name" --format '{{.Mountpoint}}' 2>/dev/null)
    
    if [[ -n "$mount_point" ]] && [[ -d "$mount_point" ]]; then
        # Get disk usage for the mount point
        local usage_info=$(df "$mount_point" 2>/dev/null | awk 'NR==2 {print $2","$3","$4","$5}' | sed 's/%//')
        
        if [[ -n "$usage_info" ]]; then
            echo "$usage_info"
        else
            echo "0,0,0,0"
        fi
    else
        echo "0,0,0,0"
    fi
}

# Check volume health
check_volume_health() {
    local volume_name=$1
    local description=$2
    
    log_info "Checking health of volume: $volume_name"
    
    local health_status="healthy"
    local issues=()
    local warnings=()
    
    # Check if volume exists
    if ! docker volume inspect "$volume_name" >/dev/null 2>&1; then
        health_status="missing"
        issues+=("Volume does not exist")
        log_error "Volume missing: $volume_name"
        return 1
    fi
    
    # Get volume usage
    local usage_info=$(get_volume_usage "$volume_name")
    IFS=',' read -r total_kb used_kb available_kb usage_percent <<< "$usage_info"
    
    # Check usage thresholds
    if [[ $usage_percent -ge $VOLUME_CRITICAL_THRESHOLD ]]; then
        health_status="critical"
        issues+=("Volume usage critical: ${usage_percent}%")
        log_error "Volume $volume_name usage critical: ${usage_percent}%"
    elif [[ $usage_percent -ge $VOLUME_USAGE_THRESHOLD ]]; then
        health_status="warning"
        warnings+=("Volume usage high: ${usage_percent}%")
        log_warn "Volume $volume_name usage high: ${usage_percent}%"
    fi
    
    # Check volume mount point accessibility
    local mount_point=$(docker volume inspect "$volume_name" --format '{{.Mountpoint}}' 2>/dev/null)
    if [[ -n "$mount_point" ]]; then
        if [[ ! -d "$mount_point" ]]; then
            health_status="error"
            issues+=("Mount point not accessible: $mount_point")
            log_error "Volume $volume_name mount point not accessible: $mount_point"
        elif [[ ! -r "$mount_point" ]]; then
            health_status="warning"
            warnings+=("Mount point not readable: $mount_point")
            log_warn "Volume $volume_name mount point not readable: $mount_point"
        fi
    fi
    
    # Check for orphaned or dangling data
    if [[ -n "$mount_point" ]] && [[ -d "$mount_point" ]]; then
        local file_count=$(find "$mount_point" -type f 2>/dev/null | wc -l)
        local dir_count=$(find "$mount_point" -type d 2>/dev/null | wc -l)
        
        # Log volume contents summary
        log_info "Volume $volume_name contains $file_count files and $dir_count directories"
        
        # Check for very old files that might indicate stale data
        local old_files=$(find "$mount_point" -type f -mtime +365 2>/dev/null | wc -l)
        if [[ $old_files -gt 0 ]]; then
            warnings+=("Found $old_files files older than 1 year")
            log_warn "Volume $volume_name has $old_files files older than 1 year"
        fi
    fi
    
    # Generate health report
    local health_report=$(cat << EOF
{
    "volume_name": "$volume_name",
    "description": "$description",
    "health_status": "$health_status",
    "usage_percent": $usage_percent,
    "total_size_kb": $total_kb,
    "used_size_kb": $used_kb,
    "available_size_kb": $available_kb,
    "mount_point": "$mount_point",
    "issues": [$(printf '"%s",' "${issues[@]}" | sed 's/,$//')]",
    "warnings": [$(printf '"%s",' "${warnings[@]}" | sed 's/,$//')]",
    "check_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
)
    
    # Save health report
    local report_dir="/logs/volume-health"
    mkdir -p "$report_dir"
    echo "$health_report" > "$report_dir/${volume_name}_$(date +%Y%m%d_%H%M%S).json"
    
    # Update metrics
    update_volume_metrics "$volume_name" "$health_status" "$usage_percent" "$used_kb"
    
    if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "warning" ]]; then
        log_info "Volume $volume_name health check completed: $health_status"
        return 0
    else
        log_error "Volume $volume_name health check failed: $health_status"
        return 1
    fi
}

# Update volume metrics for monitoring
update_volume_metrics() {
    local volume_name=$1
    local health_status=$2
    local usage_percent=$3
    local used_kb=$4
    
    local metrics_file="/tmp/volume_metrics.prom"
    local timestamp=$(date +%s)
    
    # Convert health status to numeric
    local health_numeric=1
    case "$health_status" in
        "healthy") health_numeric=1 ;;
        "warning") health_numeric=0.5 ;;
        "critical"|"error"|"missing") health_numeric=0 ;;
    esac
    
    cat >> "$metrics_file" << EOF
# HELP volume_health_status Volume health status (1=healthy, 0.5=warning, 0=critical/error)
# TYPE volume_health_status gauge
volume_health_status{volume="$volume_name"} $health_numeric

# HELP volume_usage_percent Volume usage percentage
# TYPE volume_usage_percent gauge
volume_usage_percent{volume="$volume_name"} $usage_percent

# HELP volume_used_bytes Volume used space in bytes
# TYPE volume_used_bytes gauge
volume_used_bytes{volume="$volume_name"} $((used_kb * 1024))

# HELP volume_check_timestamp Volume health check timestamp
# TYPE volume_check_timestamp gauge
volume_check_timestamp{volume="$volume_name"} $timestamp

EOF
}

# Clean up old files in volume
cleanup_volume() {
    local volume_name=$1
    local max_age_days=${2:-30}
    
    log_info "Cleaning up old files in volume: $volume_name (older than $max_age_days days)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE: No actual cleanup will be performed"
    fi
    
    local mount_point=$(docker volume inspect "$volume_name" --format '{{.Mountpoint}}' 2>/dev/null)
    
    if [[ -z "$mount_point" ]] || [[ ! -d "$mount_point" ]]; then
        log_error "Cannot access mount point for volume: $volume_name"
        return 1
    fi
    
    local cleanup_patterns=()
    local cleanup_count=0
    local space_freed=0
    
    # Define cleanup patterns based on volume type
    case "$volume_name" in
        *backup*)
            # For backup volumes, clean up old backup files
            cleanup_patterns=(
                "*.log.*"           # Old log files
                "*.tmp"             # Temporary files
                "core.*"            # Core dumps
                ".DS_Store"         # macOS metadata
                "Thumbs.db"         # Windows metadata
            )
            ;;
        *prometheus*)
            # For Prometheus, clean up old WAL files and chunks
            cleanup_patterns=(
                "wal/0*"            # Old WAL files (keep recent ones)
                "chunks_head/0*"    # Old chunk files
            )
            max_age_days=7  # Prometheus data doesn't need to be kept as long
            ;;
        *grafana*)
            # For Grafana, clean up logs and session data
            cleanup_patterns=(
                "log/grafana.log.*" # Old log files
                "data/sessions/*"   # Session files
                "plugins/*/cache"   # Plugin cache
            )
            max_age_days=14
            ;;
        *)
            # Generic cleanup patterns
            cleanup_patterns=(
                "*.log.*"
                "*.tmp"
                "*.temp"
                "core.*"
                ".DS_Store"
                "Thumbs.db"
            )
            ;;
    esac
    
    # Perform cleanup
    for pattern in "${cleanup_patterns[@]}"; do
        local files_to_clean=$(find "$mount_point" -name "$pattern" -type f -mtime +$max_age_days 2>/dev/null || true)
        
        if [[ -n "$files_to_clean" ]]; then
            while IFS= read -r file; do
                local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
                
                if [[ "$DRY_RUN" == "true" ]]; then
                    log_info "Would delete: $file ($(human_readable_size $file_size))"
                else
                    log_info "Deleting old file: $file ($(human_readable_size $file_size))"
                    rm -f "$file"
                fi
                
                ((cleanup_count++))
                space_freed=$((space_freed + file_size))
                
            done <<< "$files_to_clean"
        fi
    done
    
    # Clean up empty directories
    if [[ "$DRY_RUN" != "true" ]]; then
        find "$mount_point" -type d -empty -delete 2>/dev/null || true
    fi
    
    log_info "Volume cleanup completed: $volume_name"
    log_info "Files cleaned: $cleanup_count, Space freed: $(human_readable_size $space_freed)"
    
    # Update cleanup metrics
    cat >> "/tmp/volume_metrics.prom" << EOF
# HELP volume_cleanup_files_total Total files cleaned from volume
# TYPE volume_cleanup_files_total counter
volume_cleanup_files_total{volume="$volume_name"} $cleanup_count

# HELP volume_cleanup_bytes_total Total bytes freed from volume cleanup
# TYPE volume_cleanup_bytes_total counter
volume_cleanup_bytes_total{volume="$volume_name"} $space_freed

EOF
    
    return 0
}

# Create missing volumes
create_missing_volumes() {
    log_info "Checking for missing volumes and creating them if needed"
    
    local created_count=0
    
    for volume_name in "${!EXPECTED_VOLUMES[@]}"; do
        local description="${EXPECTED_VOLUMES[$volume_name]}"
        
        if ! docker volume inspect "$volume_name" >/dev/null 2>&1; then
            log_warn "Volume missing: $volume_name - $description"
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "DRY RUN MODE: Would create volume $volume_name"
            else
                log_info "Creating volume: $volume_name"
                
                if docker volume create "$volume_name" >/dev/null 2>&1; then
                    log_info "Volume created successfully: $volume_name"
                    ((created_count++))
                else
                    log_error "Failed to create volume: $volume_name"
                fi
            fi
        fi
    done
    
    if [[ $created_count -gt 0 ]]; then
        log_info "Created $created_count missing volumes"
    else
        log_info "No missing volumes found"
    fi
}

# Remove orphaned volumes
remove_orphaned_volumes() {
    log_info "Checking for orphaned volumes"
    
    local removed_count=0
    local all_volumes=$(docker volume ls --format "{{.Name}}")
    
    while IFS= read -r volume_name; do
        # Skip volumes that are expected or currently in use
        if [[ -n "${EXPECTED_VOLUMES[$volume_name]:-}" ]]; then
            continue
        fi
        
        # Check if volume is in use by any container
        local in_use=$(docker ps -a --filter "volume=$volume_name" --format "{{.Names}}" | wc -l)
        
        if [[ $in_use -eq 0 ]]; then
            # Check if it's a system volume or matches our naming pattern
            if [[ "$volume_name" =~ ^physics-assistant-|^physics- ]]; then
                log_warn "Found orphaned volume: $volume_name"
                
                if [[ "$DRY_RUN" == "true" ]]; then
                    log_info "DRY RUN MODE: Would remove orphaned volume $volume_name"
                else
                    log_info "Removing orphaned volume: $volume_name"
                    
                    if docker volume rm "$volume_name" >/dev/null 2>&1; then
                        log_info "Orphaned volume removed: $volume_name"
                        ((removed_count++))
                    else
                        log_error "Failed to remove orphaned volume: $volume_name"
                    fi
                fi
            fi
        fi
    done <<< "$all_volumes"
    
    if [[ $removed_count -gt 0 ]]; then
        log_info "Removed $removed_count orphaned volumes"
    else
        log_info "No orphaned volumes found"
    fi
}

# Generate volume health report
generate_volume_report() {
    log_info "Generating comprehensive volume health report"
    
    local report_file="/logs/volume-health/volume_health_report_$(date +%Y%m%d_%H%M%S).json"
    mkdir -p "$(dirname "$report_file")"
    
    local report_data="{"
    report_data+='"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",'
    report_data+='"total_expected_volumes":'${#EXPECTED_VOLUMES[@]}','
    report_data+='"volumes":['
    
    local volume_reports=()
    local healthy_count=0
    local warning_count=0
    local critical_count=0
    local missing_count=0
    
    for volume_name in "${!EXPECTED_VOLUMES[@]}"; do
        local description="${EXPECTED_VOLUMES[$volume_name]}"
        
        if docker volume inspect "$volume_name" >/dev/null 2>&1; then
            # Volume exists, check its health
            local usage_info=$(get_volume_usage "$volume_name")
            IFS=',' read -r total_kb used_kb available_kb usage_percent <<< "$usage_info"
            
            local health_status="healthy"
            if [[ $usage_percent -ge $VOLUME_CRITICAL_THRESHOLD ]]; then
                health_status="critical"
                ((critical_count++))
            elif [[ $usage_percent -ge $VOLUME_USAGE_THRESHOLD ]]; then
                health_status="warning"
                ((warning_count++))
            else
                ((healthy_count++))
            fi
            
            local mount_point=$(docker volume inspect "$volume_name" --format '{{.Mountpoint}}' 2>/dev/null)
            
            volume_reports+=('{
                "name":"'$volume_name'",
                "description":"'$description'",
                "status":"'$health_status'",
                "usage_percent":'$usage_percent',
                "total_size_mb":'$((total_kb / 1024))',
                "used_size_mb":'$((used_kb / 1024))',
                "available_size_mb":'$((available_kb / 1024))',
                "mount_point":"'$mount_point'"
            }')
        else
            # Volume is missing
            ((missing_count++))
            volume_reports+=('{
                "name":"'$volume_name'",
                "description":"'$description'",
                "status":"missing",
                "usage_percent":0,
                "total_size_mb":0,
                "used_size_mb":0,
                "available_size_mb":0,
                "mount_point":""
            }')
        fi
    done
    
    # Join volume reports
    local joined_reports=$(IFS=','; echo "${volume_reports[*]}")
    report_data+="$joined_reports"
    
    report_data+='],'
    report_data+='"summary":{'
    report_data+='"healthy":'$healthy_count','
    report_data+='"warning":'$warning_count','
    report_data+='"critical":'$critical_count','
    report_data+='"missing":'$missing_count
    report_data+='},'
    
    # Add overall status
    local overall_status="healthy"
    if [[ $critical_count -gt 0 ]] || [[ $missing_count -gt 0 ]]; then
        overall_status="critical"
    elif [[ $warning_count -gt 0 ]]; then
        overall_status="warning"
    fi
    
    report_data+='"overall_status":"'$overall_status'"'
    report_data+='}'
    
    # Save report
    echo "$report_data" | python3 -m json.tool > "$report_file" 2>/dev/null || echo "$report_data" > "$report_file"
    
    log_info "Volume health report generated: $report_file"
    log_info "Summary: $healthy_count healthy, $warning_count warning, $critical_count critical, $missing_count missing"
    
    # Return status code based on overall health
    case "$overall_status" in
        "healthy") return 0 ;;
        "warning") return 1 ;;
        "critical") return 2 ;;
    esac
}

# Monitor volume health continuously
monitor_volumes() {
    log_info "Starting continuous volume health monitoring"
    log_info "Check interval: $VOLUME_CHECK_INTERVAL seconds"
    
    while true; do
        log_info "Starting volume health check cycle"
        
        local failed_checks=0
        
        # Check each expected volume
        for volume_name in "${!EXPECTED_VOLUMES[@]}"; do
            local description="${EXPECTED_VOLUMES[$volume_name]}"
            
            if ! check_volume_health "$volume_name" "$description"; then
                ((failed_checks++))
            fi
        done
        
        # Perform cleanup if enabled
        if [[ "$CLEANUP_ENABLED" == "true" ]]; then
            for volume_name in "${!EXPECTED_VOLUMES[@]}"; do
                if [[ "$volume_name" == *backup* ]]; then
                    cleanup_volume "$volume_name" 30  # Clean backup volumes with 30-day retention
                elif [[ "$volume_name" == *prometheus* ]]; then
                    cleanup_volume "$volume_name" 7   # Clean Prometheus data with 7-day retention
                elif [[ "$volume_name" == *grafana* ]]; then
                    cleanup_volume "$volume_name" 14  # Clean Grafana data with 14-day retention
                fi
            done
        fi
        
        # Generate status report
        generate_volume_report
        
        log_info "Volume health check cycle completed"
        log_info "Failed checks: $failed_checks/${#EXPECTED_VOLUMES[@]}"
        
        # Sleep until next check
        sleep "$VOLUME_CHECK_INTERVAL"
    done
}

# Main function
main() {
    local action=${1:-"help"}
    
    case "$action" in
        "check")
            local volume_name=${2:-"all"}
            
            if [[ "$volume_name" == "all" ]]; then
                log_info "Checking health of all volumes"
                
                local failed_count=0
                for vol_name in "${!EXPECTED_VOLUMES[@]}"; do
                    local description="${EXPECTED_VOLUMES[$vol_name]}"
                    if ! check_volume_health "$vol_name" "$description"; then
                        ((failed_count++))
                    fi
                done
                
                if [[ $failed_count -gt 0 ]]; then
                    log_error "$failed_count volume health checks failed"
                    exit 1
                fi
            else
                if [[ -n "${EXPECTED_VOLUMES[$volume_name]:-}" ]]; then
                    check_volume_health "$volume_name" "${EXPECTED_VOLUMES[$volume_name]}"
                else
                    log_error "Unknown volume: $volume_name"
                    exit 1
                fi
            fi
            ;;
        
        "cleanup")
            local volume_name=${2:-"all"}
            local max_age_days=${3:-30}
            
            if [[ "$volume_name" == "all" ]]; then
                log_info "Cleaning up all volumes"
                
                for vol_name in "${!EXPECTED_VOLUMES[@]}"; do
                    cleanup_volume "$vol_name" "$max_age_days"
                done
            else
                cleanup_volume "$volume_name" "$max_age_days"
            fi
            ;;
        
        "create")
            create_missing_volumes
            ;;
        
        "remove-orphaned")
            remove_orphaned_volumes
            ;;
        
        "report")
            generate_volume_report
            ;;
        
        "monitor")
            monitor_volumes
            ;;
        
        "status")
            echo "Volume Manager Status"
            echo "===================="
            echo "Expected volumes: ${#EXPECTED_VOLUMES[@]}"
            echo "Check interval: $VOLUME_CHECK_INTERVAL seconds"
            echo "Usage threshold: $VOLUME_USAGE_THRESHOLD%"
            echo "Critical threshold: $VOLUME_CRITICAL_THRESHOLD%"
            echo "Cleanup enabled: $CLEANUP_ENABLED"
            echo "Dry run mode: $DRY_RUN"
            echo ""
            
            echo "Volume Status:"
            for volume_name in "${!EXPECTED_VOLUMES[@]}"; do
                if docker volume inspect "$volume_name" >/dev/null 2>&1; then
                    local usage_info=$(get_volume_usage "$volume_name")
                    IFS=',' read -r total_kb used_kb available_kb usage_percent <<< "$usage_info"
                    echo "  $volume_name: EXISTS (${usage_percent}% used)"
                else
                    echo "  $volume_name: MISSING"
                fi
            done
            ;;
        
        "help"|*)
            echo "Physics Assistant Volume Manager"
            echo "Usage: $0 <action> [options]"
            echo ""
            echo "Actions:"
            echo "  check [volume]       - Check health of volume(s) (default: all)"
            echo "  cleanup [volume] [days] - Clean up old files in volume(s) (default: all, 30 days)"
            echo "  create              - Create missing volumes"
            echo "  remove-orphaned     - Remove orphaned volumes"
            echo "  report              - Generate volume health report"
            echo "  monitor             - Start continuous monitoring"
            echo "  status              - Show volume manager status"
            echo "  help                - Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  VOLUME_CHECK_INTERVAL     - Check interval in seconds (default: 3600)"
            echo "  VOLUME_USAGE_THRESHOLD    - Usage warning threshold % (default: 80)"
            echo "  VOLUME_CRITICAL_THRESHOLD - Usage critical threshold % (default: 95)"
            echo "  CLEANUP_ENABLED           - Enable automatic cleanup (default: true)"
            echo "  DRY_RUN                   - Dry run mode, no changes made (default: false)"
            echo ""
            echo "Examples:"
            echo "  $0 check postgres-data"
            echo "  $0 cleanup backup-postgres 7"
            echo "  DRY_RUN=true $0 remove-orphaned"
            echo "  $0 monitor"
            ;;
    esac
}

# Execute main function
main "$@"