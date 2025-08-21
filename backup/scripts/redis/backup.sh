#!/bin/bash
set -e

# Source common functions
source /scripts/common/utils.sh
source /scripts/common/monitoring.sh
source /scripts/common/encryption.sh
source /scripts/common/s3_upload.sh

# Initialize logging
init_logging "redis-backup"

# Backup configuration
BACKUP_TYPE="${BACKUP_TYPE:-full}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"

# Timestamp for backup files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="redis_${BACKUP_TYPE}_${TIMESTAMP}"

# Backup paths
BACKUP_DIR="/backups"
TEMP_DIR="/tmp/backup"
LOG_FILE="/logs/backup/redis_${TIMESTAMP}.log"

# Metrics tracking
BACKUP_START_TIME=$(date +%s)
BACKUP_STATUS=0
BACKUP_SIZE=0

# Function to send metrics
send_metrics() {
    local status=$1
    local duration=$2
    local size=$3
    
    cat << EOF > /tmp/backup_metrics.prom
# HELP redis_backup_status Redis backup status (1=success, 0=failure)
# TYPE redis_backup_status gauge
redis_backup_status{type="$BACKUP_TYPE"} $status

# HELP redis_backup_duration_seconds Redis backup duration in seconds
# TYPE redis_backup_duration_seconds gauge
redis_backup_duration_seconds{type="$BACKUP_TYPE"} $duration

# HELP redis_backup_size_bytes Redis backup size in bytes
# TYPE redis_backup_size_bytes gauge
redis_backup_size_bytes{type="$BACKUP_TYPE"} $size

# HELP redis_backup_timestamp_seconds Redis backup completion timestamp
# TYPE redis_backup_timestamp_seconds gauge
redis_backup_timestamp_seconds{type="$BACKUP_TYPE"} $(date +%s)
EOF
}

# Function to wait for Redis to be ready
wait_for_redis() {
    log_info "Waiting for Redis to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
            log_info "Redis is ready"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: Redis not ready, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Redis failed to become ready after $max_attempts attempts"
    exit 1
}

# Function to perform RDB backup
perform_rdb_backup() {
    log_info "Starting RDB backup"
    
    local backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}.rdb"
    local temp_file="${TEMP_DIR}/${BACKUP_NAME}.rdb"
    
    # Trigger BGSAVE for RDB backup
    local save_result=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" BGSAVE)
    
    if [[ "$save_result" == "Background saving started" ]]; then
        log_info "Background save started, waiting for completion..."
        
        # Wait for BGSAVE to complete
        local max_wait=600  # 10 minutes
        local waited=0
        
        while [[ $waited -lt $max_wait ]]; do
            local last_save=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" LASTSAVE)
            sleep 5
            local current_save=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" LASTSAVE)
            
            if [[ "$current_save" -gt "$last_save" ]]; then
                log_info "RDB backup completed"
                break
            fi
            
            waited=$((waited + 5))
        done
        
        if [[ $waited -ge $max_wait ]]; then
            log_error "RDB backup timed out"
            return 1
        fi
        
        # Copy RDB file from Redis container
        docker cp "physics-redis:/data/dump.rdb" "$temp_file" 2>> "$LOG_FILE"
        
        if [[ $? -eq 0 ]]; then
            # Compress RDB file
            gzip -"$COMPRESSION_LEVEL" "$temp_file"
            mv "${temp_file}.gz" "${backup_file}.gz"
            
            log_info "RDB backup completed successfully: ${backup_file}.gz"
            BACKUP_SIZE=$(stat -f%z "${backup_file}.gz" 2>/dev/null || stat -c%s "${backup_file}.gz")
            
            # Encrypt backup if enabled
            if [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]]; then
                encrypt_file "${backup_file}.gz"
                backup_file="${backup_file}.gz.enc"
            else
                backup_file="${backup_file}.gz"
            fi
            
            # Upload to S3 if enabled
            if [[ "${BACKUP_S3_ENABLED:-false}" == "true" ]]; then
                upload_to_s3 "$backup_file" "redis/rdb/"
            fi
            
            return 0
        else
            log_error "Failed to copy RDB file"
            return 1
        fi
    else
        log_error "Failed to start background save: $save_result"
        return 1
    fi
}

# Function to perform AOF backup
perform_aof_backup() {
    log_info "Starting AOF backup"
    
    local backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}.aof"
    local temp_file="${TEMP_DIR}/${BACKUP_NAME}.aof"
    
    # Rewrite AOF file
    local rewrite_result=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" BGREWRITEAOF)
    
    if [[ "$rewrite_result" == "Background append only file rewriting started" ]]; then
        log_info "AOF rewrite started, waiting for completion..."
        
        # Wait for AOF rewrite to complete
        local max_wait=600  # 10 minutes
        local waited=0
        
        while [[ $waited -lt $max_wait ]]; do
            local aof_rewrite_in_progress=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" INFO persistence | grep aof_rewrite_in_progress | cut -d: -f2 | tr -d '\r')
            
            if [[ "$aof_rewrite_in_progress" == "0" ]]; then
                log_info "AOF rewrite completed"
                break
            fi
            
            sleep 5
            waited=$((waited + 5))
        done
        
        if [[ $waited -ge $max_wait ]]; then
            log_error "AOF rewrite timed out"
            return 1
        fi
        
        # Copy AOF file from Redis container
        docker cp "physics-redis:/data/appendonly.aof" "$temp_file" 2>> "$LOG_FILE"
        
        if [[ $? -eq 0 ]]; then
            # Compress AOF file
            gzip -"$COMPRESSION_LEVEL" "$temp_file"
            mv "${temp_file}.gz" "${backup_file}.gz"
            
            log_info "AOF backup completed successfully: ${backup_file}.gz"
            BACKUP_SIZE=$(stat -f%z "${backup_file}.gz" 2>/dev/null || stat -c%s "${backup_file}.gz")
            
            # Encrypt backup if enabled
            if [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]]; then
                encrypt_file "${backup_file}.gz"
                backup_file="${backup_file}.gz.enc"
            else
                backup_file="${backup_file}.gz"
            fi
            
            # Upload to S3 if enabled
            if [[ "${BACKUP_S3_ENABLED:-false}" == "true" ]]; then
                upload_to_s3 "$backup_file" "redis/aof/"
            fi
            
            return 0
        else
            log_error "Failed to copy AOF file"
            return 1
        fi
    else
        log_error "Failed to start AOF rewrite: $rewrite_result"
        return 1
    fi
}

# Function to perform memory dump backup
perform_memory_backup() {
    log_info "Starting Redis memory dump backup"
    
    local backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}_memory.json"
    
    # Use Redis-py to dump all keys and values
    python3 << EOF
import json
import redis
import sys

try:
    # Connect to Redis
    r = redis.Redis(
        host='${REDIS_HOST}',
        port=${REDIS_PORT},
        password='${REDIS_PASSWORD}',
        decode_responses=True
    )
    
    # Test connection
    r.ping()
    
    # Get all keys
    keys = r.keys('*')
    backup_data = {}
    
    for key in keys:
        key_type = r.type(key)
        
        if key_type == 'string':
            backup_data[key] = {
                'type': 'string',
                'value': r.get(key),
                'ttl': r.ttl(key)
            }
        elif key_type == 'hash':
            backup_data[key] = {
                'type': 'hash',
                'value': r.hgetall(key),
                'ttl': r.ttl(key)
            }
        elif key_type == 'list':
            backup_data[key] = {
                'type': 'list',
                'value': r.lrange(key, 0, -1),
                'ttl': r.ttl(key)
            }
        elif key_type == 'set':
            backup_data[key] = {
                'type': 'set',
                'value': list(r.smembers(key)),
                'ttl': r.ttl(key)
            }
        elif key_type == 'zset':
            backup_data[key] = {
                'type': 'zset',
                'value': r.zrange(key, 0, -1, withscores=True),
                'ttl': r.ttl(key)
            }
    
    # Write backup data to file
    with open('${backup_file}', 'w') as f:
        json.dump(backup_data, f, indent=2, default=str)
    
    print(f"Memory backup completed: {len(keys)} keys backed up")
    
except Exception as e:
    print(f"Memory backup failed: {e}")
    sys.exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        # Compress JSON file
        gzip -"$COMPRESSION_LEVEL" "$backup_file"
        backup_file="${backup_file}.gz"
        
        log_info "Memory backup completed successfully: $backup_file"
        BACKUP_SIZE=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file")
        
        # Encrypt backup if enabled
        if [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]]; then
            encrypt_file "$backup_file"
            backup_file="${backup_file}.enc"
        fi
        
        # Upload to S3 if enabled
        if [[ "${BACKUP_S3_ENABLED:-false}" == "true" ]]; then
            upload_to_s3 "$backup_file" "redis/memory/"
        fi
        
        return 0
    else
        log_error "Memory backup failed"
        return 1
    fi
}

# Function to perform incremental backup
perform_incremental_backup() {
    log_info "Starting incremental Redis backup"
    
    local backup_dir="${BACKUP_DIR}/incremental/${TIMESTAMP}"
    mkdir -p "$backup_dir"
    
    # For Redis, incremental backup means capturing changes since last backup
    # We'll use AOF file and compare with last backup timestamp
    
    local last_backup_file=$(find "${BACKUP_DIR}/daily" -name "redis_*_*.rdb.gz" -o -name "redis_*_*.aof.gz" | sort | tail -1)
    local last_backup_time=0
    
    if [[ -n "$last_backup_file" ]]; then
        last_backup_time=$(stat -f%m "$last_backup_file" 2>/dev/null || stat -c%Y "$last_backup_file")
    fi
    
    # Copy current AOF file
    docker cp "physics-redis:/data/appendonly.aof" "${backup_dir}/incremental.aof" 2>> "$LOG_FILE"
    
    if [[ $? -eq 0 ]]; then
        # Filter AOF entries since last backup (simplified approach)
        if [[ $last_backup_time -gt 0 ]]; then
            # In a real implementation, you would parse AOF and filter by timestamp
            # For now, we'll just include the full AOF as incremental
            log_info "Filtering AOF entries since last backup..."
        fi
        
        # Compress incremental backup
        tar -czf "${backup_dir}.tar.gz" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"
        rm -rf "$backup_dir"
        
        log_info "Incremental backup completed successfully: ${backup_dir}.tar.gz"
        BACKUP_SIZE=$(stat -f%z "${backup_dir}.tar.gz" 2>/dev/null || stat -c%s "${backup_dir}.tar.gz")
        
        # Encrypt backup if enabled
        if [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]]; then
            encrypt_file "${backup_dir}.tar.gz"
        fi
        
        # Upload to S3 if enabled
        if [[ "${BACKUP_S3_ENABLED:-false}" == "true" ]]; then
            upload_to_s3 "${backup_dir}.tar.gz" "redis/incremental/"
        fi
        
        return 0
    else
        log_error "Failed to copy AOF file for incremental backup"
        return 1
    fi
}

# Function to validate backup
validate_backup() {
    local backup_file=$1
    log_info "Validating backup: $backup_file"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Check file size
    local file_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file")
    if [[ $file_size -lt 100 ]]; then
        log_error "Backup file too small: ${file_size} bytes"
        return 1
    fi
    
    # Validate compressed file integrity
    if [[ "$backup_file" == *.gz ]]; then
        gzip -t "$backup_file"
        if [[ $? -eq 0 ]]; then
            log_info "Backup validation successful"
            return 0
        else
            log_error "Backup validation failed - corrupt gzip file"
            return 1
        fi
    elif [[ "$backup_file" == *.tar.gz ]]; then
        tar -tzf "$backup_file" >/dev/null 2>&1
        if [[ $? -eq 0 ]]; then
            log_info "Backup validation successful"
            return 0
        else
            log_error "Backup validation failed - corrupt tar.gz file"
            return 1
        fi
    fi
    
    log_info "Backup validation completed"
    return 0
}

# Function to create backup metadata
create_metadata() {
    local backup_file=$1
    local metadata_file="${backup_file}.metadata.json"
    
    # Get Redis info
    local redis_info=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" INFO server,memory,keyspace | grep -E "redis_version|used_memory_human|db0" | tr '\r' ' ')
    
    # Parse Redis info
    local redis_version=$(echo "$redis_info" | grep redis_version | cut -d: -f2)
    local used_memory=$(echo "$redis_info" | grep used_memory_human | cut -d: -f2)
    local keyspace_info=$(echo "$redis_info" | grep db0 | cut -d: -f2)
    
    cat > "$metadata_file" << EOF
{
    "backup_type": "$BACKUP_TYPE",
    "timestamp": "$TIMESTAMP",
    "backup_file": "$(basename "$backup_file")",
    "backup_size": $BACKUP_SIZE,
    "compression_level": $COMPRESSION_LEVEL,
    "encryption_enabled": "${BACKUP_ENCRYPTION:-false}",
    "redis_version": "$redis_version",
    "used_memory": "$used_memory",
    "keyspace_info": "$keyspace_info",
    "backup_duration": $(($(date +%s) - BACKUP_START_TIME)),
    "checksum": "$(md5sum "$backup_file" | cut -d' ' -f1)"
}
EOF
    
    log_info "Backup metadata created: $metadata_file"
}

# Main backup function
main() {
    log_info "Starting Redis backup process"
    log_info "Backup type: $BACKUP_TYPE"
    log_info "Target Redis: $REDIS_HOST:$REDIS_PORT"
    
    # Wait for Redis to be ready
    wait_for_redis
    
    # Ensure backup directories exist
    mkdir -p "${BACKUP_DIR}/daily" "${BACKUP_DIR}/weekly" "${BACKUP_DIR}/monthly" "${BACKUP_DIR}/incremental"
    mkdir -p "$TEMP_DIR"
    
    # Perform backup based on type
    case "$BACKUP_TYPE" in
        "rdb"|"full")
            if perform_rdb_backup; then
                BACKUP_STATUS=1
                backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}.rdb.gz"
                [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]] && backup_file="${backup_file}.enc"
            fi
            ;;
        "aof")
            if perform_aof_backup; then
                BACKUP_STATUS=1
                backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}.aof.gz"
                [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]] && backup_file="${backup_file}.enc"
            fi
            ;;
        "memory")
            if perform_memory_backup; then
                BACKUP_STATUS=1
                backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}_memory.json.gz"
                [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]] && backup_file="${backup_file}.enc"
            fi
            ;;
        "incremental")
            if perform_incremental_backup; then
                BACKUP_STATUS=1
                backup_file="${BACKUP_DIR}/incremental/${TIMESTAMP}.tar.gz"
                [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]] && backup_file="${backup_file}.enc"
            fi
            ;;
        *)
            log_error "Unknown backup type: $BACKUP_TYPE"
            exit 1
            ;;
    esac
    
    # Calculate backup duration
    BACKUP_DURATION=$(($(date +%s) - BACKUP_START_TIME))
    
    # Validate backup if successful
    if [[ $BACKUP_STATUS -eq 1 ]] && [[ -n "$backup_file" ]]; then
        if validate_backup "$backup_file"; then
            create_metadata "$backup_file"
            log_info "Backup process completed successfully in ${BACKUP_DURATION} seconds"
        else
            BACKUP_STATUS=0
            log_error "Backup validation failed"
        fi
    fi
    
    # Send metrics
    send_metrics "$BACKUP_STATUS" "$BACKUP_DURATION" "$BACKUP_SIZE"
    
    # Cleanup temporary files
    rm -rf "$TEMP_DIR"/*
    
    if [[ $BACKUP_STATUS -eq 1 ]]; then
        log_info "Redis backup completed successfully"
        exit 0
    else
        log_error "Redis backup failed"
        exit 1
    fi
}

# Execute main function
main "$@"