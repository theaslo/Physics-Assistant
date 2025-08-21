#!/bin/bash
set -e

# Source common functions
source /scripts/common/utils.sh
source /scripts/common/monitoring.sh
source /scripts/common/encryption.sh
source /scripts/common/s3_upload.sh

# Initialize logging
init_logging "postgres-backup"

# Backup configuration
BACKUP_TYPE="${BACKUP_TYPE:-full}"
BACKUP_FORMAT="${BACKUP_FORMAT:-custom}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"

# Timestamp for backup files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="postgres_${BACKUP_TYPE}_${TIMESTAMP}"

# Backup paths
BACKUP_DIR="/backups"
TEMP_DIR="/tmp/backup"
LOG_FILE="/logs/backup/postgres_${TIMESTAMP}.log"

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
# HELP postgres_backup_status PostgreSQL backup status (1=success, 0=failure)
# TYPE postgres_backup_status gauge
postgres_backup_status{type="$BACKUP_TYPE"} $status

# HELP postgres_backup_duration_seconds PostgreSQL backup duration in seconds
# TYPE postgres_backup_duration_seconds gauge
postgres_backup_duration_seconds{type="$BACKUP_TYPE"} $duration

# HELP postgres_backup_size_bytes PostgreSQL backup size in bytes
# TYPE postgres_backup_size_bytes gauge
postgres_backup_size_bytes{type="$BACKUP_TYPE"} $size

# HELP postgres_backup_timestamp_seconds PostgreSQL backup completion timestamp
# TYPE postgres_backup_timestamp_seconds gauge
postgres_backup_timestamp_seconds{type="$BACKUP_TYPE"} $(date +%s)
EOF
}

# Function to perform full backup
perform_full_backup() {
    log_info "Starting full PostgreSQL backup"
    
    local backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}.dump"
    
    # Create backup using pg_dump
    if [[ "$BACKUP_FORMAT" == "custom" ]]; then
        pg_dump \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            -F c \
            -Z "$COMPRESSION_LEVEL" \
            -j "$PARALLEL_JOBS" \
            -v \
            -f "$backup_file" 2>> "$LOG_FILE"
    else
        pg_dump \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            -F p \
            -v \
            -f "${backup_file%.dump}.sql" 2>> "$LOG_FILE"
        
        # Compress SQL file
        gzip -"$COMPRESSION_LEVEL" "${backup_file%.dump}.sql"
        backup_file="${backup_file%.dump}.sql.gz"
    fi
    
    if [[ $? -eq 0 ]]; then
        log_info "Full backup completed successfully: $backup_file"
        BACKUP_SIZE=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file")
        
        # Encrypt backup if enabled
        if [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]]; then
            encrypt_file "$backup_file"
            backup_file="${backup_file}.enc"
        fi
        
        # Upload to S3 if enabled
        if [[ "${BACKUP_S3_ENABLED:-false}" == "true" ]]; then
            upload_to_s3 "$backup_file" "postgres/daily/"
        fi
        
        return 0
    else
        log_error "Full backup failed"
        return 1
    fi
}

# Function to perform incremental backup
perform_incremental_backup() {
    log_info "Starting incremental PostgreSQL backup"
    
    # For PostgreSQL, we'll use WAL archiving for incremental backups
    # This requires WAL archiving to be enabled on the PostgreSQL server
    
    local wal_backup_dir="${BACKUP_DIR}/incremental/${TIMESTAMP}"
    mkdir -p "$wal_backup_dir"
    
    # Get current WAL position
    local current_wal=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT pg_current_wal_lsn();" | xargs)
    
    if [[ -n "$current_wal" ]]; then
        log_info "Current WAL position: $current_wal"
        echo "$current_wal" > "${wal_backup_dir}/current_wal_position.txt"
        
        # Create a base backup for incremental restore capability
        pg_basebackup \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -D "$wal_backup_dir" \
            -F tar \
            -z \
            -P \
            -v 2>> "$LOG_FILE"
        
        if [[ $? -eq 0 ]]; then
            log_info "Incremental backup completed successfully: $wal_backup_dir"
            BACKUP_SIZE=$(du -sb "$wal_backup_dir" | cut -f1)
            
            # Encrypt backup if enabled
            if [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]]; then
                find "$wal_backup_dir" -type f -exec /scripts/common/encrypt_file.sh {} \;
            fi
            
            # Upload to S3 if enabled
            if [[ "${BACKUP_S3_ENABLED:-false}" == "true" ]]; then
                upload_to_s3 "$wal_backup_dir" "postgres/incremental/"
            fi
            
            return 0
        else
            log_error "Incremental backup failed"
            return 1
        fi
    else
        log_error "Failed to get current WAL position"
        return 1
    fi
}

# Function to perform schema-only backup
perform_schema_backup() {
    log_info "Starting schema-only PostgreSQL backup"
    
    local schema_file="${BACKUP_DIR}/daily/${BACKUP_NAME}_schema.sql"
    
    pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --schema-only \
        -v \
        -f "$schema_file" 2>> "$LOG_FILE"
    
    if [[ $? -eq 0 ]]; then
        # Compress schema file
        gzip -"$COMPRESSION_LEVEL" "$schema_file"
        schema_file="${schema_file}.gz"
        
        log_info "Schema backup completed successfully: $schema_file"
        BACKUP_SIZE=$(stat -f%z "$schema_file" 2>/dev/null || stat -c%s "$schema_file")
        
        # Encrypt backup if enabled
        if [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]]; then
            encrypt_file "$schema_file"
        fi
        
        return 0
    else
        log_error "Schema backup failed"
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
    if [[ $file_size -lt 1024 ]]; then
        log_error "Backup file too small: ${file_size} bytes"
        return 1
    fi
    
    # Validate dump file integrity
    if [[ "$backup_file" == *.dump ]]; then
        pg_restore --list "$backup_file" >/dev/null 2>&1
        if [[ $? -eq 0 ]]; then
            log_info "Backup validation successful"
            return 0
        else
            log_error "Backup validation failed - corrupt dump file"
            return 1
        fi
    elif [[ "$backup_file" == *.sql.gz ]]; then
        # Test gzip integrity
        gzip -t "$backup_file"
        if [[ $? -eq 0 ]]; then
            log_info "Backup validation successful"
            return 0
        else
            log_error "Backup validation failed - corrupt gzip file"
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
    
    # Get database statistics
    local db_size=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));" | xargs)
    local table_count=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
    
    cat > "$metadata_file" << EOF
{
    "backup_type": "$BACKUP_TYPE",
    "backup_format": "$BACKUP_FORMAT",
    "timestamp": "$TIMESTAMP",
    "database_name": "$POSTGRES_DB",
    "database_size": "$db_size",
    "table_count": $table_count,
    "backup_file": "$(basename "$backup_file")",
    "backup_size": $BACKUP_SIZE,
    "compression_level": $COMPRESSION_LEVEL,
    "encryption_enabled": "${BACKUP_ENCRYPTION:-false}",
    "postgres_version": "$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT version();" | xargs)",
    "backup_duration": $(($(date +%s) - BACKUP_START_TIME)),
    "checksum": "$(md5sum "$backup_file" | cut -d' ' -f1)"
}
EOF
    
    log_info "Backup metadata created: $metadata_file"
}

# Main backup function
main() {
    log_info "Starting PostgreSQL backup process"
    log_info "Backup type: $BACKUP_TYPE"
    log_info "Target database: $POSTGRES_DB on $POSTGRES_HOST:$POSTGRES_PORT"
    
    # Ensure backup directories exist
    mkdir -p "${BACKUP_DIR}/daily" "${BACKUP_DIR}/weekly" "${BACKUP_DIR}/monthly" "${BACKUP_DIR}/incremental"
    mkdir -p "$TEMP_DIR"
    
    # Perform backup based on type
    case "$BACKUP_TYPE" in
        "full"|"daily")
            if perform_full_backup; then
                BACKUP_STATUS=1
                backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}.dump"
                [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]] && backup_file="${backup_file}.enc"
            fi
            ;;
        "incremental")
            if perform_incremental_backup; then
                BACKUP_STATUS=1
            fi
            ;;
        "schema")
            if perform_schema_backup; then
                BACKUP_STATUS=1
                backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}_schema.sql.gz"
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
        log_info "PostgreSQL backup completed successfully"
        exit 0
    else
        log_error "PostgreSQL backup failed"
        exit 1
    fi
}

# Execute main function
main "$@"