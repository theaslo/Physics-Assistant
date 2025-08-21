#!/bin/bash
set -e

# Source common functions
source /scripts/common/utils.sh
source /scripts/common/monitoring.sh
source /scripts/common/encryption.sh
source /scripts/common/s3_upload.sh

# Initialize logging
init_logging "neo4j-backup"

# Backup configuration
BACKUP_TYPE="${BACKUP_TYPE:-full}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"

# Timestamp for backup files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="neo4j_${BACKUP_TYPE}_${TIMESTAMP}"

# Backup paths
BACKUP_DIR="/backups"
TEMP_DIR="/tmp/backup"
LOG_FILE="/logs/backup/neo4j_${TIMESTAMP}.log"

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
# HELP neo4j_backup_status Neo4j backup status (1=success, 0=failure)
# TYPE neo4j_backup_status gauge
neo4j_backup_status{type="$BACKUP_TYPE"} $status

# HELP neo4j_backup_duration_seconds Neo4j backup duration in seconds
# TYPE neo4j_backup_duration_seconds gauge
neo4j_backup_duration_seconds{type="$BACKUP_TYPE"} $duration

# HELP neo4j_backup_size_bytes Neo4j backup size in bytes
# TYPE neo4j_backup_size_bytes gauge
neo4j_backup_size_bytes{type="$BACKUP_TYPE"} $size

# HELP neo4j_backup_timestamp_seconds Neo4j backup completion timestamp
# TYPE neo4j_backup_timestamp_seconds gauge
neo4j_backup_timestamp_seconds{type="$BACKUP_TYPE"} $(date +%s)
EOF
}

# Function to wait for Neo4j to be ready
wait_for_neo4j() {
    log_info "Waiting for Neo4j to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -u "$NEO4J_USER:$NEO4J_PASSWORD" \
           "http://$NEO4J_HOST:7474/db/data/" >/dev/null 2>&1; then
            log_info "Neo4j is ready"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: Neo4j not ready, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Neo4j failed to become ready after $max_attempts attempts"
    exit 1
}

# Function to perform full backup using neo4j-admin dump
perform_full_backup() {
    log_info "Starting full Neo4j backup"
    
    local backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}.dump"
    local temp_dump="${TEMP_DIR}/${BACKUP_NAME}.dump"
    
    # Create backup using neo4j-admin dump
    # Note: This requires accessing the Neo4j container's neo4j-admin command
    docker exec physics-neo4j neo4j-admin database dump \
        --database="${NEO4J_DATABASE:-neo4j}" \
        --to-path=/tmp \
        --verbose 2>> "$LOG_FILE"
    
    if [[ $? -eq 0 ]]; then
        # Copy the dump file from the container
        docker cp "physics-neo4j:/tmp/${NEO4J_DATABASE:-neo4j}.dump" "$temp_dump"
        
        # Compress the dump file
        gzip -"$COMPRESSION_LEVEL" "$temp_dump"
        mv "${temp_dump}.gz" "${backup_file}.gz"
        
        log_info "Full backup completed successfully: ${backup_file}.gz"
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
            upload_to_s3 "$backup_file" "neo4j/daily/"
        fi
        
        return 0
    else
        log_error "Full backup failed"
        return 1
    fi
}

# Function to perform Cypher export backup
perform_cypher_backup() {
    log_info "Starting Cypher export backup"
    
    local backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}_cypher.cypher"
    
    # Export all nodes and relationships using Cypher
    python3 << EOF
import json
from neo4j import GraphDatabase

def export_graph():
    driver = GraphDatabase.driver("bolt://${NEO4J_HOST}:${NEO4J_PORT}", 
                                 auth=("${NEO4J_USER}", "${NEO4J_PASSWORD}"))
    
    with driver.session() as session:
        with open("${backup_file}", "w") as f:
            # Export all nodes
            result = session.run("MATCH (n) RETURN n")
            for record in result:
                node = record["n"]
                labels = ":".join(node.labels)
                props = dict(node)
                props_str = json.dumps(props).replace('"', '\\"')
                f.write(f'CREATE (n:{labels} {props_str});\n')
            
            # Export all relationships
            result = session.run("MATCH (a)-[r]->(b) RETURN a, r, b")
            for record in result:
                rel = record["r"]
                rel_type = rel.type
                props = dict(rel)
                props_str = json.dumps(props).replace('"', '\\"') if props else ""
                f.write(f'MATCH (a), (b) WHERE id(a) = {record["a"].id} AND id(b) = {record["b"].id} ')
                if props_str:
                    f.write(f'CREATE (a)-[r:{rel_type} {props_str}]->(b);\n')
                else:
                    f.write(f'CREATE (a)-[r:{rel_type}]->(b);\n')
    
    driver.close()

try:
    export_graph()
    print("Cypher export completed successfully")
except Exception as e:
    print(f"Cypher export failed: {e}")
    exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        # Compress the Cypher file
        gzip -"$COMPRESSION_LEVEL" "$backup_file"
        backup_file="${backup_file}.gz"
        
        log_info "Cypher backup completed successfully: $backup_file"
        BACKUP_SIZE=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file")
        
        # Encrypt backup if enabled
        if [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]]; then
            encrypt_file "$backup_file"
            backup_file="${backup_file}.enc"
        fi
        
        # Upload to S3 if enabled
        if [[ "${BACKUP_S3_ENABLED:-false}" == "true" ]]; then
            upload_to_s3 "$backup_file" "neo4j/daily/"
        fi
        
        return 0
    else
        log_error "Cypher backup failed"
        return 1
    fi
}

# Function to perform incremental backup
perform_incremental_backup() {
    log_info "Starting incremental Neo4j backup"
    
    # Neo4j incremental backup using transaction logs
    local backup_dir="${BACKUP_DIR}/incremental/${TIMESTAMP}"
    mkdir -p "$backup_dir"
    
    # Get current transaction ID
    local current_tx=$(python3 << EOF
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://${NEO4J_HOST}:${NEO4J_PORT}", 
                             auth=("${NEO4J_USER}", "${NEO4J_PASSWORD}"))

with driver.session() as session:
    result = session.run("CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Transactions') YIELD attributes")
    for record in result:
        attrs = record["attributes"]
        for attr in attrs:
            if attr["name"] == "LastCommittedTxId":
                print(attr["value"])
                break

driver.close()
EOF
)
    
    if [[ -n "$current_tx" ]]; then
        log_info "Current transaction ID: $current_tx"
        echo "$current_tx" > "${backup_dir}/transaction_id.txt"
        
        # Export recent changes based on timestamp
        local incremental_file="${backup_dir}/incremental_changes.cypher"
        
        python3 << EOF
from neo4j import GraphDatabase
from datetime import datetime, timedelta

driver = GraphDatabase.driver("bolt://${NEO4J_HOST}:${NEO4J_PORT}", 
                             auth=("${NEO4J_USER}", "${NEO4J_PASSWORD}"))

# Get changes from last 24 hours (or since last backup)
last_backup_time = datetime.now() - timedelta(hours=24)
timestamp_ms = int(last_backup_time.timestamp() * 1000)

with driver.session() as session:
    with open("${incremental_file}", "w") as f:
        # Query for recently modified nodes (if timestamp property exists)
        try:
            result = session.run(
                "MATCH (n) WHERE n.lastModified > $timestamp RETURN n",
                timestamp=timestamp_ms
            )
            for record in result:
                node = record["n"]
                labels = ":".join(node.labels)
                props = dict(node)
                props_str = str(props).replace("'", '"')
                f.write(f'MERGE (n:{labels} {{id: "{props.get("id", "")}"}})')
                f.write(f' SET n = {props_str};\n')
        except:
            # Fallback: export all nodes if timestamp property doesn't exist
            result = session.run("MATCH (n) RETURN n LIMIT 1000")
            for record in result:
                node = record["n"]
                labels = ":".join(node.labels)
                props = dict(node)
                props_str = str(props).replace("'", '"')
                f.write(f'CREATE (n:{labels} {props_str});\n')

driver.close()
EOF
        
        if [[ $? -eq 0 ]]; then
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
                upload_to_s3 "${backup_dir}.tar.gz" "neo4j/incremental/"
            fi
            
            return 0
        else
            log_error "Incremental backup failed"
            return 1
        fi
    else
        log_error "Failed to get current transaction ID"
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
    
    # Get database statistics
    local db_stats=$(python3 << EOF
import json
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://${NEO4J_HOST}:${NEO4J_PORT}", 
                             auth=("${NEO4J_USER}", "${NEO4J_PASSWORD}"))

stats = {}
with driver.session() as session:
    # Get node count
    result = session.run("MATCH (n) RETURN count(n) as nodeCount")
    stats["node_count"] = result.single()["nodeCount"]
    
    # Get relationship count
    result = session.run("MATCH ()-[r]->() RETURN count(r) as relCount")
    stats["relationship_count"] = result.single()["relCount"]
    
    # Get database size
    result = session.run("CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store file sizes') YIELD attributes")
    for record in result:
        attrs = record["attributes"]
        for attr in attrs:
            if attr["name"] == "TotalStoreSize":
                stats["database_size"] = attr["value"]
                break

print(json.dumps(stats))
driver.close()
EOF
)
    
    cat > "$metadata_file" << EOF
{
    "backup_type": "$BACKUP_TYPE",
    "timestamp": "$TIMESTAMP",
    "database_name": "${NEO4J_DATABASE:-neo4j}",
    "backup_file": "$(basename "$backup_file")",
    "backup_size": $BACKUP_SIZE,
    "compression_level": $COMPRESSION_LEVEL,
    "encryption_enabled": "${BACKUP_ENCRYPTION:-false}",
    "neo4j_version": "5.11",
    "backup_duration": $(($(date +%s) - BACKUP_START_TIME)),
    "checksum": "$(md5sum "$backup_file" | cut -d' ' -f1)",
    "database_stats": $db_stats
}
EOF
    
    log_info "Backup metadata created: $metadata_file"
}

# Main backup function
main() {
    log_info "Starting Neo4j backup process"
    log_info "Backup type: $BACKUP_TYPE"
    log_info "Target database: ${NEO4J_DATABASE:-neo4j} on $NEO4J_HOST:$NEO4J_PORT"
    
    # Wait for Neo4j to be ready
    wait_for_neo4j
    
    # Ensure backup directories exist
    mkdir -p "${BACKUP_DIR}/daily" "${BACKUP_DIR}/weekly" "${BACKUP_DIR}/monthly" "${BACKUP_DIR}/incremental"
    mkdir -p "$TEMP_DIR"
    
    # Perform backup based on type
    case "$BACKUP_TYPE" in
        "full"|"daily")
            if perform_full_backup; then
                BACKUP_STATUS=1
                backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}.dump.gz"
                [[ "${BACKUP_ENCRYPTION:-false}" == "true" ]] && backup_file="${backup_file}.enc"
            fi
            ;;
        "cypher")
            if perform_cypher_backup; then
                BACKUP_STATUS=1
                backup_file="${BACKUP_DIR}/daily/${BACKUP_NAME}_cypher.cypher.gz"
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
        log_info "Neo4j backup completed successfully"
        exit 0
    else
        log_error "Neo4j backup failed"
        exit 1
    fi
}

# Execute main function
main "$@"