#!/bin/bash
set -e

# Disaster Recovery Script for Physics Assistant Platform
# This script provides comprehensive disaster recovery capabilities

# Source common functions
source /scripts/common/utils.sh
source /scripts/common/monitoring.sh
source /scripts/common/encryption.sh
source /scripts/common/s3_upload.sh

# Initialize logging
init_logging "disaster-recovery"

# Configuration
RECOVERY_PLAN_FILE="/scripts/config/recovery-plan.json"
RECOVERY_LOG_FILE="/logs/restore/disaster_recovery_$(date +%Y%m%d_%H%M%S).log"
RECOVERY_WORKSPACE="/tmp/restore/disaster_recovery"
BACKUP_SOURCE_DIR="/backups"

# Recovery configuration
RECOVERY_RTO_MINUTES=${RECOVERY_RTO_MINUTES:-60}  # Recovery Time Objective
RECOVERY_RPO_HOURS=${RECOVERY_RPO_HOURS:-4}       # Recovery Point Objective
PARALLEL_RECOVERY=${PARALLEL_RECOVERY:-true}
VALIDATION_ENABLED=${VALIDATION_ENABLED:-true}

# Recovery modes
RECOVERY_MODE=${RECOVERY_MODE:-"full"}  # full, partial, test

# Initialize disaster recovery
init_disaster_recovery() {
    log_info "Initializing disaster recovery process"
    log_info "Recovery mode: $RECOVERY_MODE"
    log_info "RTO target: $RECOVERY_RTO_MINUTES minutes"
    log_info "RPO target: $RECOVERY_RPO_HOURS hours"
    
    # Create workspace
    mkdir -p "$RECOVERY_WORKSPACE"/{postgres,neo4j,redis,application,validation}
    mkdir -p /logs/restore /scripts/config
    
    # Create default recovery plan if it doesn't exist
    if [[ ! -f "$RECOVERY_PLAN_FILE" ]]; then
        create_default_recovery_plan
    fi
    
    log_info "Disaster recovery initialized"
}

# Create default recovery plan
create_default_recovery_plan() {
    log_info "Creating default recovery plan"
    
    cat > "$RECOVERY_PLAN_FILE" << 'EOF'
{
    "recovery_plan": {
        "version": "1.0",
        "description": "Physics Assistant Platform Disaster Recovery Plan",
        "recovery_steps": [
            {
                "step": 1,
                "name": "Infrastructure Validation",
                "description": "Validate Docker containers and network connectivity",
                "timeout_minutes": 10,
                "critical": true,
                "parallel": false
            },
            {
                "step": 2,
                "name": "Database Recovery",
                "description": "Restore PostgreSQL, Neo4j, and Redis databases",
                "timeout_minutes": 30,
                "critical": true,
                "parallel": true,
                "services": ["postgres", "neo4j", "redis"]
            },
            {
                "step": 3,
                "name": "Application Data Recovery",
                "description": "Restore application files and configurations",
                "timeout_minutes": 15,
                "critical": true,
                "parallel": false
            },
            {
                "step": 4,
                "name": "Service Validation",
                "description": "Validate all services are operational",
                "timeout_minutes": 10,
                "critical": true,
                "parallel": false
            },
            {
                "step": 5,
                "name": "Data Integrity Validation",
                "description": "Perform comprehensive data integrity checks",
                "timeout_minutes": 20,
                "critical": false,
                "parallel": false
            }
        ],
        "rollback_steps": [
            {
                "step": 1,
                "name": "Stop Services",
                "description": "Stop all application services"
            },
            {
                "step": 2,
                "name": "Restore Previous State",
                "description": "Restore from previous backup if available"
            }
        ]
    }
}
EOF
    
    log_info "Default recovery plan created: $RECOVERY_PLAN_FILE"
}

# Select backup for recovery
select_recovery_backup() {
    local service=$1
    local recovery_point=${2:-"latest"}
    
    log_info "Selecting recovery backup for $service (point: $recovery_point)"
    
    local backup_dir="$BACKUP_SOURCE_DIR/$service"
    local selected_backup=""
    
    case "$recovery_point" in
        "latest")
            # Find the most recent backup
            selected_backup=$(find "$backup_dir" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -exec stat -f%m {} + 2>/dev/null | sort -n | tail -1)
            if [[ -n "$selected_backup" ]]; then
                selected_backup=$(find "$backup_dir" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -exec stat -f%m {} + 2>/dev/null | sort -n | tail -1 | xargs -I {} find "$backup_dir" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -exec stat -f%m {} \; -print | grep -A1 "^{}$" | tail -1)
            fi
            ;;
        "daily"|"weekly"|"monthly")
            # Find most recent backup of specified type
            selected_backup=$(find "$backup_dir/$recovery_point" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -exec stat -f%m {} + 2>/dev/null | sort -n | tail -1)
            if [[ -n "$selected_backup" ]]; then
                selected_backup=$(find "$backup_dir/$recovery_point" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -exec stat -f%m {} + 2>/dev/null | sort -n | tail -1 | xargs -I {} find "$backup_dir/$recovery_point" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -exec stat -f%m {} \; -print | grep -A1 "^{}$" | tail -1)
            fi
            ;;
        *)
            # Specific backup file or timestamp
            if [[ -f "$recovery_point" ]]; then
                selected_backup="$recovery_point"
            else
                # Search for backup with timestamp
                selected_backup=$(find "$backup_dir" -type f -name "*${recovery_point}*" \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) | head -1)
            fi
            ;;
    esac
    
    if [[ -n "$selected_backup" ]] && [[ -f "$selected_backup" ]]; then
        log_info "Selected backup for $service: $selected_backup"
        echo "$selected_backup"
        return 0
    else
        log_error "No suitable backup found for $service (point: $recovery_point)"
        return 1
    fi
}

# Validate infrastructure
validate_infrastructure() {
    log_info "Validating infrastructure for recovery"
    
    local validation_passed=true
    
    # Check Docker connectivity
    if ! docker version >/dev/null 2>&1; then
        log_error "Docker is not accessible"
        validation_passed=false
    else
        log_info "Docker connectivity verified"
    fi
    
    # Check required containers
    local required_containers=("physics-postgres" "physics-neo4j" "physics-redis")
    
    for container in "${required_containers[@]}"; do
        if ! docker ps -a --format "{{.Names}}" | grep -q "^${container}$"; then
            log_error "Required container not found: $container"
            validation_passed=false
        else
            log_info "Container found: $container"
        fi
    done
    
    # Check network connectivity
    local required_hosts=("postgres" "neo4j" "redis")
    
    for host in "${required_hosts[@]}"; do
        if ! wait_for_service "$host" "localhost" "80" 1 5; then
            log_warn "Service not immediately accessible: $host (will retry during recovery)"
        fi
    done
    
    # Check disk space
    local required_space_gb=10
    if ! check_disk_space "$RECOVERY_WORKSPACE" "$required_space_gb"; then
        log_error "Insufficient disk space for recovery"
        validation_passed=false
    fi
    
    # Check available memory
    local available_memory_gb=$(free -g | awk 'NR==2{print $7}')
    if [[ $available_memory_gb -lt 2 ]]; then
        log_warn "Low available memory: ${available_memory_gb}GB"
    fi
    
    if [[ "$validation_passed" == "true" ]]; then
        log_info "Infrastructure validation passed"
        return 0
    else
        log_error "Infrastructure validation failed"
        return 1
    fi
}

# Restore PostgreSQL database
restore_postgres() {
    local backup_file=$1
    local target_database=${2:-"physics_assistant"}
    
    log_info "Starting PostgreSQL restoration from: $backup_file"
    
    # Prepare backup file (decrypt if needed)
    local restore_file="$backup_file"
    if [[ "$backup_file" == *.enc ]]; then
        log_info "Decrypting backup file"
        restore_file="${RECOVERY_WORKSPACE}/postgres/$(basename "${backup_file%.enc}")"
        if ! decrypt_file "$backup_file" "$restore_file"; then
            log_error "Failed to decrypt PostgreSQL backup"
            return 1
        fi
    fi
    
    # Decompress if needed
    if [[ "$restore_file" == *.gz ]]; then
        log_info "Decompressing backup file"
        local decompressed_file="${restore_file%.gz}"
        gunzip -c "$restore_file" > "$decompressed_file"
        restore_file="$decompressed_file"
    fi
    
    # Wait for PostgreSQL to be ready
    if ! wait_for_service "postgres" "$POSTGRES_HOST" "$POSTGRES_PORT" 30 10; then
        log_error "PostgreSQL not available for restoration"
        return 1
    fi
    
    # Create restore authentication
    echo "${POSTGRES_HOST}:${POSTGRES_PORT}:${target_database}:${POSTGRES_USER}:${POSTGRES_PASSWORD}" > ~/.pgpass
    chmod 600 ~/.pgpass
    
    log_info "Dropping existing database (if exists)"
    dropdb -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" --if-exists "$target_database" 2>/dev/null || true
    
    log_info "Creating new database"
    createdb -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" "$target_database"
    
    # Restore database
    if [[ "$restore_file" == *.dump ]]; then
        log_info "Restoring from custom dump format"
        if pg_restore -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$target_database" -v "$restore_file" 2>>"$RECOVERY_LOG_FILE"; then
            log_info "PostgreSQL restoration completed successfully"
            return 0
        else
            log_error "PostgreSQL restoration failed"
            return 1
        fi
    elif [[ "$restore_file" == *.sql ]]; then
        log_info "Restoring from SQL dump"
        if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$target_database" -f "$restore_file" 2>>"$RECOVERY_LOG_FILE"; then
            log_info "PostgreSQL restoration completed successfully"
            return 0
        else
            log_error "PostgreSQL restoration failed"
            return 1
        fi
    else
        log_error "Unsupported PostgreSQL backup format: $restore_file"
        return 1
    fi
}

# Restore Neo4j database
restore_neo4j() {
    local backup_file=$1
    local target_database=${2:-"neo4j"}
    
    log_info "Starting Neo4j restoration from: $backup_file"
    
    # Prepare backup file (decrypt if needed)
    local restore_file="$backup_file"
    if [[ "$backup_file" == *.enc ]]; then
        log_info "Decrypting backup file"
        restore_file="${RECOVERY_WORKSPACE}/neo4j/$(basename "${backup_file%.enc}")"
        if ! decrypt_file "$backup_file" "$restore_file"; then
            log_error "Failed to decrypt Neo4j backup"
            return 1
        fi
    fi
    
    # Stop Neo4j service
    log_info "Stopping Neo4j service for restoration"
    docker stop physics-neo4j 2>/dev/null || true
    
    # Clear existing data
    log_info "Clearing existing Neo4j data"
    docker run --rm -v neo4j-data:/data alpine rm -rf /data/*
    
    if [[ "$restore_file" == *.dump.gz ]]; then
        # Restore from Neo4j dump
        log_info "Restoring from Neo4j dump format"
        
        # Decompress dump file
        local dump_file="${RECOVERY_WORKSPACE}/neo4j/neo4j.dump"
        gunzip -c "$restore_file" > "$dump_file"
        
        # Copy dump file to container and restore
        docker cp "$dump_file" physics-neo4j:/tmp/neo4j.dump
        
        # Start Neo4j to perform restore
        docker start physics-neo4j
        
        # Wait for Neo4j to be ready
        if ! wait_for_service "neo4j" "$NEO4J_HOST" "7474" 30 10; then
            log_error "Neo4j not available for restoration"
            return 1
        fi
        
        # Restore from dump
        if docker exec physics-neo4j neo4j-admin database load --from-path=/tmp --database="$target_database" --overwrite-destination=true 2>>"$RECOVERY_LOG_FILE"; then
            log_info "Neo4j restoration completed successfully"
            
            # Restart Neo4j
            docker restart physics-neo4j
            
            # Wait for restart
            wait_for_service "neo4j" "$NEO4J_HOST" "7474" 30 10
            
            return 0
        else
            log_error "Neo4j restoration failed"
            return 1
        fi
        
    elif [[ "$restore_file" == *.cypher.gz ]]; then
        # Restore from Cypher script
        log_info "Restoring from Cypher script"
        
        # Start Neo4j
        docker start physics-neo4j
        
        # Wait for Neo4j to be ready
        if ! wait_for_service "neo4j" "$NEO4J_HOST" "7474" 30 10; then
            log_error "Neo4j not available for restoration"
            return 1
        fi
        
        # Decompress and execute Cypher script
        local cypher_file="${RECOVERY_WORKSPACE}/neo4j/restore.cypher"
        gunzip -c "$restore_file" > "$cypher_file"
        
        # Execute Cypher script
        if python3 << EOF
from neo4j import GraphDatabase
import sys

try:
    driver = GraphDatabase.driver("bolt://${NEO4J_HOST}:${NEO4J_PORT}", 
                                 auth=("${NEO4J_USER}", "${NEO4J_PASSWORD}"))
    
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Execute restore script
        with open("${cypher_file}", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("//"):
                    session.run(line)
    
    driver.close()
    print("Cypher restoration completed successfully")
    
except Exception as e:
    print(f"Cypher restoration failed: {e}")
    sys.exit(1)
EOF
        then
            log_info "Neo4j Cypher restoration completed successfully"
            return 0
        else
            log_error "Neo4j Cypher restoration failed"
            return 1
        fi
    else
        log_error "Unsupported Neo4j backup format: $restore_file"
        return 1
    fi
}

# Restore Redis database
restore_redis() {
    local backup_file=$1
    
    log_info "Starting Redis restoration from: $backup_file"
    
    # Prepare backup file (decrypt if needed)
    local restore_file="$backup_file"
    if [[ "$backup_file" == *.enc ]]; then
        log_info "Decrypting backup file"
        restore_file="${RECOVERY_WORKSPACE}/redis/$(basename "${backup_file%.enc}")"
        if ! decrypt_file "$backup_file" "$restore_file"; then
            log_error "Failed to decrypt Redis backup"
            return 1
        fi
    fi
    
    if [[ "$restore_file" == *.rdb.gz ]]; then
        # Restore from RDB backup
        log_info "Restoring from RDB backup"
        
        # Stop Redis service
        docker stop physics-redis 2>/dev/null || true
        
        # Clear existing data
        docker run --rm -v redis-data:/data alpine rm -f /data/dump.rdb
        
        # Decompress and copy RDB file
        local rdb_file="${RECOVERY_WORKSPACE}/redis/dump.rdb"
        gunzip -c "$restore_file" > "$rdb_file"
        docker cp "$rdb_file" physics-redis:/data/dump.rdb
        
        # Start Redis
        docker start physics-redis
        
        # Wait for Redis to be ready
        if ! wait_for_service "redis" "$REDIS_HOST" "$REDIS_PORT" 30 10; then
            log_error "Redis not available after restoration"
            return 1
        fi
        
        log_info "Redis RDB restoration completed successfully"
        return 0
        
    elif [[ "$restore_file" == *.aof.gz ]]; then
        # Restore from AOF backup
        log_info "Restoring from AOF backup"
        
        # Stop Redis service
        docker stop physics-redis 2>/dev/null || true
        
        # Clear existing data
        docker run --rm -v redis-data:/data alpine sh -c "rm -f /data/appendonly.aof /data/dump.rdb"
        
        # Decompress and copy AOF file
        local aof_file="${RECOVERY_WORKSPACE}/redis/appendonly.aof"
        gunzip -c "$restore_file" > "$aof_file"
        docker cp "$aof_file" physics-redis:/data/appendonly.aof
        
        # Start Redis
        docker start physics-redis
        
        # Wait for Redis to be ready
        if ! wait_for_service "redis" "$REDIS_HOST" "$REDIS_PORT" 30 10; then
            log_error "Redis not available after restoration"
            return 1
        fi
        
        log_info "Redis AOF restoration completed successfully"
        return 0
        
    elif [[ "$restore_file" == *_memory.json.gz ]]; then
        # Restore from memory dump
        log_info "Restoring from memory dump"
        
        # Ensure Redis is running
        if ! docker ps --format "{{.Names}}" | grep -q "physics-redis"; then
            docker start physics-redis
        fi
        
        # Wait for Redis to be ready
        if ! wait_for_service "redis" "$REDIS_HOST" "$REDIS_PORT" 30 10; then
            log_error "Redis not available for restoration"
            return 1
        fi
        
        # Clear existing data
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" FLUSHALL
        
        # Decompress and restore data
        local json_file="${RECOVERY_WORKSPACE}/redis/memory_dump.json"
        gunzip -c "$restore_file" > "$json_file"
        
        # Restore data using Python script
        if python3 << EOF
import json
import redis
import sys

try:
    # Connect to Redis
    r = redis.Redis(
        host='${REDIS_HOST}',
        port=${REDIS_PORT},
        password='${REDIS_PASSWORD}',
        decode_responses=False
    )
    
    # Test connection
    r.ping()
    
    # Load backup data
    with open('${json_file}', 'r') as f:
        backup_data = json.load(f)
    
    for key, data in backup_data.items():
        key_type = data['type']
        value = data['value']
        ttl = data.get('ttl', -1)
        
        if key_type == 'string':
            r.set(key, value)
        elif key_type == 'hash':
            r.hset(key, mapping=value)
        elif key_type == 'list':
            for item in value:
                r.lpush(key, item)
        elif key_type == 'set':
            for item in value:
                r.sadd(key, item)
        elif key_type == 'zset':
            for item, score in value:
                r.zadd(key, {item: score})
        
        # Set TTL if specified
        if ttl > 0:
            r.expire(key, ttl)
    
    print(f"Memory restoration completed: {len(backup_data)} keys restored")
    
except Exception as e:
    print(f"Memory restoration failed: {e}")
    sys.exit(1)
EOF
        then
            log_info "Redis memory restoration completed successfully"
            return 0
        else
            log_error "Redis memory restoration failed"
            return 1
        fi
    else
        log_error "Unsupported Redis backup format: $restore_file"
        return 1
    fi
}

# Restore application data
restore_application_data() {
    local backup_file=$1
    
    log_info "Starting application data restoration from: $backup_file"
    
    # Prepare backup file (decrypt if needed)
    local restore_file="$backup_file"
    if [[ "$backup_file" == *.enc ]]; then
        log_info "Decrypting backup file"
        restore_file="${RECOVERY_WORKSPACE}/application/$(basename "${backup_file%.enc}")"
        if ! decrypt_file "$backup_file" "$restore_file"; then
            log_error "Failed to decrypt application backup"
            return 1
        fi
    fi
    
    # Extract application data
    if [[ "$restore_file" == *.tar.gz ]]; then
        log_info "Extracting application data"
        
        local extract_dir="${RECOVERY_WORKSPACE}/application/extracted"
        mkdir -p "$extract_dir"
        
        if tar -xzf "$restore_file" -C "$extract_dir"; then
            log_info "Application data extracted successfully"
            
            # Restore specific directories
            local restore_paths=(
                "UI/uploads:/app/uploads"
                "UI/logs:/app/logs"
                "analytics/models:/app/models"
                "analytics/exports:/app/exports"
                "database/logs:/app/database-logs"
            )
            
            for restore_path in "${restore_paths[@]}"; do
                local source_path="${extract_dir}/$(echo "$restore_path" | cut -d: -f1)"
                local target_container="$(echo "$restore_path" | cut -d: -f2 | cut -d/ -f2)"
                local target_path="$(echo "$restore_path" | cut -d: -f2)"
                
                if [[ -d "$source_path" ]]; then
                    log_info "Restoring $source_path to $target_container:$target_path"
                    
                    # Find appropriate container for restoration
                    case "$target_container" in
                        "app")
                            # Could be multiple containers, try common ones
                            for container in physics-streamlit-ui physics-agents-api physics-ml-engine; do
                                if docker ps --format "{{.Names}}" | grep -q "$container"; then
                                    docker cp "$source_path/." "$container:$target_path/"
                                    break
                                fi
                            done
                            ;;
                        *)
                            log_warn "Unknown target container: $target_container"
                            ;;
                    esac
                fi
            done
            
            log_info "Application data restoration completed successfully"
            return 0
        else
            log_error "Failed to extract application data"
            return 1
        fi
    else
        log_error "Unsupported application backup format: $restore_file"
        return 1
    fi
}

# Validate restored services
validate_restored_services() {
    log_info "Validating restored services"
    
    local validation_passed=true
    
    # Validate PostgreSQL
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "physics_assistant" -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" >/dev/null 2>&1; then
        local table_count=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "physics_assistant" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
        log_info "PostgreSQL validation passed: $table_count tables found"
    else
        log_error "PostgreSQL validation failed"
        validation_passed=false
    fi
    
    # Validate Neo4j
    if python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://${NEO4J_HOST}:${NEO4J_PORT}', auth=('${NEO4J_USER}', '${NEO4J_PASSWORD}'))
with driver.session() as session:
    result = session.run('MATCH (n) RETURN count(n) as nodeCount')
    count = result.single()['nodeCount']
    print(f'Neo4j validation passed: {count} nodes found')
driver.close()
" 2>/dev/null; then
        log_info "Neo4j validation passed"
    else
        log_error "Neo4j validation failed"
        validation_passed=false
    fi
    
    # Validate Redis
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" INFO keyspace >/dev/null 2>&1; then
        local key_count=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" DBSIZE)
        log_info "Redis validation passed: $key_count keys found"
    else
        log_error "Redis validation failed"
        validation_passed=false
    fi
    
    if [[ "$validation_passed" == "true" ]]; then
        log_info "Service validation completed successfully"
        return 0
    else
        log_error "Service validation failed"
        return 1
    fi
}

# Perform data integrity validation
perform_data_integrity_check() {
    log_info "Performing comprehensive data integrity validation"
    
    local validation_results="${RECOVERY_WORKSPACE}/validation/integrity_report.json"
    mkdir -p "$(dirname "$validation_results")"
    
    # Initialize validation report
    cat > "$validation_results" << EOF
{
    "validation_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "validation_results": {
        "postgres": {},
        "neo4j": {},
        "redis": {},
        "application": {}
    },
    "overall_status": "unknown"
}
EOF
    
    local overall_success=true
    
    # PostgreSQL integrity checks
    log_info "Validating PostgreSQL data integrity"
    local pg_check_results=$(python3 << 'EOF'
import psycopg2
import json
import os

try:
    conn = psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT'],
        database='physics_assistant',
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD']
    )
    
    cur = conn.cursor()
    
    # Check table counts
    cur.execute("SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables")
    table_stats = cur.fetchall()
    
    # Check for foreign key violations
    cur.execute("""
        SELECT conname, conrelid::regclass, confrelid::regclass 
        FROM pg_constraint 
        WHERE contype = 'f'
    """)
    foreign_keys = cur.fetchall()
    
    results = {
        "table_count": len(table_stats),
        "foreign_key_count": len(foreign_keys),
        "total_rows": sum([row[2] for row in table_stats]),
        "status": "success"
    }
    
    conn.close()
    print(json.dumps(results))
    
except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
EOF
)
    
    # Update validation results
    python3 << EOF
import json

# Load existing results
with open('$validation_results', 'r') as f:
    data = json.load(f)

# Update PostgreSQL results
pg_results = json.loads('$pg_check_results')
data['validation_results']['postgres'] = pg_results

# Write updated results
with open('$validation_results', 'w') as f:
    json.dump(data, f, indent=2)
EOF
    
    if [[ $(echo "$pg_check_results" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data.get('status', 'error'))") != "success" ]]; then
        overall_success=false
        log_error "PostgreSQL integrity validation failed"
    else
        log_info "PostgreSQL integrity validation passed"
    fi
    
    # Neo4j integrity checks
    log_info "Validating Neo4j data integrity"
    local neo4j_check_results=$(python3 << 'EOF'
from neo4j import GraphDatabase
import json
import os

try:
    driver = GraphDatabase.driver(
        f"bolt://{os.environ['NEO4J_HOST']}:{os.environ['NEO4J_PORT']}",
        auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
    )
    
    with driver.session() as session:
        # Get node count
        node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
        
        # Get relationship count
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        
        # Get node labels
        labels = session.run("CALL db.labels()").values()
        
        # Get relationship types
        rel_types = session.run("CALL db.relationshipTypes()").values()
    
    driver.close()
    
    results = {
        "node_count": node_count,
        "relationship_count": rel_count,
        "label_count": len(labels),
        "relationship_type_count": len(rel_types),
        "status": "success"
    }
    
    print(json.dumps(results))
    
except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
EOF
)
    
    # Update validation results for Neo4j
    python3 << EOF
import json

# Load existing results
with open('$validation_results', 'r') as f:
    data = json.load(f)

# Update Neo4j results
neo4j_results = json.loads('$neo4j_check_results')
data['validation_results']['neo4j'] = neo4j_results

# Write updated results
with open('$validation_results', 'w') as f:
    json.dump(data, f, indent=2)
EOF
    
    if [[ $(echo "$neo4j_check_results" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data.get('status', 'error'))") != "success" ]]; then
        overall_success=false
        log_error "Neo4j integrity validation failed"
    else
        log_info "Neo4j integrity validation passed"
    fi
    
    # Redis integrity checks
    log_info "Validating Redis data integrity"
    local redis_check_results=$(python3 << 'EOF'
import redis
import json
import os

try:
    r = redis.Redis(
        host=os.environ['REDIS_HOST'],
        port=int(os.environ['REDIS_PORT']),
        password=os.environ['REDIS_PASSWORD'],
        decode_responses=True
    )
    
    # Get database info
    info = r.info()
    keyspace = r.info('keyspace')
    
    results = {
        "key_count": r.dbsize(),
        "memory_usage": info.get('used_memory_human', 'unknown'),
        "redis_version": info.get('redis_version', 'unknown'),
        "keyspace_hits": info.get('keyspace_hits', 0),
        "keyspace_misses": info.get('keyspace_misses', 0),
        "status": "success"
    }
    
    print(json.dumps(results))
    
except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
EOF
)
    
    # Update validation results for Redis
    python3 << EOF
import json

# Load existing results
with open('$validation_results', 'r') as f:
    data = json.load(f)

# Update Redis results
redis_results = json.loads('$redis_check_results')
data['validation_results']['redis'] = redis_results

# Set overall status
if all(result.get('status') == 'success' for result in data['validation_results'].values()):
    data['overall_status'] = 'success'
else:
    data['overall_status'] = 'failure'

# Write final results
with open('$validation_results', 'w') as f:
    json.dump(data, f, indent=2)
EOF
    
    if [[ $(echo "$redis_check_results" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data.get('status', 'error'))") != "success" ]]; then
        overall_success=false
        log_error "Redis integrity validation failed"
    else
        log_info "Redis integrity validation passed"
    fi
    
    # Generate final validation report
    log_info "Data integrity validation completed"
    log_info "Validation report: $validation_results"
    
    if [[ "$overall_success" == "true" ]]; then
        log_info "Overall data integrity validation: PASSED"
        return 0
    else
        log_error "Overall data integrity validation: FAILED"
        return 1
    fi
}

# Execute disaster recovery plan
execute_recovery_plan() {
    local recovery_point=${1:-"latest"}
    
    log_info "Executing disaster recovery plan"
    log_info "Recovery point: $recovery_point"
    
    local start_time=$(date +%s)
    local recovery_successful=true
    
    # Step 1: Infrastructure Validation
    log_info "Step 1: Infrastructure Validation"
    if ! validate_infrastructure; then
        log_error "Infrastructure validation failed, aborting recovery"
        return 1
    fi
    
    # Step 2: Database Recovery (parallel if enabled)
    log_info "Step 2: Database Recovery"
    
    if [[ "$PARALLEL_RECOVERY" == "true" ]]; then
        log_info "Starting parallel database recovery"
        
        # Start background processes for each database
        (
            if backup_file=$(select_recovery_backup "postgres" "$recovery_point"); then
                restore_postgres "$backup_file"
            else
                log_error "No PostgreSQL backup found for recovery"
                exit 1
            fi
        ) &
        local postgres_pid=$!
        
        (
            if backup_file=$(select_recovery_backup "neo4j" "$recovery_point"); then
                restore_neo4j "$backup_file"
            else
                log_error "No Neo4j backup found for recovery"
                exit 1
            fi
        ) &
        local neo4j_pid=$!
        
        (
            if backup_file=$(select_recovery_backup "redis" "$recovery_point"); then
                restore_redis "$backup_file"
            else
                log_error "No Redis backup found for recovery"
                exit 1
            fi
        ) &
        local redis_pid=$!
        
        # Wait for all processes to complete
        wait $postgres_pid
        local postgres_result=$?
        
        wait $neo4j_pid
        local neo4j_result=$?
        
        wait $redis_pid
        local redis_result=$?
        
        if [[ $postgres_result -ne 0 ]] || [[ $neo4j_result -ne 0 ]] || [[ $redis_result -ne 0 ]]; then
            log_error "Database recovery failed"
            recovery_successful=false
        else
            log_info "Parallel database recovery completed successfully"
        fi
    else
        log_info "Starting sequential database recovery"
        
        # Sequential recovery
        if backup_file=$(select_recovery_backup "postgres" "$recovery_point"); then
            if ! restore_postgres "$backup_file"; then
                recovery_successful=false
            fi
        fi
        
        if backup_file=$(select_recovery_backup "neo4j" "$recovery_point"); then
            if ! restore_neo4j "$backup_file"; then
                recovery_successful=false
            fi
        fi
        
        if backup_file=$(select_recovery_backup "redis" "$recovery_point"); then
            if ! restore_redis "$backup_file"; then
                recovery_successful=false
            fi
        fi
    fi
    
    if [[ "$recovery_successful" == "false" ]]; then
        log_error "Database recovery failed, aborting"
        return 1
    fi
    
    # Step 3: Application Data Recovery
    log_info "Step 3: Application Data Recovery"
    if backup_file=$(select_recovery_backup "application" "$recovery_point"); then
        if ! restore_application_data "$backup_file"; then
            log_warn "Application data recovery failed, continuing with validation"
        fi
    else
        log_warn "No application backup found, skipping application data recovery"
    fi
    
    # Step 4: Service Validation
    log_info "Step 4: Service Validation"
    if ! validate_restored_services; then
        log_error "Service validation failed"
        recovery_successful=false
    fi
    
    # Step 5: Data Integrity Validation (if enabled)
    if [[ "$VALIDATION_ENABLED" == "true" ]]; then
        log_info "Step 5: Data Integrity Validation"
        if ! perform_data_integrity_check; then
            log_warn "Data integrity validation failed, but recovery will continue"
        fi
    fi
    
    # Calculate recovery time
    local end_time=$(date +%s)
    local recovery_duration=$((end_time - start_time))
    local recovery_minutes=$((recovery_duration / 60))
    
    log_info "Recovery completed in $recovery_duration seconds ($recovery_minutes minutes)"
    
    # Check RTO compliance
    if [[ $recovery_minutes -le $RECOVERY_RTO_MINUTES ]]; then
        log_info "Recovery completed within RTO target ($RECOVERY_RTO_MINUTES minutes)"
    else
        log_warn "Recovery exceeded RTO target: $recovery_minutes > $RECOVERY_RTO_MINUTES minutes"
    fi
    
    if [[ "$recovery_successful" == "true" ]]; then
        log_info "Disaster recovery completed successfully"
        
        # Send success notification
        send_recovery_notification "SUCCESS" "$recovery_duration"
        
        return 0
    else
        log_error "Disaster recovery failed"
        
        # Send failure notification
        send_recovery_notification "FAILURE" "$recovery_duration"
        
        return 1
    fi
}

# Send recovery notification
send_recovery_notification() {
    local status=$1
    local duration=$2
    
    if [[ "${WEBHOOK_URL:-}" ]]; then
        local message="Disaster Recovery $status completed in ${duration}s ($(($duration / 60)) minutes)"
        local color=$([ "$status" = "SUCCESS" ] && echo "good" || echo "danger")
        
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"text\": \"$message\",
                \"color\": \"$color\",
                \"fields\": [
                    {\"title\": \"Status\", \"value\": \"$status\", \"short\": true},
                    {\"title\": \"Duration\", \"value\": \"${duration}s\", \"short\": true},
                    {\"title\": \"RTO Target\", \"value\": \"${RECOVERY_RTO_MINUTES}m\", \"short\": true},
                    {\"title\": \"Host\", \"value\": \"$(hostname)\", \"short\": true}
                ]
            }" >/dev/null 2>&1 || log_warn "Failed to send recovery notification"
    fi
}

# Main disaster recovery function
main() {
    local action=${1:-"help"}
    
    case "$action" in
        "recover")
            local recovery_point=${2:-"latest"}
            
            init_disaster_recovery
            execute_recovery_plan "$recovery_point"
            ;;
        
        "validate")
            init_disaster_recovery
            validate_infrastructure
            validate_restored_services
            ;;
        
        "test")
            local recovery_point=${2:-"latest"}
            
            # Set test mode
            export RECOVERY_MODE="test"
            
            log_info "Running disaster recovery test (no actual restoration)"
            init_disaster_recovery
            
            # Test backup selection
            for service in postgres neo4j redis application; do
                if backup_file=$(select_recovery_backup "$service" "$recovery_point"); then
                    log_info "Test: Found backup for $service: $backup_file"
                else
                    log_warn "Test: No backup found for $service"
                fi
            done
            
            # Test infrastructure
            validate_infrastructure
            ;;
        
        "plan")
            init_disaster_recovery
            
            echo "Disaster Recovery Plan"
            echo "======================"
            echo "Recovery Mode: $RECOVERY_MODE"
            echo "RTO Target: $RECOVERY_RTO_MINUTES minutes"
            echo "RPO Target: $RECOVERY_RPO_HOURS hours"
            echo "Parallel Recovery: $PARALLEL_RECOVERY"
            echo "Validation Enabled: $VALIDATION_ENABLED"
            echo ""
            echo "Available Backups:"
            
            for service in postgres neo4j redis application; do
                echo "  $service:"
                if [[ -d "$BACKUP_SOURCE_DIR/$service" ]]; then
                    find "$BACKUP_SOURCE_DIR/$service" -type f \( -name "*.gz" -o -name "*.dump" -o -name "*.enc" \) -exec stat -f"%m %N" {} \; 2>/dev/null | sort -n | tail -3 | while read timestamp filepath; do
                        local date_str=$(date -r "$timestamp" 2>/dev/null || date -d@"$timestamp" 2>/dev/null || echo "Unknown")
                        echo "    $(basename "$filepath") - $date_str"
                    done
                else
                    echo "    No backups found"
                fi
            done
            ;;
        
        "help"|*)
            echo "Physics Assistant Disaster Recovery"
            echo "Usage: $0 <action> [options]"
            echo ""
            echo "Actions:"
            echo "  recover [point]  - Execute full disaster recovery"
            echo "                    point: latest, daily, weekly, monthly, or specific backup file"
            echo "  validate         - Validate infrastructure and services without recovery"
            echo "  test [point]     - Test disaster recovery process without actual restoration"
            echo "  plan             - Show recovery plan and available backups"
            echo "  help             - Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  RECOVERY_MODE            - Recovery mode: full, partial, test (default: full)"
            echo "  RECOVERY_RTO_MINUTES     - Recovery Time Objective in minutes (default: 60)"
            echo "  RECOVERY_RPO_HOURS       - Recovery Point Objective in hours (default: 4)"
            echo "  PARALLEL_RECOVERY        - Enable parallel recovery (default: true)"
            echo "  VALIDATION_ENABLED       - Enable data integrity validation (default: true)"
            echo ""
            echo "Examples:"
            echo "  $0 recover latest"
            echo "  $0 recover daily"
            echo "  $0 test weekly"
            echo "  $0 validate"
            ;;
    esac
}

# Execute main function
main "$@"