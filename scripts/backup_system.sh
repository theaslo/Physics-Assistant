#!/bin/bash
# Comprehensive Backup System for Physics Assistant
# Handles configuration, data, and complete system recovery

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKUP_BASE_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE_DIR/$TIMESTAMP"
RETENTION_DAYS=30
COMPRESSION_LEVEL=6

# Backup components
BACKUP_CONFIG=true
BACKUP_DATA=true
BACKUP_LOGS=true
BACKUP_CONTAINERS=true

# Function to print colored output
print_status() {
    echo -e "${GREEN}[BACKUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to log backup operations
log_backup() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$BACKUP_BASE_DIR/backup.log"
}

# Function to calculate directory size
calculate_size() {
    if [ -d "$1" ] || [ -f "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# Function to backup configuration files
backup_configuration() {
    print_status "Backing up configuration files..."

    local config_dir="$BACKUP_DIR/configuration"
    mkdir -p "$config_dir"

    # Core configuration files
    local config_files=(
        "docker-compose.mcp-fixed.yaml"
        ".env.production"
        ".env.production.template"
        "PRODUCTION_DEPLOYMENT.md"
        "CLAUDE.md"
        "README.md"
    )

    for file in "${config_files[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" "$config_dir/"
            print_status "✅ Backed up: $file"
        else
            print_warning "⚠️  File not found: $file"
        fi
    done

    # Backup entire mcp_tools directory
    if [ -d "mcp_tools" ]; then
        cp -r mcp_tools/ "$config_dir/"
        print_status "✅ Backed up: mcp_tools/ directory"
    fi

    # Backup scripts directory
    if [ -d "scripts" ]; then
        cp -r scripts/ "$config_dir/"
        print_status "✅ Backed up: scripts/ directory"
    fi

    log_backup "Configuration backup completed"
}

# Function to backup data files
backup_data() {
    print_status "Backing up data files..."

    local data_dir="$BACKUP_DIR/data"
    mkdir -p "$data_dir"

    # Backup data directories if they exist
    local data_dirs=(
        "data"
        "uploads"
        "exports"
    )

    for dir in "${data_dirs[@]}"; do
        if [ -d "$dir" ]; then
            cp -r "$dir" "$data_dir/"
            local size=$(calculate_size "$dir")
            print_status "✅ Backed up: $dir/ ($size)"
        fi
    done

    # Backup database data (if using local database)
    if command -v docker >/dev/null 2>&1 || command -v podman >/dev/null 2>&1; then
        backup_database_data "$data_dir"
    fi

    log_backup "Data backup completed"
}

# Function to backup database data
backup_database_data() {
    local data_dir="$1/database"
    mkdir -p "$data_dir"

    print_status "Backing up database data..."

    # Try to backup PostgreSQL data if container is running
    local postgres_container=$(docker ps --format "{{.Names}}" 2>/dev/null | grep postgres || podman ps --format "{{.Names}}" 2>/dev/null | grep postgres || echo "")

    if [ -n "$postgres_container" ]; then
        print_status "Found PostgreSQL container: $postgres_container"

        # Create SQL dump
        if docker exec "$postgres_container" pg_dump -U physics_user physics_assistant > "$data_dir/physics_assistant_dump.sql" 2>/dev/null; then
            print_status "✅ PostgreSQL database dumped"
        elif podman exec "$postgres_container" pg_dump -U physics_user physics_assistant > "$data_dir/physics_assistant_dump.sql" 2>/dev/null; then
            print_status "✅ PostgreSQL database dumped"
        else
            print_warning "⚠️  Could not dump PostgreSQL database"
        fi
    fi

    # Backup container volumes if they exist
    if command -v docker >/dev/null 2>&1; then
        docker volume ls --format "{{.Name}}" 2>/dev/null | grep physics | while read volume; do
            print_status "Found volume: $volume"
        done
    fi
}

# Function to backup logs
backup_logs() {
    print_status "Backing up logs..."

    local logs_dir="$BACKUP_DIR/logs"
    mkdir -p "$logs_dir"

    # Backup log directories
    local log_dirs=(
        "logs"
        "UI/logs"
        "database/logs"
    )

    for dir in "${log_dirs[@]}"; do
        if [ -d "$dir" ]; then
            cp -r "$dir" "$logs_dir/"
            local size=$(calculate_size "$dir")
            print_status "✅ Backed up: $dir/ ($size)"
        fi
    done

    # Backup container logs
    backup_container_logs "$logs_dir"

    log_backup "Logs backup completed"
}

# Function to backup container logs
backup_container_logs() {
    local logs_dir="$1/containers"
    mkdir -p "$logs_dir"

    print_status "Backing up container logs..."

    # Get list of physics-assistant containers
    local containers=()
    if command -v docker >/dev/null 2>&1; then
        mapfile -t containers < <(docker ps -a --format "{{.Names}}" 2>/dev/null | grep physics-assistant)
    elif command -v podman >/dev/null 2>&1; then
        mapfile -t containers < <(podman ps -a --format "{{.Names}}" 2>/dev/null | grep physics-assistant)
    fi

    for container in "${containers[@]}"; do
        if [ -n "$container" ]; then
            local log_file="$logs_dir/${container}.log"
            if docker logs "$container" > "$log_file" 2>&1; then
                print_status "✅ Backed up logs for: $container"
            elif podman logs "$container" > "$log_file" 2>&1; then
                print_status "✅ Backed up logs for: $container"
            fi
        fi
    done
}

# Function to backup container configurations
backup_containers() {
    print_status "Backing up container configurations..."

    local containers_dir="$BACKUP_DIR/containers"
    mkdir -p "$containers_dir"

    # Save container information
    if command -v docker >/dev/null 2>&1; then
        docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" > "$containers_dir/container_status.txt" 2>/dev/null || true
        docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}" > "$containers_dir/images.txt" 2>/dev/null || true
    elif command -v podman >/dev/null 2>&1; then
        podman ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" > "$containers_dir/container_status.txt" 2>/dev/null || true
        podman images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}" > "$containers_dir/images.txt" 2>/dev/null || true
    fi

    # Save docker-compose configurations
    if [ -f "docker-compose.mcp-fixed.yaml" ]; then
        cp "docker-compose.mcp-fixed.yaml" "$containers_dir/"
    fi

    # Save network information
    if command -v docker >/dev/null 2>&1; then
        docker network ls > "$containers_dir/networks.txt" 2>/dev/null || true
    elif command -v podman >/dev/null 2>&1; then
        podman network ls > "$containers_dir/networks.txt" 2>/dev/null || true
    fi

    print_status "✅ Container configurations backed up"
    log_backup "Container backup completed"
}

# Function to create backup metadata
create_backup_metadata() {
    local metadata_file="$BACKUP_DIR/backup_metadata.json"

    cat > "$metadata_file" << EOF
{
    "backup_timestamp": "$TIMESTAMP",
    "backup_date": "$(date -u '+%Y-%m-%d %H:%M:%S UTC')",
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "physics_assistant_version": "1.0.0",
    "backup_components": {
        "configuration": $BACKUP_CONFIG,
        "data": $BACKUP_DATA,
        "logs": $BACKUP_LOGS,
        "containers": $BACKUP_CONTAINERS
    },
    "backup_size": "$(calculate_size "$BACKUP_DIR")",
    "system_info": {
        "os": "$(uname -s)",
        "kernel": "$(uname -r)",
        "architecture": "$(uname -m)"
    }
}
EOF

    print_status "✅ Backup metadata created"
}

# Function to compress backup
compress_backup() {
    print_status "Compressing backup..."

    local archive_name="$BACKUP_BASE_DIR/physics_assistant_backup_$TIMESTAMP.tar.gz"
    local original_size=$(calculate_size "$BACKUP_DIR")

    tar -czf "$archive_name" -C "$BACKUP_BASE_DIR" "$(basename "$BACKUP_DIR")"

    local compressed_size=$(calculate_size "$archive_name")
    print_status "✅ Backup compressed: $archive_name"
    print_status "   Original size: $original_size"
    print_status "   Compressed size: $compressed_size"

    # Remove uncompressed backup directory
    rm -rf "$BACKUP_DIR"

    log_backup "Backup compressed to $archive_name ($compressed_size)"
}

# Function to clean old backups
cleanup_old_backups() {
    print_status "Cleaning up old backups (keeping last $RETENTION_DAYS days)..."

    if [ -d "$BACKUP_BASE_DIR" ]; then
        find "$BACKUP_BASE_DIR" -name "physics_assistant_backup_*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
        find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -mtime +$RETENTION_DAYS -name "20*" -exec rm -rf {} \; 2>/dev/null || true
    fi

    local remaining_backups=$(find "$BACKUP_BASE_DIR" -name "physics_assistant_backup_*.tar.gz" 2>/dev/null | wc -l)
    print_status "✅ Cleanup complete. $remaining_backups backup(s) remaining."

    log_backup "Cleanup completed - $remaining_backups backups remaining"
}

# Function to list existing backups
list_backups() {
    echo "📦 Physics Assistant Backups"
    echo "=============================="

    if [ ! -d "$BACKUP_BASE_DIR" ]; then
        echo "No backups found. Backup directory does not exist."
        return 0
    fi

    local backups=($(find "$BACKUP_BASE_DIR" -name "physics_assistant_backup_*.tar.gz" 2>/dev/null | sort -r))

    if [ ${#backups[@]} -eq 0 ]; then
        echo "No backup archives found."
        return 0
    fi

    echo "Available backups:"
    for backup in "${backups[@]}"; do
        local filename=$(basename "$backup")
        local timestamp=$(echo "$filename" | sed 's/physics_assistant_backup_\(.*\)\.tar\.gz/\1/')
        local size=$(calculate_size "$backup")
        local date=$(date -d "${timestamp:0:8} ${timestamp:9:2}:${timestamp:11:2}:${timestamp:13:2}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "Unknown date")

        echo "  📦 $filename"
        echo "     Date: $date"
        echo "     Size: $size"
        echo ""
    done

    echo "Total backups: ${#backups[@]}"
}

# Function to restore from backup
restore_backup() {
    local backup_file="$1"

    if [ -z "$backup_file" ]; then
        print_error "No backup file specified"
        echo "Usage: $0 restore <backup_file>"
        echo ""
        list_backups
        return 1
    fi

    if [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        return 1
    fi

    print_status "Restoring from backup: $backup_file"

    # Create restore directory
    local restore_dir="restore_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$restore_dir"

    # Extract backup
    tar -xzf "$backup_file" -C "$restore_dir"

    local extracted_dir=$(find "$restore_dir" -maxdepth 1 -type d -name "20*" | head -1)

    if [ -z "$extracted_dir" ]; then
        print_error "Could not find backup contents in archive"
        rm -rf "$restore_dir"
        return 1
    fi

    print_status "Backup extracted to: $extracted_dir"
    print_warning "⚠️  IMPORTANT: Review the extracted files before manually restoring them to avoid overwriting current configuration."
    print_warning "⚠️  Backup contents are in: $extracted_dir"

    echo ""
    echo "Restore Instructions:"
    echo "1. Review backup contents: ls -la '$extracted_dir'"
    echo "2. Stop current services: docker compose -f docker-compose.mcp-fixed.yaml down"
    echo "3. Manually copy needed files from backup"
    echo "4. Restart services: docker compose -f docker-compose.mcp-fixed.yaml up -d"
    echo "5. Run health check: ./scripts/simple_health_monitor.sh"

    log_backup "Backup restored to $extracted_dir"
}

# Function to run full backup
run_full_backup() {
    print_status "Starting full Physics Assistant backup..."
    print_status "Timestamp: $TIMESTAMP"

    # Create backup directory
    mkdir -p "$BACKUP_BASE_DIR"

    # Initialize backup log
    log_backup "Full backup started"

    # Create backup directory
    mkdir -p "$BACKUP_DIR"

    # Run backup components
    if [ "$BACKUP_CONFIG" = true ]; then
        backup_configuration
    fi

    if [ "$BACKUP_DATA" = true ]; then
        backup_data
    fi

    if [ "$BACKUP_LOGS" = true ]; then
        backup_logs
    fi

    if [ "$BACKUP_CONTAINERS" = true ]; then
        backup_containers
    fi

    # Create metadata
    create_backup_metadata

    # Compress backup
    compress_backup

    # Cleanup old backups
    cleanup_old_backups

    local final_archive="$BACKUP_BASE_DIR/physics_assistant_backup_$TIMESTAMP.tar.gz"
    local final_size=$(calculate_size "$final_archive")

    print_status "🎉 Backup completed successfully!"
    print_status "📦 Archive: $final_archive"
    print_status "📊 Size: $final_size"

    log_backup "Full backup completed successfully - $final_size"
}

# Function to show help
show_help() {
    echo "Physics Assistant Backup System"
    echo "==============================="
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  backup      Create full backup (default)"
    echo "  list        List existing backups"
    echo "  restore     Restore from backup file"
    echo "  help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Create full backup"
    echo "  $0 backup                                    # Create full backup"
    echo "  $0 list                                      # List all backups"
    echo "  $0 restore backups/physics_assistant_backup_20250916_143000.tar.gz"
    echo ""
    echo "Configuration:"
    echo "  Backup directory: $BACKUP_BASE_DIR"
    echo "  Retention: $RETENTION_DAYS days"
    echo "  Components: config($BACKUP_CONFIG) data($BACKUP_DATA) logs($BACKUP_LOGS) containers($BACKUP_CONTAINERS)"
}

# Main function
main() {
    case "${1:-backup}" in
        "backup")
            run_full_backup
            ;;
        "list")
            list_backups
            ;;
        "restore")
            restore_backup "$2"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"