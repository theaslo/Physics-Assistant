#!/bin/bash

# S3 upload utilities for backup files

# Source common utilities
source /scripts/common/utils.sh

# S3 configuration
S3_ENDPOINT="${S3_ENDPOINT:-}"
S3_STORAGE_CLASS="${S3_STORAGE_CLASS:-STANDARD_IA}"
S3_ENCRYPTION="${S3_ENCRYPTION:-AES256}"

# Initialize S3 configuration
init_s3() {
    log_info "Initializing S3 configuration"
    
    # Validate required S3 environment variables
    local required_vars=(
        "AWS_ACCESS_KEY_ID"
        "AWS_SECRET_ACCESS_KEY"
        "AWS_S3_BUCKET"
        "AWS_S3_REGION"
    )
    
    if ! validate_env_vars "${required_vars[@]}"; then
        log_error "S3 configuration incomplete"
        return 1
    fi
    
    # Configure AWS CLI
    aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
    aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
    aws configure set default.region "$AWS_S3_REGION"
    
    # Set custom endpoint if provided (for S3-compatible services)
    if [[ -n "$S3_ENDPOINT" ]]; then
        aws configure set default.s3.endpoint_url "$S3_ENDPOINT"
    fi
    
    # Test S3 connectivity
    if test_s3_connectivity; then
        log_info "S3 configuration successful"
        return 0
    else
        log_error "S3 connectivity test failed"
        return 1
    fi
}

# Test S3 connectivity
test_s3_connectivity() {
    log_info "Testing S3 connectivity"
    
    if aws s3 ls "s3://$AWS_S3_BUCKET/" >/dev/null 2>&1; then
        log_info "S3 connectivity test passed"
        return 0
    else
        log_error "S3 connectivity test failed"
        return 1
    fi
}

# Upload file to S3
upload_to_s3() {
    local local_file=$1
    local s3_prefix=$2
    local storage_class=${3:-$S3_STORAGE_CLASS}
    
    if [[ ! -f "$local_file" ]]; then
        log_error "Local file not found: $local_file"
        return 1
    fi
    
    # Initialize S3 if not already done
    if ! aws s3 ls "s3://$AWS_S3_BUCKET/" >/dev/null 2>&1; then
        if ! init_s3; then
            log_error "Failed to initialize S3"
            return 1
        fi
    fi
    
    local filename=$(basename "$local_file")
    local s3_key="${s3_prefix}${filename}"
    local s3_url="s3://$AWS_S3_BUCKET/$s3_key"
    
    log_info "Uploading $local_file to $s3_url"
    
    # Calculate file size for progress
    local file_size=$(stat -f%z "$local_file" 2>/dev/null || stat -c%s "$local_file")
    local file_size_human=$(human_readable_size "$file_size")
    
    # Upload with retry logic
    local max_attempts=3
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if aws s3 cp "$local_file" "$s3_url" \
            --storage-class "$storage_class" \
            --server-side-encryption "$S3_ENCRYPTION" \
            --metadata "source=physics-assistant-backup,timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            2>/dev/null; then
            
            log_info "Upload successful (${file_size_human}) - attempt $attempt"
            
            # Verify upload
            if verify_s3_upload "$s3_url" "$file_size"; then
                log_info "Upload verification passed"
                
                # Create upload metadata
                create_s3_metadata "$local_file" "$s3_url" "$storage_class"
                
                return 0
            else
                log_error "Upload verification failed"
                return 1
            fi
        else
            log_warn "Upload failed - attempt $attempt/$max_attempts"
            if [[ $attempt -lt $max_attempts ]]; then
                sleep $((attempt * 10))  # Increasing delay
            fi
            ((attempt++))
        fi
    done
    
    log_error "Upload failed after $max_attempts attempts"
    return 1
}

# Verify S3 upload
verify_s3_upload() {
    local s3_url=$1
    local expected_size=$2
    
    log_info "Verifying S3 upload: $s3_url"
    
    # Get object info from S3
    local s3_info=$(aws s3api head-object --bucket "$AWS_S3_BUCKET" --key "${s3_url#s3://$AWS_S3_BUCKET/}" 2>/dev/null)
    
    if [[ $? -eq 0 ]]; then
        local s3_size=$(echo "$s3_info" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('ContentLength', 0))
except:
    print(0)
")
        
        if [[ "$s3_size" == "$expected_size" ]]; then
            log_info "S3 upload verification passed - sizes match"
            return 0
        else
            log_error "S3 upload verification failed - size mismatch"
            log_error "Expected: $expected_size, Got: $s3_size"
            return 1
        fi
    else
        log_error "Failed to get S3 object info"
        return 1
    fi
}

# Create S3 upload metadata
create_s3_metadata() {
    local local_file=$1
    local s3_url=$2
    local storage_class=$3
    local metadata_file="${local_file}.s3.json"
    
    local file_size=$(stat -f%z "$local_file" 2>/dev/null || stat -c%s "$local_file")
    local checksum=$(calculate_checksum "$local_file" "md5")
    
    cat > "$metadata_file" << EOF
{
    "local_file": "$(basename "$local_file")",
    "s3_url": "$s3_url",
    "s3_bucket": "$AWS_S3_BUCKET",
    "s3_key": "${s3_url#s3://$AWS_S3_BUCKET/}",
    "storage_class": "$storage_class",
    "upload_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "file_size": $file_size,
    "md5_checksum": "$checksum",
    "aws_region": "$AWS_S3_REGION"
}
EOF
    
    log_info "S3 upload metadata created: $metadata_file"
}

# Download file from S3
download_from_s3() {
    local s3_url=$1
    local local_file=$2
    local verify_checksum=${3:-true}
    
    if [[ -z "$local_file" ]]; then
        local_file=$(basename "${s3_url#s3://$AWS_S3_BUCKET/}")
    fi
    
    log_info "Downloading $s3_url to $local_file"
    
    # Download with retry logic
    local max_attempts=3
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if aws s3 cp "$s3_url" "$local_file" 2>/dev/null; then
            log_info "Download successful - attempt $attempt"
            
            # Verify download if checksum available
            if [[ "$verify_checksum" == "true" ]]; then
                local metadata_file="${local_file}.s3.json"
                if [[ -f "$metadata_file" ]]; then
                    local expected_checksum=$(python3 -c "
import json
try:
    with open('$metadata_file', 'r') as f:
        data = json.load(f)
    print(data.get('md5_checksum', ''))
except:
    print('')
")
                    
                    if [[ -n "$expected_checksum" ]]; then
                        if verify_checksum "$local_file" "$expected_checksum" "md5"; then
                            log_info "Download verification passed"
                        else
                            log_error "Download verification failed"
                            rm -f "$local_file"
                            return 1
                        fi
                    fi
                fi
            fi
            
            return 0
        else
            log_warn "Download failed - attempt $attempt/$max_attempts"
            if [[ $attempt -lt $max_attempts ]]; then
                sleep $((attempt * 10))
            fi
            ((attempt++))
        fi
    done
    
    log_error "Download failed after $max_attempts attempts"
    return 1
}

# List S3 backups
list_s3_backups() {
    local prefix=${1:-""}
    local output_format=${2:-"table"}
    
    log_info "Listing S3 backups with prefix: $prefix"
    
    case "$output_format" in
        "json")
            aws s3api list-objects-v2 \
                --bucket "$AWS_S3_BUCKET" \
                --prefix "$prefix" \
                --query 'Contents[].{Key:Key,Size:Size,LastModified:LastModified,StorageClass:StorageClass}' \
                --output json
            ;;
        "table")
            aws s3 ls "s3://$AWS_S3_BUCKET/$prefix" --recursive --human-readable --summarize
            ;;
        "simple")
            aws s3 ls "s3://$AWS_S3_BUCKET/$prefix" --recursive | awk '{print $4}'
            ;;
        *)
            log_error "Unknown output format: $output_format"
            return 1
            ;;
    esac
}

# Delete old S3 backups based on retention policy
cleanup_s3_backups() {
    local prefix=$1
    local retention_days=${2:-30}
    
    log_info "Cleaning up S3 backups older than $retention_days days with prefix: $prefix"
    
    # Get list of objects older than retention period
    local cutoff_date=$(date -d "$retention_days days ago" -u +%Y-%m-%dT%H:%M:%SZ)
    
    local old_objects=$(aws s3api list-objects-v2 \
        --bucket "$AWS_S3_BUCKET" \
        --prefix "$prefix" \
        --query "Contents[?LastModified<'$cutoff_date'].Key" \
        --output text)
    
    if [[ -n "$old_objects" ]] && [[ "$old_objects" != "None" ]]; then
        local delete_count=0
        
        for object_key in $old_objects; do
            log_info "Deleting old backup: s3://$AWS_S3_BUCKET/$object_key"
            
            if aws s3 rm "s3://$AWS_S3_BUCKET/$object_key" >/dev/null 2>&1; then
                ((delete_count++))
            else
                log_error "Failed to delete: s3://$AWS_S3_BUCKET/$object_key"
            fi
        done
        
        log_info "Deleted $delete_count old backups from S3"
    else
        log_info "No old backups found for cleanup"
    fi
}

# Sync directory to S3
sync_directory_to_s3() {
    local local_dir=$1
    local s3_prefix=$2
    local delete_removed=${3:-false}
    
    if [[ ! -d "$local_dir" ]]; then
        log_error "Local directory not found: $local_dir"
        return 1
    fi
    
    log_info "Syncing directory $local_dir to s3://$AWS_S3_BUCKET/$s3_prefix"
    
    local sync_args=(
        "$local_dir"
        "s3://$AWS_S3_BUCKET/$s3_prefix"
        --storage-class "$S3_STORAGE_CLASS"
        --server-side-encryption "$S3_ENCRYPTION"
    )
    
    if [[ "$delete_removed" == "true" ]]; then
        sync_args+=(--delete)
    fi
    
    if aws s3 sync "${sync_args[@]}" 2>/dev/null; then
        log_info "Directory sync completed successfully"
        return 0
    else
        log_error "Directory sync failed"
        return 1
    fi
}

# Get S3 usage statistics
get_s3_usage_stats() {
    local prefix=${1:-""}
    
    log_info "Getting S3 usage statistics for prefix: $prefix"
    
    local objects_info=$(aws s3api list-objects-v2 \
        --bucket "$AWS_S3_BUCKET" \
        --prefix "$prefix" \
        --query 'Contents[].{Size:Size,StorageClass:StorageClass}' \
        --output json)
    
    if [[ -n "$objects_info" ]] && [[ "$objects_info" != "null" ]]; then
        python3 << EOF
import json

data = json.loads('$objects_info')
total_size = 0
object_count = 0
storage_classes = {}

for obj in data:
    size = obj.get('Size', 0)
    storage_class = obj.get('StorageClass', 'STANDARD')
    
    total_size += size
    object_count += 1
    
    if storage_class in storage_classes:
        storage_classes[storage_class] += size
    else:
        storage_classes[storage_class] = size

print(f"Total objects: {object_count}")
print(f"Total size: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")
print("Storage class breakdown:")
for storage_class, size in storage_classes.items():
    print(f"  {storage_class}: {size:,} bytes ({size / (1024**3):.2f} GB)")
EOF
    else
        log_info "No objects found with prefix: $prefix"
    fi
}

# Create lifecycle policy for S3 bucket
create_s3_lifecycle_policy() {
    local transition_days_ia=${1:-30}
    local transition_days_glacier=${2:-90}
    local deletion_days=${3:-365}
    
    log_info "Creating S3 lifecycle policy"
    
    local policy_file=$(create_temp_file "s3_lifecycle" ".json")
    
    cat > "$policy_file" << EOF
{
    "Rules": [
        {
            "ID": "PhysicsAssistantBackupLifecycle",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "physics-assistant/"
            },
            "Transitions": [
                {
                    "Days": $transition_days_ia,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": $transition_days_glacier,
                    "StorageClass": "GLACIER"
                }
            ],
            "Expiration": {
                "Days": $deletion_days
            }
        }
    ]
}
EOF
    
    if aws s3api put-bucket-lifecycle-configuration \
        --bucket "$AWS_S3_BUCKET" \
        --lifecycle-configuration "file://$policy_file" 2>/dev/null; then
        
        log_info "S3 lifecycle policy created successfully"
        log_info "Transition to IA: $transition_days_ia days"
        log_info "Transition to Glacier: $transition_days_glacier days"
        log_info "Deletion: $deletion_days days"
        
        rm -f "$policy_file"
        return 0
    else
        log_error "Failed to create S3 lifecycle policy"
        rm -f "$policy_file"
        return 1
    fi
}