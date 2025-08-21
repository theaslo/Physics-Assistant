#!/bin/bash

# Encryption utilities for backup files

# Source common utilities
source /scripts/common/utils.sh

# Encryption configuration
ENCRYPTION_ALGORITHM="${ENCRYPTION_ALGORITHM:-aes-256-cbc}"
ENCRYPTION_KEY_FILE="${ENCRYPTION_KEY_FILE:-/secrets/backup_encryption.key}"
ENCRYPTION_IV_FILE="${ENCRYPTION_IV_FILE:-/secrets/backup_encryption.iv}"

# Initialize encryption keys
init_encryption() {
    log_info "Initializing encryption system"
    
    # Create secrets directory
    sudo mkdir -p /secrets
    sudo chmod 700 /secrets
    
    # Generate encryption key if it doesn't exist
    if [[ ! -f "$ENCRYPTION_KEY_FILE" ]]; then
        log_info "Generating new encryption key"
        openssl rand -hex 32 | sudo tee "$ENCRYPTION_KEY_FILE" >/dev/null
        sudo chmod 600 "$ENCRYPTION_KEY_FILE"
        sudo chown backup:backup "$ENCRYPTION_KEY_FILE"
    fi
    
    # Generate IV if it doesn't exist
    if [[ ! -f "$ENCRYPTION_IV_FILE" ]]; then
        log_info "Generating new initialization vector"
        openssl rand -hex 16 | sudo tee "$ENCRYPTION_IV_FILE" >/dev/null
        sudo chmod 600 "$ENCRYPTION_IV_FILE"
        sudo chown backup:backup "$ENCRYPTION_IV_FILE"
    fi
    
    log_info "Encryption system initialized"
}

# Encrypt a file
encrypt_file() {
    local input_file=$1
    local output_file="${input_file}.enc"
    
    if [[ ! -f "$input_file" ]]; then
        log_error "Input file not found: $input_file"
        return 1
    fi
    
    # Initialize encryption if needed
    if [[ ! -f "$ENCRYPTION_KEY_FILE" ]] || [[ ! -f "$ENCRYPTION_IV_FILE" ]]; then
        init_encryption
    fi
    
    log_info "Encrypting file: $input_file"
    
    # Read encryption key and IV
    local encryption_key=$(sudo cat "$ENCRYPTION_KEY_FILE")
    local encryption_iv=$(sudo cat "$ENCRYPTION_IV_FILE")
    
    if [[ -z "$encryption_key" ]] || [[ -z "$encryption_iv" ]]; then
        log_error "Failed to read encryption key or IV"
        return 1
    fi
    
    # Encrypt the file
    if openssl enc -"$ENCRYPTION_ALGORITHM" -e \
        -K "$encryption_key" \
        -iv "$encryption_iv" \
        -in "$input_file" \
        -out "$output_file" 2>/dev/null; then
        
        # Verify encryption was successful
        if [[ -f "$output_file" ]] && [[ -s "$output_file" ]]; then
            log_info "File encrypted successfully: $output_file"
            
            # Remove original file if encryption was successful
            rm -f "$input_file"
            
            # Create metadata file with encryption info
            create_encryption_metadata "$output_file" "$input_file"
            
            return 0
        else
            log_error "Encryption failed - output file is empty or missing"
            rm -f "$output_file"
            return 1
        fi
    else
        log_error "Encryption failed for file: $input_file"
        return 1
    fi
}

# Decrypt a file
decrypt_file() {
    local input_file=$1
    local output_file=$2
    
    if [[ ! -f "$input_file" ]]; then
        log_error "Encrypted file not found: $input_file"
        return 1
    fi
    
    # If output file not specified, remove .enc extension
    if [[ -z "$output_file" ]]; then
        output_file="${input_file%.enc}"
    fi
    
    log_info "Decrypting file: $input_file"
    
    # Read encryption key and IV
    local encryption_key=$(sudo cat "$ENCRYPTION_KEY_FILE" 2>/dev/null)
    local encryption_iv=$(sudo cat "$ENCRYPTION_IV_FILE" 2>/dev/null)
    
    if [[ -z "$encryption_key" ]] || [[ -z "$encryption_iv" ]]; then
        log_error "Failed to read encryption key or IV"
        return 1
    fi
    
    # Decrypt the file
    if openssl enc -"$ENCRYPTION_ALGORITHM" -d \
        -K "$encryption_key" \
        -iv "$encryption_iv" \
        -in "$input_file" \
        -out "$output_file" 2>/dev/null; then
        
        # Verify decryption was successful
        if [[ -f "$output_file" ]] && [[ -s "$output_file" ]]; then
            log_info "File decrypted successfully: $output_file"
            return 0
        else
            log_error "Decryption failed - output file is empty or missing"
            rm -f "$output_file"
            return 1
        fi
    else
        log_error "Decryption failed for file: $input_file"
        return 1
    fi
}

# Create encryption metadata
create_encryption_metadata() {
    local encrypted_file=$1
    local original_file=$2
    local metadata_file="${encrypted_file}.metadata.json"
    
    local file_size=$(stat -f%z "$encrypted_file" 2>/dev/null || stat -c%s "$encrypted_file")
    local checksum=$(calculate_checksum "$encrypted_file" "sha256")
    
    cat > "$metadata_file" << EOF
{
    "encrypted_file": "$(basename "$encrypted_file")",
    "original_file": "$(basename "$original_file")",
    "algorithm": "$ENCRYPTION_ALGORITHM",
    "encrypted_size": $file_size,
    "encrypted_checksum": "$checksum",
    "encryption_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "encryption_version": "1.0"
}
EOF
    
    log_info "Encryption metadata created: $metadata_file"
}

# Verify encrypted file integrity
verify_encrypted_file() {
    local encrypted_file=$1
    local metadata_file="${encrypted_file}.metadata.json"
    
    if [[ ! -f "$metadata_file" ]]; then
        log_warn "No metadata file found for $encrypted_file, creating one..."
        create_encryption_metadata "$encrypted_file" "${encrypted_file%.enc}"
        return 0
    fi
    
    log_info "Verifying encrypted file integrity: $encrypted_file"
    
    # Read expected checksum from metadata
    local expected_checksum=$(python3 -c "
import json
try:
    with open('$metadata_file', 'r') as f:
        data = json.load(f)
    print(data.get('encrypted_checksum', ''))
except:
    print('')
")
    
    if [[ -z "$expected_checksum" ]]; then
        log_warn "No checksum found in metadata, skipping verification"
        return 0
    fi
    
    # Calculate current checksum
    local current_checksum=$(calculate_checksum "$encrypted_file" "sha256")
    
    if [[ "$current_checksum" == "$expected_checksum" ]]; then
        log_info "Encrypted file integrity verification passed"
        return 0
    else
        log_error "Encrypted file integrity verification failed"
        log_error "Expected: $expected_checksum"
        log_error "Current: $current_checksum"
        return 1
    fi
}

# Encrypt directory (tar + encrypt)
encrypt_directory() {
    local input_dir=$1
    local output_file=$2
    
    if [[ ! -d "$input_dir" ]]; then
        log_error "Input directory not found: $input_dir"
        return 1
    fi
    
    if [[ -z "$output_file" ]]; then
        output_file="${input_dir}.tar.enc"
    fi
    
    log_info "Encrypting directory: $input_dir"
    
    # Create temporary tar file
    local temp_tar=$(create_temp_file "dir_backup" ".tar")
    
    # Create tar archive
    if tar -cf "$temp_tar" -C "$(dirname "$input_dir")" "$(basename "$input_dir")"; then
        log_info "Directory archived successfully"
        
        # Encrypt the tar file
        if encrypt_file "$temp_tar"; then
            mv "${temp_tar}.enc" "$output_file"
            log_info "Directory encrypted successfully: $output_file"
            return 0
        else
            log_error "Failed to encrypt directory archive"
            rm -f "$temp_tar"
            return 1
        fi
    else
        log_error "Failed to create tar archive of directory: $input_dir"
        rm -f "$temp_tar"
        return 1
    fi
}

# Decrypt directory (decrypt + untar)
decrypt_directory() {
    local input_file=$1
    local output_dir=$2
    
    if [[ ! -f "$input_file" ]]; then
        log_error "Encrypted file not found: $input_file"
        return 1
    fi
    
    log_info "Decrypting directory archive: $input_file"
    
    # Create temporary file for decrypted tar
    local temp_tar=$(create_temp_file "dir_restore" ".tar")
    
    # Decrypt the file
    if decrypt_file "$input_file" "$temp_tar"; then
        log_info "Archive decrypted successfully"
        
        # Extract tar archive
        if [[ -n "$output_dir" ]]; then
            mkdir -p "$output_dir"
            if tar -xf "$temp_tar" -C "$output_dir"; then
                log_info "Directory extracted successfully to: $output_dir"
                rm -f "$temp_tar"
                return 0
            else
                log_error "Failed to extract directory archive"
                rm -f "$temp_tar"
                return 1
            fi
        else
            # Extract to current directory
            if tar -xf "$temp_tar"; then
                log_info "Directory extracted successfully"
                rm -f "$temp_tar"
                return 0
            else
                log_error "Failed to extract directory archive"
                rm -f "$temp_tar"
                return 1
            fi
        fi
    else
        log_error "Failed to decrypt directory archive"
        rm -f "$temp_tar"
        return 1
    fi
}

# Rotate encryption keys
rotate_encryption_keys() {
    log_info "Starting encryption key rotation"
    
    # Backup current keys
    local backup_suffix=$(date +%Y%m%d_%H%M%S)
    if [[ -f "$ENCRYPTION_KEY_FILE" ]]; then
        sudo cp "$ENCRYPTION_KEY_FILE" "${ENCRYPTION_KEY_FILE}.backup_${backup_suffix}"
        log_info "Current encryption key backed up"
    fi
    
    if [[ -f "$ENCRYPTION_IV_FILE" ]]; then
        sudo cp "$ENCRYPTION_IV_FILE" "${ENCRYPTION_IV_FILE}.backup_${backup_suffix}"
        log_info "Current encryption IV backed up"
    fi
    
    # Generate new keys
    openssl rand -hex 32 | sudo tee "$ENCRYPTION_KEY_FILE" >/dev/null
    openssl rand -hex 16 | sudo tee "$ENCRYPTION_IV_FILE" >/dev/null
    
    # Set proper permissions
    sudo chmod 600 "$ENCRYPTION_KEY_FILE" "$ENCRYPTION_IV_FILE"
    sudo chown backup:backup "$ENCRYPTION_KEY_FILE" "$ENCRYPTION_IV_FILE"
    
    log_info "New encryption keys generated"
    log_warn "Note: Files encrypted with old keys will need to be re-encrypted"
}

# Test encryption/decryption
test_encryption() {
    log_info "Testing encryption/decryption functionality"
    
    # Create test file
    local test_file=$(create_temp_file "encryption_test" ".txt")
    echo "This is a test file for encryption verification - $(date)" > "$test_file"
    local original_checksum=$(calculate_checksum "$test_file" "md5")
    
    # Test encryption
    if encrypt_file "$test_file"; then
        local encrypted_file="${test_file}.enc"
        log_info "Encryption test passed"
        
        # Test decryption
        if decrypt_file "$encrypted_file" "${test_file}.decrypted"; then
            local decrypted_checksum=$(calculate_checksum "${test_file}.decrypted" "md5")
            
            if [[ "$original_checksum" == "$decrypted_checksum" ]]; then
                log_info "Decryption test passed - checksums match"
                
                # Cleanup test files
                rm -f "$encrypted_file" "${test_file}.decrypted" "${encrypted_file}.metadata.json"
                
                return 0
            else
                log_error "Decryption test failed - checksums don't match"
                return 1
            fi
        else
            log_error "Decryption test failed"
            return 1
        fi
    else
        log_error "Encryption test failed"
        return 1
    fi
}

# Get encryption status
get_encryption_status() {
    local file_path=$1
    
    if [[ ! -f "$file_path" ]]; then
        echo "FILE_NOT_FOUND"
        return 1
    fi
    
    # Check if file has .enc extension
    if [[ "$file_path" == *.enc ]]; then
        # Verify it's actually encrypted by checking metadata
        local metadata_file="${file_path}.metadata.json"
        if [[ -f "$metadata_file" ]]; then
            echo "ENCRYPTED"
        else
            echo "POSSIBLY_ENCRYPTED"
        fi
    else
        echo "NOT_ENCRYPTED"
    fi
    
    return 0
}

# Bulk encrypt files in directory
bulk_encrypt_directory() {
    local source_dir=$1
    local pattern=${2:-"*"}
    
    if [[ ! -d "$source_dir" ]]; then
        log_error "Source directory not found: $source_dir"
        return 1
    fi
    
    log_info "Starting bulk encryption in directory: $source_dir"
    
    local encrypted_count=0
    local failed_count=0
    
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]] && [[ "$file" != *.enc ]] && [[ "$file" != *.metadata.json ]]; then
            log_info "Encrypting: $file"
            if encrypt_file "$file"; then
                ((encrypted_count++))
            else
                ((failed_count++))
                log_error "Failed to encrypt: $file"
            fi
        fi
    done < <(find "$source_dir" -name "$pattern" -type f -print0)
    
    log_info "Bulk encryption completed. Encrypted: $encrypted_count, Failed: $failed_count"
    
    if [[ $failed_count -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}