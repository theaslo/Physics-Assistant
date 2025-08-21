#!/bin/bash
# Start HashiCorp Vault with initialization and unsealing

set -euo pipefail

# Configuration
VAULT_CONFIG_PATH=${VAULT_CONFIG_PATH:-/vault/config}
VAULT_DATA_PATH=${VAULT_DATA_PATH:-/vault/data}
VAULT_LOGS_PATH=${VAULT_LOGS_PATH:-/vault/logs}
VAULT_ADDR=${VAULT_ADDR:-http://127.0.0.1:8200}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${VAULT_LOGS_PATH}/vault.log"
}

# Function to initialize Vault
initialize_vault() {
    log "Initializing Vault..."
    
    # Check if Vault is already initialized
    if vault status 2>/dev/null | grep -q "Initialized.*true"; then
        log "Vault is already initialized"
        return 0
    fi
    
    # Initialize Vault with 5 key shares and threshold of 3
    local init_output
    init_output=$(vault operator init \
        -key-shares=5 \
        -key-threshold=3 \
        -format=json 2>/dev/null)
    
    # Save unseal keys and root token securely
    echo "${init_output}" | jq -r '.unseal_keys_b64[]' > "${VAULT_DATA_PATH}/unseal-keys.txt"
    echo "${init_output}" | jq -r '.root_token' > "${VAULT_DATA_PATH}/root-token.txt"
    
    # Set restrictive permissions
    chmod 600 "${VAULT_DATA_PATH}/unseal-keys.txt" "${VAULT_DATA_PATH}/root-token.txt"
    
    log "Vault initialized successfully"
}

# Function to unseal Vault
unseal_vault() {
    log "Unsealing Vault..."
    
    # Check if Vault is already unsealed
    if vault status 2>/dev/null | grep -q "Sealed.*false"; then
        log "Vault is already unsealed"
        return 0
    fi
    
    # Read unseal keys
    if [[ ! -f "${VAULT_DATA_PATH}/unseal-keys.txt" ]]; then
        log "ERROR: Unseal keys not found"
        return 1
    fi
    
    # Unseal Vault with first 3 keys
    local key_count=0
    while IFS= read -r key && [[ ${key_count} -lt 3 ]]; do
        vault operator unseal "${key}"
        ((key_count++))
    done < "${VAULT_DATA_PATH}/unseal-keys.txt"
    
    log "Vault unsealed successfully"
}

# Function to configure Vault
configure_vault() {
    log "Configuring Vault..."
    
    # Authenticate with root token
    local root_token
    root_token=$(cat "${VAULT_DATA_PATH}/root-token.txt")
    vault auth "${root_token}"
    
    # Enable audit logging
    vault audit enable file file_path="${VAULT_LOGS_PATH}/audit.log" || true
    
    # Enable auth methods
    vault auth enable approle || true
    vault auth enable userpass || true
    
    # Enable secrets engines
    vault secrets enable -path=physics-assistant kv-v2 || true
    vault secrets enable -path=database database || true
    vault secrets enable pki || true
    vault secrets enable transit || true
    
    # Configure PKI for TLS certificates
    vault secrets tune -max-lease-ttl=87600h pki || true
    vault write pki/root/generate/internal \
        common_name="Physics Assistant CA" \
        ttl=87600h || true
    
    # Configure database secrets engine
    vault write database/config/postgres \
        plugin_name=postgresql-database-plugin \
        connection_url="postgresql://{{username}}:{{password}}@postgres:5432/postgres?sslmode=require" \
        allowed_roles="physics-db-role" \
        username="${DB_ADMIN_USER}" \
        password="${DB_ADMIN_PASSWORD}" || true
    
    # Create database role for dynamic credentials
    vault write database/roles/physics-db-role \
        db_name=postgres \
        creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
        default_ttl="1h" \
        max_ttl="24h" || true
    
    log "Vault configuration completed"
}

# Function to setup policies
setup_policies() {
    log "Setting up Vault policies..."
    
    # Apply policies from files
    for policy_file in "${VAULT_CONFIG_PATH}"/policies/*.hcl; do
        if [[ -f "${policy_file}" ]]; then
            local policy_name
            policy_name=$(basename "${policy_file}" .hcl)
            vault policy write "${policy_name}" "${policy_file}"
            log "Applied policy: ${policy_name}"
        fi
    done
}

# Function to create application roles
create_app_roles() {
    log "Creating application roles..."
    
    # Create role for database API
    vault write auth/approle/role/database-api \
        token_policies="database-api-policy" \
        token_ttl=1h \
        token_max_ttl=4h \
        bind_secret_id=true
    
    # Create role for dashboard API
    vault write auth/approle/role/dashboard-api \
        token_policies="dashboard-api-policy" \
        token_ttl=1h \
        token_max_ttl=4h \
        bind_secret_id=true
    
    # Create role for physics agents
    vault write auth/approle/role/physics-agents \
        token_policies="physics-agents-policy" \
        token_ttl=1h \
        token_max_ttl=4h \
        bind_secret_id=true
    
    log "Application roles created"
}

# Function to store initial secrets
store_initial_secrets() {
    log "Storing initial secrets..."
    
    # Store database passwords
    vault kv put physics-assistant/database \
        postgres_password="${POSTGRES_PASSWORD}" \
        physics_db_password="${PHYSICS_DB_PASSWORD}" \
        neo4j_password="${NEO4J_PASSWORD}" \
        redis_password="${REDIS_PASSWORD}"
    
    # Store API keys and tokens
    vault kv put physics-assistant/api-keys \
        ollama_api_key="${OLLAMA_API_KEY:-}" \
        monitoring_token="${MONITORING_TOKEN:-}" \
        grafana_admin_password="${GRAFANA_ADMIN_PASSWORD}"
    
    # Store TLS certificates
    vault kv put physics-assistant/tls \
        ca_cert="@${VAULT_CONFIG_PATH}/tls/ca.crt" \
        server_cert="@${VAULT_CONFIG_PATH}/tls/server.crt" \
        server_key="@${VAULT_CONFIG_PATH}/tls/server.key"
    
    log "Initial secrets stored"
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up Vault monitoring..."
    
    # Enable Prometheus metrics
    vault write sys/metrics/config \
        enabled=true \
        enable_hostname_label=false \
        prometheus_retention_time="30s"
    
    log "Vault monitoring configured"
}

# Main execution
main() {
    log "Starting HashiCorp Vault for Physics Assistant"
    
    # Create necessary directories
    mkdir -p "${VAULT_DATA_PATH}" "${VAULT_LOGS_PATH}"
    
    # Start Vault server in background
    vault server -config="${VAULT_CONFIG_PATH}/vault.hcl" &
    local vault_pid=$!
    
    # Wait for Vault to start
    log "Waiting for Vault to start..."
    local retry_count=0
    while ! vault status >/dev/null 2>&1 && [[ ${retry_count} -lt 30 ]]; do
        sleep 2
        ((retry_count++))
    done
    
    if [[ ${retry_count} -eq 30 ]]; then
        log "ERROR: Vault failed to start within timeout"
        exit 1
    fi
    
    # Initialize and configure Vault
    initialize_vault
    unseal_vault
    configure_vault
    setup_policies
    create_app_roles
    store_initial_secrets
    setup_monitoring
    
    log "Vault setup completed successfully"
    log "Vault server running with PID: ${vault_pid}"
    
    # Keep the container running
    wait ${vault_pid}
}

# Execute main function
main "$@"