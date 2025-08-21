# HashiCorp Vault Configuration for Physics Assistant
# Production-ready configuration with high availability

ui = true
disable_mlock = true

# Storage backend - using PostgreSQL for production
storage "postgresql" {
  connection_url = "postgres://vault_user:${VAULT_DB_PASSWORD}@postgres:5432/vault_db?sslmode=require"
  table          = "vault_kv_store"
  max_parallel   = 128
}

# Listener configuration with TLS
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_cert_file = "/vault/config/tls/vault.crt"
  tls_key_file  = "/vault/config/tls/vault.key"
  tls_min_version = "tls12"
  tls_cipher_suites = "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384"
}

# API listener for internal services
listener "tcp" {
  address       = "0.0.0.0:8201"
  tls_disable   = true
}

# Cluster configuration for HA
cluster_addr = "https://vault:8201"
api_addr = "https://vault:8200"

# Telemetry for monitoring
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
  enable_hostname_label = false
}

# Performance and security settings
max_lease_ttl = "768h"
default_lease_ttl = "768h"

# Enable audit logging
audit {
  file {
    file_path = "/vault/logs/audit.log"
    log_raw = false
    format = "json"
  }
}

# Plugin directory
plugin_directory = "/vault/plugins"

# Seal configuration for auto-unseal (optional)
# seal "transit" {
#   address            = "https://vault-cluster:8200"
#   token              = "${VAULT_TRANSIT_TOKEN}"
#   disable_renewal    = "false"
#   key_name           = "physics-assistant-seal"
#   mount_path         = "transit/"
# }

# License path (for Enterprise)
# license_path = "/vault/config/vault.hclic"