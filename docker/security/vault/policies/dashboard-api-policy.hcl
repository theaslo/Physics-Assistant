# Vault Policy for Dashboard API Service

# Allow reading database credentials
path "physics-assistant/data/database" {
  capabilities = ["read"]
}

# Allow reading monitoring credentials
path "physics-assistant/data/monitoring" {
  capabilities = ["read"]
}

# Allow reading API keys
path "physics-assistant/data/api-keys" {
  capabilities = ["read"]
}

# Allow reading TLS certificates
path "physics-assistant/data/tls" {
  capabilities = ["read"]
}

# Allow generating dynamic database credentials
path "database/creds/physics-db-role" {
  capabilities = ["read"]
}

# Allow token self-renewal
path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Allow token self-lookup
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

# Allow AppRole authentication
path "auth/approle/login" {
  capabilities = ["create", "update"]
}