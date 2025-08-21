# Vault Policy for Physics Agents Service

# Allow reading agent-specific configurations
path "physics-assistant/data/agents/*" {
  capabilities = ["read"]
}

# Allow reading model credentials
path "physics-assistant/data/models" {
  capabilities = ["read"]
}

# Allow reading API keys
path "physics-assistant/data/api-keys" {
  capabilities = ["read"]
}

# Allow reading MCP service configurations
path "physics-assistant/data/mcp/*" {
  capabilities = ["read"]
}

# Allow reading TLS certificates
path "physics-assistant/data/tls" {
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