# HashiCorp Vault for Secrets Management
FROM hashicorp/vault:1.15

# Install additional tools
USER root
RUN apk add --no-cache \
    bash \
    curl \
    jq \
    openssl

# Create vault directories
RUN mkdir -p /vault/config /vault/policies /vault/scripts /vault/data /vault/logs

# Copy vault configuration and scripts
COPY security/vault/config/ /vault/config/
COPY security/vault/policies/ /vault/policies/
COPY security/vault/scripts/ /vault/scripts/

# Set permissions
RUN chmod +x /vault/scripts/*.sh && \
    chown -R vault:vault /vault

USER vault

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD vault status || exit 1

# Expose Vault port
EXPOSE 8200

# Set environment variables
ENV VAULT_ADDR=http://127.0.0.1:8200
ENV VAULT_CONFIG_PATH=/vault/config

CMD ["/vault/scripts/start-vault.sh"]