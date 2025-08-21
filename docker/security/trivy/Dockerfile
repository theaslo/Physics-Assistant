# Trivy Security Scanner Container
FROM aquasec/trivy:latest

# Install additional tools for scanning
USER root
RUN apk add --no-cache \
    bash \
    curl \
    jq \
    git

# Create scanner directories
RUN mkdir -p /scanner/reports /scanner/policies /scanner/scripts

# Copy security scanning scripts
COPY security/trivy/scripts/ /scanner/scripts/
COPY security/trivy/policies/ /scanner/policies/

# Set permissions
RUN chmod +x /scanner/scripts/*.sh

# Create non-root user for scanner
RUN addgroup -g 1001 scanner && \
    adduser -D -u 1001 -G scanner scanner && \
    chown -R scanner:scanner /scanner

USER scanner
WORKDIR /scanner

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD trivy version || exit 1

CMD ["/scanner/scripts/scan-all.sh"]