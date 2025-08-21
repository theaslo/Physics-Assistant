# Redis Backup Container
FROM redis:7-alpine

# Install required packages
RUN apk add --no-cache \
    bash \
    curl \
    aws-cli \
    gnupg \
    dcron \
    tzdata \
    python3 \
    py3-pip

# Install Python packages for monitoring and notifications
RUN pip3 install --no-cache-dir \
    requests \
    psutil \
    prometheus-client \
    boto3 \
    redis

# Create backup user and directories
RUN adduser -D -s /bin/bash backup && \
    mkdir -p /backups /scripts /logs /tmp/backup && \
    chown -R backup:backup /backups /scripts /logs /tmp/backup

# Copy backup scripts
COPY backup/scripts/redis/ /scripts/
COPY backup/scripts/common/ /scripts/common/

# Make scripts executable
RUN chmod +x /scripts/*.sh /scripts/common/*.sh

# Set up sudo for backup user
RUN echo "backup ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Copy entrypoint
COPY backup/containers/redis-backup/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
    CMD /scripts/common/health_check.sh redis

USER backup
WORKDIR /backups

# Expose metrics port
EXPOSE 8082

ENTRYPOINT ["/entrypoint.sh"]