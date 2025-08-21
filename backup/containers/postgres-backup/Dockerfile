# PostgreSQL Backup Container
FROM postgres:15-alpine

# Install required packages
RUN apk add --no-cache \
    bash \
    curl \
    aws-cli \
    gnupg \
    cronie \
    tzdata \
    python3 \
    py3-pip \
    postgresql-client

# Install Python packages for monitoring and notifications
RUN pip3 install --no-cache-dir \
    requests \
    psutil \
    prometheus-client \
    boto3

# Create backup user and directories
RUN adduser -D -s /bin/bash backup && \
    mkdir -p /backups /scripts /logs /tmp/backup && \
    chown -R backup:backup /backups /scripts /logs /tmp/backup

# Copy backup scripts
COPY backup/scripts/postgres/ /scripts/
COPY backup/scripts/common/ /scripts/common/

# Make scripts executable
RUN chmod +x /scripts/*.sh /scripts/common/*.sh

# Set up cron for backup user
RUN echo "backup ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Copy entrypoint
COPY backup/containers/postgres-backup/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
    CMD /scripts/common/health_check.sh postgres

USER backup
WORKDIR /backups

# Expose metrics port
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]