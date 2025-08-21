# Neo4j Backup Container
FROM neo4j:5.11-community

# Switch to root for installation
USER root

# Install required packages
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    awscli \
    gnupg \
    cron \
    tzdata \
    python3 \
    python3-pip \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for monitoring and notifications
RUN pip3 install --no-cache-dir \
    requests \
    psutil \
    prometheus-client \
    boto3 \
    neo4j

# Create backup user and directories
RUN useradd -m -s /bin/bash backup && \
    mkdir -p /backups /scripts /logs /tmp/backup && \
    chown -R backup:backup /backups /scripts /logs /tmp/backup

# Copy backup scripts
COPY backup/scripts/neo4j/ /scripts/
COPY backup/scripts/common/ /scripts/common/

# Make scripts executable
RUN chmod +x /scripts/*.sh /scripts/common/*.sh

# Set up sudo for backup user
RUN echo "backup ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Copy entrypoint
COPY backup/containers/neo4j-backup/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
    CMD /scripts/common/health_check.sh neo4j

USER backup
WORKDIR /backups

# Expose metrics port
EXPOSE 8081

ENTRYPOINT ["/entrypoint.sh"]