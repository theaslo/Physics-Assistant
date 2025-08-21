# Backup Restoration Container
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    wget \
    awscli \
    postgresql-client \
    redis-tools \
    python3 \
    python3-pip \
    jq \
    tar \
    gzip \
    openssl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    requests \
    psutil \
    prometheus-client \
    boto3 \
    neo4j \
    redis

# Install Neo4j client tools
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - && \
    echo 'deb https://debian.neo4j.com stable latest' | tee /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && apt-get install -y neo4j-client && \
    rm -rf /var/lib/apt/lists/*

# Create restore user and directories
RUN useradd -m -s /bin/bash restore && \
    mkdir -p /backups /scripts /logs /tmp/restore && \
    chown -R restore:restore /backups /scripts /logs /tmp/restore

# Copy restore scripts
COPY backup/scripts/restore/ /scripts/
COPY backup/scripts/common/ /scripts/common/

# Make scripts executable
RUN chmod +x /scripts/*.sh /scripts/common/*.sh

# Set up sudo for restore user
RUN echo "restore ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Copy entrypoint
COPY backup/containers/backup-restore/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
    CMD /scripts/common/health_check.sh restore

USER restore
WORKDIR /backups

# Expose metrics port
EXPOSE 8083

ENTRYPOINT ["/entrypoint.sh"]