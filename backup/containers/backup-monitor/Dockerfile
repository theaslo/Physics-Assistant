# Backup Monitoring Container
FROM python:3.11-slim

# Install required packages
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    requests \
    psutil \
    prometheus-client \
    schedule \
    pyyaml \
    jinja2

# Create monitor user and directories
RUN useradd -m -s /bin/bash monitor && \
    mkdir -p /backups /scripts /logs /tmp/monitor && \
    chown -R monitor:monitor /backups /scripts /logs /tmp/monitor

# Copy monitoring scripts
COPY backup/scripts/monitoring/ /scripts/
COPY backup/scripts/common/ /scripts/common/

# Make scripts executable
RUN chmod +x /scripts/*.py /scripts/*.sh /scripts/common/*.sh

# Copy entrypoint
COPY backup/containers/backup-monitor/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=1m --retries=3 \
    CMD /scripts/health_check.py

USER monitor
WORKDIR /scripts

# Expose metrics port
EXPOSE 8084

ENTRYPOINT ["/entrypoint.sh"]