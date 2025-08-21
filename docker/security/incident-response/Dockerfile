# Security Incident Response and Monitoring System
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    netcat-openbsd \
    dnsutils \
    tcpdump \
    nmap \
    iptables \
    fail2ban \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /incident-response

# Copy requirements and install Python dependencies
COPY security/incident-response/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy incident response scripts and configuration
COPY security/incident-response/src/ ./src/
COPY security/incident-response/config/ ./config/
COPY security/incident-response/scripts/ ./scripts/
COPY security/incident-response/playbooks/ ./playbooks/

# Set permissions
RUN chmod +x ./scripts/*.sh

# Create non-root user
RUN useradd -m -u 1001 security && \
    chown -R security:security /incident-response

USER security

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python src/health_check.py || exit 1

# Environment variables
ENV PYTHONPATH=/incident-response/src
ENV ALERT_WEBHOOK_URL=""
ENV SLACK_WEBHOOK_URL=""
ENV EMAIL_SMTP_SERVER=""
ENV EMAIL_SMTP_PORT="587"

CMD ["python", "src/incident_response_service.py"]