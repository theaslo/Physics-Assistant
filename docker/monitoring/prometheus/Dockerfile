# Prometheus Container for Physics Assistant Monitoring
FROM docker.io/prom/prometheus:latest

LABEL maintainer="Physics Assistant Team"
LABEL description="Prometheus monitoring for Physics Assistant platform"

USER root

# Check if we have addgroup (Alpine) or groupadd (Ubuntu/Debian)
# and create physicsapp user accordingly
RUN if command -v addgroup >/dev/null; then \
        addgroup -g 1001 physicsapp && \
        adduser -D -u 1001 -G physicsapp physicsapp; \
    else \
        groupadd -g 1001 physicsapp && \
        useradd -r -u 1001 -g physicsapp physicsapp; \
    fi

# Copy Prometheus configuration
COPY docker/monitoring/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
COPY docker/monitoring/prometheus/alert_rules.yml /etc/prometheus/alert_rules.yml

# Create necessary directories
RUN mkdir -p /prometheus/data /etc/prometheus/rules && \
    chown -R physicsapp:physicsapp /prometheus /etc/prometheus

# Copy startup script
COPY docker/monitoring/prometheus/start.sh /start.sh
RUN chmod +x /start.sh

# Health check
# CMD curl -f http://localhost:9090/-/healthy || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 9090

# Volume for data persistence
VOLUME ["/prometheus/data"]

# Set the startup command
CMD ["/start.sh"]