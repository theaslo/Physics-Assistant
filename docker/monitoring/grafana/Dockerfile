# Grafana Container for Physics Assistant Visualization
FROM grafana/grafana:latest

LABEL maintainer="Physics Assistant Team"
LABEL description="Grafana dashboards for Physics Assistant platform"

USER root

# Install additional plugins
RUN grafana-cli plugins install grafana-clock-panel && \
    grafana-cli plugins install grafana-worldmap-panel && \
    grafana-cli plugins install grafana-piechart-panel

# Create application user
RUN addgroup -g 1001 physicsapp && \
    adduser -D -u 1001 -G physicsapp physicsapp

# Copy Grafana configuration
COPY docker/monitoring/grafana/grafana.ini /etc/grafana/grafana.ini
COPY docker/monitoring/grafana/provisioning/ /etc/grafana/provisioning/
COPY docker/monitoring/grafana/dashboards/ /var/lib/grafana/dashboards/

# Create necessary directories
RUN mkdir -p /var/lib/grafana/plugins /var/log/grafana && \
    chown -R physicsapp:physicsapp /var/lib/grafana /var/log/grafana /etc/grafana

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 3000

# Volume for data persistence
VOLUME ["/var/lib/grafana"]

# Environment variables
ENV GF_SECURITY_ADMIN_PASSWORD=physics_admin_2024
ENV GF_USERS_ALLOW_SIGN_UP=false
ENV GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-worldmap-panel,grafana-piechart-panel