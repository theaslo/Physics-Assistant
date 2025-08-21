# Alertmanager Container for Physics Assistant Notifications
FROM prom/alertmanager:latest

LABEL maintainer="Physics Assistant Team"
LABEL description="Alertmanager for Physics Assistant platform notifications"

USER root

# Install additional tools
RUN apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Create application user
RUN addgroup -g 1001 physicsapp && \
    adduser -D -u 1001 -G physicsapp physicsapp

# Copy Alertmanager configuration
COPY docker/monitoring/alertmanager/alertmanager.yml /etc/alertmanager/alertmanager.yml

# Create necessary directories
RUN mkdir -p /alertmanager/data && \
    chown -R physicsapp:physicsapp /alertmanager /etc/alertmanager

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9093/-/healthy || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 9093

# Volume for data persistence
VOLUME ["/alertmanager/data"]

# Start Alertmanager
CMD ["alertmanager", "--config.file=/etc/alertmanager/alertmanager.yml", "--storage.path=/alertmanager/data", "--web.listen-address=0.0.0.0:9093", "--cluster.listen-address=0.0.0.0:9094"]