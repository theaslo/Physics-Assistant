# Alertmanager Container for Physics Assistant Notifications
FROM docker.io/prom/alertmanager:latest

LABEL maintainer="Physics Assistant Team"
LABEL description="Alertmanager for Physics Assistant platform notifications"

USER root

# Create physicsapp user
RUN if command -v addgroup >/dev/null; then \
        addgroup -g 1001 physicsapp && \
        adduser -D -u 1001 -G physicsapp physicsapp; \
    else \
        groupadd -g 1001 physicsapp && \
        useradd -r -u 1001 -g physicsapp physicsapp; \
    fi

# Copy Alertmanager configuration
COPY docker/monitoring/alertmanager/alertmanager.yml /etc/alertmanager/alertmanager.yml

# Create necessary directories
RUN mkdir -p /alertmanager/data && \
    chown -R physicsapp:physicsapp /alertmanager /etc/alertmanager

# Health check
# CMD curl -f http://localhost:9093/-/healthy || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 9093

# Volume for data persistence
VOLUME ["/alertmanager/data"]

# Start Alertmanager
CMD ["alertmanager", "--config.file=/etc/alertmanager/alertmanager.yml", "--storage.path=/alertmanager/data", "--web.listen-address=0.0.0.0:9093", "--cluster.listen-address=0.0.0.0:9094"]