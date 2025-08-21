# Redis Container for Physics Assistant Caching and Sessions
FROM redis:7-alpine

LABEL maintainer="Physics Assistant Team"
LABEL description="Redis cache and session store for Physics Assistant platform"

# Install additional tools
RUN apk add --no-cache \
    curl \
    bash \
    && rm -rf /var/cache/apk/*

# Create application user
RUN addgroup -g 1001 physicsapp && \
    adduser -D -u 1001 -G physicsapp physicsapp

# Copy Redis configuration
COPY docker/database/redis/redis.conf /usr/local/etc/redis/redis.conf
COPY docker/database/redis/scripts/ /opt/physics-assistant/scripts/

# Create directories for persistence and logs
RUN mkdir -p /data/redis /var/log/redis && \
    chown -R redis:redis /data/redis /var/log/redis && \
    chmod +x /opt/physics-assistant/scripts/*.sh

# Set proper permissions
RUN chown -R redis:redis /usr/local/etc/redis/ && \
    chown -R redis:redis /opt/physics-assistant/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD redis-cli --raw incr ping_counter || exit 1

# Switch to redis user
USER redis

# Expose port
EXPOSE 6379

# Volume for data persistence
VOLUME ["/data"]

# Start Redis with custom configuration
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]