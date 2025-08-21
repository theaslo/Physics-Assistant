# PostgreSQL Container for Physics Assistant
FROM postgres:16-alpine

LABEL maintainer="Physics Assistant Team"
LABEL description="PostgreSQL database for Physics Assistant platform"

# Install additional tools for backup and monitoring
RUN apk add --no-cache \
    curl \
    postgresql-contrib \
    && rm -rf /var/cache/apk/*

# Create application user
RUN addgroup -g 1001 physicsapp && \
    adduser -D -u 1001 -G physicsapp physicsapp

# Copy initialization scripts
COPY docker/database/postgresql/init/ /docker-entrypoint-initdb.d/
COPY database/schema/ /opt/physics-assistant/schema/

# Set proper permissions
RUN chmod +x /docker-entrypoint-initdb.d/*.sh && \
    chown -R postgres:postgres /opt/physics-assistant/

# Create directories for backups and logs
RUN mkdir -p /var/lib/postgresql/backups /var/log/postgresql && \
    chown -R postgres:postgres /var/lib/postgresql/backups /var/log/postgresql

# Copy custom PostgreSQL configuration
COPY docker/database/postgresql/postgresql.conf /etc/postgresql/postgresql.conf
COPY docker/database/postgresql/pg_hba.conf /etc/postgresql/pg_hba.conf

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD pg_isready -U $POSTGRES_USER -d $POSTGRES_DB || exit 1

# Expose port
EXPOSE 5432

# Set environment variables
ENV POSTGRES_INITDB_ARGS="--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
ENV POSTGRES_HOST_AUTH_METHOD=md5

# Volume for data persistence
VOLUME ["/var/lib/postgresql/data", "/var/lib/postgresql/backups"]