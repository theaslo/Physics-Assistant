# Database Query Optimizer and Performance Monitor
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /optimizer

# Copy requirements and install Python dependencies
COPY optimization/query-optimizer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy optimizer scripts and configuration
COPY optimization/query-optimizer/src/ ./src/
COPY optimization/query-optimizer/config/ ./config/
COPY optimization/query-optimizer/scripts/ ./scripts/

# Set permissions
RUN chmod +x ./scripts/*.sh

# Create non-root user
RUN useradd -m -u 1001 optimizer && \
    chown -R optimizer:optimizer /optimizer

USER optimizer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python src/health_check.py || exit 1

# Environment variables
ENV PYTHONPATH=/optimizer/src
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV NEO4J_HOST=neo4j
ENV NEO4J_PORT=7687

CMD ["python", "src/optimizer_service.py"]