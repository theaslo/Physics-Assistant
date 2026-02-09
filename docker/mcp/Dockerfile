# Multi-stage build for MCP Physics Tools
# Build stage
FROM docker.io/python:3.13-slim AS builder

LABEL maintainer="Physics Assistant Team"
LABEL description="MCP Physics Tools servers for Physics Assistant platform"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_PYTHON=python3.13

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy project files
COPY mcp_tools/ ./mcp_tools/

# Set working directory to mcp_tools for UV
WORKDIR /app/mcp_tools

# Install dependencies using UV
RUN uv sync --no-dev --frozen

# Production stage
FROM docker.io/python:3.13-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/mcp_tools
ENV PATH="/app/mcp_tools/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/mcp_tools/.venv"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with specific UID/GID for security
RUN groupadd -r mcp -g 1001 && \
    useradd -r -g mcp -u 1001 -m -s /bin/bash mcp

# Create necessary directories
RUN mkdir -p /app/logs /app/temp && \
    chown -R mcp:mcp /app

# Copy virtual environment and application code from builder
COPY --from=builder /app/mcp_tools /app/mcp_tools

# Set proper ownership
RUN chown -R mcp:mcp /app/mcp_tools /home/mcp

# Copy startup script
COPY docker/mcp/start.sh /app/start.sh
RUN chmod +x /app/start.sh && \
    chown mcp:mcp /app/start.sh

# Health check - uses MCP_PORT env var (set in docker-compose)
# Note: MCP servers don't have /health, so we check if port is open
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD sh -c 'nc -z localhost ${MCP_PORT:-10100}' || exit 1

# Switch to non-root user
USER mcp

# Set working directory
WORKDIR /app/mcp_tools

# Expose the default MCP port
EXPOSE 10100

# Default command - will be overridden in docker-compose for specific services
CMD ["/app/start.sh"]