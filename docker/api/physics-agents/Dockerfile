# Physics Agents API Container for Physics Assistant
FROM python:3.11-slim

LABEL maintainer="Physics Assistant Team"
LABEL description="Physics agents API server with MCP tools integration"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including UV
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    netcat-openbsd \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Create application user
RUN groupadd -r physicsapp -g 1001 && \
    useradd -r -g physicsapp -u 1001 -m physicsapp

# Set working directory
WORKDIR /app

# Copy project files for UV
COPY UI/api/pyproject.toml UI/api/uv.lock ./
COPY agents/pyproject.toml ./agents/
COPY mcp_tools/pyproject.toml ./mcp_tools/

# Install dependencies using UV
RUN uv sync --no-dev

# Copy application code
COPY UI/api/ ./api/
COPY agents/ ./agents/
COPY mcp_tools/ ./mcp_tools/

# Create necessary directories
RUN mkdir -p /app/logs /app/temp && \
    chown -R physicsapp:physicsapp /app

# Copy startup script
COPY docker/api/physics-agents/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 8000

# Set the startup command
CMD ["/app/start.sh"]