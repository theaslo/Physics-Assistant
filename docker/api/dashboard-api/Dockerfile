# Dashboard API Server Container for Physics Assistant
FROM python:3.11-slim

LABEL maintainer="Physics Assistant Team"
LABEL description="Dashboard API server with analytics for Physics Assistant platform"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r physicsapp -g 1001 && \
    useradd -r -g physicsapp -u 1001 -m physicsapp

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY database/dashboard_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r dashboard_requirements.txt

# Copy application code
COPY database/ ./database/

# Create necessary directories
RUN mkdir -p /app/logs /app/exports && \
    chown -R physicsapp:physicsapp /app

# Copy startup script
COPY docker/api/dashboard-api/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 8002

# Set the startup command
CMD ["/app/start.sh"]