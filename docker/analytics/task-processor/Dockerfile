# Background Task Processor Container for Physics Assistant
FROM python:3.11-slim

LABEL maintainer="Physics Assistant Team"
LABEL description="Background task processor for Physics Assistant analytics"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0

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
COPY database/analytics/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Celery and task processing dependencies
RUN pip install --no-cache-dir \
    celery==5.3.4 \
    redis==5.0.1 \
    kombu==5.3.4 \
    flower==2.0.1 \
    billiard==4.2.0

# Copy application code
COPY database/analytics/ ./analytics/

# Create necessary directories
RUN mkdir -p /app/logs /app/temp /app/results && \
    chown -R physicsapp:physicsapp /app

# Copy startup script
COPY docker/analytics/task-processor/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=60s --retries=3 \
    CMD celery -A analytics.tasks status || exit 1

# Switch to application user
USER physicsapp

# Set the startup command
CMD ["/app/start.sh"]