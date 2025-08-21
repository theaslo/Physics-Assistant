# Flower Task Monitor Container for Physics Assistant
FROM python:3.11-slim

LABEL maintainer="Physics Assistant Team"
LABEL description="Flower task monitoring for Physics Assistant analytics"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CELERY_BROKER_URL=redis://redis:6379/0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r physicsapp -g 1001 && \
    useradd -r -g physicsapp -u 1001 -m physicsapp

# Install Flower
RUN pip install --no-cache-dir \
    flower==2.0.1 \
    celery==5.3.4 \
    redis==5.0.1

# Copy startup script
COPY docker/analytics/flower-monitor/start.sh /start.sh
RUN chmod +x /start.sh && \
    chown physicsapp:physicsapp /start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5555/ || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 5555

# Set the startup command
CMD ["/start.sh"]