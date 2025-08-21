# ML Analytics Engine Container for Physics Assistant
FROM python:3.11-slim

LABEL maintainer="Physics Assistant Team"
LABEL description="Machine learning analytics engine for Physics Assistant platform"

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
    git \
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

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.24.4 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    joblib==1.3.2 \
    xgboost==2.0.2 \
    lightgbm==4.1.0 \
    optuna==3.4.0 \
    mlflow==2.8.1

# Copy application code
COPY database/analytics/ ./analytics/

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/exports /app/temp && \
    chown -R physicsapp:physicsapp /app

# Copy startup script
COPY docker/analytics/ml-engine/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')" || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 8003

# Set the startup command
CMD ["/app/start.sh"]