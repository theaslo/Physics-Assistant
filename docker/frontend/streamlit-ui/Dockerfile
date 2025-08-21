# Streamlit UI Container for Physics Assistant
FROM python:3.11-slim

LABEL maintainer="Physics Assistant Team"
LABEL description="Streamlit frontend UI for Physics Assistant platform"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gcc \
    g++ \
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
COPY UI/frontend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY UI/frontend/ ./frontend/
COPY UI/api/agent.py ./api/

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/temp && \
    chown -R physicsapp:physicsapp /app

# Copy configuration and startup script
COPY docker/frontend/streamlit-ui/streamlit_config.toml /app/.streamlit/config.toml
COPY docker/frontend/streamlit-ui/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Switch to application user
USER physicsapp

# Expose port
EXPOSE 8501

# Set the startup command
CMD ["/app/start.sh"]