# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy the entire mcp_tools directory to maintain proper structure
COPY mcp_tools/ ./mcp_tools/

# Set working directory to mcp_tools where pyproject.toml expects to find README.md
WORKDIR /app/mcp_tools

# Install Python dependencies using uv
RUN uv sync --frozen

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash mcp
RUN chown -R mcp:mcp /app/mcp_tools
USER mcp

# Expose the default MCP port
EXPOSE 10100

# Set environment variables
ENV PYTHONPATH=/app/mcp_tools
ENV PYTHONUNBUFFERED=1

# Default command - this will be overridden in docker-compose for specific services
CMD ["uv", "run", "python", "-m", "physics_mcp_tools.forces_mcp_server"]