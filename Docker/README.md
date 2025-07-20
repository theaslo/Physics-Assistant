# Physics MCP Tools Docker Configuration

This directory contains Docker configuration for running the Physics MCP Tools services.

## Services

The docker-compose.yaml defines three MCP services:

1. **Forces MCP Server** - Port 10100 (host) → 10100 (container)
2. **Kinematics MCP Server** - Port 10101 (host) → 10100 (container)  
3. **Circuit MCP Server** - Port 10102 (host) → 10100 (container)

## Requirements

- Docker
- Docker Compose plugin (not the standalone docker-compose)

## Usage

### Build and start all services
```bash
cd Docker
docker compose up --build -d
```

### Check running services
```bash
docker compose ps
```

### View logs
```bash
# All services
docker compose logs

# Specific service
docker compose logs mcp-forces
docker compose logs mcp-kinematics
docker compose logs mcp-circuit
```

### Stop services
```bash
docker compose down
```

### Rebuild and restart
```bash
docker compose down
docker compose up --build -d
```

## Service URLs

Once running, the services will be available at:

- Forces MCP: http://localhost:10100
- Kinematics MCP: http://localhost:10101
- Circuit MCP: http://localhost:10102

## Configuration

Each service runs with:
- Python 3.13
- UV package manager for dependencies
- Streamable HTTP transport
- Non-root user for security
- Automatic restart on failure

## Troubleshooting

### Check container logs
```bash
docker compose logs [service-name]
```

### Exec into container
```bash
docker compose exec mcp-forces bash
```

### Rebuild from scratch
```bash
docker compose down
docker system prune -f
docker compose up --build -d
```
