# Physics Assistant Docker Deployment Guide

## Overview

This guide covers the complete containerization and deployment of the Physics Assistant platform using Docker and Docker Compose. The platform includes database services, API servers, frontend applications, analytics engines, and monitoring tools.

## Architecture

### Container Services

#### Database Services
- **PostgreSQL**: Primary relational database for user data and interactions
- **Neo4j**: Knowledge graph database for physics concepts and relationships  
- **Redis**: Cache and session store for performance optimization

#### API Services
- **Database API**: Core data access layer (Port 8001)
- **Dashboard API**: Analytics and dashboard backend (Port 8002)
- **Physics Agents API**: AI tutoring agents with MCP integration (Port 8000)

#### MCP (Model Context Protocol) Services
- **Forces MCP**: Force calculations and vector analysis
- **Kinematics MCP**: Motion problems and calculations
- **Math MCP**: Mathematical utilities and helpers
- **Energy MCP**: Work-energy theorem and conservation
- **Momentum MCP**: Linear momentum and collisions
- **Angular Motion MCP**: Rotational dynamics

#### Frontend Services
- **Streamlit UI**: Main student interface (Port 8501)
- **React Dashboard**: Analytics dashboard (Port 5173)
- **Nginx Gateway**: Load balancer and reverse proxy (Ports 80/443)

#### Analytics Services
- **ML Engine**: Machine learning analytics (Port 8003)
- **Task Processor**: Background task processing with Celery
- **Flower Monitor**: Task monitoring interface (Port 5555)

#### Monitoring Services
- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Visualization dashboards (Port 3000)
- **Alertmanager**: Alert routing and notifications (Port 9093)

## Deployment Environments

### Development Environment

For local development with hot reloading and debugging capabilities:

```bash
# Copy and configure environment file
cp .env.development.template .env.development
# Edit .env.development with your settings

# Deploy development environment
./scripts/deploy-docker.sh development
```

**Development Features:**
- Hot reload for frontend services
- Direct port access to all services
- Relaxed security settings
- Debug logging enabled
- Smaller resource limits

**Access URLs:**
- Main Application: http://localhost:8501
- Analytics Dashboard: http://localhost:5173
- Database API: http://localhost:8001
- Dashboard API: http://localhost:8002
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

### Production Environment

For production deployment with security hardening and monitoring:

```bash
# Copy and configure environment file
cp .env.production.template .env.production
# Edit .env.production with secure passwords and settings

# Deploy production environment
./scripts/deploy-docker.sh production
```

**Production Features:**
- Security hardening with non-root users
- Resource limits and health checks
- Comprehensive monitoring and alerting
- Load balancing and high availability
- SSL/TLS termination at gateway

**Access URLs:**
- Main Application: http://localhost
- Analytics Dashboard: http://localhost/dashboard
- Monitoring: http://localhost:3000

## Configuration

### Environment Variables

#### Required Variables (Production)
```bash
# Database passwords (use strong passwords!)
POSTGRES_PASSWORD=secure_postgres_password
PHYSICS_DB_PASSWORD=secure_physics_db_password
NEO4J_PASSWORD=secure_neo4j_password
REDIS_PASSWORD=secure_redis_password

# Application secrets
SECRET_KEY=your-super-secret-key
JWT_SECRET_KEY=your-jwt-secret-key

# Monitoring
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
```

#### Optional Configuration
```bash
# API Configuration
API_HOST=0.0.0.0
CORS_ORIGINS=https://your-domain.com

# External Services
SMTP_HOST=smtp.your-domain.com
SMTP_USER=alerts@your-domain.com
SMTP_PASSWORD=your-smtp-password

# Resource Limits
MAX_UPLOAD_SIZE=100MB
REQUEST_TIMEOUT=300
```

### Volume Management

#### Persistent Volumes
- `postgres-data`: PostgreSQL database files
- `neo4j-data`: Neo4j knowledge graph data
- `redis-data`: Redis persistence files
- `prometheus-data`: Metrics storage
- `grafana-data`: Dashboard configurations

#### Backup Locations
- Database backups: `./database/backups`
- Application logs: `./logs`
- ML models: `./analytics/models`

## Security Features

### Container Security
- Non-root users (UID 1001)
- Read-only root filesystems where possible
- Security scanning and vulnerability management
- Resource limits and quotas

### Network Security
- Internal Docker network isolation
- Service-to-service authentication
- Rate limiting and CORS protection
- SSL/TLS encryption in production

### Data Security
- Encrypted database connections
- Secure password policies
- Secret management
- Access logging and monitoring

## Monitoring and Observability

### Metrics Collection
- **Prometheus**: Collects metrics from all services
- **Custom metrics**: Application-specific performance indicators
- **System metrics**: CPU, memory, disk, network usage

### Visualization
- **Grafana dashboards**: Pre-configured monitoring views
- **Real-time analytics**: Live student progress and system health
- **Alert visualization**: Visual alert status and history

### Alerting
- **Critical alerts**: Service failures, database issues
- **Warning alerts**: High resource usage, performance degradation
- **Info alerts**: Backup completion, maintenance windows

### Log Management
- Centralized logging with structured JSON format
- Log rotation and retention policies
- Error tracking and debugging support

## Scaling and Performance

### Horizontal Scaling
```bash
# Scale API services
docker-compose -f docker-compose.production.yml up -d --scale physics-agents-api=3

# Scale MCP services
docker-compose -f docker-compose.production.yml up -d --scale mcp-forces=2
```

### Resource Optimization
- Container resource limits prevent resource exhaustion
- Health checks ensure service reliability
- Connection pooling for database efficiency
- Caching strategies with Redis

## Maintenance Operations

### Backup Procedures
```bash
# Database backup
docker exec physics-postgres pg_dump -U physics_user physics_assistant > backup.sql

# Neo4j backup
docker exec physics-neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j-backup.dump

# Redis backup
docker exec physics-redis redis-cli BGSAVE
```

### Log Management
```bash
# View service logs
docker-compose -f docker-compose.production.yml logs -f [service-name]

# Log rotation (automatic via configuration)
docker exec physics-postgres logrotate /etc/logrotate.d/postgresql
```

### Health Checks
```bash
# Check all service health
docker-compose -f docker-compose.production.yml ps

# Individual service health
curl -f http://localhost:8001/health  # Database API
curl -f http://localhost:8002/health  # Dashboard API
curl -f http://localhost:8000/health  # Physics Agents API
```

## Troubleshooting

### Common Issues

#### Service Start Failures
1. Check environment variables are set correctly
2. Verify network connectivity between services
3. Check resource availability and limits
4. Review service logs for specific errors

#### Database Connection Issues
1. Verify database containers are running and healthy
2. Check network connectivity and port accessibility
3. Validate credentials and connection strings
4. Review database logs for connection errors

#### Performance Issues
1. Monitor resource usage with Grafana dashboards
2. Check for memory leaks or CPU spikes
3. Review slow query logs in databases
4. Analyze network latency and throughput

### Log Analysis
```bash
# Critical error search
docker-compose logs | grep -i error

# Database connection issues
docker-compose logs postgres | grep -i connection

# MCP service debugging
docker-compose logs mcp-forces | grep -i exception
```

### Recovery Procedures

#### Database Recovery
```bash
# PostgreSQL recovery from backup
docker exec -i physics-postgres psql -U physics_user physics_assistant < backup.sql

# Neo4j recovery from dump
docker exec physics-neo4j neo4j-admin load --from=/backups/neo4j-backup.dump --database=neo4j --force
```

#### Service Recovery
```bash
# Restart specific service
docker-compose -f docker-compose.production.yml restart [service-name]

# Rebuild and restart service
docker-compose -f docker-compose.production.yml up -d --build [service-name]

# Full system restart
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d
```

## Best Practices

### Security
1. Use strong, unique passwords for all services
2. Regular security updates and vulnerability scanning
3. Network segmentation and access controls
4. Backup encryption and secure storage

### Performance
1. Monitor resource usage and scale accordingly
2. Implement caching strategies at multiple layers
3. Optimize database queries and indexes
4. Use connection pooling and persistent connections

### Reliability
1. Implement comprehensive health checks
2. Set up monitoring and alerting
3. Regular backup testing and recovery drills
4. Capacity planning and resource management

### Operations
1. Automate deployment and scaling procedures
2. Document configuration changes and updates
3. Implement CI/CD pipelines for updates
4. Regular monitoring and maintenance schedules

## Support and Documentation

### Additional Resources
- Container logs: Available through Docker Compose
- Monitoring dashboards: Access via Grafana
- API documentation: Available at service endpoints
- System metrics: Available via Prometheus

### Getting Help
1. Check service logs for error messages
2. Review monitoring dashboards for system health
3. Consult this documentation for common issues
4. Check Docker and service-specific documentation