# Physics Assistant Production Deployment Guide

This guide provides comprehensive instructions for deploying the Physics Assistant platform in production using Docker Compose with high availability, load balancing, monitoring, and operational excellence.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Backup & Recovery](#backup--recovery)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Overview

The Physics Assistant production deployment includes:

- **High Availability**: Load balanced services with health checks and automatic failover
- **Security**: SSL/TLS termination, security headers, rate limiting, and access controls
- **Monitoring**: Comprehensive observability with Prometheus, Grafana, and centralized logging
- **Backup**: Automated backup system with encryption and S3 storage
- **Scaling**: Horizontal scaling capabilities for all application tiers
- **Performance**: Optimized configurations for production workloads

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended, CentOS 8+, or similar)
- **CPU**: Minimum 8 cores (16+ recommended for production)
- **Memory**: Minimum 16GB RAM (32GB+ recommended)
- **Storage**: Minimum 100GB SSD (500GB+ recommended)
- **Network**: Public IP with DNS configuration for SSL certificates

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+ (or docker-compose 1.29+)
- curl, jq, bc (for scripts and health checks)
- Valid domain name with DNS pointing to the server

### Optional Requirements

- AWS S3 bucket for backups
- SMTP server for email notifications
- Slack webhook for alerting

## Architecture

### Service Tiers

```
┌─────────────────────────────────────────────────────────────┐
│                    Internet/Load Balancer                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  Nginx Load Balancer                       │
│              (SSL/TLS Termination)                         │
└─────────┬───────────────────────┬───────────────────────────┘
          │                       │
┌─────────┴─────────┐    ┌───────┴──────────────────────────┐
│  Frontend Tier    │    │         API Tier                │
│                   │    │                                 │
│ • Streamlit UI    │    │ • Physics Agents API (2x)      │
│   (2 instances)   │    │ • Database API (2x)            │
│ • React Dashboard │    │ • Dashboard API                 │
└─────────┬─────────┘    └───────┬──────────────────────────┘
          │                      │
          └──────────┬───────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   MCP Services Tier                        │
│                                                            │
│ • Forces • Kinematics • Math • Energy • Momentum • Angular │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  Database Tier                             │
│                                                            │
│ • PostgreSQL (Primary) • Neo4j Cluster • Redis Cluster    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                Analytics & Monitoring                      │
│                                                            │
│ • ML Engine • Task Processor • Prometheus • Grafana      │
│ • Alertmanager • Loki • Backup Service                    │
└─────────────────────────────────────────────────────────────┘
```

### Network Architecture

- **physics-network**: Main application network (172.20.0.0/16)
- **database-network**: Internal database communication (172.21.0.0/16)
- **monitoring-network**: Monitoring and logging services (172.22.0.0/16)

## Quick Start

### 1. Clone and Configure

```bash
# Clone the repository
git clone <repository-url>
cd Physics-Assistant

# Copy production environment file
cp .env.production .env.production.local

# Edit the configuration
nano .env.production.local
```

### 2. Update Configuration

**Required changes in `.env.production.local`:**

```bash
# Domain configuration
DOMAIN_NAME=your-domain.com
CERT_EMAIL=admin@your-domain.com

# Database passwords (CHANGE THESE!)
POSTGRES_PASSWORD=your_secure_postgres_password
PHYSICS_DB_PASSWORD=your_secure_physics_password
NEO4J_PASSWORD=your_secure_neo4j_password
REDIS_PASSWORD=your_secure_redis_password

# Security secrets (CHANGE THESE!)
JWT_SECRET=your_jwt_secret_key_32_chars_minimum
ENCRYPTION_KEY=your_encryption_key_32_chars_long
SESSION_SECRET=your_session_secret_64_chars_long

# Monitoring
GRAFANA_ADMIN_PASSWORD=your_secure_grafana_password

# Optional: Backup configuration
S3_ACCESS_KEY_ID=your_aws_access_key
S3_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your-backup-bucket
```

### 3. Deploy

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Deploy the platform
./scripts/deploy-production.sh --env production

# Validate deployment
./scripts/validate-deployment.sh --env production

# Check system health
./scripts/health-check.sh --env production --verbose
```

### 4. Access the Platform

- **Main Application**: https://your-domain.com
- **Analytics Dashboard**: https://your-domain.com/dashboard
- **Monitoring**: https://monitoring.your-domain.com
- **Grafana**: https://monitoring.your-domain.com (admin/your_grafana_password)

## Configuration

### Environment Variables

| Category | Variable | Description | Default |
|----------|----------|-------------|---------|
| **Domain** | `DOMAIN_NAME` | Primary domain name | localhost |
| | `SSL_ENABLED` | Enable SSL/TLS | true |
| | `CERT_EMAIL` | Email for SSL certificates | admin@example.com |
| **Database** | `POSTGRES_PASSWORD` | PostgreSQL root password | Required |
| | `PHYSICS_DB_PASSWORD` | Application database password | Required |
| | `NEO4J_PASSWORD` | Neo4j database password | Required |
| | `REDIS_PASSWORD` | Redis password | Required |
| **Security** | `JWT_SECRET` | JWT signing secret | Required |
| | `ENCRYPTION_KEY` | Data encryption key | Required |
| | `SESSION_SECRET` | Session encryption secret | Required |
| **Performance** | `MAX_WORKERS` | Application workers | 4 |
| | `WORKER_MEMORY_LIMIT` | Worker memory limit | 2G |
| | `NGINX_WORKER_PROCESSES` | Nginx workers | auto |
| **Monitoring** | `GRAFANA_ADMIN_PASSWORD` | Grafana admin password | Required |
| | `ALERT_WEBHOOK_URL` | Slack webhook for alerts | Optional |

### Resource Limits

Default resource allocations:

```yaml
# Database Services
PostgreSQL: 2G RAM, 1.0 CPU
Neo4j: 3G RAM, 1.5 CPU
Redis: 1G RAM, 0.5 CPU

# API Services
Physics Agents API: 2G RAM, 1.5 CPU (per instance)
Database API: 1G RAM, 1.0 CPU (per instance)
Dashboard API: 1G RAM, 1.0 CPU

# Frontend Services
Streamlit UI: 1G RAM, 1.0 CPU (per instance)
React Dashboard: 512M RAM, 0.5 CPU

# MCP Services
Each MCP Service: 512M RAM, 0.5 CPU

# Monitoring
Prometheus: 1G RAM, 0.5 CPU
Grafana: 512M RAM, 0.5 CPU
```

## Deployment

### Production Deployment

```bash
# Full production deployment with all checks
./scripts/deploy-production.sh

# Deploy specific environment
./scripts/deploy-production.sh --env staging

# Force deployment (skip confirmations)
./scripts/deploy-production.sh --force

# Skip backup (faster deployment)
./scripts/deploy-production.sh --skip-backup
```

### Rolling Updates

```bash
# Update specific service
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d --no-deps physics-agents-api-1

# Scale services
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d --scale streamlit-ui=3
```

### Blue-Green Deployment

For zero-downtime deployments:

1. Deploy to staging environment
2. Validate staging deployment
3. Switch traffic using load balancer
4. Monitor and rollback if needed

## Monitoring & Observability

### Prometheus Metrics

Access Prometheus at `http://localhost:9090` or `https://monitoring.your-domain.com/prometheus`

**Key Metrics:**
- Application response times
- Database performance
- System resource usage
- Error rates and status codes

### Grafana Dashboards

Access Grafana at `http://localhost:3000` or `https://monitoring.your-domain.com`

**Pre-configured Dashboards:**
- Physics Assistant Overview
- System Metrics
- Database Performance
- API Performance
- User Analytics

### Centralized Logging

Logs are aggregated using Loki and accessible through Grafana:

- Application logs (structured JSON)
- Nginx access/error logs
- Database logs
- System logs

### Alerting

Alertmanager handles notifications for:
- Service downtime
- High resource usage
- Database connection failures
- SSL certificate expiration
- Backup failures

## Security

### SSL/TLS Configuration

- **Certificate Management**: Automatic Let's Encrypt certificates
- **Security Headers**: HSTS, CSP, X-Frame-Options, etc.
- **Cipher Suites**: Modern, secure cipher configurations
- **OCSP Stapling**: Enabled for performance

### Network Security

- **Internal Networks**: Database tier isolated on internal network
- **Rate Limiting**: API endpoints protected with rate limits
- **Access Controls**: Service-to-service authentication
- **Firewall Rules**: Minimal exposed ports (80, 443, monitoring)

### Data Protection

- **Encryption at Rest**: Database encryption enabled
- **Encryption in Transit**: All communication encrypted
- **Secrets Management**: Environment variable protection
- **Backup Encryption**: All backups encrypted before storage

## Backup & Recovery

### Automated Backups

- **Schedule**: Daily at 2 AM (configurable)
- **Retention**: 30 days (configurable)
- **Encryption**: AES-256 encryption
- **Storage**: Local and S3 (optional)

### Backup Components

- PostgreSQL databases (full dump)
- Neo4j knowledge graph
- Redis data
- Application configuration
- SSL certificates

### Recovery Procedures

```bash
# List available backups
ls -la ./backups/

# Restore from specific backup
./scripts/restore-backup.sh --backup-date 2024-08-15

# Disaster recovery (full system restore)
./scripts/disaster-recovery.sh --backup-path s3://your-bucket/backup-20240815
```

## Scaling

### Horizontal Scaling

Scale individual services:

```bash
# Scale frontend services
docker-compose up -d --scale streamlit-ui-1=2 --scale streamlit-ui-2=2

# Scale API services
docker-compose up -d --scale physics-agents-api-1=2 --scale physics-agents-api-2=2

# Scale MCP services
docker-compose up -d --scale mcp-forces=3
```

### Vertical Scaling

Update resource limits in environment file:

```bash
# Increase memory limits
POSTGRES_MEMORY_LIMIT=4G
NEO4J_MEMORY_LIMIT=6G

# Increase CPU limits
POSTGRES_CPU_LIMIT=2.0
NEO4J_CPU_LIMIT=3.0
```

### Load Testing

```bash
# Run load tests
./scripts/load-test.sh --concurrent-users 100 --duration 300

# Monitor performance during load test
./scripts/health-check.sh --verbose
```

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check service logs
docker-compose logs physics-agents-api-1

# Check resource usage
docker stats

# Validate configuration
./scripts/validate-deployment.sh --quick
```

#### Database Connection Issues

```bash
# Check database health
docker-compose exec postgres-primary pg_isready

# Check database logs
docker-compose logs postgres-primary

# Test database connectivity
./scripts/health-check.sh --verbose
```

#### SSL Certificate Issues

```bash
# Check certificate status
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Renew certificates manually
docker-compose exec nginx-loadbalancer certbot renew

# Check certificate expiry
./scripts/validate-deployment.sh --env production
```

### Performance Issues

#### High Memory Usage

```bash
# Check memory usage by service
docker stats --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Optimize database settings
# Edit postgresql.conf and restart
```

#### Slow Response Times

```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null https://your-domain.com

# Analyze slow queries in PostgreSQL
docker-compose exec postgres-primary psql -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
```

### Log Analysis

```bash
# View real-time logs
docker-compose logs -f physics-agents-api-1

# Search logs with patterns
docker-compose logs | grep ERROR

# Access centralized logs in Grafana
# Navigate to Explore -> Loki -> Query logs
```

## Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor system health
- Check backup completion
- Review error logs

#### Weekly
- Update system packages
- Review security logs
- Validate backup integrity

#### Monthly
- Update SSL certificates (automated)
- Review and rotate secrets
- Performance optimization review

### Maintenance Scripts

```bash
# System health check
./scripts/health-check.sh --verbose --output health-report.json

# Backup validation
./scripts/validate-backup.sh --latest

# Security audit
./scripts/security-audit.sh

# Performance report
./scripts/performance-report.sh --duration 7d
```

### Updates and Upgrades

#### Application Updates

```bash
# Pull latest images
docker-compose pull

# Deploy updates with validation
./scripts/deploy-production.sh --force

# Validate update
./scripts/validate-deployment.sh
```

#### System Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Docker
curl -fsSL https://get.docker.com | sh

# Restart services after system updates
docker-compose restart
```

## Support and Documentation

### Additional Resources

- [Docker Compose Reference](./docker-compose.prod.yml)
- [Environment Configuration](./.env.production)
- [Monitoring Configuration](./docker/monitoring/)
- [Backup Documentation](./backup/)

### Getting Help

- **Health Checks**: `./scripts/health-check.sh --help`
- **Deployment Issues**: `./scripts/deploy-production.sh --help`
- **Validation**: `./scripts/validate-deployment.sh --help`

### Performance Monitoring

- **Grafana**: https://monitoring.your-domain.com
- **Prometheus**: https://monitoring.your-domain.com/prometheus
- **Application Logs**: Available in Grafana Explore

---

## License

This deployment configuration is part of the Physics Assistant project. See LICENSE file for details.

## Contributing

For deployment improvements and bug fixes, please follow the contributing guidelines in the main repository.