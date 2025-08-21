# Physics Assistant Backup System Deployment Guide

## Overview

This comprehensive backup and disaster recovery system provides automated, monitored, and validated backup capabilities for the Physics Assistant platform. The system includes:

- **Automated Backup Services**: PostgreSQL, Neo4j, Redis, and application data
- **Backup Scheduling & Retention**: Configurable policies with automatic cleanup
- **Disaster Recovery**: Complete system restoration capabilities
- **Monitoring & Alerting**: Prometheus integration with real-time health checks
- **Backup Validation**: Integrity checking and restoration testing
- **Volume Management**: Docker volume health monitoring and maintenance

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Physics Assistant Platform                   │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│   PostgreSQL    │     Neo4j       │     Redis       │   Apps   │
│      Data       │      Data       │      Data       │   Data   │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backup Services Layer                        │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│  PostgreSQL     │     Neo4j       │     Redis       │   App    │
│   Backup        │    Backup       │    Backup       │  Backup  │
│   Service       │    Service      │    Service      │ Service  │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Management & Monitoring                      │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│   Scheduler     │    Monitor      │   Validator     │  Volume  │
│    Service      │    Service      │    Service      │ Manager  │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage & Recovery                           │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│   Local         │      S3         │   Disaster      │  Health  │
│   Storage       │    Storage      │   Recovery      │   Logs   │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
```

## Quick Start

### 1. Environment Configuration

Create the backup environment configuration file:

```bash
# /opt/physics-assistant/backup/.env
POSTGRES_PASSWORD=your_secure_postgres_password
NEO4J_PASSWORD=your_secure_neo4j_password
PHYSICS_DB_PASSWORD=your_physics_db_password

# Backup Configuration
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
BACKUP_ENCRYPTION=true

# S3 Configuration (optional)
BACKUP_S3_ENABLED=false
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET=physics-assistant-backups
AWS_S3_REGION=us-west-2

# Monitoring Configuration
BACKUP_EMAIL_ENABLED=false
BACKUP_EMAIL_TO=admin@yourorganization.com
WEBHOOK_URL=https://hooks.slack.com/your-webhook-url

# Recovery Configuration
RECOVERY_RTO_MINUTES=60
RECOVERY_RPO_HOURS=4
PARALLEL_RECOVERY=true
VALIDATION_ENABLED=true
```

### 2. Deploy Backup Services

```bash
# Navigate to the project directory
cd /home/atk21004admin/Physics-Assistant

# Create backup storage directories
sudo mkdir -p /opt/physics-assistant/backups/{postgres,neo4j,redis,application}
sudo mkdir -p /opt/physics-assistant/logs/{backup,restore,monitoring,validation}
sudo chown -R 1000:1000 /opt/physics-assistant

# Deploy backup services
docker-compose -f backup/docker-compose.backup.yml up -d
```

### 3. Verify Deployment

```bash
# Check service status
docker-compose -f backup/docker-compose.backup.yml ps

# Check logs
docker-compose -f backup/docker-compose.backup.yml logs backup-monitor

# Test backup functionality
docker exec physics-postgres-backup /scripts/scheduler.sh backup postgres daily
```

## Service Components

### Backup Services

#### PostgreSQL Backup Service
- **Container**: `physics-postgres-backup`
- **Schedule**: Daily at 2:00 AM (configurable)
- **Features**: Full, incremental, and schema-only backups
- **Formats**: Custom dump, SQL script
- **Location**: `/backups/postgres/`

#### Neo4j Backup Service
- **Container**: `physics-neo4j-backup`
- **Schedule**: Daily at 3:00 AM (configurable)
- **Features**: Database dump, Cypher export
- **Formats**: Binary dump, Cypher script
- **Location**: `/backups/neo4j/`

#### Redis Backup Service
- **Container**: `physics-redis-backup`
- **Schedule**: Daily at 4:00 AM (configurable)
- **Features**: RDB, AOF, memory dump
- **Formats**: RDB file, AOF file, JSON export
- **Location**: `/backups/redis/`

#### Application Backup Service
- **Container**: `physics-application-backup`
- **Schedule**: Daily at 5:00 AM (configurable)
- **Features**: File system backup, configuration backup
- **Formats**: Compressed tar archive
- **Location**: `/backups/application/`

### Management Services

#### Backup Monitor
- **Container**: `physics-backup-monitor`
- **Port**: 8084 (Prometheus metrics)
- **Features**: Health checking, alerting, reporting
- **Metrics**: Backup status, age, size, success rates

#### Backup Validator
- **Features**: File integrity, content validation, restoration testing
- **Database**: SQLite validation history
- **Reports**: Comprehensive validation reports

#### Volume Manager
- **Features**: Volume health monitoring, cleanup, orphan removal
- **Thresholds**: 80% warning, 95% critical
- **Cleanup**: Automated old file removal

#### Disaster Recovery
- **RTO Target**: 60 minutes (configurable)
- **RPO Target**: 4 hours (configurable)
- **Features**: Parallel recovery, validation, rollback

## Configuration

### Backup Schedules

Edit `/scripts/config/backup-schedule.conf`:

```bash
# PostgreSQL Backups
POSTGRES_DAILY_SCHEDULE="0 2 * * *"    # 2:00 AM daily
POSTGRES_WEEKLY_SCHEDULE="0 1 * * 0"   # 1:00 AM Sunday
POSTGRES_MONTHLY_SCHEDULE="0 0 1 * *"  # Midnight 1st of month

# Neo4j Backups
NEO4J_DAILY_SCHEDULE="0 3 * * *"       # 3:00 AM daily
NEO4J_WEEKLY_SCHEDULE="0 2 * * 0"      # 2:00 AM Sunday

# Redis Backups
REDIS_DAILY_SCHEDULE="0 4 * * *"       # 4:00 AM daily
REDIS_WEEKLY_SCHEDULE="0 3 * * 0"      # 3:00 AM Sunday

# Application Backups
APPLICATION_DAILY_SCHEDULE="0 5 * * *" # 5:00 AM daily
APPLICATION_WEEKLY_SCHEDULE="0 4 * * 0" # 4:00 AM Sunday
```

### Retention Policies

```bash
# Retention periods (in days)
POSTGRES_DAILY_RETENTION=7      # Keep 7 daily backups
POSTGRES_WEEKLY_RETENTION=28    # Keep 4 weekly backups
POSTGRES_MONTHLY_RETENTION=365  # Keep 12 monthly backups

NEO4J_DAILY_RETENTION=7
NEO4J_WEEKLY_RETENTION=28
NEO4J_MONTHLY_RETENTION=365

REDIS_DAILY_RETENTION=7
REDIS_WEEKLY_RETENTION=28
REDIS_MONTHLY_RETENTION=365

APPLICATION_DAILY_RETENTION=7
APPLICATION_WEEKLY_RETENTION=28
APPLICATION_MONTHLY_RETENTION=365
```

## Operations

### Manual Backup Operations

```bash
# Manual backup execution
./backup/scripts/scheduler.sh backup postgres daily
./backup/scripts/scheduler.sh backup neo4j weekly
./backup/scripts/scheduler.sh backup redis monthly

# Backup validation
python3 ./backup/scripts/validation/backup_validator.py --file /backups/postgres/daily/postgres_daily_20240816.dump.gz
python3 ./backup/scripts/validation/backup_validator.py --all --report

# Backup cleanup
./backup/scripts/scheduler.sh cleanup postgres daily
./backup/scripts/scheduler.sh cleanup all all
```

### Disaster Recovery Operations

```bash
# Full system recovery (latest backups)
./backup/scripts/restore/disaster_recovery.sh recover latest

# Recovery from specific backup point
./backup/scripts/restore/disaster_recovery.sh recover daily
./backup/scripts/restore/disaster_recovery.sh recover weekly

# Test recovery process (no actual restoration)
./backup/scripts/restore/disaster_recovery.sh test latest

# View recovery plan
./backup/scripts/restore/disaster_recovery.sh plan
```

### Volume Management

```bash
# Check volume health
./backup/scripts/volume_manager.sh check all
./backup/scripts/volume_manager.sh check postgres-data

# Volume cleanup
./backup/scripts/volume_manager.sh cleanup all 30
./backup/scripts/volume_manager.sh cleanup backup-postgres 7

# Create missing volumes
./backup/scripts/volume_manager.sh create

# Remove orphaned volumes
./backup/scripts/volume_manager.sh remove-orphaned

# Generate health report
./backup/scripts/volume_manager.sh report
```

### Monitoring and Alerting

```bash
# Check backup monitor status
curl http://localhost:8084/metrics

# View backup reports
ls /logs/monitoring/backup_report_*.json
cat /logs/monitoring/backup_report_latest.json | jq

# Check validation results
sqlite3 /logs/validation/validation_results.db "SELECT * FROM validation_results ORDER BY validation_timestamp DESC LIMIT 10;"
```

## Monitoring Integration

### Prometheus Metrics

The backup system exposes comprehensive metrics on port 8084:

```yaml
# Backup status metrics
backup_status{service="postgres",type="daily"} 1
backup_age_hours{service="postgres",type="daily"} 12.5
backup_size_bytes{service="postgres",type="daily"} 1048576000

# Volume health metrics
volume_health_status{volume="postgres-data"} 1
volume_usage_percent{volume="postgres-data"} 65
volume_used_bytes{volume="postgres-data"} 2147483648

# Validation metrics
backup_validation_status{service="postgres",type="daily"} 1
backup_validation_duration{service="postgres",type="daily"} 45.2
```

### Grafana Dashboards

Import the provided Grafana dashboard for backup monitoring:

1. Navigate to Grafana at `http://localhost:3000`
2. Import dashboard from `/backup/monitoring/grafana-dashboard-backup.json`
3. Configure alerts for critical backup failures

### Alert Rules

Key alerting rules configured in Prometheus:

```yaml
groups:
  - name: backup.rules
    rules:
      - alert: BackupFailed
        expr: backup_status == 0
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Backup failed for {{ $labels.service }}"
          
      - alert: BackupStale
        expr: backup_age_hours > 26
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Backup is stale for {{ $labels.service }}"
          
      - alert: VolumeUsageHigh
        expr: volume_usage_percent > 90
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Volume usage high: {{ $labels.volume }}"
```

## Security

### Encryption

All backups are encrypted using AES-256-CBC:

```bash
# Encryption key management
/scripts/common/encryption.sh test
/scripts/common/encryption.sh rotate-keys

# Backup encryption status
grep -r "encrypted.*true" /backups/*/metadata.json
```

### Access Control

- Backup services run with minimal privileges
- Volume access restricted to backup users
- Encryption keys stored securely in `/secrets/`
- Database credentials use environment variables

### Network Security

- Internal Docker network isolation
- No external ports except monitoring endpoint
- S3 uploads use encrypted connections
- Webhook notifications use HTTPS

## Troubleshooting

### Common Issues

#### Backup Service Not Starting
```bash
# Check container status
docker-compose -f backup/docker-compose.backup.yml ps

# Check logs
docker-compose -f backup/docker-compose.backup.yml logs postgres-backup

# Check volume mounts
docker inspect physics-postgres-backup | jq '.[0].Mounts'
```

#### Backup Validation Failures
```bash
# Check validation logs
tail -f /logs/validation/backup_validator.log

# Manual validation
python3 /scripts/validation/backup_validator.py --file /path/to/backup

# Check database connectivity
docker exec physics-postgres-backup pg_isready -h postgres
```

#### Volume Space Issues
```bash
# Check volume usage
./backup/scripts/volume_manager.sh status

# Clean up old backups
./backup/scripts/scheduler.sh cleanup all all

# Check system disk space
df -h /opt/physics-assistant
```

#### Recovery Failures
```bash
# Test recovery in dry-run mode
RECOVERY_MODE=test ./backup/scripts/restore/disaster_recovery.sh recover latest

# Check recovery logs
tail -f /logs/restore/disaster_recovery_*.log

# Validate infrastructure
./backup/scripts/restore/disaster_recovery.sh validate
```

### Log Locations

- **Backup Logs**: `/logs/backup/`
- **Restore Logs**: `/logs/restore/`
- **Monitoring Logs**: `/logs/monitoring/`
- **Validation Logs**: `/logs/validation/`
- **Volume Health**: `/logs/volume-health/`

### Performance Tuning

#### Backup Performance
```bash
# Adjust compression level (1-9)
export COMPRESSION_LEVEL=6

# Parallel backup jobs
export PARALLEL_JOBS=4

# Timeout adjustments
export BACKUP_TIMEOUT=7200  # 2 hours
```

#### Storage Optimization
```bash
# Enable S3 lifecycle policies
aws s3api put-bucket-lifecycle-configuration \
  --bucket physics-assistant-backups \
  --lifecycle-configuration file://s3-lifecycle.json

# Backup retention tuning
export BACKUP_RETENTION_DAYS=30
export CLEANUP_ENABLED=true
```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Review backup reports and validation results
2. **Monthly**: Test disaster recovery procedures
3. **Quarterly**: Review and update retention policies
4. **Annually**: Rotate encryption keys and access credentials

### Backup Health Checks

```bash
# Daily automated checks via cron
0 8 * * * /path/to/backup/scripts/scheduler.sh status | mail -s "Daily Backup Status" admin@company.com

# Weekly validation reports
0 9 * * 1 python3 /path/to/backup/scripts/validation/backup_validator.py --all --report | mail -s "Weekly Backup Validation" admin@company.com
```

### Updates and Upgrades

1. Update Docker images for backup services
2. Review and update backup scripts
3. Test updated components in staging environment
4. Deploy updates with minimal downtime
5. Validate backup and recovery functionality

## Support

For issues and support:

1. Check the troubleshooting section above
2. Review service logs for error messages
3. Validate configuration and environment variables
4. Test backup and recovery procedures in isolation
5. Consult the backup system documentation

The backup system provides comprehensive protection for the Physics Assistant platform with automated backup, monitoring, validation, and recovery capabilities. Regular testing and maintenance ensure reliable operation and quick recovery in case of data loss or system failures.