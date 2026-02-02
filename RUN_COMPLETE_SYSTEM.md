# Physics Assistant - Complete System Startup Guide

## Overview
This guide provides step-by-step instructions for running the complete Physics Assistant system including database, UI, and all MCP servers for comprehensive data collection and physics tutoring.

## System Components
- **6 MCP Physics Servers** (Forces, Kinematics, Math, Momentum, Energy, Angular Motion)
- **PostgreSQL Database** (Data collection and analytics)
- **Redis Cache** (Session management and performance)
- **Neo4j Graph Database** (Learning path analytics)
- **Streamlit UI** (Student interface)
- **Health Monitoring** (System observability)

## Prerequisites

### System Requirements
- Docker or Podman installed
- 8GB RAM minimum (16GB recommended)
- 4 CPU cores minimum
- 50GB available disk space

### Verify Prerequisites
```bash
# Check Docker/Podman
docker --version || podman --version

# Check available resources
free -h
df -h .
```

## Step 1: Initial Setup

### Clone and Navigate
```bash
git clone <repository>
cd Physics-Assistant
```

### Setup Environment
```bash
# Copy environment template
cp .env.production.template .env.production

# Edit configuration (IMPORTANT: Update all CHANGE_ME values)
nano .env.production
```

### Required Environment Variables
Update these in `.env.production`:
```bash
# Database Configuration
PHYSICS_DB_PASSWORD=your_secure_database_password
POSTGRES_PASSWORD=your_secure_database_password

# Redis Configuration
REDIS_PASSWORD=your_secure_redis_password

# Neo4j Configuration
NEO4J_AUTH=neo4j/your_secure_neo4j_password

# Security
SECRET_KEY=your_secret_key_for_ui
JWT_SECRET=your_jwt_secret_key

# Email Alerts (Optional)
ALERT_EMAIL=your-email@domain.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-smtp-username
SMTP_PASSWORD=your-smtp-password
```

## Step 2: Start Complete System

### Option A: Production Compose (Recommended)
```bash
# Start all services including databases
docker-compose -f docker-compose.production.yml up -d

# OR with Podman
podman-compose -f docker-compose.production.yml up -d
```

### Option B: Database + MCP Servers Separately
```bash
# Start databases first
docker-compose -f docker-compose.production.yml --profile database up -d

# Wait 30 seconds for databases to initialize
sleep 30

# Start MCP servers
docker-compose -f docker-compose.mcp-fixed.yaml up -d

# Start UI separately (if not included in production compose)
cd UI
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

## Step 3: Verify System Health

### Check All Services
```bash
# Run comprehensive health check
./scripts/simple_health_monitor.sh

# Check container status
docker ps -a | grep physics
# OR
podman ps -a | grep physics
```

### Expected Services Running
- `physics-postgres` (Port 5432)
- `physics-redis` (Port 6379)
- `physics-neo4j` (Port 7474, 7687)
- `physics-mcp-forces` (Port 10100)
- `physics-mcp-kinematics` (Port 10101)
- `physics-mcp-math` (Port 10103)
- `physics-mcp-momentum` (Port 10104)
- `physics-mcp-energy` (Port 10105)
- `physics-mcp-angular-motion` (Port 10106)
- `physics-ui` (Port 8501)

### Test MCP Server Connectivity
```bash
# Quick connectivity test
python database/fixes/test_physics_workflow.py

# Individual server tests
curl http://localhost:10100/  # Forces
curl http://localhost:10101/  # Kinematics
curl http://localhost:10103/  # Math
curl http://localhost:10104/  # Momentum
curl http://localhost:10105/  # Energy
curl http://localhost:10106/  # Angular Motion
```

### Test Database Connectivity
```bash
# PostgreSQL
docker exec physics-postgres psql -U physics_user -d physics_assistant -c "SELECT version();"

# Redis
docker exec physics-redis redis-cli ping

# Neo4j (via HTTP)
curl -u neo4j:your_password http://localhost:7474/db/data/
```

## Step 4: Access the System

### Student Interface
- **URL**: http://localhost:8501
- **Purpose**: Student physics tutoring interface
- **Features**: Chat, problem solving, visualizations

### Database Interfaces
- **PostgreSQL**: localhost:5432 (use pgAdmin or similar)
- **Redis**: localhost:6379 (use Redis CLI or GUI)
- **Neo4j Browser**: http://localhost:7474

## Step 5: Data Collection Verification

### Check Data Flow
```bash
# Monitor database logs
docker logs physics-postgres -f

# Check data insertion
docker exec physics-postgres psql -U physics_user -d physics_assistant -c "
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
"

# Verify data collection
docker exec physics-postgres psql -U physics_user -d physics_assistant -c "
SELECT COUNT(*) FROM student_interactions;
SELECT COUNT(*) FROM tool_usage_logs;
"
```

### Test Data Collection
1. Open UI at http://localhost:8501
2. Ask a physics question (e.g., "Calculate projectile motion for v0=20m/s at 45°")
3. Verify data appears in database tables
4. Check interaction logs

## Step 6: Monitoring and Maintenance

### Continuous Health Monitoring
```bash
# Start continuous monitoring
./scripts/simple_health_monitor.sh monitor

# Check recent alerts
./scripts/simple_health_monitor.sh alerts

# View system summary
./scripts/simple_health_monitor.sh summary
```

### Backup System
```bash
# Create full backup
./scripts/backup_system.sh

# List existing backups
./scripts/backup_system.sh list

# Restore from backup (if needed)
./scripts/backup_system.sh restore backups/physics_assistant_backup_YYYYMMDD_HHMMSS.tar.gz
```

## Troubleshooting

### Common Issues

#### Database Connection Failed
```bash
# Check database status
docker logs physics-postgres

# Restart database
docker restart physics-postgres

# Wait for initialization
sleep 30
```

#### MCP Server Offline
```bash
# Check server logs
docker logs physics-mcp-forces

# Restart specific server
docker restart physics-mcp-forces

# Rebuild if needed
docker-compose -f docker-compose.mcp-fixed.yaml build mcp-forces
docker-compose -f docker-compose.mcp-fixed.yaml up -d mcp-forces
```

#### UI Not Accessible
```bash
# Check UI container
docker logs physics-ui

# Check port binding
netstat -tlnp | grep 8501

# Restart UI
docker restart physics-ui
```

#### Out of Memory
```bash
# Check resource usage
docker stats

# Increase container limits in docker-compose.yml
# Or add more system RAM
```

### Performance Optimization

#### For High Load
```bash
# Scale MCP servers
docker-compose -f docker-compose.production.yml up -d --scale mcp-forces=3 --scale mcp-kinematics=3

# Add load balancer (Nginx configuration example in PRODUCTION_DEPLOYMENT.md)
```

#### Database Optimization
```bash
# Optimize PostgreSQL
docker exec physics-postgres psql -U physics_user -d physics_assistant -c "
VACUUM ANALYZE;
REINDEX DATABASE physics_assistant;
"

# Monitor database performance
docker exec physics-postgres psql -U physics_user -d physics_assistant -c "
SELECT * FROM pg_stat_activity WHERE state = 'active';
"
```

## Data Collection Schema

### Student Interactions Table
```sql
CREATE TABLE student_interactions (
    id SERIAL PRIMARY KEY,
    session_id UUID,
    user_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question TEXT,
    response TEXT,
    tools_used JSONB,
    satisfaction_rating INTEGER,
    problem_type VARCHAR(100),
    difficulty_level VARCHAR(50)
);
```

### Tool Usage Logs
```sql
CREATE TABLE tool_usage_logs (
    id SERIAL PRIMARY KEY,
    interaction_id INTEGER REFERENCES student_interactions(id),
    tool_name VARCHAR(100),
    parameters JSONB,
    result JSONB,
    execution_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## System Shutdown

### Graceful Shutdown
```bash
# Stop all services
docker-compose -f docker-compose.production.yml down

# OR stop individual components
docker-compose -f docker-compose.mcp-fixed.yaml down
docker stop physics-postgres physics-redis physics-neo4j physics-ui
```

### Complete Cleanup (Caution: Removes all data)
```bash
# Stop and remove all containers and volumes
docker-compose -f docker-compose.production.yml down -v

# Remove all physics-related containers and volumes
docker system prune -f
docker volume prune -f
```

## Support and Logs

### Log Locations
- **Health Monitor**: `logs/health_monitor.log`
- **Alerts**: `logs/alerts.log`
- **Application**: `logs/physics_assistant.log`
- **Container Logs**: `docker logs <container_name>`

### Getting Help
```bash
# System status
./scripts/simple_health_monitor.sh summary

# Recent alerts
./scripts/simple_health_monitor.sh alerts

# Resource usage
docker stats

# Detailed health check
./scripts/simple_health_monitor.sh check
```

---

**Success Criteria**: When all components are running and you can:
1. Access UI at http://localhost:8501
2. Ask physics questions and get responses
3. See data appearing in PostgreSQL database
4. All health checks pass
5. No critical alerts in monitoring logs

**Deployment Date**: 2025-09-17
**Version**: 1.0.0 - Complete System with Database Integration