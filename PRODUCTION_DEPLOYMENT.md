# Physics Assistant - Production Deployment Guide

## Overview
Complete deployment guide for the Physics Assistant platform optimized for fine-tuning data collection.

## System Architecture

### Core Components
- **6 MCP Physics Servers**: 53 specialized physics tools across 6 domains
- **Streamlit UI**: Student interaction interface with comprehensive logging
- **Database Infrastructure**: PostgreSQL for interaction data storage
- **Container Orchestration**: Docker/Podman with compose configuration

### Production-Optimized Architecture
```
Student Interaction → UI (Streamlit) → MCP Tools → Results
                       ↓
                   Database Logging
                       ↓
                 Fine-Tuning Data
```

## Quick Start Deployment

### Prerequisites
- Docker/Podman installed
- Python 3.13+
- UV package manager
- 8GB RAM minimum
- 4 CPU cores recommended

### 1. Clone and Setup
```bash
git clone <repository>
cd Physics-Assistant
```

### 2. Start MCP Servers
```bash
# Start all 6 physics MCP servers
docker compose -f docker-compose.mcp-fixed.yaml up -d

# Verify all servers running
docker compose -f docker-compose.mcp-fixed.yaml ps
```

### 3. Verify System Health
```bash
# Run end-to-end test
python database/fixes/test_physics_workflow.py

# Expected output: "🎉 PHASE 2.3 COMPLETE: System ready for production physics tutoring!"
```

## MCP Server Details

### Server Endpoints
- **Forces**: localhost:10100 (12 tools - force analysis, equilibrium, free body diagrams)
- **Kinematics**: localhost:10101 (7 tools - motion analysis, projectiles)
- **Math**: localhost:10103 (10 tools - calculations, conversions)
- **Momentum**: localhost:10104 (8 tools - collisions, impulse)
- **Energy**: localhost:10105 (8 tools - kinetic, potential, work-energy)
- **Angular Motion**: localhost:10106 (7 tools - rotational dynamics)

### Health Monitoring
```bash
# Check individual server status
for port in 10100 10101 10103 10104 10105 10106; do
  curl -s http://localhost:$port/ >/dev/null && echo "Port $port: ✅" || echo "Port $port: ❌"
done
```

## Data Collection for Fine-Tuning

### Interaction Data Points
The UI captures complete training data:
1. **Student Queries**: Physics questions and problem statements
2. **Tool Selection**: Which MCP tools are chosen for each problem
3. **Input Parameters**: Values and variables passed to tools
4. **Calculation Results**: Tool outputs and explanations
5. **User Feedback**: Correctness ratings and helpfulness scores
6. **Learning Context**: Problem types, difficulty levels, topics

### Data Format
```json
{
  "session_id": "unique_id",
  "user_query": "A ball is thrown at 20 m/s at 45°...",
  "tools_used": ["kinematics.projectile_motion", "energy.kinetic_energy"],
  "parameters": {"initial_velocity": 20, "angle": 45},
  "results": {"max_height": "10.2m", "time_of_flight": "2.89s"},
  "user_feedback": {"correct": true, "helpful": 4/5},
  "timestamp": "2025-09-16T14:30:00Z"
}
```

## Environment Configuration

### Production Environment Variables
```bash
# Core Configuration
ENVIRONMENT=production
PHYSICS_ASSISTANT_MODE=production
LOG_LEVEL=info

# MCP Server Configuration
MCP_FORCES_URL=http://localhost:10100
MCP_KINEMATICS_URL=http://localhost:10101
MCP_MATH_URL=http://localhost:10103
MCP_MOMENTUM_URL=http://localhost:10104
MCP_ENERGY_URL=http://localhost:10105
MCP_ANGULAR_URL=http://localhost:10106

# Database Configuration (if using database logging)
DATABASE_URL=postgresql://physics_user:password@localhost:5432/physics_assistant
REDIS_URL=redis://localhost:6379

# Security
SESSION_SECRET=your_secure_session_secret_here
CSRF_PROTECTION=true
RATE_LIMITING=true
```

## Performance Optimization

### Resource Requirements
- **Development**: 4GB RAM, 2 CPU cores
- **Production**: 8GB RAM, 4 CPU cores
- **High Load**: 16GB RAM, 8 CPU cores

### Container Resource Limits
```yaml
# In docker-compose.mcp-fixed.yaml
services:
  mcp-forces:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'
```

### Performance Monitoring
```bash
# Monitor container resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check system performance
top -p $(pgrep -d',' python)
```

## Security Configuration

### Container Security
- Run containers as non-root users
- Use read-only file systems where possible
- Implement network segmentation
- Regular security updates

### API Security
- Rate limiting on all endpoints
- Input validation and sanitization
- CORS configuration for production domains
- SSL/TLS encryption for all communications

## Backup and Recovery

### Data Backup Strategy
```bash
# Backup interaction data (if using database)
pg_dump physics_assistant > backup_$(date +%Y%m%d).sql

# Backup container configurations
cp docker-compose.mcp-fixed.yaml backups/
cp -r mcp_tools/ backups/
```

### Recovery Procedures
```bash
# Restore from backup
psql physics_assistant < backup_20250916.sql

# Restart services
docker compose -f docker-compose.mcp-fixed.yaml restart
```

## Scaling and Load Handling

### Horizontal Scaling
- Deploy multiple MCP server instances behind load balancer
- Use container orchestration (Kubernetes) for auto-scaling
- Implement database read replicas for heavy read workloads

### Load Balancing Configuration
```nginx
upstream physics_mcp_forces {
    server localhost:10100;
    server localhost:10200;  # Additional instance
    server localhost:10300;  # Additional instance
}
```

## Troubleshooting

### Common Issues
1. **MCP Server Not Responding**
   ```bash
   # Check logs
   docker logs physics-assistant-forces

   # Restart specific server
   docker restart physics-assistant-forces
   ```

2. **High Memory Usage**
   ```bash
   # Check memory usage per container
   docker stats --no-stream

   # Restart high-memory containers
   docker restart <container_name>
   ```

3. **Database Connection Issues**
   ```bash
   # Test database connectivity
   pg_isready -h localhost -p 5432 -U physics_user
   ```

### Health Check Commands
```bash
# Full system health check
python database/fixes/test_physics_workflow.py

# Individual server tests
curl http://localhost:10100/ && echo "Forces: OK"
curl http://localhost:10101/ && echo "Kinematics: OK"
curl http://localhost:10103/ && echo "Math: OK"
curl http://localhost:10104/ && echo "Momentum: OK"
curl http://localhost:10105/ && echo "Energy: OK"
curl http://localhost:10106/ && echo "Angular: OK"
```

## Monitoring and Alerts

### Key Metrics to Monitor
- MCP server response times
- Container resource usage (CPU, memory)
- Student interaction rates
- Error rates and success metrics
- Data collection quality and completeness

### Alert Thresholds
- Response time > 2 seconds
- Memory usage > 80%
- Error rate > 5%
- Any MCP server offline

## Fine-Tuning Data Pipeline

### Data Collection Flow
1. Student interacts with UI
2. UI logs all interactions to database/files
3. Regular data export for model training
4. Privacy-aware data processing
5. Model fine-tuning with physics-specific datasets

### Data Export for Training
```bash
# Export interaction data for fine-tuning
python scripts/export_training_data.py --format jsonl --output training_data.jsonl
```

## Production Checklist

- [ ] All 6 MCP servers deployed and healthy
- [ ] UI configured for production environment
- [ ] Database connections tested and secure
- [ ] Logging and monitoring configured
- [ ] Backup procedures implemented
- [ ] Security measures in place
- [ ] Performance optimization applied
- [ ] Data collection pipeline validated
- [ ] Health check automation deployed
- [ ] Documentation updated and accessible

## Support and Maintenance

### Regular Maintenance Tasks
- Weekly health checks and performance reviews
- Monthly security updates and patches
- Quarterly capacity planning and scaling review
- Data quality audits for fine-tuning pipeline

### Contact Information
- Technical Issues: See container logs and health checks
- Performance Problems: Monitor resource usage and scaling
- Data Quality: Validate interaction logging and export procedures

---

**Deployment Status**: ✅ Ready for Production
**Last Updated**: 2025-09-16
**Version**: 1.0.0