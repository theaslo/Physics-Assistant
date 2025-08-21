# Physics Assistant - Complete Deployment Guide

A comprehensive physics education platform with AI tutoring, database integration, and advanced analytics. This guide explains how to deploy all components: API, UI, MCP servers, and databases.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Physics Assistant Platform                   │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│   Frontend UI   │      APIs       │   MCP Servers   │ Database │
│   (Streamlit)   │    (FastAPI)    │   (Physics)     │ Systems  │
│   + Dashboard   │   + Analytics   │   Tools         │ + Cache  │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
```

### Core Components
- **Frontend**: Streamlit web UI + React analytics dashboard
- **APIs**: Physics agents API + Database API + Dashboard API
- **MCP Servers**: 6 specialized physics calculation microservices
- **Databases**: PostgreSQL + Neo4j + Redis for complete data management
- **Analytics**: Advanced ML-powered learning analytics and tutoring

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, or similar)
- **CPU**: Minimum 4 cores (8+ recommended)
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: Minimum 50GB (100GB+ recommended)

### Software Requirements
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose (if not included)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Python and UV for local development (optional)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 🚀 Quick Start Deployment Options

### Option 1: Development Environment (Fastest)
```bash
# Clone and start development environment
git clone <repository-url>
cd Physics-Assistant

# Start core services only
docker compose -f docker-compose.development.yml up -d

# Access the application
echo "🌐 Frontend UI: http://localhost:8501"
echo "📊 Analytics: http://localhost:3000"
echo "📖 API Docs: http://localhost:8000/docs"
```

### Option 2: Full Production Deployment
```bash
# 1. Configure environment
cp .env.production .env.production.local
# Edit .env.production.local with your settings

# 2. Deploy complete system
./scripts/deploy-production.sh --env production

# 3. Validate deployment
./scripts/validate-deployment.sh --env production

# 4. Check system health
./scripts/health-check.sh --env production
```

### Option 3: Component-by-Component Setup
Follow the detailed sections below for custom deployment.

## 🗄️ Database Setup

### PostgreSQL (Primary Database)
```bash
# Start PostgreSQL container
docker compose -f database/docker-compose.yml up -d postgres

# Initialize schema
cd database
python setup_schema.py

# Verify connection
python scripts/test_connection.py
```

### Neo4j (Knowledge Graph)
```bash
# Start Neo4j container
docker compose -f database/docker-compose.yml up -d neo4j

# Initialize physics knowledge graph
python setup_complete_knowledge_graph.py

# Access Neo4j browser: http://localhost:7474
# Default credentials: neo4j/physics123
```

### Redis (Caching)
```bash
# Start Redis container
docker compose -f database/docker-compose.yml up -d redis

# Test Redis connection
docker exec -it physics-redis redis-cli ping
```

### Database API Server
```bash
# Start database API
cd database
python api_server.py

# API available at: http://localhost:8001
# Documentation: http://localhost:8001/docs
```

## 🔧 API Services Setup

### Physics Agents API
```bash
# Start physics agents API
cd UI/api
uv run python main.py

# API available at: http://localhost:8000
# Endpoints:
# - POST /physics/kinematics
# - POST /physics/forces  
# - POST /physics/energy
# - POST /physics/momentum
# - POST /physics/angular-motion
```

### Analytics Dashboard API
```bash
# Start analytics API
cd database
python dashboard_api_server.py

# API available at: http://localhost:8002
# Real-time analytics endpoints for learning data
```

## 🖥️ Frontend UI Setup

### Streamlit Main UI
```bash
# Start Streamlit frontend
cd UI/frontend
uv run streamlit run app.py --server.port 8501

# Access at: http://localhost:8501
# Features:
# - Physics problem solving interface
# - File upload for diagrams
# - Real-time LaTeX rendering
# - Progress tracking
```

### React Analytics Dashboard
```bash
# Build and start React dashboard
cd dashboard-ui
npm install
npm run build
npm run preview

# Access at: http://localhost:3000
# Features:
# - Real-time learning analytics
# - Student progress visualization
# - System performance metrics
```

## ⚙️ MCP Servers Setup

### MCP Physics Tools
```bash
# Start all MCP physics servers
cd mcp_tools

# Individual MCP servers:
uv run python -m physics_mcp_tools.kinematics_mcp_server
uv run python -m physics_mcp_tools.forces_mcp_server
uv run python -m physics_mcp_tools.energy_mcp_server
uv run python -m physics_mcp_tools.momentum_mcp_server
uv run python -m physics_mcp_tools.angular_motion_mcp_server
uv run python -m physics_mcp_tools.math_mcp_server

# Or use Docker:
docker compose -f docker-compose.yml up mcp-kinematics mcp-forces mcp-energy mcp-momentum mcp-angular mcp-math
```

### MCP Server Capabilities
- **Kinematics**: Motion calculations, trajectory analysis
- **Forces**: Vector analysis, equilibrium, friction
- **Energy**: Work-energy theorem, conservation laws  
- **Momentum**: Collisions, impulse calculations
- **Angular Motion**: Rotational dynamics, torque
- **Math Helper**: Trigonometry, algebra, calculus

## 🧠 Advanced Analytics (Phase 6)

### Intelligent Tutoring System
```bash
# Start adaptive tutoring system
cd database/analytics
python start_phase6_2_tutoring.py

# Features:
# - Real-time difficulty adjustment
# - Learning style detection
# - Personalized problem generation
# - Access: http://localhost:8502
```

### Predictive Analytics Engine
```bash
# Start predictive analytics
cd database/analytics
python start_phase_6_3_system.py

# Features:
# - Performance prediction (>85% accuracy)
# - Early warning system
# - Time-to-mastery estimation
# - Access: http://localhost:8503
```

## 🐳 Docker Deployment Options

### Development (Quick Start)
```bash
docker compose -f docker-compose.development.yml up -d
```

### Production (Full Features)
```bash
docker compose -f docker-compose.production.yml up -d
```

### Production Optimized (High Performance)
```bash
docker compose -f docker-compose.production-optimized.yml up -d
```

## 📊 Monitoring & Observability

### Start Monitoring Stack
```bash
# Start Prometheus, Grafana, and logging
cd database/monitoring
./start_monitoring.sh

# Access points:
# - Grafana: http://localhost:3001 (admin/admin)
# - Prometheus: http://localhost:9090
```

### System Health Monitoring
```bash
# Check all services health
./scripts/health-check.sh --verbose

# Monitor specific component
./scripts/health-check.sh --component database
./scripts/health-check.sh --component api
./scripts/health-check.sh --component mcp
```

## 🔒 Security & Backup

### Backup System
```bash
# Deploy automated backup system
cd backup
./deploy-backup-system.sh production

# Features:
# - Automated PostgreSQL, Neo4j, Redis backups
# - Encryption and cloud storage
# - Disaster recovery procedures
```

### Security Hardening
```bash
# Apply security configurations
docker compose -f docker-compose.production-optimized.yml up -d

# Features:
# - Container vulnerability scanning
# - Network security policies
# - Secrets management with Vault
# - SSL/TLS termination
```

## 🌐 Access Points Summary

After successful deployment:

| Service | URL | Description |
|---------|-----|-------------|
| Main UI | http://localhost:8501 | Streamlit physics assistant interface |
| Analytics Dashboard | http://localhost:3000 | React-based learning analytics |
| Physics API | http://localhost:8000 | RESTful physics agents API |
| Database API | http://localhost:8001 | Database operations API |
| Dashboard API | http://localhost:8002 | Analytics and dashboard API |
| Tutoring System | http://localhost:8502 | Intelligent adaptive tutoring |
| Predictive Analytics | http://localhost:8503 | Performance prediction system |
| API Documentation | http://localhost:8000/docs | Swagger/OpenAPI docs |
| Neo4j Browser | http://localhost:7474 | Graph database interface |
| Database Admin | http://localhost:8080 | Adminer database management |
| Grafana Monitoring | http://localhost:3001 | System monitoring dashboards |
| Prometheus Metrics | http://localhost:9090 | Metrics collection interface |

## 🧪 Testing & Validation

### API Testing
```bash
# Test physics API endpoints
curl -X POST "http://localhost:8000/physics/kinematics" \
  -H "Content-Type: application/json" \
  -d '{"problem": "A ball is thrown upward with initial velocity 20 m/s. Find maximum height."}'

# Test database API
curl "http://localhost:8001/health"

# Test MCP integration
curl -X POST "http://localhost:8000/physics/forces" \
  -H "Content-Type: application/json" \
  -d '{"problem": "Find the tension in a rope supporting a 10kg mass."}'
```

### System Validation
```bash
# Comprehensive deployment validation
./scripts/validate-deployment.sh --env production --verbose

# Test specific components
python database/test_dashboard_api.py
python database/analytics/test_analytics_suite.py
```

## 🔧 Configuration Files

### Environment Configuration
```bash
# Development
.env.development

# Production
.env.production

# Production Local (create from template)
cp .env.production .env.production.local
# Edit with your specific settings
```

### Key Configuration Options
```bash
# Database settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=physics_assistant
NEO4J_URI=bolt://localhost:7687
REDIS_URL=redis://localhost:6379

# API settings
API_PORT=8000
DATABASE_API_PORT=8001
DASHBOARD_API_PORT=8002

# UI settings
STREAMLIT_PORT=8501
REACT_PORT=3000

# Security
JWT_SECRET_KEY=your_secure_key_here
ENCRYPTION_KEY=your_encryption_key_here
```

## 🚨 Troubleshooting

### Common Issues

**Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8501
# Kill process using port
sudo kill -9 $(lsof -t -i:8501)
```

**Database Connection Issues**
```bash
# Check database containers
docker compose -f database/docker-compose.yml ps

# Check database logs
docker compose -f database/docker-compose.yml logs postgres
docker compose -f database/docker-compose.yml logs neo4j
```

**MCP Server Issues**
```bash
# Check MCP server status
docker compose ps | grep mcp

# Restart specific MCP server
docker compose restart mcp-kinematics
```

**Memory Issues**
```bash
# Check system resources
docker stats

# Adjust memory limits in docker-compose files
# Look for 'mem_limit' settings
```

### Log Analysis
```bash
# View application logs
docker compose logs -f streamlit-ui
docker compose logs -f physics-api
docker compose logs -f database-api

# View system logs
journalctl -u docker -f
```

## 📚 Additional Resources

### Documentation
- `/PRODUCTION_DEPLOYMENT_GUIDE.md` - Detailed production setup
- `/database/analytics/PHASE_6_2_README.md` - Intelligent tutoring system
- `/database/analytics/PHASE_6_3_README.md` - Predictive analytics
- `/backup/backup-system-deployment.md` - Backup system guide

### Development
- `/UI/frontend/README.md` - Frontend development guide
- `/UI/api/agent_README.md` - Physics agents development
- `/mcp_tools/README.md` - MCP tools development
- `/database/README.md` - Database development guide

### Performance Tuning
- Adjust Docker memory limits in compose files
- Configure PostgreSQL for your workload in `docker/database/postgresql/postgresql.conf`
- Optimize Redis configuration in `docker/database/redis/redis.conf`
- Tune Neo4j settings in `docker/database/neo4j/neo4j.conf`

## 📞 Support

For deployment issues:
1. Check the troubleshooting section above
2. Review component-specific logs
3. Validate your configuration files
4. Ensure all prerequisites are met
5. Check system resources (CPU, memory, disk)

The Physics Assistant platform provides a comprehensive educational AI system with enterprise-grade deployment capabilities, advanced analytics, and production-ready monitoring and backup systems.