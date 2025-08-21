# Physics Assistant Database

This directory contains the PostgreSQL database setup and management tools for the Physics Assistant educational platform. The database tracks student interactions with physics tutoring agents, stores user sessions, and maintains learning progress analytics.

## Architecture Overview

The database consists of core tables for:
- **User Management**: Users, sessions, roles, and preferences
- **Interaction Logging**: All user interactions with MCP tools and agents
- **Message Storage**: Chat messages, system communications, and LaTeX content
- **Agent Tracking**: Specific MCP tool and physics agent invocations
- **Progress Tracking**: Learning analytics and achievement systems
- **File Management**: Uploaded physics diagrams and problem files

## Quick Start

1. **Setup environment**:
   ```bash
   make setup
   ```

2. **Start database services**:
   ```bash
   make start
   ```

3. **Run migrations**:
   ```bash
   make migrate
   ```

4. **Test connections**:
   ```bash
   make test
   ```

## Database Schema

### Core Tables

- `users` - User accounts with authentication and profile data
- `user_sessions` - Active session tracking with expiration
- `user_preferences` - User settings and learning preferences
- `user_progress` - Learning progress by physics topic
- `interactions` - Log of all system interactions
- `messages` - Chat messages and system communications
- `agent_calls` - Specific MCP tool and agent invocations
- `file_uploads` - Metadata for uploaded files and diagrams

### Key Features

- **UUID Primary Keys** - All tables use UUIDs for better distributed system support
- **JSONB Storage** - Flexible metadata and parameter storage
- **Comprehensive Indexing** - Optimized for common query patterns
- **Audit Trails** - Automatic timestamps and update tracking
- **Data Integrity** - Foreign key constraints and proper relationships

## Management Commands

Use the provided Makefile for common database operations:

### Service Management
```bash
make start          # Start database services
make stop           # Stop database services  
make status         # Check service status
make logs           # View database logs
```

### Database Operations
```bash
make migrate        # Run pending migrations
make migrate-status # Show migration status
make test           # Test database connections
make reset          # Reset database (destroys data)
```

### Backup Operations
```bash
make backup         # Create full backup
make backup-schema  # Schema-only backup
make backup-data    # Data-only backup
make backup-list    # List available backups
make restore BACKUP_FILE=file.sql.gz  # Restore from backup
```

### Development Shortcuts
```bash
make dev-setup      # Complete development setup
make quick-reset    # Quick development reset
make daily-backup   # Automated backup routine
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Database connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=physics_assistant
POSTGRES_USER=physics_user
POSTGRES_PASSWORD=your_secure_password

# Redis cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Backup settings
BACKUP_RETENTION_DAYS=30
```

## Tools and Scripts

### Migration Tool (`scripts/migrate.py`)
- Apply schema changes with version tracking
- Rollback support for failed migrations
- Force reapply for development

### Connection Test (`scripts/test_connection.py`)
- Comprehensive connection testing
- CRUD operation validation
- Performance metrics and latency testing

### Backup Tool (`scripts/backup.py`)
- Automated backup creation with compression
- Retention policy enforcement
- Restore operations with safety checks

## Docker Services

The setup includes:
- **PostgreSQL 15** - Main database with optimized configuration
- **Adminer** - Web-based database administration (port 8080)
- **Redis** - Caching and session storage

## Security Considerations

- Passwords use bcrypt hashing with salt
- Database connections use SCRAM-SHA-256 authentication
- Environment variables for all sensitive configuration
- Network isolation through Docker networks
- Regular security updates via Alpine base images

## Performance Features

- Connection pooling configured for high concurrent usage
- Optimized indexes for common query patterns
- JSONB storage for flexible metadata with GIN indexes
- Efficient pagination support with UUID-based ordering
- Automatic query logging for performance monitoring

## Monitoring and Maintenance

### Health Checks
- Automated container health monitoring
- Connection pool status tracking
- Query performance metrics

### Backup Strategy
- Automated daily backups with compression
- Configurable retention policies
- Point-in-time recovery capability
- Schema and data separation options

### Scaling Considerations
- Read replica support ready
- Horizontal partitioning preparation
- Connection pooling for high concurrency
- Stateless application design support

## Development Workflow

1. **Local Development**:
   ```bash
   make dev-setup     # Initialize everything
   make test          # Verify setup
   ```

2. **Schema Changes**:
   ```bash
   # Create new migration file in schema/
   make migrate       # Apply changes
   ```

3. **Testing**:
   ```bash
   make quick-reset   # Clean slate
   make test          # Verify functionality
   ```

4. **Backup Before Major Changes**:
   ```bash
   make backup        # Safety backup
   # Make changes
   make test          # Verify
   ```

## Production Deployment

For production deployments:

1. Use `make prod-setup` instead of `dev-setup`
2. Update `.env` with production credentials
3. Configure external PostgreSQL if not using Docker
4. Set up automated backups with `make daily-backup`
5. Monitor with provided health checks

## Troubleshooting

### Connection Issues
```bash
make status        # Check service status
make logs          # View error logs
make test          # Run diagnostics
```

### Migration Problems
```bash
make migrate-status    # Check migration state
make migrate-force     # Force reapply if needed
```

### Performance Issues
- Check `pg_stat_activity` for long-running queries
- Review index usage with query explain plans
- Monitor connection pool utilization
- Use Adminer for real-time query analysis

For additional help, check the individual script files for detailed options and error handling.