#!/usr/bin/env python3
"""
Unified Database Manager for Physics Assistant
Provides comprehensive access to PostgreSQL, Neo4j, and Redis databases
"""
import asyncio
import os
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import asyncpg
import redis
from neo4j import GraphDatabase, Driver as Neo4jDriver
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).resolve().parent / "schema" / "01_core_tables.sql"

CANONICAL_AGENT_TYPES = {
    "kinematics",
    "forces",
    "energy",
    "momentum",
    "angular_motion",
    "math",
    "math_helper",
    "thermodynamics",
    "waves",
    "electromagnetism",
    "optics",
    "modern_physics",
}

AGENT_TYPE_ALIASES = {
    "force": "forces",
    "forces": "forces",
    "kinematic": "kinematics",
    "kinematics": "kinematics",
    "angular": "angular_motion",
    "angular_motion": "angular_motion",
    "angularmotion": "angular_motion",
    "math": "math",
    "math_helper": "math",
    "momentum": "momentum",
    "energy": "energy",
    "thermodynamics": "thermodynamics",
    "thermodynamic": "thermodynamics",
    "waves": "waves",
    "wave": "waves",
    "electromagnetism": "electromagnetism",
    "electromagnetic": "electromagnetism",
    "em": "electromagnetism",
    "optics": "optics",
    "optic": "optics",
    "modern_physics": "modern_physics",
    "modernphysics": "modern_physics",
}

INTERACTION_TYPE_ALIASES = {
    "chat": "chat",
    "mcp_tool": "mcp_tool",
    "tool_call": "mcp_tool",
    "agent_call": "agent_call",
    "agent": "agent_call",
    "file_upload": "file_upload",
    "calculation": "calculation",
}


def normalize_agent_type(agent_type: Optional[str]) -> Optional[str]:
    """Normalize UI, Strands, MCP, and legacy agent IDs to enum values."""
    if not agent_type:
        return None

    normalized = agent_type.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)

    for prefix in ("physics_", "mcp_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    for suffix in ("_agent", "_mcp_server", "_server"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    return AGENT_TYPE_ALIASES.get(normalized, normalized)


def normalize_interaction_type(interaction_type: Optional[str]) -> str:
    """Normalize caller interaction types to the Postgres enum values."""
    if not interaction_type:
        return "chat"
    normalized = interaction_type.strip().lower().replace("-", "_").replace(" ", "_")
    return INTERACTION_TYPE_ALIASES.get(normalized, normalized)


def stable_user_uuid(user_id: str) -> uuid.UUID:
    """Return a stable UUID for UUID and external string user IDs alike."""
    try:
        return uuid.UUID(str(user_id))
    except (TypeError, ValueError):
        return uuid.uuid5(uuid.NAMESPACE_URL, f"physics-assistant:user:{user_id}")


def _external_username(user_id: str, resolved_user_id: uuid.UUID) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(user_id).strip().lower()).strip("_")
    slug = slug or "external"
    return f"api_{slug[:72]}_{resolved_user_id.hex[:8]}"

class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self, env_file: str = ".env.example"):
        load_dotenv(env_file)
        
        # PostgreSQL configuration
        self.postgres = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'physics_assistant'),
            'user': os.getenv('POSTGRES_USER', 'physics_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'physics_secure_password_2024'),
            'min_connections': int(os.getenv('DB_POOL_MIN_CONNECTIONS', 5)),
            'max_connections': int(os.getenv('DB_POOL_MAX_CONNECTIONS', 20)),
        }
        
        # Neo4j configuration
        self.neo4j = {
            'uri': f"bolt://{os.getenv('NEO4J_HOST', 'localhost')}:{os.getenv('NEO4J_PORT', '7687')}",
            'user': os.getenv('NEO4J_USER', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'physics_graph_password_2024'),
        }
        
        # Redis configuration  
        self.redis = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD', 'redis_secure_password_2024'),
            'decode_responses': True,
        }

class PostgreSQLManager:
    """PostgreSQL database connection manager"""
    
    def __init__(self, config: dict):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                min_size=self.config['min_connections'],
                max_size=self.config['max_connections'],
                command_timeout=30
            )
            logger.info("✅ PostgreSQL connection pool initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
            return False

    async def ensure_schema(self) -> bool:
        """Apply additive, idempotent schema objects needed at runtime."""
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        try:
            schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")
            async with self.get_connection() as conn:
                await conn.execute(schema_sql)
            logger.info("✅ PostgreSQL schema verified")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to verify PostgreSQL schema: {e}")
            return False
    
    async def close(self):
        """Close PostgreSQL connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("🔒 PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get PostgreSQL connection from pool"""
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def health_check(self) -> dict:
        """Check PostgreSQL health"""
        try:
            async with self.get_connection() as conn:
                version = await conn.fetchval("SELECT version()")
                table_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                
                return {
                    'status': 'healthy',
                    'version': version,
                    'tables': table_count,
                    'pool_size': self.pool.get_size() if self.pool else 0,
                    'pool_idle': self.pool.get_idle_size() if self.pool else 0
                }
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

class Neo4jManager:
    """Neo4j graph database connection manager"""
    
    def __init__(self, config: dict):
        self.config = config
        self.driver: Optional[Neo4jDriver] = None
    
    async def initialize(self):
        """Initialize Neo4j driver"""
        try:
            self.driver = GraphDatabase.driver(
                self.config['uri'],
                auth=(self.config['user'], self.config['password'])
            )
            
            # Test connection
            await asyncio.to_thread(self.driver.verify_connectivity)
            logger.info("✅ Neo4j driver initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j: {e}")
            return False
    
    async def close(self):
        """Close Neo4j driver"""
        if self.driver:
            await asyncio.to_thread(self.driver.close)
            logger.info("🔒 Neo4j driver closed")
    
    def get_session(self):
        """Get Neo4j session"""
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self.driver.session()
    
    async def run_query(self, query: str, parameters: dict = None) -> List[dict]:
        """Execute Neo4j query and return results"""
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        def _run_query():
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        
        return await asyncio.to_thread(_run_query)
    
    async def health_check(self) -> dict:
        """Check Neo4j health"""
        try:
            def _health_check():
                with self.driver.session() as session:
                    # Get database info
                    result = session.run("CALL db.info()")
                    db_info = result.single()
                    
                    # Count nodes and relationships
                    node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                    rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                    
                    return {
                        'status': 'healthy',
                        'database_info': dict(db_info) if db_info else {},
                        'nodes': node_count,
                        'relationships': rel_count
                    }
            
            return await asyncio.to_thread(_health_check)
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

class RedisManager:
    """Redis cache connection manager"""
    
    def __init__(self, config: dict):
        self.config = config
        self.client: Optional[redis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.client = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                password=self.config['password'],
                decode_responses=self.config['decode_responses'],
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await asyncio.to_thread(self.client.ping)
            logger.info("✅ Redis client initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await asyncio.to_thread(self.client.close)
            logger.info("🔒 Redis connection closed")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        return await asyncio.to_thread(self.client.get, key)
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with optional TTL"""
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        return await asyncio.to_thread(self.client.set, key, value, ex=ttl)
    
    async def delete(self, key: str) -> int:
        """Delete key from Redis"""
        if not self.client:
            raise RuntimeError("Redis client not initialized")
        return await asyncio.to_thread(self.client.delete, key)
    
    async def health_check(self) -> dict:
        """Check Redis health"""
        try:
            def _health_check():
                info = self.client.info()
                memory_info = self.client.info('memory')
                
                return {
                    'status': 'healthy',
                    'version': info.get('redis_version'),
                    'memory_used': memory_info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'keyspace': info.get('db0', {})
                }
            
            return await asyncio.to_thread(_health_check)
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

class DatabaseManager:
    """Unified database manager for Physics Assistant"""
    
    def __init__(self, config_file: str = ".env.example"):
        self.config = DatabaseConfig(config_file)
        self.postgres = PostgreSQLManager(self.config.postgres)
        self.neo4j = Neo4jManager(self.config.neo4j)
        self.redis = RedisManager(self.config.redis)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all database connections"""
        logger.info("🚀 Initializing Physics Assistant Database Manager")
        
        # Initialize all databases
        results = await asyncio.gather(
            self.postgres.initialize(),
            self.neo4j.initialize(), 
            self.redis.initialize(),
            return_exceptions=True
        )
        
        postgres_ready, neo4j_ready, redis_ready = results

        if postgres_ready is True:
            postgres_ready = await self.postgres.ensure_schema()

        normalized_results = [postgres_ready, neo4j_ready, redis_ready]
        success_count = sum(1 for result in normalized_results if result is True)
        
        if success_count == 3:
            self._initialized = True
            logger.info("✅ All databases initialized successfully")
            return True
        else:
            logger.warning(f"⚠️ Only {success_count}/3 databases initialized")
            return False
    
    async def close(self):
        """Close all database connections"""
        if self._initialized:
            await asyncio.gather(
                self.postgres.close(),
                self.neo4j.close(),
                self.redis.close(),
                return_exceptions=True
            )
            self._initialized = False
            logger.info("🔒 All database connections closed")
    
    async def health_check(self) -> dict:
        """Comprehensive health check for all databases"""
        if not self._initialized:
            return {'status': 'uninitialized', 'databases': {}}
        
        health_checks = await asyncio.gather(
            self.postgres.health_check(),
            self.neo4j.health_check(),
            self.redis.health_check(),
            return_exceptions=True
        )
        
        postgres_health, neo4j_health, redis_health = health_checks
        
        # Count healthy databases
        healthy_count = sum(1 for health in health_checks 
                           if isinstance(health, dict) and health.get('status') == 'healthy')
        
        overall_status = 'healthy' if healthy_count == 3 else 'degraded' if healthy_count > 0 else 'unhealthy'
        
        return {
            'status': overall_status,
            'healthy_databases': f"{healthy_count}/3",
            'timestamp': datetime.now().isoformat(),
            'databases': {
                'postgresql': postgres_health if isinstance(postgres_health, dict) else {'status': 'error', 'error': str(postgres_health)},
                'neo4j': neo4j_health if isinstance(neo4j_health, dict) else {'status': 'error', 'error': str(neo4j_health)},
                'redis': redis_health if isinstance(redis_health, dict) else {'status': 'error', 'error': str(redis_health)}
            }
        }
    
    # Convenience methods for common operations
    
    async def _ensure_user(self, conn, user_id: str) -> uuid.UUID:
        """Create a stable synthetic user row for external UI/agent IDs."""
        resolved_user_id = stable_user_uuid(user_id)
        username = _external_username(user_id, resolved_user_id)
        email = f"{username}@physics-assistant.local"

        await conn.execute(
            """
            INSERT INTO users (id, email, username, password_hash, is_verified, metadata)
            VALUES ($1, $2, $3, $4, TRUE, $5::jsonb)
            ON CONFLICT (id) DO NOTHING
            """,
            resolved_user_id,
            email,
            username,
            "external-user",
            json.dumps({"external_user_id": str(user_id)}),
        )

        return resolved_user_id

    async def _resolve_session_id(self, conn, session_id: Optional[str], user_id: uuid.UUID) -> Optional[uuid.UUID]:
        if not session_id:
            return None

        try:
            resolved_session_id = uuid.UUID(str(session_id))
        except (TypeError, ValueError):
            return None

        session_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM user_sessions WHERE id = $1 AND user_id = $2)",
            resolved_session_id,
            user_id,
        )
        return resolved_session_id if session_exists else None

    async def log_interaction(
        self,
        user_id: str,
        agent_type: str,
        message: str = "",
        response: str = "",
        session_id: str = None,
        metadata: dict = None,
        interaction_type: str = "chat",
        execution_time_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_data: Optional[dict] = None,
        response_data: Optional[dict] = None,
    ) -> str:
        """Log user interaction to PostgreSQL"""
        normalized_agent_type = normalize_agent_type(agent_type)
        normalized_interaction_type = normalize_interaction_type(interaction_type)

        if normalized_agent_type not in CANONICAL_AGENT_TYPES:
            raise ValueError(f"Unsupported agent_type: {agent_type}")

        query = """
        INSERT INTO interactions (
            user_id, session_id, type, agent_type, request_data, response_data,
            execution_time_ms, success, error_message, metadata, created_at
        )
        VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10::jsonb, $11)
        RETURNING id
        """

        request_payload = request_data or {'message': message}
        response_payload = response_data or {'response': response}
        metadata_payload = metadata or {}
        
        async with self.postgres.get_connection() as conn:
            resolved_user_id = await self._ensure_user(conn, user_id)
            resolved_session_id = await self._resolve_session_id(conn, session_id, resolved_user_id)

            interaction_id = await conn.fetchval(
                query, 
                resolved_user_id,
                resolved_session_id,
                normalized_interaction_type,
                normalized_agent_type,
                json.dumps(request_payload),
                json.dumps(response_payload),
                execution_time_ms,
                success,
                error_message,
                json.dumps(metadata_payload),
                datetime.now()  # created_at
            )
            return str(interaction_id)
    
    async def get_physics_concepts(self, category: str = None) -> List[dict]:
        """Get physics concepts from Neo4j graph"""
        query = "MATCH (c:Concept) "
        if category:
            query += "WHERE c.category = $category "
        query += "RETURN c.name as name, c.description as description, c.category as category"
        
        return await self.neo4j.run_query(query, {'category': category} if category else None)
    
    async def cache_user_session(self, session_id: str, user_data: dict, ttl: int = 3600) -> bool:
        """Cache user session data in Redis"""
        return await self.redis.set(f"session:{session_id}", json.dumps(user_data), ttl)
    
    async def get_user_session(self, session_id: str) -> Optional[dict]:
        """Get cached user session from Redis"""
        session_data = await self.redis.get(f"session:{session_id}")
        return json.loads(session_data) if session_data else None

# Async context manager for easy usage
@asynccontextmanager
async def get_db_manager(config_file: str = ".env.example"):
    """Async context manager for database operations"""
    db_manager = DatabaseManager(config_file)
    
    try:
        success = await db_manager.initialize()
        if not success:
            raise RuntimeError("Failed to initialize database manager")
        yield db_manager
    finally:
        await db_manager.close()

# Example usage functions
async def example_usage():
    """Example of how to use the database manager"""
    async with get_db_manager() as db:
        # Health check
        health = await db.health_check()
        print("Health check:", health)
        
        # Get a sample user UUID from database
        async with db.postgres.get_connection() as conn:
            sample_user = await conn.fetchrow("SELECT id, username FROM users LIMIT 1")
            if not sample_user:
                print("⚠️ No users found, skipping interaction logging test")
                return
        
        user_id = str(sample_user['id'])
        print(f"Using sample user: {sample_user['username']} ({user_id})")
        
        # Get or create a user session for the test
        async with db.postgres.get_connection() as conn:
            session_record = await conn.fetchrow("SELECT id FROM user_sessions WHERE user_id = $1 LIMIT 1", sample_user['id'])
            test_session_id = str(session_record['id']) if session_record else None
        
        # Log interaction
        interaction_id = await db.log_interaction(
            user_id=user_id,
            agent_type="kinematics", 
            message="What is velocity?",
            response="Velocity is the rate of change of position.",
            session_id=test_session_id
        )
        print(f"Logged interaction: {interaction_id}")
        
        # Get physics concepts
        concepts = await db.get_physics_concepts("mechanics")
        print(f"Found {len(concepts)} mechanics concepts")
        for concept in concepts[:3]:  # Show first 3
            print(f"  - {concept['name']}: {concept['description']}")
        
        # Cache session
        await db.cache_user_session("test_session", {"user_id": user_id, "level": "beginner"})
        session = await db.get_user_session("test_session")
        print("Cached session:", session)

if __name__ == "__main__":
    asyncio.run(example_usage())
