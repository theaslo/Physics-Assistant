#!/usr/bin/env python3
"""
Unified Database Manager for Physics Assistant
Provides comprehensive access to PostgreSQL, Neo4j, and Redis databases
"""
import asyncio
import os
import json
import logging
import uuid
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
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
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
        
        success_count = sum(1 for result in results if result is True)
        
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
    
    async def log_interaction(self, user_id: str, agent_type: str, message: str, 
                            response: str, session_id: str = None, metadata: dict = None) -> str:
        """Log user interaction to PostgreSQL"""
        query = """
        INSERT INTO interactions (user_id, session_id, type, agent_type, request_data, response_data, success, metadata, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        """
        
        request_data = {'message': message}
        response_data = {'response': response}
        
        async with self.postgres.get_connection() as conn:
            interaction_id = await conn.fetchval(
                query, 
                user_id,
                session_id,
                'chat',  # type
                agent_type,  # agent_type
                json.dumps(request_data),  # request_data
                json.dumps(response_data),  # response_data
                True,  # success
                json.dumps(metadata or {}),  # metadata
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