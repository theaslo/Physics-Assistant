#!/usr/bin/env python3
"""
Database connection test script for Physics Assistant
Tests PostgreSQL and Redis connections with comprehensive error handling
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import asyncpg
    import redis
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required packages. Install with: pip install asyncpg redis python-dotenv")
    print(f"Error: {e}")
    sys.exit(1)


class DatabaseTester:
    def __init__(self):
        self.load_config()
        self.test_results = {
            'postgres': {'connected': False, 'error': None, 'latency': None},
            'redis': {'connected': False, 'error': None, 'latency': None}
        }
    
    def load_config(self):
        """Load configuration from environment variables"""
        # Try to load from .env file if it exists
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'physics_assistant'),
            'user': os.getenv('POSTGRES_USER', 'physics_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'physics_secure_password_2024')
        }
        
        self.redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD', 'redis_secure_password_2024')
        }
    
    async def test_postgres_connection(self):
        """Test PostgreSQL database connection"""
        print("Testing PostgreSQL connection...")
        
        try:
            start_time = datetime.now()
            
            # Attempt to connect
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            # Test basic query
            result = await conn.fetchval('SELECT version()')
            
            # Test table existence
            tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
            """
            tables = await conn.fetch(tables_query)
            
            # Test user count
            user_count = await conn.fetchval('SELECT COUNT(*) FROM users')
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            await conn.close()
            
            self.test_results['postgres'] = {
                'connected': True,
                'error': None,
                'latency': round(latency, 2),
                'version': result,
                'tables': [row['table_name'] for row in tables],
                'user_count': user_count
            }
            
            print(f"✅ PostgreSQL connection successful (latency: {latency:.2f}ms)")
            print(f"   Database version: {result}")
            print(f"   Tables found: {len(tables)}")
            print(f"   User records: {user_count}")
            
        except Exception as e:
            self.test_results['postgres'] = {
                'connected': False,
                'error': str(e),
                'latency': None
            }
            print(f"❌ PostgreSQL connection failed: {e}")
    
    def test_redis_connection(self):
        """Test Redis connection"""
        print("\nTesting Redis connection...")
        
        try:
            start_time = datetime.now()
            
            # Create Redis client
            client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                password=self.redis_config['password'],
                decode_responses=True
            )
            
            # Test connection with ping
            client.ping()
            
            # Test basic operations
            test_key = 'physics_assistant_test'
            client.set(test_key, 'connection_test', ex=60)
            test_value = client.get(test_key)
            client.delete(test_key)
            
            # Get Redis info
            info = client.info()
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            self.test_results['redis'] = {
                'connected': True,
                'error': None,
                'latency': round(latency, 2),
                'version': info.get('redis_version'),
                'memory_used': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients')
            }
            
            print(f"✅ Redis connection successful (latency: {latency:.2f}ms)")
            print(f"   Redis version: {info.get('redis_version')}")
            print(f"   Memory used: {info.get('used_memory_human')}")
            print(f"   Connected clients: {info.get('connected_clients')}")
            
            client.close()
            
        except Exception as e:
            self.test_results['redis'] = {
                'connected': False,
                'error': str(e),
                'latency': None
            }
            print(f"❌ Redis connection failed: {e}")
    
    async def test_crud_operations(self):
        """Test basic CRUD operations on the database"""
        if not self.test_results['postgres']['connected']:
            print("\n⚠️ Skipping CRUD tests - PostgreSQL connection failed")
            return
        
        print("\nTesting basic CRUD operations...")
        
        try:
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            # Test user creation (CREATE)
            test_email = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}@test.com"
            user_id = await conn.fetchval(
                """
                INSERT INTO users (email, username, password_hash, first_name, last_name) 
                VALUES ($1, $2, $3, $4, $5) 
                RETURNING id
                """,
                test_email, f"testuser_{datetime.now().strftime('%H%M%S')}", 
                "test_hash", "Test", "User"
            )
            
            # Test user retrieval (READ)
            user = await conn.fetchrow(
                "SELECT email, username, first_name, last_name FROM users WHERE id = $1",
                user_id
            )
            
            # Test user update (UPDATE)
            await conn.execute(
                "UPDATE users SET first_name = $1 WHERE id = $2",
                "Updated Test", user_id
            )
            
            # Test session creation
            session_id = await conn.fetchval(
                """
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES ($1, $2, NOW() + INTERVAL '1 day')
                RETURNING id
                """,
                user_id, f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Test interaction logging
            interaction_id = await conn.fetchval(
                """
                INSERT INTO interactions (user_id, session_id, type, agent_type, request_data, success)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                user_id, session_id, 'chat', 'kinematics', '{"test": true}', True
            )
            
            # Cleanup (DELETE)
            await conn.execute("DELETE FROM interactions WHERE id = $1", interaction_id)
            await conn.execute("DELETE FROM user_sessions WHERE id = $1", session_id)
            await conn.execute("DELETE FROM users WHERE id = $1", user_id)
            
            await conn.close()
            
            print("✅ CRUD operations test successful")
            print(f"   Created user: {user['email']}")
            print(f"   Created session and interaction records")
            print("   Successfully cleaned up test data")
            
        except Exception as e:
            print(f"❌ CRUD operations test failed: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("DATABASE CONNECTION TEST SUMMARY")
        print("="*60)
        
        postgres_status = "✅ Connected" if self.test_results['postgres']['connected'] else "❌ Failed"
        redis_status = "✅ Connected" if self.test_results['redis']['connected'] else "❌ Failed"
        
        print(f"PostgreSQL: {postgres_status}")
        if self.test_results['postgres']['connected']:
            print(f"  Latency: {self.test_results['postgres']['latency']}ms")
            print(f"  Tables: {len(self.test_results['postgres'].get('tables', []))}")
        else:
            print(f"  Error: {self.test_results['postgres']['error']}")
        
        print(f"\nRedis: {redis_status}")
        if self.test_results['redis']['connected']:
            print(f"  Latency: {self.test_results['redis']['latency']}ms")
        else:
            print(f"  Error: {self.test_results['redis']['error']}")
        
        all_connected = (self.test_results['postgres']['connected'] and 
                        self.test_results['redis']['connected'])
        
        print(f"\nOverall Status: {'✅ All systems operational' if all_connected else '❌ Some systems failing'}")
        print("="*60)
        
        return all_connected


async def main():
    """Main test function"""
    print("Physics Assistant Database Connection Tester")
    print("=" * 50)
    
    tester = DatabaseTester()
    
    # Test PostgreSQL
    await tester.test_postgres_connection()
    
    # Test Redis
    tester.test_redis_connection()
    
    # Test CRUD operations
    await tester.test_crud_operations()
    
    # Print summary
    success = tester.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())