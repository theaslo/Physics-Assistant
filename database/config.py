"""
Database configuration for Physics Assistant
Provides connection settings and database utilities
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class DatabaseConfig:
    """Database configuration manager"""
    
    def __init__(self, env_file: Optional[Path] = None):
        """Initialize database configuration
        
        Args:
            env_file: Path to .env file. If None, looks in current directory.
        """
        self.load_environment(env_file)
        self._validate_config()
    
    def load_environment(self, env_file: Optional[Path] = None):
        """Load environment variables from .env file"""
        if load_dotenv is None:
            print("Warning: python-dotenv not installed. Using system environment variables only.")
            return
        
        if env_file is None:
            env_file = Path(__file__).parent / '.env'
        
        if env_file.exists():
            load_dotenv(env_file)
    
    @property
    def postgres_config(self) -> Dict[str, Any]:
        """Get PostgreSQL connection configuration"""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'physics_assistant'),
            'user': os.getenv('POSTGRES_USER', 'physics_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'physics_secure_password_2024'),
            'ssl': os.getenv('POSTGRES_SSL_MODE', 'prefer')
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Get Redis connection configuration"""
        return {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD', 'redis_secure_password_2024'),
            'db': int(os.getenv('REDIS_DB', 0))
        }
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL"""
        config = self.postgres_config
        return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    
    @property
    def async_database_url(self) -> str:
        """Get async PostgreSQL connection URL"""
        config = self.postgres_config
        return f"postgresql+asyncpg://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL"""
        config = self.redis_config
        password_part = f":{config['password']}@" if config['password'] else ""
        return f"redis://{password_part}{config['host']}:{config['port']}/{config['db']}"
    
    @property
    def connection_pool_config(self) -> Dict[str, Any]:
        """Get connection pool configuration"""
        return {
            'min_connections': int(os.getenv('DB_POOL_MIN_CONNECTIONS', 5)),
            'max_connections': int(os.getenv('DB_POOL_MAX_CONNECTIONS', 20)),
            'acquire_timeout': int(os.getenv('DB_POOL_ACQUIRE_TIMEOUT', 30000)),
            'idle_timeout': int(os.getenv('DB_POOL_IDLE_TIMEOUT', 600000))
        }
    
    def _validate_config(self):
        """Validate configuration values"""
        required_vars = [
            'POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get formatted connection information for display"""
        postgres = self.postgres_config
        redis = self.redis_config
        
        return {
            'postgres': {
                'host': postgres['host'],
                'port': postgres['port'],
                'database': postgres['database'],
                'user': postgres['user'],
                'password_set': bool(postgres['password'])
            },
            'redis': {
                'host': redis['host'],
                'port': redis['port'],
                'password_set': bool(redis['password'])
            },
            'urls': {
                'database_url': self.database_url.replace(postgres['password'], '***'),
                'redis_url': self.redis_url.replace(redis['password'], '***') if redis['password'] else self.redis_url
            }
        }


# Global configuration instance
config = DatabaseConfig()


def get_database_config() -> DatabaseConfig:
    """Get the global database configuration instance"""
    return config


def parse_database_url(url: str) -> Dict[str, Any]:
    """Parse database URL into connection parameters
    
    Args:
        url: Database connection URL
        
    Returns:
        Dictionary of connection parameters
    """
    parsed = urlparse(url)
    
    return {
        'host': parsed.hostname or 'localhost',
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/') if parsed.path else '',
        'user': parsed.username or '',
        'password': parsed.password or ''
    }


# Environment-based configuration shortcuts
def is_development() -> bool:
    """Check if running in development environment"""
    return os.getenv('ENV', 'development').lower() == 'development'


def is_production() -> bool:
    """Check if running in production environment"""
    return os.getenv('ENV', 'development').lower() == 'production'


def get_debug_mode() -> bool:
    """Get database debug mode setting"""
    return os.getenv('DB_DEBUG_LOG', 'false').lower() in ('true', '1', 'yes')


if __name__ == '__main__':
    # Configuration test
    import json
    
    try:
        config = get_database_config()
        info = config.get_connection_info()
        
        print("Database Configuration")
        print("=" * 50)
        print(json.dumps(info, indent=2))
        print()
        print("Connection URLs:")
        print(f"  Database: {config.database_url}")
        print(f"  Redis: {config.redis_url}")
        print()
        print("Environment:")
        print(f"  Development: {is_development()}")
        print(f"  Production: {is_production()}")
        print(f"  Debug mode: {get_debug_mode()}")
        
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Please check your .env file and ensure all required variables are set.")