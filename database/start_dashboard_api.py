#!/usr/bin/env python3
"""
Production startup script for Dashboard API Server
Configures middleware, initializes services, and starts the server with proper settings.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dashboard_api_server import app
from dashboard_middleware import (
    RateLimitingMiddleware,
    ResponseCompressionMiddleware,
    CacheOptimizationMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    PerformanceMonitoringMiddleware
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard_api.log')
    ]
)
logger = logging.getLogger(__name__)

class DashboardAPIConfig:
    """Configuration for Dashboard API Server"""
    
    def __init__(self):
        # Server configuration
        self.host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
        self.port = int(os.getenv("DASHBOARD_PORT", "8001"))
        self.reload = os.getenv("DASHBOARD_RELOAD", "false").lower() == "true"
        self.workers = int(os.getenv("DASHBOARD_WORKERS", "1"))
        
        # Redis configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        
        # Cache configuration
        self.cache_ttl_default = int(os.getenv("CACHE_TTL_DEFAULT", "300"))
        self.cache_max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        
        # Rate limiting configuration
        self.rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.rate_limit_requests_per_hour = int(os.getenv("RATE_LIMIT_RPH", "1000"))
        
        # Performance configuration
        self.compression_enabled = os.getenv("COMPRESSION_ENABLED", "true").lower() == "true"
        self.compression_min_size = int(os.getenv("COMPRESSION_MIN_SIZE", "1000"))
        self.slow_request_threshold = float(os.getenv("SLOW_REQUEST_THRESHOLD", "2.0"))
        
        # Security configuration
        self.security_headers_enabled = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"
        
        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.request_logging_enabled = os.getenv("REQUEST_LOGGING_ENABLED", "true").lower() == "true"

async def setup_redis_connection(config: DashboardAPIConfig) -> redis.Redis:
    """Setup Redis connection with configuration"""
    try:
        redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            max_connections=20
        )
        
        # Test connection
        await redis_client.ping()
        logger.info(f"✅ Redis connected successfully at {config.redis_host}:{config.redis_port}")
        return redis_client
        
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        logger.info("📝 Continuing without Redis - using memory-only caching")
        return None

def setup_middleware(app: FastAPI, config: DashboardAPIConfig, redis_client: redis.Redis = None):
    """Setup all middleware in correct order"""
    
    # Performance monitoring (should be first to capture all requests)
    if config.slow_request_threshold > 0:
        app.add_middleware(
            PerformanceMonitoringMiddleware,
            slow_request_threshold=config.slow_request_threshold
        )
        logger.info("✅ Performance monitoring middleware enabled")
    
    # Request logging
    if config.request_logging_enabled:
        app.add_middleware(
            RequestLoggingMiddleware,
            log_level=config.log_level
        )
        logger.info("✅ Request logging middleware enabled")
    
    # Security headers
    if config.security_headers_enabled:
        app.add_middleware(SecurityHeadersMiddleware)
        logger.info("✅ Security headers middleware enabled")
    
    # Cache optimization
    app.add_middleware(
        CacheOptimizationMiddleware,
        redis_client=redis_client
    )
    logger.info("✅ Cache optimization middleware enabled")
    
    # Response compression
    if config.compression_enabled:
        app.add_middleware(
            ResponseCompressionMiddleware,
            minimum_size=config.compression_min_size,
            compression_level=6
        )
        logger.info("✅ Response compression middleware enabled")
    
    # Rate limiting (should be early to protect against abuse)
    if config.rate_limit_enabled:
        app.add_middleware(
            RateLimitingMiddleware,
            redis_client=redis_client
        )
        logger.info("✅ Rate limiting middleware enabled")

def validate_environment():
    """Validate environment and dependencies"""
    required_modules = [
        'fastapi',
        'uvicorn',
        'redis',
        'pandas',
        'numpy',
        'prometheus_client'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"❌ Missing required modules: {missing_modules}")
        logger.error("Install with: pip install -r dashboard_requirements.txt")
        sys.exit(1)
    
    logger.info("✅ All required modules available")

def setup_logging(config: DashboardAPIConfig):
    """Setup enhanced logging configuration"""
    
    log_level = getattr(logging, config.log_level.upper())
    
    # Configure root logger
    logging.getLogger().setLevel(log_level)
    
    # Configure specific loggers
    loggers = [
        'dashboard_api',
        'dashboard_api.requests',
        'dashboard_api.performance',
        'uvicorn.access'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
    
    logger.info(f"✅ Logging configured at {config.log_level} level")

async def startup_checks():
    """Perform startup health checks"""
    checks = {
        "memory": True,
        "disk_space": True,
        "dependencies": True
    }
    
    # Check available memory (basic check)
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.warning(f"⚠️ High memory usage: {memory.percent}%")
        else:
            logger.info(f"✅ Memory usage: {memory.percent}%")
    except ImportError:
        logger.info("📝 psutil not available - skipping memory check")
    
    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage('.')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        if free_percent < 10:
            logger.warning(f"⚠️ Low disk space: {free_percent:.1f}% free")
        else:
            logger.info(f"✅ Disk space: {free_percent:.1f}% free")
    except Exception as e:
        logger.warning(f"⚠️ Could not check disk space: {e}")
    
    return all(checks.values())

def print_startup_banner(config: DashboardAPIConfig):
    """Print startup banner with configuration"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  Physics Assistant Dashboard API                 ║
    ║                            Version 2.0.0                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║ Host: {config.host:<15} Port: {config.port:<10} Workers: {config.workers:<5} ║
    ║ Redis: {config.redis_host}:{config.redis_port:<10} Cache TTL: {config.cache_ttl_default:<8} ║
    ║ Rate Limiting: {'Enabled' if config.rate_limit_enabled else 'Disabled':<10} Compression: {'Enabled' if config.compression_enabled else 'Disabled':<8} ║
    ║ Log Level: {config.log_level:<10} Request Logging: {'Enabled' if config.request_logging_enabled else 'Disabled':<8} ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def main():
    """Main startup function"""
    
    print("🚀 Starting Physics Assistant Dashboard API Server...")
    
    # Load configuration
    config = DashboardAPIConfig()
    
    # Print startup banner
    print_startup_banner(config)
    
    # Validate environment
    validate_environment()
    
    # Setup logging
    setup_logging(config)
    
    # Perform startup checks
    logger.info("🔍 Performing startup checks...")
    checks_passed = await startup_checks()
    
    if not checks_passed:
        logger.error("❌ Startup checks failed")
        sys.exit(1)
    
    # Setup Redis connection
    logger.info("🔗 Setting up Redis connection...")
    redis_client = await setup_redis_connection(config)
    
    # Setup middleware
    logger.info("⚙️ Setting up middleware...")
    setup_middleware(app, config, redis_client)
    
    # Configure uvicorn
    uvicorn_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        reload=config.reload,
        workers=config.workers if not config.reload else 1,
        access_log=True,
        server_header=False,  # Hide server header for security
        date_header=True
    )
    
    # Start server
    logger.info("🌟 Starting Dashboard API Server...")
    server = uvicorn.Server(uvicorn_config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("⏹️ Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise
    finally:
        # Cleanup
        if redis_client:
            await redis_client.close()
            logger.info("✅ Redis connection closed")
        
        logger.info("👋 Dashboard API Server stopped")

def run_server():
    """Synchronous entry point for running the server"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config":
            # Print configuration and exit
            config = DashboardAPIConfig()
            print(f"Configuration:")
            for attr in dir(config):
                if not attr.startswith('_'):
                    print(f"  {attr}: {getattr(config, attr)}")
            sys.exit(0)
        elif sys.argv[1] == "--test":
            # Run tests
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pytest", "test_dashboard_api.py", "-v"])
            sys.exit(result.returncode)
        elif sys.argv[1] == "--help":
            print("""
Dashboard API Server Startup Script

Usage:
  python start_dashboard_api.py          # Start the server
  python start_dashboard_api.py --config # Show configuration
  python start_dashboard_api.py --test   # Run tests
  python start_dashboard_api.py --help   # Show this help

Environment Variables:
  DASHBOARD_HOST              # Server host (default: 0.0.0.0)
  DASHBOARD_PORT              # Server port (default: 8001)
  DASHBOARD_RELOAD            # Enable reload (default: false)
  DASHBOARD_WORKERS           # Number of workers (default: 1)
  REDIS_HOST                  # Redis host (default: localhost)
  REDIS_PORT                  # Redis port (default: 6379)
  REDIS_DB                    # Redis database (default: 0)
  REDIS_PASSWORD              # Redis password (default: none)
  CACHE_TTL_DEFAULT           # Default cache TTL (default: 300)
  RATE_LIMIT_ENABLED          # Enable rate limiting (default: true)
  COMPRESSION_ENABLED         # Enable compression (default: true)
  LOG_LEVEL                   # Log level (default: INFO)
            """)
            sys.exit(0)
    
    # Default: run the server
    run_server()