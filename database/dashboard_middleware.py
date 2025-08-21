#!/usr/bin/env python3
"""
Advanced middleware for Dashboard API Server
Includes rate limiting, caching optimization, compression, and security features.
"""

import time
import json
import gzip
import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Callable
import logging

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
import redis.asyncio as redis

# Configure logging
logger = logging.getLogger(__name__)

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with sliding window and user-based limits"""
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.redis_client = redis_client
        self.memory_store = defaultdict(lambda: deque())
        
        # Rate limit configurations
        self.limits = {
            "dashboard_summary": {"requests": 100, "window": 3600},  # 100 requests per hour
            "timeseries": {"requests": 50, "window": 3600},          # 50 requests per hour
            "aggregation": {"requests": 30, "window": 3600},         # 30 requests per hour
            "export": {"requests": 5, "window": 3600},               # 5 exports per hour
            "websocket": {"requests": 10, "window": 60},             # 10 connections per minute
            "default": {"requests": 200, "window": 3600}             # Default: 200 per hour
        }
        
        # IP-based global limits
        self.global_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/dashboard/health", "/dashboard/metrics"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self.get_client_id(request)
        endpoint_type = self.get_endpoint_type(request.url.path)
        
        # Check rate limits
        is_allowed, retry_after = await self.check_rate_limit(client_id, endpoint_type)
        
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "limit_type": endpoint_type,
                    "timestamp": datetime.now().isoformat()
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self.get_remaining_requests(client_id, endpoint_type)
        response.headers["X-RateLimit-Limit"] = str(self.limits[endpoint_type]["requests"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self.limits[endpoint_type]["window"]))
        
        return response
    
    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from token or session
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            client_ip = x_forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def get_endpoint_type(self, path: str) -> str:
        """Determine endpoint type for rate limiting"""
        if "/dashboard/summary" in path:
            return "dashboard_summary"
        elif "/dashboard/timeseries" in path:
            return "timeseries"
        elif "/dashboard/aggregation" in path:
            return "aggregation"
        elif "/dashboard/export" in path:
            return "export"
        elif "/dashboard/ws/" in path:
            return "websocket"
        else:
            return "default"
    
    async def check_rate_limit(self, client_id: str, endpoint_type: str) -> tuple[bool, int]:
        """Check if request is within rate limits"""
        limit_config = self.limits[endpoint_type]
        current_time = time.time()
        window_start = current_time - limit_config["window"]
        
        if self.redis_client:
            # Use Redis for distributed rate limiting
            return await self.check_redis_rate_limit(client_id, endpoint_type, current_time, window_start)
        else:
            # Use memory-based rate limiting
            return self.check_memory_rate_limit(client_id, endpoint_type, current_time, window_start)
    
    async def check_redis_rate_limit(self, client_id: str, endpoint_type: str, current_time: float, window_start: float) -> tuple[bool, int]:
        """Redis-based rate limiting using sliding window"""
        try:
            key = f"ratelimit:{client_id}:{endpoint_type}"
            limit_config = self.limits[endpoint_type]
            
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, limit_config["window"])
            
            results = await pipe.execute()
            current_requests = results[1]
            
            if current_requests >= limit_config["requests"]:
                # Get oldest request time to calculate retry_after
                oldest = await self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(oldest[0][1] + limit_config["window"] - current_time)
                else:
                    retry_after = limit_config["window"]
                return False, retry_after
            
            return True, 0
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fall back to memory-based limiting
            return self.check_memory_rate_limit(client_id, endpoint_type, current_time, window_start)
    
    def check_memory_rate_limit(self, client_id: str, endpoint_type: str, current_time: float, window_start: float) -> tuple[bool, int]:
        """Memory-based rate limiting"""
        key = f"{client_id}:{endpoint_type}"
        requests = self.memory_store[key]
        limit_config = self.limits[endpoint_type]
        
        # Remove old requests
        while requests and requests[0] < window_start:
            requests.popleft()
        
        if len(requests) >= limit_config["requests"]:
            retry_after = int(requests[0] + limit_config["window"] - current_time)
            return False, retry_after
        
        # Add current request
        requests.append(current_time)
        return True, 0
    
    async def get_remaining_requests(self, client_id: str, endpoint_type: str) -> int:
        """Get remaining requests for client"""
        limit_config = self.limits[endpoint_type]
        current_time = time.time()
        window_start = current_time - limit_config["window"]
        
        if self.redis_client:
            try:
                key = f"ratelimit:{client_id}:{endpoint_type}"
                # Clean old entries and count
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                results = await pipe.execute()
                current_requests = results[1]
                return max(0, limit_config["requests"] - current_requests)
            except:
                pass
        
        # Fall back to memory
        key = f"{client_id}:{endpoint_type}"
        requests = self.memory_store[key]
        
        # Clean old requests
        while requests and requests[0] < window_start:
            requests.popleft()
        
        return max(0, limit_config["requests"] - len(requests))

class ResponseCompressionMiddleware(BaseHTTPMiddleware):
    """Advanced response compression middleware"""
    
    def __init__(self, app, minimum_size: int = 1000, compression_level: int = 6):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        self.compressible_types = {
            "application/json",
            "text/plain",
            "text/html",
            "text/css",
            "text/javascript",
            "application/javascript"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process response with compression"""
        response = await call_next(request)
        
        # Skip compression for certain responses
        if (response.status_code < 200 or 
            response.status_code >= 300 or
            "content-encoding" in response.headers):
            return response
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Check content type
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in self.compressible_types):
            return response
        
        # Get response body
        if hasattr(response, 'body'):
            body = response.body
        else:
            # For streaming responses
            return response
        
        # Check minimum size
        if len(body) < self.minimum_size:
            return response
        
        # Compress response
        compressed_body = gzip.compress(body, compresslevel=self.compression_level)
        
        # Only use compression if it actually reduces size
        if len(compressed_body) >= len(body):
            return response
        
        # Create compressed response
        response.headers["content-encoding"] = "gzip"
        response.headers["content-length"] = str(len(compressed_body))
        response.headers["vary"] = "Accept-Encoding"
        
        # Replace body
        response.body = compressed_body
        
        return response

class CacheOptimizationMiddleware(BaseHTTPMiddleware):
    """Cache optimization and ETags middleware"""
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.redis_client = redis_client
        
        # Cache control settings by endpoint type
        self.cache_settings = {
            "dashboard_summary": {"max_age": 300, "stale_while_revalidate": 600},
            "timeseries": {"max_age": 900, "stale_while_revalidate": 1800},
            "aggregation": {"max_age": 600, "stale_while_revalidate": 1200},
            "student_insights": {"max_age": 300, "stale_while_revalidate": 600},
            "class_overview": {"max_age": 1800, "stale_while_revalidate": 3600},
            "mock": {"max_age": 86400, "stale_while_revalidate": 172800}  # Mock data can be cached longer
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with cache optimization"""
        
        # Skip caching for non-GET requests
        if request.method != "GET":
            return await call_next(request)
        
        endpoint_type = self.get_cache_type(request.url.path)
        
        # Generate cache key
        cache_key = self.generate_cache_key(request)
        
        # Check if-none-match header for ETags
        if_none_match = request.headers.get("if-none-match")
        
        # Check for cached ETag
        if if_none_match and self.redis_client:
            try:
                cached_etag = await self.redis_client.get(f"etag:{cache_key}")
                if cached_etag and cached_etag.decode() == if_none_match:
                    return Response(status_code=304)  # Not Modified
            except:
                pass
        
        # Process request
        response = await call_next(request)
        
        # Only cache successful responses
        if response.status_code != 200:
            return response
        
        # Generate ETag
        if hasattr(response, 'body') and response.body:
            etag = hashlib.md5(response.body).hexdigest()
            response.headers["etag"] = f'"{etag}"'
            
            # Store ETag in Redis
            if self.redis_client:
                try:
                    await self.redis_client.setex(f"etag:{cache_key}", 3600, etag)
                except:
                    pass
        
        # Add cache control headers
        if endpoint_type in self.cache_settings:
            settings = self.cache_settings[endpoint_type]
            cache_control = f"public, max-age={settings['max_age']}, stale-while-revalidate={settings['stale_while_revalidate']}"
            response.headers["cache-control"] = cache_control
        
        # Add cache-related headers
        response.headers["x-cache-type"] = endpoint_type
        response.headers["x-cache-key"] = cache_key[:32]  # Truncated for header size
        
        return response
    
    def get_cache_type(self, path: str) -> str:
        """Determine cache type from path"""
        if "/dashboard/summary" in path:
            return "dashboard_summary"
        elif "/dashboard/timeseries" in path:
            return "timeseries"
        elif "/dashboard/aggregation" in path:
            return "aggregation"
        elif "/dashboard/student-insights" in path:
            return "student_insights"
        elif "/dashboard/class-overview" in path:
            return "class_overview"
        elif "/dashboard/mock" in path:
            return "mock"
        else:
            return "default"
    
    def generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request"""
        # Include path, query parameters, and relevant headers
        key_components = [
            request.url.path,
            str(request.query_params),
            request.headers.get("x-user-id", "anonymous")
        ]
        
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response"""
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add API-specific headers
        response.headers["X-API-Version"] = "2.0.0"
        response.headers["X-Dashboard-API"] = "Physics-Assistant"
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Advanced request logging middleware"""
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.logger = logging.getLogger("dashboard_api.requests")
        self.log_level = getattr(logging, log_level.upper())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details"""
        start_time = time.time()
        
        # Log request
        request_id = hashlib.md5(f"{time.time()}{request.client.host}".encode()).hexdigest()[:8]
        
        self.logger.log(
            self.log_level,
            f"Request {request_id}: {request.method} {request.url.path} from {request.client.host}"
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            self.logger.error(f"Request {request_id} failed: {str(e)}")
            raise
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        self.logger.log(
            self.log_level,
            f"Response {request_id}: {response.status_code} in {duration:.3f}s"
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Performance monitoring and alerting middleware"""
    
    def __init__(self, app, slow_request_threshold: float = 2.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.logger = logging.getLogger("dashboard_api.performance")
        
        # Track performance metrics
        self.request_times = defaultdict(list)
        self.error_counts = defaultdict(int)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance"""
        start_time = time.time()
        endpoint = request.url.path
        
        try:
            response = await call_next(request)
        except Exception as e:
            # Track errors
            self.error_counts[endpoint] += 1
            self.logger.error(f"Error in {endpoint}: {str(e)}")
            raise
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Track request times
        self.request_times[endpoint].append(duration)
        
        # Keep only recent measurements (last 100 requests per endpoint)
        if len(self.request_times[endpoint]) > 100:
            self.request_times[endpoint] = self.request_times[endpoint][-100:]
        
        # Log slow requests
        if duration > self.slow_request_threshold:
            self.logger.warning(f"Slow request: {endpoint} took {duration:.3f}s")
        
        # Add performance headers
        response.headers["X-Processing-Time"] = f"{duration:.3f}"
        
        # Calculate and add average response time for this endpoint
        if self.request_times[endpoint]:
            avg_time = sum(self.request_times[endpoint]) / len(self.request_times[endpoint])
            response.headers["X-Average-Response-Time"] = f"{avg_time:.3f}"
        
        return response
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for endpoint, times in self.request_times.items():
            if times:
                stats[endpoint] = {
                    "avg_response_time": sum(times) / len(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "request_count": len(times),
                    "error_count": self.error_counts.get(endpoint, 0),
                    "slow_requests": len([t for t in times if t > self.slow_request_threshold])
                }
        
        return stats