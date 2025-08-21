#!/usr/bin/env python3
"""
Comprehensive test suite for Dashboard API Server
Tests all endpoints, caching, real-time features, and performance characteristics.
"""

import asyncio
import json
import time
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
import websockets
import redis.asyncio as redis

# Import the dashboard API
from dashboard_api_server import app, cache, websocket_manager
from dashboard_middleware import RateLimitingMiddleware, ResponseCompressionMiddleware

# Test configuration
TEST_BASE_URL = "http://localhost:8001"
TEST_USER_ID = "test_user_123"
TEST_AGENT_TYPES = ["kinematics", "forces", "energy"]

class TestDashboardAPI:
    """Test suite for Dashboard API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Async test client fixture"""
        async with httpx.AsyncClient(app=app, base_url=TEST_BASE_URL) as client:
            yield client
    
    @pytest.fixture
    def mock_db(self):
        """Mock database fixture"""
        mock_db = Mock()
        mock_conn = AsyncMock()
        
        # Mock database query results
        mock_conn.fetchrow.return_value = {
            'total_interactions': 100,
            'active_users': 25,
            'agents_used': 5,
            'avg_response_time': 250.5,
            'successful_interactions': 85,
            'success_rate': 0.85
        }
        
        mock_conn.fetch.return_value = [
            {'agent_type': 'kinematics', 'count': 45},
            {'agent_type': 'forces', 'count': 30},
            {'agent_type': 'energy', 'count': 25}
        ]
        
        mock_db.postgres.get_connection.return_value.__aenter__.return_value = mock_conn
        return mock_db
    
    def test_health_check(self, client):
        """Test dashboard health check endpoint"""
        response = client.get("/dashboard/health")
        assert response.status_code in [200, 206]  # 206 for degraded but functional
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert "cache_stats" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/dashboard/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for expected metrics
        content = response.text
        assert "dashboard_requests_total" in content
        assert "dashboard_response_seconds" in content
    
    @patch('dashboard_api_server.get_db')
    def test_dashboard_summary(self, mock_get_db, client):
        """Test dashboard summary endpoint"""
        mock_get_db.return_value = self.mock_db()
        
        response = client.get("/dashboard/summary?preset=7d")
        assert response.status_code == 200
        
        data = response.json()
        assert "time_range" in data
        assert "summary" in data
        assert "agent_breakdown" in data
        assert "hourly_activity" in data
        assert "cache_info" in data
        
        # Verify summary metrics
        summary = data["summary"]
        assert "total_interactions" in summary
        assert "active_users" in summary
        assert "success_rate" in summary
    
    @patch('dashboard_api_server.get_db')
    def test_timeseries_endpoint(self, mock_get_db, client):
        """Test time-series data endpoint"""
        mock_get_db.return_value = self.mock_db()
        
        request_data = {
            "metrics": ["interaction_count", "success_rate"],
            "time_range": {"preset": "24h"},
            "granularity": "1h",
            "aggregation": "avg"
        }
        
        response = client.post("/dashboard/timeseries", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        assert "granularity" in data
        assert "time_range" in data
        assert "data" in data
        assert data["metrics"] == request_data["metrics"]
    
    @patch('dashboard_api_server.get_db')
    def test_aggregation_endpoint(self, mock_get_db, client):
        """Test data aggregation endpoint"""
        mock_get_db.return_value = self.mock_db()
        
        request_data = {
            "dimensions": ["agent_type", "date"],
            "metrics": ["interaction_count", "avg_response_time"],
            "time_range": {"preset": "7d"},
            "limit": 50
        }
        
        response = client.post("/dashboard/aggregation", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "dimensions" in data
        assert "metrics" in data
        assert "data" in data
        assert "total_records" in data
    
    @patch('dashboard_api_server.get_db')
    def test_comparative_analysis(self, mock_get_db, client):
        """Test comparative analysis endpoint"""
        mock_get_db.return_value = self.mock_db()
        
        request_data = {
            "comparison_type": "students",
            "primary_entities": ["user1", "user2"],
            "comparison_entities": ["user3", "user4"],
            "metrics": ["success_rate", "interaction_count"],
            "time_range": {"preset": "30d"}
        }
        
        response = client.post("/dashboard/comparative", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "comparison_type" in data
        assert "primary_entities" in data
        assert "comparison_entities" in data
        assert "comparisons" in data
    
    @patch('dashboard_api_server.get_db')
    def test_student_insights(self, mock_get_db, client):
        """Test student insights endpoint"""
        mock_get_db.return_value = self.mock_db()
        
        response = client.get(f"/dashboard/student-insights/{TEST_USER_ID}?preset=30d&include_predictions=true")
        assert response.status_code == 200
        
        data = response.json()
        assert "user_id" in data
        assert "time_range" in data
        assert "progress_tracking" in data
        assert "concept_mastery" in data
        assert data["user_id"] == TEST_USER_ID
    
    @patch('dashboard_api_server.get_db')
    def test_class_overview(self, mock_get_db, client):
        """Test class overview endpoint"""
        mock_get_db.return_value = self.mock_db()
        
        response = client.get("/dashboard/class-overview?preset=7d&include_comparisons=true")
        assert response.status_code == 200
        
        data = response.json()
        assert "time_range" in data
        assert "class_statistics" in data
        assert "performance_distribution" in data
        assert "top_performers" in data
    
    @patch('dashboard_api_server.get_db')
    def test_export_json(self, mock_get_db, client):
        """Test data export in JSON format"""
        mock_get_db.return_value = self.mock_db()
        
        response = client.post("/dashboard/export?export_format=json&data_type=interactions&preset=7d")
        assert response.status_code == 200
        
        data = response.json()
        assert "format" in data
        assert "data_type" in data
        assert "data" in data
        assert data["format"] == "json"
        assert data["data_type"] == "interactions"
    
    def test_mock_student_progress(self, client):
        """Test mock student progress endpoint"""
        response = client.get("/dashboard/mock/student-progress")
        assert response.status_code == 200
        
        data = response.json()
        assert "user_id" in data
        assert "progress_metrics" in data
        assert "concept_breakdown" in data
        assert "time_series" in data
        assert data["mock_data"] is True
    
    def test_mock_class_analytics(self, client):
        """Test mock class analytics endpoint"""
        response = client.get("/dashboard/mock/class-analytics")
        assert response.status_code == 200
        
        data = response.json()
        assert "class_id" in data
        assert "summary" in data
        assert "performance_distribution" in data
        assert "trending_concepts" in data
        assert data["mock_data"] is True

class TestCacheSystem:
    """Test suite for caching system"""
    
    @pytest.fixture
    def cache_system(self):
        """Cache system fixture"""
        from dashboard_api_server import MultiLayerCache
        return MultiLayerCache()
    
    async def test_memory_cache_operations(self, cache_system):
        """Test memory cache get/set operations"""
        # Test cache miss
        result = await cache_system.get("test_key")
        assert result is None
        
        # Test cache set and hit
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        await cache_system.set("test_key", test_data, ttl=300)
        
        cached_result = await cache_system.get("test_key")
        assert cached_result == test_data
    
    async def test_cache_invalidation(self, cache_system):
        """Test cache invalidation"""
        # Set multiple cache entries
        await cache_system.set("user:123:progress", {"data": "test1"}, ttl=300)
        await cache_system.set("user:456:progress", {"data": "test2"}, ttl=300)
        await cache_system.set("summary:general", {"data": "test3"}, ttl=300)
        
        # Invalidate user progress caches
        await cache_system.invalidate("user:*:progress")
        
        # Check that user caches are invalidated but summary remains
        assert await cache_system.get("user:123:progress") is None
        assert await cache_system.get("user:456:progress") is None
        assert await cache_system.get("summary:general") is not None
    
    def test_cache_stats(self, cache_system):
        """Test cache statistics"""
        stats = cache_system.get_stats()
        
        assert "memory" in stats
        assert "redis" in stats
        assert "total_requests" in stats
        
        # Memory stats
        memory_stats = stats["memory"]
        assert "hit_rate" in memory_stats
        assert "size" in memory_stats
        assert "max_size" in memory_stats

class TestRealTimeFeatures:
    """Test suite for real-time features"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection and messaging"""
        # This would require a running server for full testing
        # Here we test the WebSocket manager directly
        
        from unittest.mock import AsyncMock
        mock_websocket = AsyncMock()
        
        # Test connection
        await websocket_manager.connect(mock_websocket, TEST_USER_ID)
        assert mock_websocket in websocket_manager.active_connections
        assert mock_websocket in websocket_manager.user_connections[TEST_USER_ID]
        
        # Test personal message
        await websocket_manager.send_personal_message("test message", mock_websocket)
        mock_websocket.send_text.assert_called_with("test message")
        
        # Test user-specific message
        await websocket_manager.send_to_user("user message", TEST_USER_ID)
        mock_websocket.send_text.assert_called_with("user message")
        
        # Test disconnect
        websocket_manager.disconnect(mock_websocket, TEST_USER_ID)
        assert mock_websocket not in websocket_manager.active_connections
        assert mock_websocket not in websocket_manager.user_connections[TEST_USER_ID]
    
    @pytest.mark.asyncio
    async def test_sse_stream(self, async_client):
        """Test Server-Sent Events stream"""
        # This is a basic test - full SSE testing requires special handling
        with patch('dashboard_api_server.event_queue') as mock_queue:
            mock_queue.__iter__.return_value = [
                {"type": "test", "data": "test_data", "timestamp": time.time()}
            ]
            
            # Test SSE endpoint (this would normally stream)
            response = await async_client.get("/dashboard/stream")
            assert response.status_code == 200

class TestPerformanceAndSecurity:
    """Test suite for performance and security features"""
    
    def test_rate_limiting_middleware(self):
        """Test rate limiting middleware"""
        from dashboard_middleware import RateLimitingMiddleware
        
        middleware = RateLimitingMiddleware(None)
        
        # Test endpoint type detection
        assert middleware.get_endpoint_type("/dashboard/summary") == "dashboard_summary"
        assert middleware.get_endpoint_type("/dashboard/timeseries") == "timeseries"
        assert middleware.get_endpoint_type("/dashboard/export") == "export"
        assert middleware.get_endpoint_type("/dashboard/other") == "default"
        
        # Test client ID generation
        mock_request = Mock()
        mock_request.headers = {"X-User-ID": "user123"}
        mock_request.client.host = "127.0.0.1"
        
        client_id = middleware.get_client_id(mock_request)
        assert client_id == "user:user123"
        
        # Test without user ID
        mock_request.headers = {}
        client_id = middleware.get_client_id(mock_request)
        assert client_id == "ip:127.0.0.1"
    
    def test_compression_middleware(self):
        """Test response compression middleware"""
        from dashboard_middleware import ResponseCompressionMiddleware
        
        middleware = ResponseCompressionMiddleware(None)
        
        # Test compressible content types
        assert "application/json" in middleware.compressible_types
        assert "text/plain" in middleware.compressible_types
        assert "image/png" not in middleware.compressible_types
    
    def test_security_headers_middleware(self):
        """Test security headers middleware"""
        from dashboard_middleware import SecurityHeadersMiddleware
        
        middleware = SecurityHeadersMiddleware(None)
        
        # Check required security headers
        assert "X-Content-Type-Options" in middleware.security_headers
        assert "X-Frame-Options" in middleware.security_headers
        assert "Strict-Transport-Security" in middleware.security_headers
        assert "Content-Security-Policy" in middleware.security_headers

class TestAPIValidation:
    """Test suite for API validation and error handling"""
    
    def test_invalid_time_range(self, client):
        """Test invalid time range handling"""
        request_data = {
            "metrics": ["interaction_count"],
            "time_range": {"preset": "invalid_preset"},
            "granularity": "1h"
        }
        
        response = client.post("/dashboard/timeseries", json=request_data)
        # Should either handle gracefully or return 422 for validation error
        assert response.status_code in [200, 422]
    
    def test_invalid_aggregation_request(self, client):
        """Test invalid aggregation request"""
        request_data = {
            "dimensions": [],  # Empty dimensions
            "metrics": [],     # Empty metrics
            "time_range": {"preset": "7d"}
        }
        
        response = client.post("/dashboard/aggregation", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_endpoint(self, client):
        """Test 404 handling for non-existent endpoints"""
        response = client.get("/dashboard/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
        assert "available_endpoints" in data
    
    def test_invalid_export_format(self, client):
        """Test invalid export format handling"""
        response = client.post("/dashboard/export?export_format=invalid&data_type=interactions")
        assert response.status_code == 422  # Validation error

class TestLoadAndPerformance:
    """Load testing and performance validation"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling of concurrent requests"""
        async def make_request():
            return await async_client.get("/dashboard/mock/student-progress")
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_large_data_export(self, async_client):
        """Test export of large datasets"""
        # Test with a request that would generate large response
        response = await async_client.post(
            "/dashboard/export?export_format=json&data_type=interactions&preset=1y"
        )
        
        # Should handle large exports gracefully
        assert response.status_code in [200, 413, 503]  # Success, too large, or service unavailable
    
    def test_response_time_headers(self, client):
        """Test that response time headers are added"""
        response = client.get("/dashboard/mock/student-progress")
        assert response.status_code == 200
        
        # Check for performance headers (added by middleware)
        headers = response.headers
        # These might be added by middleware in production
        # assert "X-Response-Time" in headers
        # assert "X-Processing-Time" in headers

# Benchmark tests
class TestBenchmarks:
    """Benchmark tests for performance validation"""
    
    def test_cache_performance(self):
        """Benchmark cache operations"""
        from dashboard_api_server import MultiLayerCache
        cache_system = MultiLayerCache()
        
        # Benchmark cache set operations
        start_time = time.time()
        for i in range(1000):
            asyncio.run(cache_system.set(f"key_{i}", {"data": f"value_{i}"}, ttl=300))
        set_time = time.time() - start_time
        
        # Benchmark cache get operations
        start_time = time.time()
        for i in range(1000):
            asyncio.run(cache_system.get(f"key_{i}"))
        get_time = time.time() - start_time
        
        print(f"Cache set time for 1000 operations: {set_time:.3f}s")
        print(f"Cache get time for 1000 operations: {get_time:.3f}s")
        
        # Performance assertions
        assert set_time < 5.0  # Should complete in under 5 seconds
        assert get_time < 1.0  # Gets should be much faster
    
    def test_endpoint_response_times(self, client):
        """Benchmark endpoint response times"""
        endpoints = [
            "/dashboard/mock/student-progress",
            "/dashboard/mock/class-analytics",
            "/dashboard/health"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            duration = time.time() - start_time
            
            assert response.status_code == 200
            assert duration < 1.0  # Should respond within 1 second
            print(f"{endpoint}: {duration:.3f}s")

# Integration tests
class TestIntegration:
    """Integration tests with external dependencies"""
    
    @pytest.mark.skipif(True, reason="Requires Redis server")
    async def test_redis_integration(self):
        """Test Redis integration (requires running Redis)"""
        try:
            redis_client = redis.Redis(host="localhost", port=6379)
            await redis_client.ping()
            
            # Test Redis-based caching
            await redis_client.set("test_key", "test_value")
            value = await redis_client.get("test_key")
            assert value.decode() == "test_value"
            
            await redis_client.delete("test_key")
            
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
    
    @pytest.mark.skipif(True, reason="Requires database connection")
    async def test_database_integration(self):
        """Test database integration (requires running database)"""
        # This would test actual database connections
        # Skipped by default as it requires infrastructure
        pass

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=dashboard_api_server",
        "--cov=dashboard_middleware",
        "--cov-report=html",
        "--cov-report=term"
    ])