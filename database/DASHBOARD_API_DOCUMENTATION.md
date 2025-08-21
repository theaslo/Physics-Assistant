# Physics Assistant Dashboard API Documentation

## Version 2.0.0

A comprehensive backend API system designed specifically for analytics dashboard consumption with advanced caching, real-time streaming, and performance optimization features.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Authentication](#authentication)
5. [Core Endpoints](#core-endpoints)
6. [Real-time Features](#real-time-features)
7. [Caching System](#caching-system)
8. [Performance Features](#performance-features)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)
11. [Monitoring](#monitoring)
12. [Deployment](#deployment)

## Overview

The Dashboard API Server provides specialized backend endpoints optimized for analytics dashboard consumption. It builds upon the existing analytics infrastructure to offer:

- **Dashboard-optimized REST endpoints** with data aggregation
- **Advanced multi-layer caching** (memory + Redis)
- **Real-time streaming** via WebSockets and Server-Sent Events
- **Time-series and comparative analytics**
- **Data export capabilities** (JSON, CSV, Excel)
- **Performance optimization** and monitoring
- **Comprehensive security** and rate limiting

### Key Features

- 🚀 **High Performance**: Sub-second response times with advanced caching
- 🔄 **Real-time Updates**: WebSocket and SSE support for live data
- 📊 **Rich Analytics**: Time-series, aggregations, and comparative analysis
- 🛡️ **Security First**: Rate limiting, security headers, and authentication
- 📈 **Scalable**: Designed for high-concurrency dashboard loads
- 🧪 **Development-Friendly**: Mock endpoints and comprehensive testing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard API Server                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Middleware    │  │   Core API      │  │   Cache     │  │
│  │   - Rate Limit  │  │   - Dashboard   │  │   - Memory  │  │
│  │   - Security    │  │   - Analytics   │  │   - Redis   │  │
│  │   - Compression │  │   - Real-time   │  │   - Multi   │  │
│  │   - Monitoring  │  │   - Export      │  │     Layer   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Analytics Engine                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Learning Analytics • Concept Mastery • Path Optimizer │  │
│  │  Educational Data Mining • Real-time Processing        │  │
│  └─────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Database Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PostgreSQL  │  │   Neo4j     │  │       Redis         │  │
│  │ Interactions│  │ Knowledge   │  │   Cache & Sessions  │  │
│  │ Analytics   │  │   Graph     │  │   Rate Limiting     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r dashboard_requirements.txt

# Set environment variables (optional)
export DASHBOARD_PORT=8001
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Start the server
python start_dashboard_api.py
```

### Basic Usage

```python
import httpx

# Connect to the API
client = httpx.Client(base_url="http://localhost:8001")

# Get dashboard summary
response = client.get("/dashboard/summary?preset=7d")
summary = response.json()

# Get time-series data
timeseries_request = {
    "metrics": ["interaction_count", "success_rate"],
    "time_range": {"preset": "24h"},
    "granularity": "1h"
}
response = client.post("/dashboard/timeseries", json=timeseries_request)
timeseries = response.json()
```

### WebSocket Connection

```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8001/dashboard/ws/user_123');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

## Authentication

The API supports multiple authentication methods:

### Header-based Authentication

```http
GET /dashboard/summary
Authorization: Bearer <token>
X-User-ID: user_123
```

### Query Parameter Authentication

```http
GET /dashboard/summary?user_id=user_123&token=<token>
```

**Note**: Authentication implementation depends on your existing user management system.

## Core Endpoints

### Dashboard Summary

Get comprehensive dashboard overview with key metrics.

**Endpoint**: `GET /dashboard/summary`

**Parameters**:
- `preset` (optional): Time range preset (`1h`, `6h`, `24h`, `7d`, `30d`, `90d`, `1y`)
- `start_date` (optional): Custom start date (ISO format)
- `end_date` (optional): Custom end date (ISO format)
- `user_ids` (optional): Filter by specific users
- `agent_types` (optional): Filter by agent types
- `force_refresh` (optional): Skip cache and regenerate data

**Response**:
```json
{
  "time_range": {
    "start": "2025-01-08T00:00:00",
    "end": "2025-01-15T00:00:00",
    "preset": "7d"
  },
  "summary": {
    "total_interactions": 1250,
    "active_users": 45,
    "agents_used": 6,
    "avg_response_time": 245.3,
    "success_rate": 0.87,
    "successful_interactions": 1087
  },
  "agent_breakdown": [
    {"agent_type": "kinematics", "interaction_count": 450},
    {"agent_type": "forces", "interaction_count": 320},
    {"agent_type": "energy", "interaction_count": 280}
  ],
  "hourly_activity": [
    {
      "hour": "2025-01-15T10:00:00",
      "interactions": 85,
      "avg_response_time": 235.2
    }
  ],
  "cache_info": {
    "generated_at": "2025-01-15T10:30:00",
    "cache_key": "dashboard:summary:all:7d"
  }
}
```

### Time-Series Data

Get time-series analytics data for charts and graphs.

**Endpoint**: `POST /dashboard/timeseries`

**Request Body**:
```json
{
  "metrics": ["interaction_count", "success_rate", "avg_response_time"],
  "time_range": {
    "preset": "24h"
  },
  "granularity": "1h",
  "filters": {
    "user_ids": ["user1", "user2"],
    "agent_types": ["kinematics", "forces"],
    "success_only": false
  },
  "aggregation": "avg"
}
```

**Response**:
```json
{
  "metrics": ["interaction_count", "success_rate"],
  "granularity": "1h",
  "time_range": {
    "start": "2025-01-14T10:00:00",
    "end": "2025-01-15T10:00:00"
  },
  "data": [
    {
      "timestamp": "2025-01-14T10:00:00",
      "interaction_count": 45,
      "success_rate": 0.89,
      "avg_response_time": 234.5,
      "unique_users": 12
    },
    {
      "timestamp": "2025-01-14T11:00:00",
      "interaction_count": 52,
      "success_rate": 0.85,
      "avg_response_time": 245.3,
      "unique_users": 15
    }
  ],
  "generated_at": "2025-01-15T10:30:00"
}
```

### Data Aggregation

Get aggregated data grouped by dimensions.

**Endpoint**: `POST /dashboard/aggregation`

**Request Body**:
```json
{
  "dimensions": ["agent_type", "date"],
  "metrics": ["interaction_count", "avg_response_time", "success_rate"],
  "time_range": {"preset": "7d"},
  "filters": {
    "agent_types": ["kinematics", "forces"]
  },
  "limit": 100
}
```

**Response**:
```json
{
  "dimensions": ["agent_type", "date"],
  "metrics": ["interaction_count", "avg_response_time"],
  "data": [
    {
      "agent_type": "kinematics",
      "date": "2025-01-14",
      "interaction_count": 85,
      "avg_response_time": 234.5,
      "success_rate": 0.89
    },
    {
      "agent_type": "forces", 
      "date": "2025-01-14",
      "interaction_count": 62,
      "avg_response_time": 256.2,
      "success_rate": 0.82
    }
  ],
  "total_records": 14,
  "generated_at": "2025-01-15T10:30:00"
}
```

### Comparative Analysis

Compare performance between different entities.

**Endpoint**: `POST /dashboard/comparative`

**Request Body**:
```json
{
  "comparison_type": "students",
  "primary_entities": ["user_123", "user_456"],
  "comparison_entities": ["user_789", "user_012"],
  "metrics": ["success_rate", "interaction_count", "avg_response_time"],
  "time_range": {"preset": "30d"}
}
```

**Response**:
```json
{
  "comparison_type": "students",
  "primary_entities": ["user_123", "user_456"],
  "comparison_entities": ["user_789", "user_012"],
  "comparisons": [
    {
      "entity_id": "user_123",
      "entity_type": "primary",
      "metrics": {
        "success_rate": 0.89,
        "interaction_count": 145,
        "avg_response_time": 234.5
      }
    },
    {
      "entity_id": "user_789",
      "entity_type": "comparison", 
      "metrics": {
        "success_rate": 0.76,
        "interaction_count": 98,
        "avg_response_time": 267.8
      }
    }
  ],
  "generated_at": "2025-01-15T10:30:00"
}
```

### Student Insights

Get comprehensive insights for individual students.

**Endpoint**: `GET /dashboard/student-insights/{user_id}`

**Parameters**:
- `preset`: Time range preset
- `include_predictions`: Include ML predictions

**Response**:
```json
{
  "user_id": "user_123",
  "time_range": {
    "start": "2024-12-15T00:00:00",
    "end": "2025-01-15T00:00:00"
  },
  "progress_tracking": {
    "overall_score": 0.78,
    "learning_velocity": 0.65,
    "engagement_score": 0.82,
    "concepts_mastered": 12,
    "total_concepts": 20
  },
  "concept_mastery": [
    {
      "concept": "Kinematics",
      "mastery_score": 0.89,
      "confidence": [0.85, 0.93]
    },
    {
      "concept": "Forces",
      "mastery_score": 0.72,
      "confidence": [0.68, 0.76]
    }
  ],
  "predictions": {
    "success_rate": {
      "predicted_value": 0.85,
      "confidence": 0.78,
      "factors": ["engagement_trend", "concept_mastery", "learning_velocity"]
    }
  },
  "generated_at": "2025-01-15T10:30:00"
}
```

### Class Overview

Get class-level analytics and performance distribution.

**Endpoint**: `GET /dashboard/class-overview`

**Parameters**:
- `class_ids`: Specific class IDs to analyze
- `preset`: Time range preset
- `include_comparisons`: Include comparative metrics

**Response**:
```json
{
  "time_range": {
    "start": "2025-01-08T00:00:00",
    "end": "2025-01-15T00:00:00"
  },
  "class_statistics": {
    "total_students": 45,
    "total_interactions": 1250,
    "avg_response_time": 245.3,
    "class_success_rate": 0.82,
    "most_used_agent": "kinematics"
  },
  "performance_distribution": {
    "percentiles": {
      "25th": 0.65,
      "50th": 0.78,
      "75th": 0.89,
      "90th": 0.94
    },
    "student_count": 45
  },
  "top_performers": [
    {
      "user_id": "user_123",
      "success_rate": 0.94,
      "interaction_count": 89
    }
  ],
  "generated_at": "2025-01-15T10:30:00"
}
```

### Data Export

Export dashboard data in various formats.

**Endpoint**: `POST /dashboard/export`

**Parameters**:
- `export_format`: Format (`json`, `csv`, `excel`)
- `data_type`: Data type (`interactions`, `analytics`, `summary`)
- `preset`: Time range preset

**Response** (JSON format):
```json
{
  "format": "json",
  "data_type": "interactions",
  "time_range": {
    "start": "2025-01-08T00:00:00",
    "end": "2025-01-15T00:00:00"
  },
  "data": [
    {
      "id": "12345",
      "user_id": "user_123",
      "agent_type": "kinematics",
      "created_at": "2025-01-14T10:30:00",
      "success": true,
      "execution_time_ms": 245
    }
  ],
  "export_timestamp": "2025-01-15T10:30:00"
}
```

**Response** (CSV/Excel formats return file downloads)

## Real-time Features

### WebSocket Connection

Connect to real-time dashboard updates.

**Endpoint**: `WS /dashboard/ws/{user_id}`

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8001/dashboard/ws/user_123');

ws.onopen = function() {
    console.log('Connected to real-time updates');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'metrics_update':
            updateDashboardMetrics(data.data);
            break;
        case 'alert':
            showAlert(data.data);
            break;
        case 'student_progress':
            updateStudentProgress(data.data);
            break;
    }
};
```

**Message Types**:
- `metrics_update`: System metrics and performance data
- `alert`: Real-time alerts and notifications
- `student_progress`: Student progress updates
- `heartbeat`: Connection health check

### Server-Sent Events (SSE)

Stream real-time events to the dashboard.

**Endpoint**: `GET /dashboard/stream`

**Parameters**:
- `user_id` (optional): Filter events for specific user

**Usage**:
```javascript
const eventSource = new EventSource('/dashboard/stream?user_id=user_123');

eventSource.addEventListener('dashboard_update', function(event) {
    const data = JSON.parse(event.data);
    console.log('Dashboard update:', data);
});

eventSource.addEventListener('heartbeat', function(event) {
    console.log('Connection alive');
});
```

**Event Types**:
- `dashboard_update`: Dashboard data changes
- `heartbeat`: Keep-alive messages
- `alert`: System alerts and warnings

## Caching System

The API implements a sophisticated multi-layer caching system for optimal performance.

### Cache Layers

1. **Memory Cache**: Fastest access, limited size
2. **Redis Cache**: Persistent, shared across instances
3. **Database**: Fallback for cache misses

### Cache Control

#### Manual Cache Management

**Get Cache Statistics**:
```http
GET /dashboard/cache/stats
```

Response:
```json
{
  "cache_statistics": {
    "memory": {
      "hit_rate": 0.85,
      "size": 456,
      "max_size": 1000
    },
    "redis": {
      "hit_rate": 0.78,
      "memory_usage": "15.2MB",
      "connected": true
    }
  },
  "total_cache_requests": 1543
}
```

**Invalidate Cache**:
```http
POST /dashboard/cache/invalidate
Content-Type: application/json

{
  "pattern": "user:123",
  "cache_layer": "all"
}
```

**Warm Cache**:
```http
POST /dashboard/cache/warm
Content-Type: application/json

{
  "cache_types": ["summary", "timeseries"]
}
```

### Cache Headers

Responses include cache-related headers:

```http
HTTP/1.1 200 OK
Cache-Control: public, max-age=300, stale-while-revalidate=600
ETag: "d4f5c7a8b9e1f2d3"
X-Cache-Type: dashboard_summary
X-Cache-Key: abc123def456
```

## Performance Features

### Response Compression

Automatic GZIP compression for responses > 1KB:

```http
Accept-Encoding: gzip, deflate
```

Response includes:
```http
Content-Encoding: gzip
Vary: Accept-Encoding
```

### Background Processing

Long-running operations are processed in the background:

```python
# Cache warming runs automatically every 5 minutes
# Real-time event processing runs continuously
# Performance monitoring tracks all requests
```

### Performance Headers

Responses include performance metrics:

```http
X-Processing-Time: 0.234
X-Response-Time: 0.234s
X-Average-Response-Time: 0.187
X-Request-ID: abc12345
```

## Error Handling

### Standard Error Response

```json
{
  "error": "Rate limit exceeded",
  "detail": "Too many requests from this client",
  "code": "RATE_LIMIT_EXCEEDED",
  "timestamp": "2025-01-15T10:30:00",
  "request_id": "abc12345"
}
```

### HTTP Status Codes

- `200`: Success
- `206`: Partial success (some services degraded)
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not found
- `422`: Validation error
- `429`: Rate limit exceeded
- `500`: Internal server error
- `503`: Service unavailable

### Error Types

- `VALIDATION_ERROR`: Invalid request parameters
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `CACHE_ERROR`: Caching system issue
- `DATABASE_ERROR`: Database connectivity issue
- `ANALYTICS_UNAVAILABLE`: Analytics engine offline

## Rate Limiting

### Limits by Endpoint Type

| Endpoint Type | Requests per Hour | Burst Limit |
|---------------|-------------------|-------------|
| Dashboard Summary | 100 | 10 |
| Time-series | 50 | 5 |
| Aggregation | 30 | 3 |
| Export | 5 | 1 |
| WebSocket | 10 connections/min | - |
| Default | 200 | 20 |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642334400
Retry-After: 3600
```

### Rate Limit Response

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 3600,
  "limit_type": "dashboard_summary",
  "timestamp": "2025-01-15T10:30:00"
}
```

## Monitoring

### Health Check

**Endpoint**: `GET /dashboard/health`

```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00",
  "services": {
    "redis": "healthy",
    "cache": "healthy", 
    "websockets": "5 active",
    "background_processor": "running"
  },
  "cache_stats": {
    "memory": {"hit_rate": 0.85},
    "redis": {"hit_rate": 0.78}
  }
}
```

### Prometheus Metrics

**Endpoint**: `GET /dashboard/metrics`

Available metrics:
- `dashboard_requests_total`: Total requests by endpoint and cache status
- `dashboard_response_seconds`: Response time histogram
- `dashboard_cache_hit_rate`: Cache hit rate by layer
- `dashboard_websockets_active`: Active WebSocket connections
- `dashboard_aggregation_seconds`: Data aggregation time

## Mock Data Endpoints

For frontend development and testing:

### Mock Student Progress

**Endpoint**: `GET /dashboard/mock/student-progress`

```json
{
  "user_id": "mock_student_123",
  "progress_metrics": {
    "overall_score": 0.75,
    "concepts_mastered": 12,
    "total_concepts": 20,
    "learning_velocity": 0.65,
    "engagement_score": 0.82
  },
  "concept_breakdown": [
    {"concept": "Kinematics", "mastery": 0.89, "confidence": 0.85},
    {"concept": "Forces", "mastery": 0.72, "confidence": 0.78}
  ],
  "time_series": [
    {"date": "2025-01-01", "score": 0.45},
    {"date": "2025-01-07", "score": 0.52}
  ],
  "mock_data": true
}
```

### Mock Class Analytics

**Endpoint**: `GET /dashboard/mock/class-analytics`

```json
{
  "class_id": "physics_101",
  "summary": {
    "total_students": 45,
    "active_students": 38,
    "avg_progress": 0.67,
    "completion_rate": 0.73
  },
  "performance_distribution": {
    "excellent": 8,
    "good": 15,
    "average": 12,
    "needs_help": 8,
    "at_risk": 2
  },
  "mock_data": true
}
```

## Deployment

### Environment Variables

```bash
# Server Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8001
DASHBOARD_WORKERS=4
DASHBOARD_RELOAD=false

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password

# Cache Configuration
CACHE_TTL_DEFAULT=300
CACHE_MAX_SIZE=1000

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_RPH=1000

# Performance
COMPRESSION_ENABLED=true
COMPRESSION_MIN_SIZE=1000
SLOW_REQUEST_THRESHOLD=2.0

# Security
SECURITY_HEADERS_ENABLED=true

# Logging
LOG_LEVEL=INFO
REQUEST_LOGGING_ENABLED=true
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY dashboard_requirements.txt .
RUN pip install -r dashboard_requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "start_dashboard_api.py"]
```

### Production Checklist

- [ ] Set environment variables
- [ ] Configure Redis connection
- [ ] Enable rate limiting
- [ ] Set up monitoring
- [ ] Configure logging
- [ ] Test all endpoints
- [ ] Verify cache performance
- [ ] Test WebSocket connections
- [ ] Validate security headers
- [ ] Load test the API

### Scaling Considerations

1. **Horizontal Scaling**: Use multiple workers/instances
2. **Load Balancing**: Distribute requests across instances
3. **Redis Clustering**: Scale cache layer
4. **Database Optimization**: Optimize queries and connections
5. **CDN**: Cache static responses at edge locations

## API Client Examples

### Python Client

```python
import httpx
import asyncio
import websockets
import json

class DashboardAPIClient:
    def __init__(self, base_url="http://localhost:8001", user_id=None):
        self.base_url = base_url
        self.user_id = user_id
        self.client = httpx.AsyncClient(base_url=base_url)
    
    async def get_summary(self, preset="7d"):
        response = await self.client.get(f"/dashboard/summary?preset={preset}")
        return response.json()
    
    async def get_timeseries(self, metrics, preset="24h", granularity="1h"):
        data = {
            "metrics": metrics,
            "time_range": {"preset": preset},
            "granularity": granularity
        }
        response = await self.client.post("/dashboard/timeseries", json=data)
        return response.json()
    
    async def connect_websocket(self, callback):
        uri = f"ws://localhost:8001/dashboard/ws/{self.user_id}"
        async with websockets.connect(uri) as websocket:
            async for message in websocket:
                data = json.loads(message)
                await callback(data)

# Usage
async def main():
    client = DashboardAPIClient(user_id="user_123")
    
    # Get dashboard summary
    summary = await client.get_summary("7d")
    print(f"Total interactions: {summary['summary']['total_interactions']}")
    
    # Get time-series data
    timeseries = await client.get_timeseries(
        metrics=["interaction_count", "success_rate"],
        preset="24h"
    )
    print(f"Data points: {len(timeseries['data'])}")

asyncio.run(main())
```

### JavaScript Client

```javascript
class DashboardAPIClient {
    constructor(baseUrl = 'http://localhost:8001', userId = null) {
        this.baseUrl = baseUrl;
        this.userId = userId;
    }
    
    async getSummary(preset = '7d') {
        const response = await fetch(`${this.baseUrl}/dashboard/summary?preset=${preset}`);
        return response.json();
    }
    
    async getTimeseries(metrics, preset = '24h', granularity = '1h') {
        const response = await fetch(`${this.baseUrl}/dashboard/timeseries`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                metrics,
                time_range: {preset},
                granularity
            })
        });
        return response.json();
    }
    
    connectWebSocket(callback) {
        const ws = new WebSocket(`ws://localhost:8001/dashboard/ws/${this.userId}`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            callback(data);
        };
        
        return ws;
    }
    
    connectSSE(callback) {
        const eventSource = new EventSource(`${this.baseUrl}/dashboard/stream?user_id=${this.userId}`);
        
        eventSource.addEventListener('dashboard_update', function(event) {
            const data = JSON.parse(event.data);
            callback(data);
        });
        
        return eventSource;
    }
}

// Usage
const client = new DashboardAPIClient('http://localhost:8001', 'user_123');

// Get dashboard summary
client.getSummary('7d').then(summary => {
    console.log('Total interactions:', summary.summary.total_interactions);
});

// Connect to real-time updates
client.connectWebSocket(data => {
    console.log('Real-time update:', data);
});
```

## Support and Contributing

For issues, feature requests, or contributions, please refer to the main Physics Assistant repository.

### API Versioning

The API uses semantic versioning. Current version: `2.0.0`

- Major version: Breaking changes
- Minor version: New features, backward compatible
- Patch version: Bug fixes

### Changelog

#### Version 2.0.0
- Initial release of Dashboard API Server
- Multi-layer caching system
- Real-time WebSocket and SSE support
- Comprehensive analytics endpoints
- Advanced middleware and security features