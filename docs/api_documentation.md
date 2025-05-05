# API Documentation

## Overview

This document provides comprehensive documentation for the Social Media Engagement Analysis API. The API enables researchers to analyze engagement metrics across multiple social media platforms using our agentic AI system.

## Authentication

The API uses JWT (JSON Web Token) authentication.

### Obtaining a Token

```
POST /api/auth/token
```

**Request Body:**
```json
{
  "username": "researcher@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using Authentication

Include the token in the Authorization header for all protected endpoints:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Rate Limiting

API requests are subject to rate limiting based on user roles:
- Researchers: 100 requests per hour
- Administrators: 300 requests per hour

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1620000000
```

## Endpoints

### 1. Analysis Endpoints

#### Initiate Content Analysis

```
POST /api/analysis/content
```

Initiates analysis of a single social media content item.

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "platform": "youtube",
  "preset_id": "standard_analysis",
  "custom_settings": {
    "depth": "detailed",
    "features_enabled": {
      "video_analysis": true,
      "audio_analysis": true,
      "text_analysis": true,
      "temporal_analysis": true,
      "engagement_scoring": true
    }
  }
}
```

**Response:**
```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "estimated_completion_time": "2023-05-01T15:30:00Z",
  "request_details": { ... }
}
```

#### Batch Analysis

```
POST /api/analysis/batch
```

Submits multiple content items for analysis.

**Request Body:**
```json
{
  "items": [
    {
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "platform": "youtube"
    },
    {
      "url": "https://www.reddit.com/r/science/comments/abcdef",
      "platform": "reddit"
    }
  ],
  "preset_id": "comparative_analysis",
  "custom_settings": null,
  "priority": 1
}
```

**Response:**
```json
{
  "batch_id": "b1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "item_count": 2,
  "estimated_completion_time": "2023-05-01T16:45:00Z"
}
```

#### Get Analysis Results

```
GET /api/analysis/results/{analysis_id}
```

Retrieves results for a specific analysis.

**Response:**
```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "platform": "youtube",
  "completed_at": "2023-05-01T15:25:32Z",
  "results": {
    "engagement_score": 0.87,
    "confidence": 0.92,
    "modality_scores": {
      "video": 0.85,
      "audio": 0.91,
      "text": 0.82
    },
    "temporal_analysis": [ ... ],
    "detailed_metrics": { ... }
  }
}
```

### 2. Agent Management Endpoints

#### Get Agent Status

```
GET /api/agents/status
```

Provides status information for all analysis agents.

**Response:**
```json
{
  "agents": [
    {
      "id": "video_agent",
      "status": "active",
      "current_load": 0.45,
      "queue_length": 3,
      "uptime": "2d 5h 32m"
    },
    {
      "id": "audio_agent",
      "status": "active",
      "current_load": 0.32,
      "queue_length": 2,
      "uptime": "2d 5h 30m"
    },
    ...
  ]
}
```

#### Configure Agent

```
PUT /api/agents/{agent_id}/config
```

Updates configuration for a specific agent.

**Request Body:**
```json
{
  "enabled": true,
  "priority": 2,
  "parameters": {
    "model_size": "large",
    "batch_size": 32,
    "features_enabled": ["all"],
    "device": "gpu"
  }
}
```

**Response:**
```json
{
  "id": "video_agent",
  "status": "reconfiguring",
  "config": {
    "enabled": true,
    "priority": 2,
    "parameters": { ... }
  },
  "estimated_ready_time": "2023-05-01T15:05:00Z"
}
```

### 3. Analysis Configuration Endpoints

#### Create Analysis Preset

```
POST /api/config/presets
```

Creates a new analysis configuration preset.

**Request Body:**
```json
{
  "name": "Deep Video Analysis",
  "description": "Detailed analysis focusing on video content",
  "depth": "detailed",
  "features_enabled": {
    "video_analysis": true,
    "audio_analysis": true,
    "text_analysis": true,
    "temporal_analysis": true,
    "engagement_scoring": true,
    "comparative_analysis": false,
    "hitl_validation": true
  },
  "platform_specific_settings": {
    "youtube": {
      "include_comments": true,
      "comment_depth": 100,
      "include_channel_context": true
    }
  }
}
```

**Response:**
```json
{
  "preset_id": "p1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Deep Video Analysis",
  "created_at": "2023-05-01T14:30:22Z",
  "created_by": "researcher@example.com"
}
```

#### List Analysis Presets

```
GET /api/config/presets
```

Lists all available analysis configuration presets.

**Response:**
```json
{
  "presets": [
    {
      "preset_id": "p1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "name": "Deep Video Analysis",
      "description": "Detailed analysis focusing on video content",
      "created_at": "2023-05-01T14:30:22Z",
      "created_by": "researcher@example.com"
    },
    ...
  ]
}
```

### 4. User Management Endpoints

#### Create User

```
POST /api/users
```

Creates a new user account (admin only).

**Request Body:**
```json
{
  "username": "new_researcher",
  "email": "researcher@university.edu",
  "full_name": "Jane Researcher",
  "password": "secure_password",
  "role": "researcher"
}
```

**Response:**
```json
{
  "user_id": "u1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "username": "new_researcher",
  "email": "researcher@university.edu",
  "role": "researcher",
  "created_at": "2023-05-01T14:00:00Z"
}
```

### 5. Debug Endpoints

#### Get System Diagnostics

```
GET /api/debug/diagnostics
```

Retrieves system diagnostic information (admin only).

**Response:**
```json
{
  "system_status": "healthy",
  "component_statuses": {
    "database": "connected",
    "redis": "connected",
    "agents": {
      "video_agent": "running",
      "audio_agent": "running",
      "text_agent": "running",
      "engagement_agent": "running",
      "hitl_agent": "running",
      "coordinator_agent": "running"
    }
  },
  "performance_metrics": {
    "cpu_usage": 0.65,
    "memory_usage": 0.72,
    "disk_usage": 0.43,
    "request_latency_ms": 142
  },
  "queue_metrics": {
    "pending_analyses": 12,
    "processing_analyses": 4,
    "average_completion_time_s": 124
  },
  "error_rates": {
    "last_hour": 0.02,
    "last_day": 0.015
  }
}
```

## Error Handling

All API errors return standard HTTP status codes with a JSON response body:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested analysis could not be found",
    "details": "Analysis ID a1b2c3d4 does not exist in the system",
    "request_id": "req-1234567890"
  }
}
```

Common error codes:
- `400 Bad Request`: Invalid input parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error 