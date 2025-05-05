# API Documentation

This document outlines the API endpoints and data structures for the social media engagement analysis system.

## Base URL

All API endpoints are relative to the base URL:

```
http://localhost:8000
```

## Authentication

Authentication is not implemented in this initial version. In a production deployment, endpoints would require authentication using OAuth 2.0 or JWT tokens.

## Common Response Formats

All responses follow a standard format:

### Success Response

```json
{
  "data": {
    // Response data specific to the endpoint
  },
  "meta": {
    "timestamp": "2023-07-14T12:34:56Z",
    "version": "1.0.0"
  }
}
```

### Error Response

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      // Optional additional details about the error
    }
  },
  "meta": {
    "timestamp": "2023-07-14T12:34:56Z",
    "version": "1.0.0"
  }
}
```

## Content Analysis Endpoints

### Submit Content for Analysis

Initiates analysis of a content item from a social media platform.

**Endpoint:** `POST /api/content/analyze`

**Request Body:**

```json
{
  "url": "https://www.youtube.com/watch?v=abc123",
  "platform": "youtube",
  "analysis_depth": "standard" // optional, can be "minimal", "standard", or "detailed"
}
```

**Response:**

```json
{
  "data": {
    "content_id": "analysis_12345",
    "status": "processing",
    "estimated_time_seconds": 120
  },
  "meta": {
    "timestamp": "2023-07-14T12:34:56Z",
    "version": "1.0.0"
  }
}
```

### Get Content Metadata

Retrieves metadata and extracted features for a specific content item.

**Endpoint:** `GET /api/content/{content_id}`

**Response:**

```json
{
  "data": {
    "id": "analysis_12345",
    "content_type": "VIDEO",
    "metadata": {
      "title": "Understanding User Engagement",
      "description": "This video explores factors affecting user engagement...",
      "creator_id": "creator_123",
      "creator_name": "Academic Research Channel",
      "platform": "YOUTUBE",
      "published_at": "2023-05-15T14:30:00Z",
      "url": "https://www.youtube.com/watch?v=12345abcde",
      "tags": ["engagement", "research"],
      "category": "Education",
      "language": "en",
      "duration_seconds": 600
    },
    "video_features": {
      "resolution": "1080p",
      "fps": 30.0,
      "scene_transitions": [10.5, 25.2, 42.8, 60.1, 75.3, 90.6],
      "visual_complexity": {
        "spatial_complexity": 0.72,
        "temporal_complexity": 0.65,
        "information_density": 0.68
      },
      // Additional video features
    },
    "audio_features": {
      // Audio features if available
    },
    "text_features": {
      // Text features if available
    }
  },
  "meta": {
    "timestamp": "2023-07-14T12:34:56Z",
    "version": "1.0.0"
  }
}
```

### Get Engagement Metrics

Retrieves calculated engagement metrics for a specific content item.

**Endpoint:** `GET /api/metrics/{content_id}`

**Response:**

```json
{
  "data": {
    "content_id": "analysis_12345",
    "composite_score": 0.76,
    "dimensions": {
      "focused_attention": {
        "value": 0.82,
        "confidence": 0.95,
        "contributing_factors": {
          "scene_transitions": 0.65,
          "audio_tempo": 0.78,
          "narrative_coherence": 0.88
        },
        "temporal_pattern": "SUSTAINED"
      },
      "emotional_response": {
        "value": 0.71,
        "confidence": 0.92,
        "contributing_factors": {
          "emotional_tone": 0.65,
          "visual_sentiment": 0.78,
          "narrative_tension": 0.63
        },
        "temporal_pattern": "PEAK_AND_VALLEY"
      },
      // Additional engagement dimensions
    },
    "platform_specific": {
      "youtube_retention_index": 0.72,
      "predicted_shareability": 0.65
    },
    "temporal_pattern": "SUSTAINED",
    "analysis_version": "1.0.3"
  },
  "meta": {
    "timestamp": "2023-07-14T12:34:56Z",
    "version": "1.0.0"
  }
}
```

## Agent Monitoring Endpoints

### Get All Agents Status

Retrieves status information for all analysis agents.

**Endpoint:** `GET /api/agents`

**Response:**

```json
{
  "data": [
    {
      "agent_id": "video_agent_default",
      "agent_type": "video_agent",
      "status": "ready",
      "capabilities": [
        "scene_transition_detection",
        "visual_complexity_analysis",
        "motion_intensity_measurement"
      ],
      "performance_metrics": {
        "avg_processing_time": 45.2,
        "success_rate": 0.98
      }
    },
    {
      "agent_id": "audio_agent_default",
      "agent_type": "audio_agent",
      "status": "processing",
      "capabilities": [
        "speech_detection",
        "music_analysis",
        "emotional_tone_analysis"
      ],
      "performance_metrics": {
        "avg_processing_time": 32.7,
        "success_rate": 0.96
      }
    },
    // Additional agents
  ],
  "meta": {
    "timestamp": "2023-07-14T12:34:56Z",
    "version": "1.0.0"
  }
}
```

### Get Specific Agent Status

Retrieves detailed status information for a specific agent.

**Endpoint:** `GET /api/agents/{agent_id}`

**Response:**

```json
{
  "data": {
    "agent_id": "video_agent_default",
    "agent_type": "video_agent",
    "status": "ready",
    "capabilities": [
      "scene_transition_detection",
      "visual_complexity_analysis",
      "motion_intensity_measurement",
      "color_scheme_analysis",
      "production_quality_assessment",
      "thumbnail_effectiveness_analysis"
    ],
    "performance_metrics": {
      "avg_processing_time": 45.2,
      "success_rate": 0.98,
      "memory_usage_mb": 420.5,
      "processed_jobs": 152
    },
    "current_jobs": [
      {
        "job_id": "job_789",
        "content_id": "analysis_56789",
        "progress": 0.45,
        "started_at": "2023-07-14T12:30:00Z"
      }
    ]
  },
  "meta": {
    "timestamp": "2023-07-14T12:34:56Z",
    "version": "1.0.0"
  }
}
```

## Job Management Endpoints

### Get Job Status

Retrieves the status of a specific analysis job.

**Endpoint:** `GET /api/jobs/{job_id}`

**Response:**

```json
{
  "data": {
    "job_id": "job_456",
    "content_id": "analysis_12345",
    "status": "processing",
    "progress": 0.65,
    "created_at": "2023-07-14T12:20:00Z",
    "updated_at": "2023-07-14T12:32:00Z",
    "tasks": {
      "video_agent": {
        "status": "completed",
        "agent_id": "video_agent_default",
        "started_at": "2023-07-14T12:21:00Z",
        "completed_at": "2023-07-14T12:25:00Z"
      },
      "audio_agent": {
        "status": "processing",
        "agent_id": "audio_agent_default",
        "started_at": "2023-07-14T12:26:00Z",
        "completed_at": null
      },
      "text_agent": {
        "status": "pending",
        "agent_id": null,
        "started_at": null,
        "completed_at": null
      }
    },
    "errors": []
  },
  "meta": {
    "timestamp": "2023-07-14T12:34:56Z",
    "version": "1.0.0"
  }
}
```

### Cancel Job

Cancels a running analysis job.

**Endpoint:** `POST /api/jobs/{job_id}/cancel`

**Response:**

```json
{
  "data": {
    "job_id": "job_456",
    "status": "cancelled",
    "cancelled_at": "2023-07-14T12:35:00Z"
  },
  "meta": {
    "timestamp": "2023-07-14T12:35:00Z",
    "version": "1.0.0"
  }
}
```

## Data Models

### Content Types

| Value | Description |
|-------|-------------|
| `VIDEO` | Video content (e.g., YouTube videos) |
| `TEXT` | Text-only content (e.g., Reddit text posts) |
| `IMAGE` | Image content (e.g., Instagram posts) |
| `AUDIO` | Audio content (e.g., podcasts) |
| `MIXED` | Content with multiple modalities |

### Platforms

| Value | Description |
|-------|-------------|
| `YOUTUBE` | YouTube platform |
| `REDDIT` | Reddit platform |
| `TWITTER` | Twitter platform |
| `INSTAGRAM` | Instagram platform |
| `TIKTOK` | TikTok platform |

### Engagement Dimensions

| Value | Description |
|-------|-------------|
| `aesthetic_appeal` | Visual and sensory appeal |
| `focused_attention` | Concentration and absorption |
| `perceived_usability` | Ease of use and control |
| `endurability` | Likelihood of remembering and returning |
| `novelty` | Curiosity, surprise, and newness |
| `involvement` | Interest and motivation |
| `social_presence` | Sense of connection with others |
| `shareability` | Likelihood of sharing content |
| `emotional_response` | Affective reactions |

### Temporal Patterns

| Value | Description |
|-------|-------------|
| `sustained` | Consistently high engagement |
| `declining` | Starts high, gradually decreases |
| `increasing` | Builds up over time |
| `u_shaped` | High at beginning and end, lower in middle |
| `inverted_u` | Peaks in middle, lower at start and end |
| `fluctuating` | Varies significantly throughout |
| `cliff` | Sudden drop after initial period |
| `peak_and_valley` | Multiple peaks and valleys |

### Agent Status

| Value | Description |
|-------|-------------|
| `ready` | Agent is ready to process tasks |
| `processing` | Agent is currently processing a task |
| `error` | Agent has encountered an error |
| `idle` | Agent is initialized but not actively processing |
| `learning` | Agent is updating its models |

## Error Codes

| Code | Description |
|------|-------------|
| `CONTENT_NOT_FOUND` | The specified content ID was not found |
| `INVALID_URL` | The URL provided is not valid or not supported |
| `PLATFORM_NOT_SUPPORTED` | The specified platform is not supported |
| `AGENT_UNAVAILABLE` | Required agent is not available for processing |
| `ANALYSIS_FAILED` | Analysis failed due to content processing issues |
| `INVALID_PARAMETERS` | Invalid parameters provided in the request |
| `UNAUTHORIZED` | Authentication required or insufficient permissions |
| `RATE_LIMITED` | Too many requests, rate limit exceeded | 