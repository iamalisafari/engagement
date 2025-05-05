"""
API Main Entry Point

This module defines the main FastAPI application and routes
for the social media engagement analysis system.
"""

import logging
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import config
from ..database.init_db import init_db
from ..models.content import Content, ContentMetadata, ContentType, Platform
from ..models.engagement import EngagementDimension, EngagementMetrics

# Import routers
from .debug_routes import router as debug_router
from .agent_routes import router as agent_router
from .analysis_routes import router as analysis_router
from .auth_routes import router as auth_router
from .auth import add_rate_limit_headers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Initialize database
logger.info("Initializing database...")
init_db(add_sample_data=False)  # Don't add sample data on startup

# Initialize FastAPI application
app = FastAPI(
    title="Social Media Engagement Analysis API",
    description="API for analyzing engagement across social media platforms",
    version="0.1.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema models for API requests and responses
class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis."""
    url: str
    platform: Platform
    analysis_depth: Optional[str] = "standard"  # standard, detailed, or minimal


class ContentAnalysisResponse(BaseModel):
    """Response model for content analysis."""
    content_id: str
    status: str
    estimated_time_seconds: int


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str
    agent_type: str
    status: str
    capabilities: List[str]
    performance_metrics: Optional[Dict[str, float]] = None


# Dependency to get the database session
def get_db():
    """Dependency for database session."""
    db = next(config.get_db())
    try:
        yield db
    finally:
        db.close()


# Add middleware to add rate limit headers
@app.middleware("http")
async def rate_limit_headers_middleware(request: Request, call_next):
    """Add rate limit headers to responses."""
    response = await call_next(request)
    # Add rate limit headers if available
    return await add_rate_limit_headers(request, response)


# Error handler for custom error formats
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom error handler for HTTP exceptions."""
    return Response(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail, "status_code": exc.status_code},
        headers=exc.headers,
    )


# Routes
@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "Social Media Engagement Analysis API",
        "version": "0.1.0",
        "documentation": "/docs",
        "status": "operational"
    }


@app.post("/api/content/analyze", response_model=ContentAnalysisResponse)
async def analyze_content(request: ContentAnalysisRequest, db: Session = Depends(get_db)):
    """
    Analyze content from a social media URL.
    
    This endpoint initiates the analysis process by:
    1. Retrieving content metadata from the platform
    2. Distributing analysis tasks to appropriate agents
    3. Returning an ID for tracking analysis progress
    
    The analysis is performed asynchronously; results can be retrieved
    using the returned content_id.
    """
    logger.info(f"Analysis request received for {request.url}")
    
    # This would typically involve:
    # 1. Validating the URL
    # 2. Checking if the content has already been analyzed
    # 3. Dispatching the analysis job to the coordinator agent
    
    # For demo purposes, we return a simulated response
    return ContentAnalysisResponse(
        content_id="analysis_12345",
        status="processing",
        estimated_time_seconds=120
    )


@app.get("/api/content/{content_id}", response_model=dict)
async def get_content(content_id: str, db: Session = Depends(get_db)):
    """
    Retrieve content metadata by ID.
    
    This endpoint returns the metadata and features extracted
    from a specific content item, without engagement metrics.
    """
    # In a real implementation, this would fetch from the database
    if content_id != "analysis_12345":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Content with ID {content_id} not found"
        )
        
    # Simulated response for demonstration
    return {
        "id": content_id,
        "content_type": ContentType.VIDEO,
        "metadata": {
            "title": "Understanding User Engagement",
            "description": "This video explores factors affecting user engagement...",
            "creator_id": "creator_123",
            "creator_name": "Academic Research Channel",
            "platform": Platform.YOUTUBE,
            "published_at": "2023-05-15T14:30:00Z",
            "url": "https://www.youtube.com/watch?v=12345abcde",
            "tags": ["engagement", "research"],
            "category": "Education",
            "language": "en",
            "duration_seconds": 600
        },
        # Other fields would be included in a real implementation
    }


@app.get("/api/metrics/{content_id}", response_model=dict)
async def get_engagement_metrics(content_id: str, db: Session = Depends(get_db)):
    """
    Retrieve engagement metrics for a specific content.
    
    This endpoint returns comprehensive engagement analytics,
    including all dimensions, temporal patterns, and benchmarks.
    """
    # In a real implementation, this would fetch from the database
    if content_id != "analysis_12345":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics for content ID {content_id} not found"
        )
        
    # Simulated response for demonstration
    return {
        "content_id": content_id,
        "composite_score": 0.76,
        "dimensions": {
            EngagementDimension.FOCUSED_ATTENTION: {
                "value": 0.82,
                "confidence": 0.95,
                "contributing_factors": {
                    "scene_transitions": 0.65,
                    "audio_tempo": 0.78, 
                    "narrative_coherence": 0.88
                },
                "temporal_pattern": "SUSTAINED"
            },
            EngagementDimension.EMOTIONAL_RESPONSE: {
                "value": 0.71,
                "confidence": 0.92,
                "contributing_factors": {
                    "emotional_tone": 0.65,
                    "visual_sentiment": 0.78,
                    "narrative_tension": 0.63
                },
                "temporal_pattern": "PEAK_AND_VALLEY"
            }
        },
        "temporal_pattern": "SUSTAINED",
        "analysis_version": "1.0.3"
    }


@app.get("/api/agents", response_model=List[AgentStatusResponse])
async def get_agent_status(db: Session = Depends(get_db)):
    """
    Retrieve status information for all agents.
    
    This endpoint returns current status, capabilities, and performance
    metrics for all agents in the system.
    """
    # In a real implementation, this would query the agents
    return [
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
            "status": "ready",
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
        {
            "agent_id": "text_agent_default",
            "agent_type": "text_agent",
            "status": "ready",
            "capabilities": [
                "sentiment_analysis",
                "topic_modeling",
                "readability_scoring"
            ],
            "performance_metrics": {
                "avg_processing_time": 12.3,
                "success_rate": 0.99
            }
        }
    ]


# Include routers
app.include_router(auth_router)  # Authentication routes should be included first
app.include_router(agent_router)
app.include_router(analysis_router)
app.include_router(debug_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 