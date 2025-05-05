"""
Agent Control API Routes

API endpoints for controlling and managing agents in the system.
Allows starting, stopping, configuring, and monitoring agents.
"""

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import config
from ..utils.debugging import get_agent_diagnostics, DebugLevel

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Model for agent configuration
class AgentConfig(BaseModel):
    """Configuration settings for an agent."""
    enabled: bool = True
    priority: int = 1
    parameters: Dict[str, Any] = {}

# Model for agent action request
class AgentActionRequest(BaseModel):
    """Request model for agent actions."""
    action: str  # start, stop, restart, pause, resume
    parameters: Optional[Dict[str, Any]] = None

# Get DB session dependency
def get_db():
    """Dependency for database session."""
    db = next(config.get_db())
    try:
        yield db
    finally:
        db.close()

# Routes
@router.get("/")
async def list_agents(
    agent_type: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all agents in the system with optional filtering.
    
    Args:
        agent_type: Filter by agent type (e.g., video_agent, audio_agent)
        status: Filter by agent status (e.g., ready, processing, error)
    """
    # In a real implementation, this would query the database
    # For now, return mock data
    agents = [
        {
            "agent_id": "video_agent_default",
            "agent_type": "video_agent",
            "status": "ready",
            "enabled": True,
            "capabilities": [
                "scene_transition_detection",
                "visual_complexity_analysis",
                "motion_intensity_measurement"
            ],
            "performance_metrics": {
                "avg_processing_time": 45.2,
                "success_rate": 0.98
            },
            "version": "1.0.0"
        },
        {
            "agent_id": "audio_agent_default",
            "agent_type": "audio_agent",
            "status": "ready",
            "enabled": True,
            "capabilities": [
                "speech_detection",
                "music_analysis",
                "emotional_tone_analysis"
            ],
            "performance_metrics": {
                "avg_processing_time": 32.7,
                "success_rate": 0.96
            },
            "version": "1.0.0"
        },
        {
            "agent_id": "text_agent_default",
            "agent_type": "text_agent",
            "status": "ready",
            "enabled": True,
            "capabilities": [
                "sentiment_analysis",
                "topic_modeling",
                "readability_scoring"
            ],
            "performance_metrics": {
                "avg_processing_time": 12.3,
                "success_rate": 0.99
            },
            "version": "1.0.0"
        },
        {
            "agent_id": "hitl_agent_default",
            "agent_type": "hitl_agent",
            "status": "ready",
            "enabled": True,
            "capabilities": [
                "human_feedback_collection",
                "model_adjustment",
                "inter_rater_reliability"
            ],
            "performance_metrics": {
                "avg_response_time": 120.5,
                "feedback_quality": 0.92
            },
            "version": "1.0.0"
        },
        {
            "agent_id": "engagement_scoring_default",
            "agent_type": "engagement_scoring_agent",
            "status": "ready",
            "enabled": True,
            "capabilities": [
                "feature_weighting",
                "temporal_pattern_recognition",
                "context_specific_scoring"
            ],
            "performance_metrics": {
                "avg_processing_time": 18.9,
                "accuracy": 0.94
            },
            "version": "1.0.0"
        },
        {
            "agent_id": "coordinator_default",
            "agent_type": "coordinator",
            "status": "ready",
            "enabled": True,
            "capabilities": [
                "task_scheduling",
                "inter_agent_communication",
                "performance_monitoring"
            ],
            "performance_metrics": {
                "avg_processing_time": 5.8,
                "success_rate": 0.99
            },
            "version": "1.0.0"
        }
    ]
    
    # Apply filters
    if agent_type:
        agents = [a for a in agents if a["agent_type"] == agent_type]
    if status:
        agents = [a for a in agents if a["status"] == status]
        
    return agents


@router.get("/{agent_id}")
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """
    Get detailed information about a specific agent.
    
    Args:
        agent_id: ID of the agent to retrieve
    """
    # In a real implementation, this would query the database
    # For now, check if agent_id matches mock data
    agent_map = {
        "video_agent_default": {
            "agent_id": "video_agent_default",
            "agent_type": "video_agent",
            "status": "ready",
            "enabled": True,
            "capabilities": [
                "scene_transition_detection",
                "visual_complexity_analysis",
                "motion_intensity_measurement"
            ],
            "performance_metrics": {
                "avg_processing_time": 45.2,
                "success_rate": 0.98
            },
            "version": "1.0.0",
            "configuration": {
                "model_size": "medium",
                "batch_size": 16,
                "features_enabled": ["all"],
                "device": "gpu"
            },
            "description": "Analyzes video content for engagement indicators using computer vision techniques"
        }
    }
    
    if agent_id not in agent_map:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        
    return agent_map[agent_id]


@router.post("/{agent_id}/action")
async def perform_agent_action(
    agent_id: str,
    request: AgentActionRequest,
    db: Session = Depends(get_db)
):
    """
    Perform an action on an agent (start, stop, restart, etc.)
    
    Args:
        agent_id: ID of the agent to act upon
        request: Action details
    """
    # Validate agent exists (in a real system, we'd check the database)
    if agent_id not in ["video_agent_default", "audio_agent_default", "text_agent_default", 
                        "hitl_agent_default", "engagement_scoring_default", "coordinator_default"]:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    
    # Validate action
    valid_actions = ["start", "stop", "restart", "pause", "resume"]
    if request.action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}")
    
    # In a real implementation, this would communicate with the agent
    # Log the action for debugging
    diagnostics = get_agent_diagnostics(agent_id, "unknown")
    diagnostics.log(
        DebugLevel.INFO,
        f"Agent action requested: {request.action}",
        data={"parameters": request.parameters}
    )
    
    # Return success response with new status
    new_status = {
        "start": "processing",
        "stop": "idle",
        "restart": "processing",
        "pause": "paused",
        "resume": "processing"
    }.get(request.action, "ready")
    
    return {
        "agent_id": agent_id,
        "action": request.action,
        "status": "success",
        "new_agent_status": new_status,
        "message": f"Agent {agent_id} {request.action} command successful"
    }


@router.put("/{agent_id}/config")
async def update_agent_config(
    agent_id: str,
    config: AgentConfig,
    db: Session = Depends(get_db)
):
    """
    Update configuration for a specific agent.
    
    Args:
        agent_id: ID of the agent to configure
        config: New configuration settings
    """
    # Validate agent exists
    if agent_id not in ["video_agent_default", "audio_agent_default", "text_agent_default", 
                        "hitl_agent_default", "engagement_scoring_default", "coordinator_default"]:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    
    # In a real implementation, this would update the agent's configuration
    # Log the configuration change for debugging
    diagnostics = get_agent_diagnostics(agent_id, "unknown")
    diagnostics.log(
        DebugLevel.INFO,
        f"Agent configuration updated",
        data={"new_config": config.dict()}
    )
    
    return {
        "agent_id": agent_id,
        "status": "success",
        "message": "Configuration updated successfully",
        "config": config
    }


@router.post("/create")
async def create_agent(
    agent_type: str = Body(..., embed=True),
    name: str = Body(..., embed=True),
    config: AgentConfig = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """
    Create a new agent instance.
    
    Args:
        agent_type: Type of agent to create
        name: Name for the new agent
        config: Initial configuration settings
    """
    # Validate agent type
    valid_types = ["video_agent", "audio_agent", "text_agent", "hitl_agent", "engagement_scoring_agent", "coordinator"]
    if agent_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid agent type. Must be one of: {', '.join(valid_types)}")
    
    # In a real implementation, this would create a new agent
    agent_id = f"{agent_type}_{name}"
    
    # Log the agent creation for debugging
    diagnostics = get_agent_diagnostics(agent_id, agent_type)
    diagnostics.log(
        DebugLevel.INFO,
        f"New agent created",
        data={"type": agent_type, "name": name, "config": config.dict()}
    )
    
    return {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "status": "ready",
        "message": "Agent created successfully",
        "config": config
    } 