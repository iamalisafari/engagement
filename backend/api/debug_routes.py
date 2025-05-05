"""
Debug API Routes

API endpoints for retrieving debugging and diagnostics information
from the agent system.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List, Optional

from ..utils.debugging import (
    export_all_diagnostics, 
    get_all_diagnostics, 
    get_agent_diagnostics,
    DebugLevel
)

router = APIRouter(prefix="/debug", tags=["debugging"])


@router.get("/agents")
async def get_agents_debug_info():
    """Get debugging information for all agents."""
    try:
        return get_all_diagnostics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving agent data: {str(e)}")


@router.get("/agent/{agent_id}")
async def get_agent_debug_info(agent_id: str):
    """Get debugging information for a specific agent."""
    try:
        # This will return an error if the agent doesn't exist
        diagnostics = get_all_diagnostics().get(agent_id)
        if not diagnostics:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        return diagnostics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving agent data: {str(e)}")


@router.get("/logs")
async def get_debug_logs(
    agents: Optional[List[str]] = Query(None),
    levels: Optional[List[DebugLevel]] = Query(None),
    limit: int = 100
):
    """
    Get debug logs filtered by agent and level.
    
    Args:
        agents: List of agent IDs to filter by. If empty, include all agents.
        levels: List of log levels to filter by. If empty, include all levels.
        limit: Maximum number of logs to return.
    """
    try:
        all_diagnostics = get_all_diagnostics()
        
        # Collect logs from all agents
        logs = []
        for agent_id, diagnostics in all_diagnostics.items():
            if agents and agent_id not in agents:
                continue
                
            # Get logs from this agent's diagnostics
            for log in diagnostics.get("recent_logs", []):
                if levels and log.get("level") not in levels:
                    continue
                logs.append(log)
        
        # Sort by timestamp (newest first) and limit
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        logs = logs[:limit]
        
        return {"logs": logs, "total": len(logs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")


@router.post("/export")
async def export_debug_data(agents: Optional[List[str]] = Query(None)):
    """
    Export diagnostics data for specified agents or all agents.
    
    Args:
        agents: List of agent IDs to export. If empty, export all agents.
    """
    try:
        filepaths = export_all_diagnostics()
        return {"exported_files": filepaths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting diagnostics: {str(e)}")


@router.post("/log")
async def add_debug_log(
    agent_id: str,
    message: str,
    level: DebugLevel = DebugLevel.INFO,
    data: Optional[Dict] = None
):
    """
    Add a debug log entry for an agent.
    
    This endpoint allows manual addition of log entries, particularly
    useful for frontend-detected issues.
    """
    try:
        # Get agent diagnostics (will create if doesn't exist)
        diagnostics = get_agent_diagnostics(agent_id, "unknown")
        diagnostics.log(level, message, data)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding log: {str(e)}")


@router.get("/download/{agent_id}")
async def download_agent_diagnostics(agent_id: str):
    """
    Generate and download diagnostics file for a specific agent.
    """
    try:
        all_diagnostics = get_all_diagnostics()
        if agent_id not in all_diagnostics:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
        # Export diagnostics to file
        filepath = get_agent_diagnostics(agent_id, all_diagnostics[agent_id]["agent_type"]).export_full_diagnostics()
        
        # Return file for download
        return FileResponse(
            filepath, 
            filename=f"{agent_id}_diagnostics.json",
            media_type="application/json"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating diagnostics file: {str(e)}") 