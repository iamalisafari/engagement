"""
Coordinator Agent Implementation

This module implements the coordinator agent that orchestrates the workflow
across agents, following a distributed systems architecture
pattern for coordinating multi-modal analysis.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ...models.content import Content, ContentType, Platform
from ...models.engagement import EngagementDimension, EngagementMetrics, EngagementScore, TemporalPattern
from ..base_agent import AgentMessage, AgentStatus, BaseAgent


class CoordinatorAgent(BaseAgent):
    """
    Agent responsible for orchestrating the analysis workflow across
    specialized analysis agents. Implements a distributed coordination
    pattern based on principles from distributed AI systems.
    """
    
    def __init__(self, agent_id: str = "coordinator_default"):
        """Initialize the coordinator agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type="coordinator",
            description="Orchestrates analysis workflow across specialized agents",
            version="0.1.0"
        )
        
        # Define agent capabilities
        self.update_capabilities([
            "task_orchestration",
            "result_aggregation",
            "job_scheduling",
            "dependency_management",
            "timeout_handling",
            "error_recovery"
        ])
        
        self.logger = logging.getLogger(f"agent.coordinator.{agent_id}")
        self.update_status(AgentStatus.READY)
        
        # Keep track of available agent instances
        self._available_agents: Dict[str, List[str]] = {
            "video_agent": [],
            "audio_agent": [],
            "text_agent": [],
            "hitl_agent": []
        }
        
        # Keep track of active analysis jobs
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Dispatch queue for pending analysis tasks
        self._dispatch_queue: List[Dict[str, Any]] = []
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request to coordinate content analysis across agents.
        
        Args:
            input_data: Dict containing analysis request parameters
                Required keys:
                - content_id: ID of the content to analyze
                - content_type: Type of content (VIDEO, TEXT, etc.)
                - content_url: URL to the content
                - platform: Platform the content is from
                Optional keys:
                - agent_config: Configuration options for specific agents
                - priority: Priority of the analysis (higher = process sooner)
                - callback_url: URL to call when analysis is complete
        
        Returns:
            Dict containing the job ID and initial status
        """
        self.update_status(AgentStatus.PROCESSING)
        self.logger.info(f"Coordinating analysis for content {input_data.get('content_id')}")
        
        try:
            # Generate a job ID
            job_id = str(uuid.uuid4())
            
            # Create a job record
            job = {
                "job_id": job_id,
                "content_id": input_data.get("content_id"),
                "content_type": input_data.get("content_type"),
                "content_url": input_data.get("content_url"),
                "platform": input_data.get("platform"),
                "priority": input_data.get("priority", 0),
                "callback_url": input_data.get("callback_url"),
                "status": "queued",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "progress": 0.0,
                "errors": [],
                "agent_tasks": {},
                "results": {}
            }
            
            # Determine which agents need to be involved based on content type
            agents_needed = self._determine_required_agents(input_data)
            
            # Create agent task records
            for agent_type in agents_needed:
                job["agent_tasks"][agent_type] = {
                    "status": "pending",
                    "agent_id": None,
                    "started_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None
                }
            
            # Store the job
            self._active_jobs[job_id] = job
            
            # Add to dispatch queue
            self._dispatch_queue.append({
                "job_id": job_id,
                "priority": job["priority"]
            })
            
            # Sort queue by priority (higher = first)
            self._dispatch_queue.sort(key=lambda x: x["priority"], reverse=True)
            
            # Schedule dispatch (would be an ongoing process in a real implementation)
            asyncio.create_task(self._dispatch_job(job_id))
            
            # Return initial status
            result = {
                "job_id": job_id,
                "content_id": job["content_id"],
                "status": job["status"],
                "created_at": job["created_at"].isoformat(),
                "estimated_completion_time": self._estimate_completion_time(job)
            }
            
            self.update_status(AgentStatus.READY)
            return result
            
        except Exception as e:
            self.logger.error(f"Error coordinating analysis: {e}")
            self.update_status(AgentStatus.ERROR)
            return {
                "error": str(e),
                "content_id": input_data.get("content_id")
            }
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Handle incoming messages from other agents.
        
        Args:
            message: The message to handle
        """
        if message.message_type == "agent_available":
            # Register an agent as available
            agent_type = message.content.get("agent_type")
            agent_id = message.content.get("agent_id")
            
            if agent_type and agent_id:
                if agent_type in self._available_agents:
                    if agent_id not in self._available_agents[agent_type]:
                        self._available_agents[agent_type].append(agent_id)
                        self.logger.info(f"Agent {agent_id} of type {agent_type} is now available")
                        
                        # Acknowledge receipt
                        await self.send_message(
                            recipient_id=agent_id,
                            message_type="availability_acknowledged",
                            content={"status": "acknowledged"},
                            correlation_id=message.correlation_id
                        )
                        
                        # Check dispatch queue
                        await self._process_dispatch_queue()
                else:
                    self.logger.warning(f"Unknown agent type: {agent_type}")
        
        elif message.message_type == "agent_task_complete":
            # Process completed task from an agent
            job_id = message.content.get("job_id")
            agent_type = message.content.get("agent_type")
            agent_id = message.content.get("agent_id")
            result = message.content.get("result")
            
            if job_id and agent_type and agent_id:
                if job_id in self._active_jobs:
                    job = self._active_jobs[job_id]
                    
                    if agent_type in job["agent_tasks"]:
                        # Update task status
                        job["agent_tasks"][agent_type]["status"] = "completed"
                        job["agent_tasks"][agent_type]["completed_at"] = datetime.now()
                        job["agent_tasks"][agent_type]["result"] = result
                        
                        # Store the agent result
                        job["results"][agent_type] = result
                        
                        # Make agent available again
                        if agent_id not in self._available_agents[agent_type]:
                            self._available_agents[agent_type].append(agent_id)
                        
                        # Update job progress
                        self._update_job_progress(job_id)
                        
                        # Check if all tasks are complete
                        if self._is_job_complete(job_id):
                            # Process final results
                            await self._finalize_job(job_id)
                        
                        # Acknowledge receipt
                        await self.send_message(
                            recipient_id=agent_id,
                            message_type="task_receipt_acknowledged",
                            content={"status": "acknowledged", "job_id": job_id},
                            correlation_id=message.correlation_id
                        )
                        
                        # Check dispatch queue
                        await self._process_dispatch_queue()
                    else:
                        self.logger.warning(f"Agent type {agent_type} not found in job {job_id}")
                else:
                    self.logger.warning(f"Job {job_id} not found")
        
        elif message.message_type == "agent_task_error":
            # Process error from an agent
            job_id = message.content.get("job_id")
            agent_type = message.content.get("agent_type")
            agent_id = message.content.get("agent_id")
            error = message.content.get("error")
            
            if job_id and agent_type and agent_id:
                if job_id in self._active_jobs:
                    job = self._active_jobs[job_id]
                    
                    if agent_type in job["agent_tasks"]:
                        # Update task status
                        job["agent_tasks"][agent_type]["status"] = "error"
                        job["agent_tasks"][agent_type]["completed_at"] = datetime.now()
                        job["agent_tasks"][agent_type]["error"] = error
                        
                        # Add to job errors
                        job["errors"].append({
                            "agent_type": agent_type,
                            "agent_id": agent_id,
                            "error": error,
                            "timestamp": datetime.now()
                        })
                        
                        # Make agent available again
                        if agent_id not in self._available_agents[agent_type]:
                            self._available_agents[agent_type].append(agent_id)
                        
                        # Implement error recovery strategy
                        await self._handle_task_error(job_id, agent_type, error)
                        
                        # Acknowledge receipt
                        await self.send_message(
                            recipient_id=agent_id,
                            message_type="error_receipt_acknowledged",
                            content={"status": "acknowledged", "job_id": job_id},
                            correlation_id=message.correlation_id
                        )
                        
                        # Check dispatch queue
                        await self._process_dispatch_queue()
                    else:
                        self.logger.warning(f"Agent type {agent_type} not found in job {job_id}")
                else:
                    self.logger.warning(f"Job {job_id} not found")
        
        elif message.message_type == "job_status_request":
            # Handle request for job status
            job_id = message.content.get("job_id")
            
            if job_id:
                status = self._get_job_status(job_id)
                await self.send_message(
                    recipient_id=message.sender_id,
                    message_type="job_status_response",
                    content=status,
                    correlation_id=message.correlation_id
                )
            else:
                await self.send_message(
                    recipient_id=message.sender_id,
                    message_type="error_response",
                    content={"error": "Missing job_id parameter"},
                    correlation_id=message.correlation_id
                )
        
        elif message.message_type == "status_request":
            # Handle request for coordinator status
            await self.send_message(
                recipient_id=message.sender_id,
                message_type="status_response",
                content={
                    "status": self.get_status().value,
                    "active_jobs": len(self._active_jobs),
                    "queued_jobs": len(self._dispatch_queue),
                    "available_agents": {k: len(v) for k, v in self._available_agents.items()}
                },
                correlation_id=message.correlation_id
            )
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
    
    def _determine_required_agents(self, input_data: Dict[str, Any]) -> Set[str]:
        """
        Determine which agent types are needed for a particular content analysis.
        
        Args:
            input_data: Analysis request data
            
        Returns:
            Set of agent type strings
        """
        content_type = input_data.get("content_type", "")
        required_agents = set()
        
        if content_type == ContentType.VIDEO:
            required_agents.add("video_agent")
            required_agents.add("audio_agent")
            required_agents.add("text_agent")  # For captions/comments
        elif content_type == ContentType.AUDIO:
            required_agents.add("audio_agent")
            required_agents.add("text_agent")  # For transcripts
        elif content_type == ContentType.TEXT:
            required_agents.add("text_agent")
        elif content_type == ContentType.IMAGE:
            required_agents.add("video_agent")  # Handles image analysis too
        elif content_type == ContentType.MIXED:
            # For mixed content, involve all agents
            required_agents.add("video_agent")
            required_agents.add("audio_agent")
            required_agents.add("text_agent")
        
        # HITL agent is optional and based on configuration
        if input_data.get("agent_config", {}).get("use_hitl", False):
            required_agents.add("hitl_agent")
            
        return required_agents
    
    def _estimate_completion_time(self, job: Dict[str, Any]) -> str:
        """
        Estimate when a job will be completed.
        
        Args:
            job: Job record
            
        Returns:
            ISO format timestamp of estimated completion time
        """
        # In a real implementation, this would use historical data
        # For now, use a simple estimate based on content type
        # Add 2 minutes per agent type involved
        minutes_per_agent = 2
        agent_count = len(job["agent_tasks"])
        total_minutes = agent_count * minutes_per_agent
        
        # Add some time for coordination and finalization
        total_minutes += 1
        
        # Calculate estimated completion time
        completion_time = job["created_at"] + datetime.timedelta(minutes=total_minutes)
        return completion_time.isoformat()
    
    async def _dispatch_job(self, job_id: str) -> None:
        """
        Dispatch tasks for a specific job to available agents.
        
        Args:
            job_id: The ID of the job to dispatch
        """
        if job_id not in self._active_jobs:
            self.logger.warning(f"Job {job_id} not found for dispatch")
            return
            
        job = self._active_jobs[job_id]
        job["status"] = "processing"
        job["updated_at"] = datetime.now()
        
        # For each agent type needed, find an available agent and dispatch
        for agent_type, task in job["agent_tasks"].items():
            if task["status"] == "pending" and self._available_agents[agent_type]:
                # Get an available agent
                agent_id = self._available_agents[agent_type].pop(0)
                
                # Update task status
                task["status"] = "dispatched"
                task["agent_id"] = agent_id
                task["started_at"] = datetime.now()
                
                # Prepare task message
                task_message = {
                    "job_id": job_id,
                    "content_id": job["content_id"],
                    "content_url": job["content_url"],
                    "platform": job["platform"],
                    "agent_config": job.get("agent_config", {}).get(agent_type, {})
                }
                
                # Send task to agent
                await self.send_message(
                    recipient_id=agent_id,
                    message_type="process_request",
                    content=task_message,
                    priority=job["priority"]
                )
                
                self.logger.info(f"Dispatched {agent_type} task for job {job_id} to agent {agent_id}")
            else:
                # If no agent is available, leave in queue
                self.logger.info(f"No {agent_type} agent available for job {job_id}, leaving in queue")
    
    async def _process_dispatch_queue(self) -> None:
        """Process the dispatch queue, sending tasks to available agents."""
        # Check each job in the queue
        for i, queue_item in enumerate(self._dispatch_queue[:]):
            job_id = queue_item["job_id"]
            
            if job_id in self._active_jobs:
                job = self._active_jobs[job_id]
                
                # Check if there are pending tasks that can be dispatched
                can_dispatch = False
                for agent_type, task in job["agent_tasks"].items():
                    if task["status"] == "pending" and self._available_agents[agent_type]:
                        can_dispatch = True
                        break
                
                if can_dispatch:
                    # Remove from queue and dispatch
                    self._dispatch_queue.pop(i)
                    await self._dispatch_job(job_id)
                    
                    # Only process one job at a time to avoid index issues
                    break
    
    def _update_job_progress(self, job_id: str) -> None:
        """
        Update the progress of a job based on completed tasks.
        
        Args:
            job_id: The ID of the job to update
        """
        if job_id not in self._active_jobs:
            return
            
        job = self._active_jobs[job_id]
        tasks = job["agent_tasks"]
        
        # Count completed tasks
        completed = sum(1 for task in tasks.values() if task["status"] == "completed")
        total = len(tasks)
        
        # Calculate progress (0.0 to 1.0)
        if total > 0:
            job["progress"] = completed / total
        else:
            job["progress"] = 0.0
            
        job["updated_at"] = datetime.now()
    
    def _is_job_complete(self, job_id: str) -> bool:
        """
        Check if all tasks for a job are complete.
        
        Args:
            job_id: The ID of the job to check
            
        Returns:
            True if all tasks are complete, False otherwise
        """
        if job_id not in self._active_jobs:
            return False
            
        job = self._active_jobs[job_id]
        tasks = job["agent_tasks"]
        
        # Check if all tasks are completed
        return all(task["status"] == "completed" for task in tasks.values())
    
    async def _finalize_job(self, job_id: str) -> None:
        """
        Finalize a job by aggregating results and calculating engagement metrics.
        
        Args:
            job_id: The ID of the job to finalize
        """
        if job_id not in self._active_jobs:
            return
            
        job = self._active_jobs[job_id]
        
        # Mark job as completed
        job["status"] = "completed"
        job["updated_at"] = datetime.now()
        job["progress"] = 1.0
        
        # Aggregate results from all agents
        # In a real implementation, this would calculate engagement metrics
        # based on the various agent outputs
        
        # For demonstration purposes, create a simulated engagement metrics result
        engagement_result = self._calculate_engagement_metrics(job)
        
        # Store the final result
        job["final_result"] = engagement_result
        
        # In a real implementation, this would store results in a database
        # and potentially trigger a callback to notify of completion
        
        self.logger.info(f"Finalized job {job_id}")
    
    async def _handle_task_error(self, job_id: str, agent_type: str, error: str) -> None:
        """
        Handle an error from an agent task.
        
        Args:
            job_id: The ID of the job with the error
            agent_type: The type of agent that encountered the error
            error: The error message
        """
        if job_id not in self._active_jobs:
            return
            
        job = self._active_jobs[job_id]
        
        # Log the error
        self.logger.error(f"Error in {agent_type} for job {job_id}: {error}")
        
        # Check if this is a critical agent
        # For demonstration, all agents are considered critical
        critical_error = True
        
        if critical_error:
            # Mark job as failed
            job["status"] = "failed"
            job["updated_at"] = datetime.now()
            
            # In a real implementation, this would potentially trigger
            # a notification or recovery process
        else:
            # For non-critical errors, can continue with other agents
            # Update progress
            self._update_job_progress(job_id)
            
            # Check if all other tasks are complete
            all_others_complete = True
            for other_type, task in job["agent_tasks"].items():
                if other_type != agent_type and task["status"] != "completed":
                    all_others_complete = False
                    break
                    
            if all_others_complete:
                # Finalize job even with this error
                await self._finalize_job(job_id)
    
    def _get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the current status of a job.
        
        Args:
            job_id: The ID of the job to check
            
        Returns:
            Dict containing status information
        """
        if job_id not in self._active_jobs:
            return {"error": f"Job {job_id} not found"}
            
        job = self._active_jobs[job_id]
        
        # Create a status response
        return {
            "job_id": job_id,
            "content_id": job["content_id"],
            "status": job["status"],
            "progress": job["progress"],
            "created_at": job["created_at"].isoformat(),
            "updated_at": job["updated_at"].isoformat(),
            "tasks": {
                agent_type: {
                    "status": task["status"],
                    "agent_id": task["agent_id"],
                    "started_at": task["started_at"].isoformat() if task["started_at"] else None,
                    "completed_at": task["completed_at"].isoformat() if task["completed_at"] else None
                }
                for agent_type, task in job["agent_tasks"].items()
            },
            "errors": [
                {
                    "agent_type": error["agent_type"],
                    "error": error["error"],
                    "timestamp": error["timestamp"].isoformat()
                }
                for error in job["errors"]
            ] if job["errors"] else []
        }
    
    def _calculate_engagement_metrics(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate engagement metrics from agent results.
        
        In a real implementation, this would apply sophisticated algorithms
        to combine results from different agents and calculate metrics
        based on the User Engagement Scale framework.
        
        Args:
            job: The job record containing agent results
            
        Returns:
            Dict containing engagement metrics
        """
        # Simulated engagement metrics calculation
        # This is a placeholder for the actual calculation that would
        # be implemented based on research findings
        
        # Example dimension scores (would be calculated from agent results)
        dimensions = {
            EngagementDimension.AESTHETIC_APPEAL: EngagementScore(
                value=0.68,
                confidence=0.85,
                contributing_factors={
                    "visual_quality": 0.75,
                    "color_harmony": 0.62,
                    "production_value": 0.83
                },
                temporal_pattern=TemporalPattern.SUSTAINED
            ),
            EngagementDimension.FOCUSED_ATTENTION: EngagementScore(
                value=0.73,
                confidence=0.92,
                contributing_factors={
                    "scene_transitions": 0.65,
                    "audio_clarity": 0.78,
                    "narrative_coherence": 0.85
                },
                temporal_pattern=TemporalPattern.DECLINING
            ),
            EngagementDimension.EMOTIONAL_RESPONSE: EngagementScore(
                value=0.81,
                confidence=0.88,
                contributing_factors={
                    "emotional_tone": 0.74,
                    "music_emotion": 0.83,
                    "narrative_tension": 0.76
                },
                temporal_pattern=TemporalPattern.PEAK_AND_VALLEY
            )
        }
        
        # Calculate composite score (weighted average)
        weights = {
            EngagementDimension.AESTHETIC_APPEAL: 0.3,
            EngagementDimension.FOCUSED_ATTENTION: 0.4,
            EngagementDimension.EMOTIONAL_RESPONSE: 0.3
        }
        
        weighted_sum = sum(
            dimensions[dim].value * weights.get(dim, 0)
            for dim in dimensions
        )
        
        weight_sum = sum(weights.get(dim, 0) for dim in dimensions)
        composite_score = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        # Determine overall temporal pattern (would be more sophisticated in reality)
        # For now, use most common pattern or the pattern of highest weighted dimension
        temporal_pattern = TemporalPattern.SUSTAINED  # Default
        
        # Create engagement metrics object
        metrics = EngagementMetrics(
            content_id=job["content_id"],
            composite_score=composite_score,
            dimensions=dimensions,
            platform_specific={
                "youtube_retention_index": 0.72,
                "predicted_shareability": 0.65
            },
            temporal_pattern=temporal_pattern,
            analysis_version="1.0.0"
        )
        
        # Return as dict for storage
        return metrics.dict()  # Convert Pydantic model to dict 