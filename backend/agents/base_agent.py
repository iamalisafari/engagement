"""
Base Agent Implementation

This module defines the BaseAgent abstract class that establishes
the common interface and functionality for all agent implementations
in the system.
"""

import abc
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Possible states for an agent."""
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    IDLE = "idle"
    LEARNING = "learning"


class AgentMessage(BaseModel):
    """
    Standardized message format for inter-agent communication,
    based on agent communication protocols in distributed AI systems.
    """
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: int = 0  # Higher number = higher priority


class AgentMetadata(BaseModel):
    """Metadata about the agent instance and its capabilities."""
    agent_id: str
    agent_type: str
    version: str
    capabilities: List[str]
    description: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    status: AgentStatus = AgentStatus.IDLE
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the system.
    
    This implements the agent pattern with standardized interfaces
    for processing, communication, and lifecycle management.
    """
    
    def __init__(self, agent_id: str, agent_type: str, description: str, version: str = "0.1.0"):
        """Initialize the base agent with metadata."""
        self.metadata = AgentMetadata(
            agent_id=agent_id,
            agent_type=agent_type,
            version=version,
            capabilities=[],
            description=description
        )
        self.logger = logging.getLogger(f"agent.{agent_type}.{agent_id}")
        self._message_queue: List[AgentMessage] = []
        self._processing_state: Dict[str, Any] = {}
        
    @abc.abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and return results.
        
        Args:
            input_data: The data to process
            
        Returns:
            Dict containing processing results
        """
        raise NotImplementedError
    
    async def send_message(self, recipient_id: str, message_type: str, content: Dict[str, Any], 
                         priority: int = 0, correlation_id: Optional[str] = None) -> None:
        """Send a message to another agent."""
        message = AgentMessage(
            sender_id=self.metadata.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            priority=priority,
            correlation_id=correlation_id
        )
        # In a real implementation, this would use a message broker/queue
        self.logger.info(f"Sending message of type {message_type} to {recipient_id}")
        # Placeholder for message sending implementation
        
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive and queue a message from another agent."""
        self._message_queue.append(message)
        self.logger.info(f"Received message of type {message.message_type} from {message.sender_id}")
        
    async def process_messages(self) -> None:
        """Process all messages in the queue."""
        # Sort by priority (higher numbers = higher priority)
        self._message_queue.sort(key=lambda m: m.priority, reverse=True)
        
        while self._message_queue:
            message = self._message_queue.pop(0)
            await self._handle_message(message)
            
    @abc.abstractmethod
    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Handle a specific message.
        
        Args:
            message: The message to handle
        """
        raise NotImplementedError
    
    def update_status(self, status: AgentStatus) -> None:
        """Update the agent's current status."""
        self.metadata.status = status
        self.metadata.last_updated = datetime.now()
        
    def get_status(self) -> AgentStatus:
        """Get the agent's current status."""
        return self.metadata.status
    
    def update_capabilities(self, capabilities: List[str]) -> None:
        """Update the agent's capabilities list."""
        self.metadata.capabilities = capabilities
        self.metadata.last_updated = datetime.now()
        
    def get_metadata(self) -> AgentMetadata:
        """Get the agent's metadata."""
        return self.metadata
    
    async def train(self, training_data: Any) -> None:
        """
        Train or update the agent's models.
        
        Args:
            training_data: The data to train on
        """
        self.logger.info(f"Training not implemented for {self.metadata.agent_type}")
        
    def update_performance_metric(self, metric_name: str, value: float) -> None:
        """
        Update a performance metric for this agent.
        
        Args:
            metric_name: Name of the metric
            value: Value of the metric
        """
        self.metadata.performance_metrics[metric_name] = value
        self.logger.info(f"Updated performance metric {metric_name}: {value}")
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get all performance metrics for this agent."""
        return self.metadata.performance_metrics 