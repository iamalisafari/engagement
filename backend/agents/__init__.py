"""
Agent Package

This package implements the agent-based architecture for social media
engagement analysis, following principles from distributed AI systems
and cognitive architectures.
"""

from .base_agent import BaseAgent, AgentMessage, AgentStatus, AgentMetadata
from .audio_agent import AudioAgent
from .video_agent import VideoAgent
from .text_agent import TextAgent
from .hitl_agent import HITLAgent
from .engagement_scoring_agent import EngagementScoringAgent
from .coordinator_agent import CoordinatorAgent

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "AgentStatus",
    "AgentMetadata",
    "AudioAgent",
    "VideoAgent",
    "TextAgent",
    "HITLAgent",
    "EngagementScoringAgent",
    "CoordinatorAgent"
] 