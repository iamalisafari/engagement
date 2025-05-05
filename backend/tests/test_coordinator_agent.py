"""
Unit tests for the coordinator agent.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from ..agents.coordinator_agent.coordinator_agent import CoordinatorAgent
from ..agents.base import AgentStatus, AgentResult


class TestCoordinatorAgent:
    """Tests for the CoordinatorAgent implementation."""
    
    @pytest.fixture
    def coordinator_agent(self):
        """Create a CoordinatorAgent instance for testing."""
        return CoordinatorAgent()
    
    @pytest.fixture
    def sample_analysis_request(self):
        """Sample analysis request for testing."""
        return {
            "analysis_id": "test-analysis-123",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "platform": "youtube",
            "preset_id": "standard_analysis",
            "custom_settings": {
                "depth": "detailed",
                "features_enabled": {
                    "video_analysis": True,
                    "audio_analysis": True,
                    "text_analysis": True,
                    "temporal_analysis": True,
                    "engagement_scoring": True
                }
            },
            "priority": 2,
            "created_at": datetime.utcnow().isoformat(),
            "user_id": "user-123"
        }
    
    @patch("backend.agents.coordinator_agent.coordinator_agent.VideoAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.AudioAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.TextAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.EngagementScoringAgent")
    async def test_process_analysis_request(self, mock_engagement, mock_text, mock_audio, 
                                    mock_video, coordinator_agent, sample_analysis_request):
        """Test processing an analysis request."""
        # Set up agent mocks
        video_agent = AsyncMock()
        video_agent.process.return_value = AgentResult(
            success=True,
            score=0.75,
            confidence=0.85,
            features={"scene_analysis": {"scene_count": 8}}
        )
        
        audio_agent = AsyncMock()
        audio_agent.process.return_value = AgentResult(
            success=True,
            score=0.82,
            confidence=0.78,
            features={"speech_analysis": {"speech_clarity": 0.88}}
        )
        
        text_agent = AsyncMock()
        text_agent.process.return_value = AgentResult(
            success=True,
            score=0.68,
            confidence=0.92,
            features={"sentiment_analysis": {"sentiment_score": 0.65}}
        )
        
        engagement_agent = AsyncMock()
        engagement_agent.process_results.return_value = AgentResult(
            success=True,
            score=0.76,
            confidence=0.88,
            features={
                "modality_weights": {
                    "video": 0.4,
                    "audio": 0.35,
                    "text": 0.25
                }
            }
        )
        
        # Set up the mock agent constructor returns
        mock_video.return_value = video_agent
        mock_audio.return_value = audio_agent
        mock_text.return_value = text_agent
        mock_engagement.return_value = engagement_agent
        
        # Mock content extraction
        coordinator_agent._extract_content = AsyncMock()
        coordinator_agent._extract_content.return_value = {
            "metadata": {
                "title": "Test Video",
                "duration": 210
            }
        }
        
        # Mock storage
        coordinator_agent._store_result = AsyncMock()
        
        # Execute the method
        result = await coordinator_agent.process_analysis_request(sample_analysis_request)
        
        # Verify the result
        assert result["success"] is True
        assert result["analysis_id"] == sample_analysis_request["analysis_id"]
        assert result["score"] == 0.76
        assert result["confidence"] == 0.88
        
        # Verify agent interactions
        video_agent.process.assert_called_once()
        audio_agent.process.assert_called_once()
        text_agent.process.assert_called_once()
        engagement_agent.process_results.assert_called_once()
        
        # Verify storage
        coordinator_agent._store_result.assert_called_once()
    
    @patch("backend.agents.coordinator_agent.coordinator_agent.VideoAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.AudioAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.TextAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.EngagementScoringAgent")
    async def test_partial_failure_handling(self, mock_engagement, mock_text, mock_audio, 
                                    mock_video, coordinator_agent, sample_analysis_request):
        """Test handling when one agent fails but others succeed."""
        # Set up agent mocks
        video_agent = AsyncMock()
        video_agent.process.return_value = AgentResult(
            success=True,
            score=0.75,
            confidence=0.85,
            features={"scene_analysis": {"scene_count": 8}}
        )
        
        audio_agent = AsyncMock()
        audio_agent.process.return_value = AgentResult(
            success=False,
            error="Audio processing failed",
            confidence=0
        )
        
        text_agent = AsyncMock()
        text_agent.process.return_value = AgentResult(
            success=True,
            score=0.68,
            confidence=0.92,
            features={"sentiment_analysis": {"sentiment_score": 0.65}}
        )
        
        engagement_agent = AsyncMock()
        engagement_agent.process_results.return_value = AgentResult(
            success=True,
            score=0.72,
            confidence=0.80,
            features={
                "modality_weights": {
                    "video": 0.6,
                    "text": 0.4
                }
            },
            metadata={
                "warnings": {
                    "audio": "Audio processing failed"
                }
            }
        )
        
        # Set up the mock agent constructor returns
        mock_video.return_value = video_agent
        mock_audio.return_value = audio_agent
        mock_text.return_value = text_agent
        mock_engagement.return_value = engagement_agent
        
        # Mock content extraction
        coordinator_agent._extract_content = AsyncMock()
        coordinator_agent._extract_content.return_value = {
            "metadata": {
                "title": "Test Video",
                "duration": 210
            }
        }
        
        # Mock storage
        coordinator_agent._store_result = AsyncMock()
        
        # Execute the method
        result = await coordinator_agent.process_analysis_request(sample_analysis_request)
        
        # Verify the result
        assert result["success"] is True  # Overall success despite one agent failing
        assert result["analysis_id"] == sample_analysis_request["analysis_id"]
        assert result["score"] == 0.72
        assert result["confidence"] == 0.80
        assert "warnings" in result
        assert "audio" in result["warnings"]
        
        # Verify engagement agent was called with only successful modality results
        call_args = engagement_agent.process_results.call_args[0][0]
        assert "video" in call_args
        assert "text" in call_args
        assert "audio" not in call_args
    
    @patch("backend.agents.coordinator_agent.coordinator_agent.VideoAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.AudioAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.TextAgent")
    @patch("backend.agents.coordinator_agent.coordinator_agent.EngagementScoringAgent")
    async def test_extraction_failure(self, mock_engagement, mock_text, mock_audio, 
                             mock_video, coordinator_agent, sample_analysis_request):
        """Test handling when content extraction fails."""
        # Mock content extraction to fail
        coordinator_agent._extract_content = AsyncMock()
        coordinator_agent._extract_content.side_effect = Exception("Failed to fetch content")
        
        # Mock storage
        coordinator_agent._store_result = AsyncMock()
        
        # Execute the method
        result = await coordinator_agent.process_analysis_request(sample_analysis_request)
        
        # Verify the result
        assert result["success"] is False
        assert result["analysis_id"] == sample_analysis_request["analysis_id"]
        assert "error" in result
        assert "Failed to fetch content" in result["error"]
        
        # Verify no agents were initialized
        mock_video.assert_not_called()
        mock_audio.assert_not_called()
        mock_text.assert_not_called()
        mock_engagement.assert_not_called()
        
        # Verify error result was stored
        coordinator_agent._store_result.assert_called_once()
    
    async def test_task_prioritization(self, coordinator_agent):
        """Test the task prioritization algorithm."""
        # Create sample tasks with different priorities
        tasks = [
            {"analysis_id": "task1", "priority": 1, "created_at": "2023-05-01T10:00:00Z"},
            {"analysis_id": "task2", "priority": 3, "created_at": "2023-05-01T10:30:00Z"},
            {"analysis_id": "task3", "priority": 2, "created_at": "2023-05-01T09:45:00Z"},
            {"analysis_id": "task4", "priority": 2, "created_at": "2023-05-01T10:15:00Z"},
            {"analysis_id": "task5", "priority": 1, "created_at": "2023-05-01T09:30:00Z"},
        ]
        
        # Get prioritized tasks
        prioritized = coordinator_agent.prioritize_tasks(tasks)
        
        # Verify order: priority first, then creation time
        assert prioritized[0]["analysis_id"] == "task2"  # Highest priority
        assert prioritized[1]["analysis_id"] == "task3"  # Priority 2, earlier time
        assert prioritized[2]["analysis_id"] == "task4"  # Priority 2, later time
        assert prioritized[3]["analysis_id"] == "task5"  # Priority 1, earlier time
        assert prioritized[4]["analysis_id"] == "task1"  # Priority 1, later time
    
    async def test_agent_health_check(self, coordinator_agent):
        """Test the agent health checking functionality."""
        # Setup mock agents
        coordinator_agent.agents = {
            "video": MagicMock(status=AgentStatus.IDLE, last_error=None),
            "audio": MagicMock(status=AgentStatus.PROCESSING, last_error=None),
            "text": MagicMock(status=AgentStatus.ERROR, last_error="Test error"),
            "engagement": MagicMock(status=AgentStatus.IDLE, last_error=None)
        }
        
        # Get health status
        health_status = coordinator_agent.get_agent_health()
        
        # Verify status reporting
        assert health_status["video"]["status"] == "idle"
        assert health_status["audio"]["status"] == "processing"
        assert health_status["text"]["status"] == "error"
        assert health_status["text"]["error"] == "Test error"
        assert health_status["engagement"]["status"] == "idle"
        
        # Verify overall health
        assert health_status["overall_health"] == "degraded"  # Because text agent is in error state 