"""
Unit tests for the agent system components.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

from ..agents.base import BaseAgent, AgentResult, AgentStatus
from ..agents.video import VideoAgent
from ..agents.audio import AudioAgent
from ..agents.text import TextAgent
from ..agents.engagement import EngagementScoringAgent
from ..agents.hitl import HITLAgent


class TestBaseAgent:
    """Tests for the BaseAgent abstract class and common functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization with default parameters."""
        # Create a concrete implementation of BaseAgent for testing
        class TestAgent(BaseAgent):
            agent_type = "test"
            
            async def process(self, content, settings):
                return AgentResult(
                    success=True,
                    score=0.5,
                    confidence=0.8,
                    features={"test_feature": 1.0},
                    metadata={"processing_time": 0.1}
                )
        
        agent = TestAgent()
        
        assert agent.agent_type == "test"
        assert agent.status == AgentStatus.IDLE
        assert agent.version == "1.0.0"  # Default version
        
    def test_agent_status_transitions(self):
        """Test agent status transitions."""
        # Create a concrete implementation
        class TestAgent(BaseAgent):
            agent_type = "test"
            
            async def process(self, content, settings):
                return AgentResult(success=True, score=0.5, confidence=0.8)
        
        agent = TestAgent()
        
        # Initial state
        assert agent.status == AgentStatus.IDLE
        
        # Transition to processing
        agent.start_processing()
        assert agent.status == AgentStatus.PROCESSING
        
        # Transition to idle
        agent.complete_processing()
        assert agent.status == AgentStatus.IDLE
        
        # Transition to error
        agent.set_error("Test error")
        assert agent.status == AgentStatus.ERROR
        assert agent.last_error == "Test error"
        
        # Reset from error
        agent.reset()
        assert agent.status == AgentStatus.IDLE
        assert agent.last_error is None
    
    def test_agent_result_serialization(self):
        """Test AgentResult serialization to JSON."""
        result = AgentResult(
            success=True,
            score=0.75,
            confidence=0.9,
            features={
                "feature1": 0.8,
                "feature2": 0.6
            },
            metadata={
                "processing_time": 0.35,
                "model_version": "1.2.3"
            }
        )
        
        # Serialize to JSON
        json_result = result.json()
        
        # Deserialize and check
        decoded = json.loads(json_result)
        
        assert decoded["success"] is True
        assert decoded["score"] == 0.75
        assert decoded["confidence"] == 0.9
        assert decoded["features"]["feature1"] == 0.8
        assert decoded["features"]["feature2"] == 0.6
        assert decoded["metadata"]["processing_time"] == 0.35
        assert decoded["metadata"]["model_version"] == "1.2.3"


class TestVideoAgent:
    """Tests for the VideoAgent implementation."""
    
    @pytest.fixture
    def video_agent(self):
        """Create a VideoAgent instance for testing."""
        return VideoAgent()
    
    @pytest.fixture
    def sample_video_content(self):
        """Sample video content for testing."""
        return {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "platform": "youtube",
            "metadata": {
                "title": "Test Video",
                "duration": 210,  # 3:30 in seconds
                "width": 1920,
                "height": 1080
            }
        }
    
    @pytest.fixture
    def sample_video_settings(self):
        """Sample settings for video processing."""
        return {
            "features_enabled": {
                "scene_detection": True,
                "motion_analysis": True,
                "color_analysis": True,
                "object_detection": False,
                "thumbnail_analysis": True
            },
            "depth": "standard"
        }
    
    @patch("backend.agents.video.VideoAgent._extract_frames")
    @patch("backend.agents.video.VideoAgent._analyze_scenes")
    @patch("backend.agents.video.VideoAgent._analyze_motion")
    @patch("backend.agents.video.VideoAgent._analyze_colors")
    @patch("backend.agents.video.VideoAgent._analyze_thumbnail")
    async def test_video_processing(self, mock_thumbnail, mock_colors, mock_motion, 
                             mock_scenes, mock_extract, video_agent,
                             sample_video_content, sample_video_settings):
        """Test the full video processing pipeline."""
        # Set up mocks
        mock_extract.return_value = [MagicMock() for _ in range(10)]  # 10 frames
        mock_scenes.return_value = {
            "scene_count": 5,
            "avg_scene_duration": 42.0,
            "scene_transitions": [30, 75, 120, 165]
        }
        mock_motion.return_value = {
            "motion_intensity": 0.65,
            "motion_consistency": 0.72,
            "motion_peaks": [45, 120, 180]
        }
        mock_colors.return_value = {
            "color_palette": ["#FF0000", "#00FF00", "#0000FF"],
            "color_diversity": 0.68,
            "brightness": 0.72,
            "contrast": 0.81
        }
        mock_thumbnail.return_value = {
            "thumbnail_clarity": 0.85,
            "thumbnail_appeal": 0.79,
            "text_presence": True,
            "face_presence": True
        }
        
        # Call the process method
        result = await video_agent.process(sample_video_content, sample_video_settings)
        
        # Verify the result
        assert result.success is True
        assert 0 <= result.score <= 1
        assert 0 <= result.confidence <= 1
        
        # Verify features
        assert "scene_analysis" in result.features
        assert "motion_analysis" in result.features
        assert "color_analysis" in result.features
        assert "thumbnail_analysis" in result.features
        
        # Verify the mocks were called correctly
        mock_extract.assert_called_once_with(sample_video_content, sample_video_settings)
        mock_scenes.assert_called_once()
        mock_motion.assert_called_once()
        mock_colors.assert_called_once()
        mock_thumbnail.assert_called_once()
    
    @patch("backend.agents.video.VideoAgent._extract_frames")
    async def test_video_processing_error_handling(self, mock_extract, video_agent,
                                           sample_video_content, sample_video_settings):
        """Test error handling during video processing."""
        # Set up mock to raise an exception
        mock_extract.side_effect = Exception("Failed to extract frames")
        
        # Call the process method
        result = await video_agent.process(sample_video_content, sample_video_settings)
        
        # Verify the result indicates failure
        assert result.success is False
        assert result.score is None
        assert result.error == "Failed to extract frames"
        assert video_agent.status == AgentStatus.ERROR
        assert video_agent.last_error == "Failed to extract frames"


class TestEngagementScoringAgent:
    """Tests for the EngagementScoringAgent implementation."""
    
    @pytest.fixture
    def scoring_agent(self):
        """Create an EngagementScoringAgent instance for testing."""
        return EngagementScoringAgent()
    
    @pytest.fixture
    def sample_modality_results(self):
        """Sample results from modality agents."""
        return {
            "video": AgentResult(
                success=True,
                score=0.75,
                confidence=0.85,
                features={
                    "scene_analysis": {
                        "scene_count": 8,
                        "avg_scene_duration": 32.5
                    },
                    "motion_analysis": {
                        "motion_intensity": 0.68
                    },
                    "color_analysis": {
                        "color_diversity": 0.72
                    }
                }
            ),
            "audio": AgentResult(
                success=True,
                score=0.82,
                confidence=0.78,
                features={
                    "speech_analysis": {
                        "speech_clarity": 0.88,
                        "emotional_tone": 0.75
                    },
                    "music_analysis": {
                        "tempo": 120,
                        "energy": 0.82
                    }
                }
            ),
            "text": AgentResult(
                success=True,
                score=0.68,
                confidence=0.92,
                features={
                    "sentiment_analysis": {
                        "sentiment_score": 0.65,
                        "subjectivity": 0.45
                    },
                    "topic_analysis": {
                        "topic_relevance": 0.78,
                        "topic_diversity": 0.62
                    }
                }
            )
        }
    
    @pytest.fixture
    def sample_content_metadata(self):
        """Sample content metadata for context."""
        return {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "platform": "youtube",
            "metadata": {
                "title": "Test Video",
                "channel": "Test Channel",
                "published_at": datetime(2023, 1, 15),
                "view_count": 12500,
                "like_count": 850,
                "comment_count": 320,
                "subscriber_count": 50000
            }
        }
    
    @pytest.fixture
    def sample_scoring_settings(self):
        """Sample settings for engagement scoring."""
        return {
            "weighting_strategy": "adaptive",
            "platform_normalization": True,
            "include_historical_context": True,
            "content_category": "educational"
        }
    
    async def test_engagement_scoring(self, scoring_agent, sample_modality_results,
                               sample_content_metadata, sample_scoring_settings):
        """Test the engagement scoring algorithm."""
        # Call the process method
        result = await scoring_agent.process_results(
            sample_modality_results,
            sample_content_metadata,
            sample_scoring_settings
        )
        
        # Verify the result
        assert result.success is True
        assert 0 <= result.score <= 1
        assert 0 <= result.confidence <= 1
        
        # Verify the result structure
        assert "modality_weights" in result.features
        assert "temporal_patterns" in result.features
        assert "platform_specific" in result.features
        
        # Verify modality weights
        weights = result.features["modality_weights"]
        assert "video" in weights
        assert "audio" in weights
        assert "text" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.001  # Weights should sum to 1
    
    async def test_partial_modality_results(self, scoring_agent, sample_modality_results,
                                     sample_content_metadata, sample_scoring_settings):
        """Test scoring with partial modality results (missing one modality)."""
        # Remove one modality
        del sample_modality_results["audio"]
        
        # Call the process method
        result = await scoring_agent.process_results(
            sample_modality_results,
            sample_content_metadata,
            sample_scoring_settings
        )
        
        # Verify the result
        assert result.success is True
        assert 0 <= result.score <= 1
        
        # Verify weights only include available modalities
        weights = result.features["modality_weights"]
        assert "video" in weights
        assert "text" in weights
        assert "audio" not in weights
        assert abs(sum(weights.values()) - 1.0) < 0.001  # Weights should still sum to 1
    
    async def test_failed_modality(self, scoring_agent, sample_modality_results,
                            sample_content_metadata, sample_scoring_settings):
        """Test scoring when one modality failed."""
        # Set one modality as failed
        sample_modality_results["audio"] = AgentResult(
            success=False,
            error="Audio processing failed",
            confidence=0
        )
        
        # Call the process method
        result = await scoring_agent.process_results(
            sample_modality_results,
            sample_content_metadata,
            sample_scoring_settings
        )
        
        # Verify the result
        assert result.success is True  # Overall process still succeeds
        assert 0 <= result.score <= 1
        
        # Verify weights only include successful modalities
        weights = result.features["modality_weights"]
        assert "video" in weights
        assert "text" in weights
        assert "audio" not in weights
        assert abs(sum(weights.values()) - 1.0) < 0.001
        
        # Verify metadata includes warning about failed modality
        assert "warnings" in result.metadata
        assert "audio" in result.metadata["warnings"]


# Additional test classes would be implemented for other agents (AudioAgent, TextAgent, HITLAgent)
# following similar patterns to those demonstrated above. 