"""
Content Models

This module defines data models for representing multi-modal content
from social media platforms, based on Media Richness Theory 
(Daft & Lengel, 1986) which categorizes media by their capacity to
facilitate shared meaning.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl


class Platform(str, Enum):
    """Social media platforms supported by the system."""
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    # Will be expanded as needed


class ContentType(str, Enum):
    """Content types based on modality components."""
    VIDEO = "video"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MIXED = "mixed"


class ContentMetadata(BaseModel):
    """
    Metadata for content items, capturing contextual factors
    that influence engagement as per Social Presence Theory
    (Short et al., 1976).
    """
    title: str
    description: Optional[str] = None
    creator_id: str
    creator_name: str
    platform: Platform
    published_at: datetime
    url: HttpUrl
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    language: str = "en"
    duration_seconds: Optional[int] = None  # For time-based content
    

class VideoFeatures(BaseModel):
    """
    Features extracted from video content, based on visual engagement
    factors identified in multimedia engagement research.
    """
    resolution: str  # e.g., "1080p"
    fps: float  # Frames per second
    scene_transitions: List[float] = Field(default_factory=list)  # Timestamps of transitions
    visual_complexity: Dict[str, float] = Field(default_factory=dict)  # Various complexity metrics
    motion_intensity: Dict[str, float] = Field(default_factory=dict)  # Motion metrics over time
    color_scheme: Dict[str, float] = Field(default_factory=dict)  # Color palette analysis
    production_quality: float = 0.0  # Normalized score (0-1)
    thumbnail_data: Optional[Dict[str, float]] = None  # Thumbnail analysis data


class AudioFeatures(BaseModel):
    """
    Features extracted from audio content, capturing elements that
    influence cognitive processing as per Information Processing Theory.
    """
    sample_rate: int
    bit_depth: int
    speech_segments: List[Dict[str, Union[float, str]]] = Field(default_factory=list)
    music_segments: List[Dict[str, Union[float, str]]] = Field(default_factory=list)
    volume_dynamics: Dict[str, float] = Field(default_factory=dict)
    voice_characteristics: Optional[Dict[str, float]] = None
    emotional_tone: Dict[str, float] = Field(default_factory=dict)
    audio_quality: float = 0.0  # Normalized score (0-1)


class TextFeatures(BaseModel):
    """
    Features extracted from textual content using NLP techniques,
    based on linguistic factors that affect engagement.
    """
    word_count: int
    sentiment: Dict[str, float] = Field(default_factory=dict)
    topics: List[Dict[str, float]] = Field(default_factory=list)
    readability_scores: Dict[str, float] = Field(default_factory=dict)
    linguistic_complexity: float = 0.0
    keywords: List[Dict[str, float]] = Field(default_factory=list)
    emotional_content: Dict[str, float] = Field(default_factory=dict)


class Content(BaseModel):
    """
    Comprehensive content model integrating all modalities,
    following the principles of Media Richness Theory to represent
    varying richness levels across different content types.
    """
    id: str
    content_type: ContentType
    metadata: ContentMetadata
    video_features: Optional[VideoFeatures] = None
    audio_features: Optional[AudioFeatures] = None
    text_features: Optional[TextFeatures] = None
    raw_data_location: Optional[str] = None  # Storage location for raw content
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "id": "yt_12345abcde",
                "content_type": "VIDEO",
                "metadata": {
                    "title": "Understanding User Engagement in Social Media",
                    "description": "This video explores the factors affecting user engagement...",
                    "creator_id": "UC12345abcde",
                    "creator_name": "Academic Research Channel",
                    "platform": "YOUTUBE",
                    "published_at": "2023-05-15T14:30:00Z",
                    "url": "https://www.youtube.com/watch?v=12345abcde",
                    "tags": ["user engagement", "social media", "research"],
                    "category": "Education",
                    "language": "en",
                    "duration_seconds": 843
                }
            }
        } 