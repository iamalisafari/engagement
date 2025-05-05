"""
Engagement Models

This module defines models for quantifying engagement metrics based on the
User Engagement Scale framework (O'Brien & Toms, 2010) and extended with
platform-specific metrics.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EngagementDimension(str, Enum):
    """
    Core engagement dimensions based on User Engagement Scale (UES),
    as defined by O'Brien & Toms (2010), with additional dimensions
    for social media context.
    """
    AESTHETIC_APPEAL = "aesthetic_appeal"  # Visual and sensory appeal
    FOCUSED_ATTENTION = "focused_attention"  # Concentration and absorption
    PERCEIVED_USABILITY = "perceived_usability"  # Ease of use and control
    ENDURABILITY = "endurability"  # Likelihood of remembering and returning
    NOVELTY = "novelty"  # Curiosity, surprise, and newness
    INVOLVEMENT = "involvement"  # Interest and motivation
    SOCIAL_PRESENCE = "social_presence"  # Sense of connection with others
    SHAREABILITY = "shareability"  # Likelihood of sharing content
    EMOTIONAL_RESPONSE = "emotional_response"  # Affective reactions


class TemporalPattern(str, Enum):
    """
    Patterns of engagement over time, based on theoretical models
    of attention spans and engagement trajectories.
    """
    SUSTAINED = "sustained"  # Consistently high engagement
    DECLINING = "declining"  # Starts high, gradually decreases
    INCREASING = "increasing"  # Builds up over time
    U_SHAPED = "u_shaped"  # High at beginning and end, lower in middle
    INVERTED_U = "inverted_u"  # Peaks in middle, lower at start and end
    FLUCTUATING = "fluctuating"  # Varies significantly throughout
    CLIFF = "cliff"  # Sudden drop after initial period
    PEAK_AND_VALLEY = "peak_and_valley"  # Multiple peaks and valleys


class EngagementScore(BaseModel):
    """
    Detailed engagement score for a specific dimension,
    with statistical information for academic rigor.
    """
    value: float = Field(..., ge=0.0, le=1.0)  # Normalized score (0-1)
    confidence: float = Field(..., ge=0.0, le=1.0)  # Statistical confidence
    contributing_factors: Dict[str, float] = Field(default_factory=dict)
    temporal_distribution: Optional[List[float]] = None  # Time-series values if available
    temporal_pattern: Optional[TemporalPattern] = None
    benchmark_percentile: Optional[float] = None  # Percentile relative to benchmark


class EngagementMetrics(BaseModel):
    """
    Comprehensive engagement metrics for content,
    structured according to validated engagement dimensions
    from academic literature.
    """
    content_id: str
    composite_score: float = Field(..., ge=0.0, le=1.0)
    dimensions: Dict[EngagementDimension, EngagementScore]
    platform_specific: Dict[str, float] = Field(default_factory=dict)
    demographic_breakdown: Optional[Dict[str, Dict[str, float]]] = None
    comparative_metrics: Optional[Dict[str, float]] = None
    temporal_pattern: TemporalPattern
    created_at: datetime = Field(default_factory=datetime.now)
    analysis_version: str  # Version of the analysis methodology
    
    class Config:
        schema_extra = {
            "example": {
                "content_id": "yt_12345abcde",
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
                    }
                },
                "temporal_pattern": "SUSTAINED",
                "analysis_version": "1.0.3"
            }
        } 