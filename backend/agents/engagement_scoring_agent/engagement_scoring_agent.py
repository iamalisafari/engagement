"""
Engagement Scoring Agent Implementation

This module implements a specialized agent for synthesizing multi-modal
features into coherent engagement metrics based on the User Engagement
Scale framework (O'Brien & Toms, 2010).
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..base_agent import AgentMessage, AgentStatus, BaseAgent


class EngagementScoringAgent(BaseAgent):
    """
    Agent responsible for scoring content engagement based on multi-modal features.
    
    This agent synthesizes features from different modalities (video, audio, text)
    into comprehensive engagement metrics based on established engagement frameworks.
    """
    
    def __init__(self, agent_id: str = "engagement_scoring_agent_default"):
        """Initialize the engagement scoring agent with default capabilities."""
        super().__init__(
            agent_id=agent_id,
            agent_type="engagement_scoring_agent",
            description="Synthesizes multi-modal features into engagement metrics",
            version="0.1.0"
        )
        
        # Define agent capabilities
        self.update_capabilities([
            "feature_weighting",
            "temporal_pattern_recognition",
            "platform_specific_normalization",
            "comparative_benchmarking",
            "engagement_dimension_scoring",
            "cross_modal_integration"
        ])
        
        self.logger = logging.getLogger(f"agent.engagement.{agent_id}")
        self.update_status(AgentStatus.READY)
        
        # Initialize feature weights - these would be learned/calibrated in a real implementation
        self._feature_weights = self._initialize_feature_weights()
        
        # Initialize platform-specific normalization parameters
        self._platform_norms = self._initialize_platform_norms()
        
        # Initialize benchmarks for comparative analysis
        self._benchmarks = self._initialize_benchmarks()
        
        # Initialize temporal pattern templates
        self._temporal_patterns = self._initialize_temporal_patterns()
    
    def _initialize_feature_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize default feature weights for engagement scoring.
        
        In a production system, these would be learned from training data
        and continuously updated based on human feedback.
        
        Returns:
            Nested dictionary of feature weights by modality and dimension
        """
        return {
            "video": {
                "aesthetic_appeal": {
                    "color_harmony": 0.25,
                    "visual_complexity.spatial_complexity": 0.15,
                    "production_quality": 0.30,
                    "visual_complexity.information_density": 0.15,
                    "color_scheme.contrast_avg": 0.15
                },
                "focused_attention": {
                    "scene_transitions": 0.20,
                    "motion_intensity.motion_consistency": 0.25,
                    "visual_complexity.temporal_complexity": 0.25,
                    "motion_intensity.camera_stability": 0.15,
                    "visual_complexity.edge_density": 0.15
                },
                "perceived_usability": {
                    "visual_complexity.information_density": 0.35,
                    "motion_intensity.motion_intensity_avg": 0.25,
                    "scene_transitions": 0.20,
                    "production_quality": 0.20
                },
                "novelty": {
                    "visual_complexity.spatial_complexity": 0.35,
                    "color_scheme.color_diversity": 0.25,
                    "motion_intensity.dynamic_range": 0.25,
                    "thumbnail_data.visual_salience": 0.15
                }
            },
            "audio": {
                "aesthetic_appeal": {
                    "audio_quality": 0.35,
                    "emotional_tone.enthusiasm": 0.25,
                    "volume_dynamics.volume_consistency": 0.20,
                    "voice_characteristics.articulation_clarity": 0.20
                },
                "focused_attention": {
                    "voice_characteristics.speech_rate": 0.25,
                    "emotional_tone.tension": 0.20,
                    "volume_dynamics.sudden_changes": 0.20,
                    "voice_characteristics.pause_frequency": 0.20,
                    "emotional_tone.arousal": 0.15
                },
                "emotional_response": {
                    "emotional_tone.valence": 0.30,
                    "emotional_tone.emotion_intensity": 0.25,
                    "voice_characteristics.pitch_range": 0.20,
                    "music_segments.*.energy": 0.25
                }
            },
            "text": {
                "perceived_usability": {
                    "readability_scores.flesch_reading_ease": 0.30,
                    "linguistic_complexity": 0.30,
                    "sentiment.objectivity": 0.20,
                    "word_count": 0.20
                },
                "emotional_response": {
                    "emotional_content.emotional_intensity": 0.30,
                    "sentiment.compound": 0.25,
                    "emotional_content.joy": 0.15,
                    "emotional_content.anticipation": 0.15,
                    "sentiment.subjectivity": 0.15
                },
                "novelty": {
                    "topics.*.relevance": 0.40,
                    "keywords.*.relevance": 0.30,
                    "linguistic_complexity": 0.30
                }
            }
        }
    
    def _initialize_platform_norms(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize platform-specific normalization parameters.
        
        These specify how engagement metrics should be adjusted based on the
        platform context and typical engagement patterns on each platform.
        
        Returns:
            Dictionary of normalization parameters by platform
        """
        return {
            "youtube": {
                "baseline_engagement": 0.65,
                "dimension_weights": {
                    "aesthetic_appeal": 0.25,
                    "focused_attention": 0.30,
                    "emotional_response": 0.20,
                    "perceived_usability": 0.15,
                    "novelty": 0.10
                },
                "modality_weights": {
                    "video": 0.50,
                    "audio": 0.30,
                    "text": 0.20
                },
                "retention_curve_templates": {
                    "educational": [1.0, 0.85, 0.75, 0.70, 0.68, 0.65],
                    "entertainment": [1.0, 0.90, 0.82, 0.76, 0.72, 0.65],
                    "tutorial": [1.0, 0.92, 0.88, 0.82, 0.75, 0.68]
                }
            },
            "reddit": {
                "baseline_engagement": 0.58,
                "dimension_weights": {
                    "aesthetic_appeal": 0.10,
                    "focused_attention": 0.25,
                    "emotional_response": 0.25,
                    "perceived_usability": 0.25,
                    "novelty": 0.15
                },
                "modality_weights": {
                    "video": 0.30,
                    "audio": 0.15,
                    "text": 0.55
                },
                "comment_engagement_weight": 0.45  # Reddit-specific factor
            }
        }
    
    def _initialize_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize benchmarks for comparative analysis.
        
        These represent typical engagement scores for different content types
        and categories, allowing content to be compared against appropriate
        reference points.
        
        Returns:
            Dictionary of benchmarks by platform and content category
        """
        return {
            "youtube": {
                "education": {
                    "aesthetic_appeal": 0.72,
                    "focused_attention": 0.68,
                    "emotional_response": 0.65,
                    "perceived_usability": 0.75,
                    "novelty": 0.62,
                    "composite_score": 0.68
                },
                "entertainment": {
                    "aesthetic_appeal": 0.78,
                    "focused_attention": 0.72,
                    "emotional_response": 0.82,
                    "perceived_usability": 0.65,
                    "novelty": 0.68,
                    "composite_score": 0.73
                },
                "tutorial": {
                    "aesthetic_appeal": 0.68,
                    "focused_attention": 0.78,
                    "emotional_response": 0.62,
                    "perceived_usability": 0.82,
                    "novelty": 0.58,
                    "composite_score": 0.70
                }
            },
            "reddit": {
                "discussion": {
                    "aesthetic_appeal": 0.58,
                    "focused_attention": 0.72,
                    "emotional_response": 0.68,
                    "perceived_usability": 0.75,
                    "novelty": 0.65,
                    "composite_score": 0.68
                },
                "research": {
                    "aesthetic_appeal": 0.55,
                    "focused_attention": 0.78,
                    "emotional_response": 0.62,
                    "perceived_usability": 0.85,
                    "novelty": 0.72,
                    "composite_score": 0.70
                }
            }
        }
    
    def _initialize_temporal_patterns(self) -> Dict[str, List[float]]:
        """
        Initialize temporal pattern templates for pattern recognition.
        
        These templates represent common engagement patterns over time,
        which can be matched against actual engagement trajectories.
        
        Returns:
            Dictionary of engagement pattern templates
        """
        return {
            "sustained": [0.85, 0.84, 0.82, 0.83, 0.81, 0.80],
            "declining": [0.90, 0.85, 0.75, 0.65, 0.58, 0.52],
            "increasing": [0.65, 0.70, 0.78, 0.82, 0.85, 0.88],
            "u_shaped": [0.85, 0.75, 0.65, 0.62, 0.72, 0.82],
            "peak_and_valley": [0.75, 0.85, 0.75, 0.88, 0.72, 0.82],
            "step_function": [0.65, 0.65, 0.85, 0.85, 0.85, 0.82]
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multi-modal features to generate engagement metrics.
        
        This implements a weighted feature integration approach based on the
        User Engagement Scale framework (O'Brien & Toms, 2010).
        
        Args:
            input_data: Dict containing features and processing parameters
                Required keys:
                - content_id: Unique identifier for the content
                - platform: Platform the content is from (e.g., "youtube", "reddit")
                - content_category: Category of the content (e.g., "education", "entertainment")
                - features: Dict containing video_features, audio_features, text_features
                Optional keys:
                - temporal_features: List of feature sets at different time points
                - benchmark_category: Category to benchmark against (overrides content_category)
                - custom_weights: Custom feature weights to override defaults
        
        Returns:
            Dict containing engagement metrics
        """
        self.update_status(AgentStatus.PROCESSING)
        self.logger.info(f"Generating engagement metrics for {input_data.get('content_id', 'unknown')}")
        
        try:
            # Extract required information
            content_id = input_data.get("content_id", "unknown")
            platform = input_data.get("platform", "youtube").lower()
            category = input_data.get("content_category", "education").lower()
            benchmark_category = input_data.get("benchmark_category", category).lower()
            features = input_data.get("features", {})
            
            # Get platform-specific normalization parameters
            platform_norm = self._platform_norms.get(
                platform, self._platform_norms["youtube"]  # Default to YouTube if platform not found
            )
            
            # Calculate scores for each engagement dimension
            dimension_scores = self._calculate_dimension_scores(features, platform, platform_norm)
            
            # Calculate composite engagement score
            composite_score = self._calculate_composite_score(dimension_scores, platform_norm)
            
            # Analyze temporal pattern if temporal features provided
            temporal_pattern = "not_applicable"
            temporal_data = {}
            if "temporal_features" in input_data:
                temporal_pattern, temporal_data = self._analyze_temporal_pattern(
                    input_data["temporal_features"], platform
                )
            
            # Compare to benchmarks
            benchmark_comparison = self._compare_to_benchmarks(
                dimension_scores, composite_score, platform, benchmark_category
            )
            
            # Construct response
            result = {
                "content_id": content_id,
                "composite_score": composite_score,
                "dimensions": dimension_scores,
                "temporal_pattern": temporal_pattern,
                "benchmark_comparison": benchmark_comparison,
                "platform_specific": self._get_platform_specific_metrics(
                    features, platform, composite_score, temporal_pattern
                ),
                "analysis_version": self.metadata.version
            }
            
            if temporal_data:
                result["temporal_data"] = temporal_data
                
            self.update_status(AgentStatus.READY)
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating engagement metrics: {e}")
            self.update_status(AgentStatus.ERROR)
            return {
                "error": str(e),
                "content_id": input_data.get("content_id", "unknown")
            }
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Handle incoming messages from other agents.
        
        Args:
            message: The message to handle
        """
        if message.message_type == "process_request":
            result = await self.process(message.content)
            await self.send_message(
                recipient_id=message.sender_id,
                message_type="process_response",
                content=result,
                correlation_id=message.correlation_id
            )
        elif message.message_type == "update_weights_request":
            # Handle request to update weights based on feedback
            self._update_feature_weights(message.content)
            await self.send_message(
                recipient_id=message.sender_id,
                message_type="update_weights_response",
                content={"status": "weights_updated"},
                correlation_id=message.correlation_id
            )
        elif message.message_type == "status_request":
            await self.send_message(
                recipient_id=message.sender_id,
                message_type="status_response",
                content={"status": self.get_status().value},
                correlation_id=message.correlation_id
            )
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
    
    def _calculate_dimension_scores(
        self, features: Dict[str, Any], platform: str, platform_norm: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate scores for each engagement dimension based on features.
        
        This implements a weighted combination of features for each dimension,
        based on the User Engagement Scale framework.
        
        Args:
            features: Features extracted from content
            platform: Platform the content is from
            platform_norm: Platform-specific normalization parameters
            
        Returns:
            Dictionary of dimension scores with confidence and contributing factors
        """
        dimension_scores = {}
        
        # Get modality weights for this platform
        modality_weights = platform_norm.get("modality_weights", {
            "video": 0.4, "audio": 0.3, "text": 0.3  # Default weights
        })
        
        # Process each engagement dimension
        for dimension in ["aesthetic_appeal", "focused_attention", "emotional_response", 
                         "perceived_usability", "novelty"]:
            # Calculate dimension score from each modality
            modality_scores = {}
            
            # Process video features
            if "video_features" in features and "video" in self._feature_weights:
                video_dim_weights = self._feature_weights["video"].get(dimension, {})
                if video_dim_weights:
                    video_score = self._apply_feature_weights(features["video_features"], video_dim_weights)
                    modality_scores["video"] = video_score
            
            # Process audio features
            if "audio_features" in features and "audio" in self._feature_weights:
                audio_dim_weights = self._feature_weights["audio"].get(dimension, {})
                if audio_dim_weights:
                    audio_score = self._apply_feature_weights(features["audio_features"], audio_dim_weights)
                    modality_scores["audio"] = audio_score
            
            # Process text features
            if "text_features" in features and "text" in self._feature_weights:
                text_dim_weights = self._feature_weights["text"].get(dimension, {})
                if text_dim_weights:
                    text_score = self._apply_feature_weights(features["text_features"], text_dim_weights)
                    modality_scores["text"] = text_score
            
            # Combine modality scores using platform-specific modality weights
            dimension_value = 0.0
            contributing_factors = {}
            confidence = 0.0
            
            if modality_scores:
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for modality, score in modality_scores.items():
                    if modality in modality_weights:
                        weight = modality_weights[modality]
                        weighted_sum += score * weight
                        weight_sum += weight
                        contributing_factors[f"{modality}_contribution"] = score
                
                if weight_sum > 0:
                    dimension_value = weighted_sum / weight_sum
                    confidence = min(1.0, 0.5 + (len(modality_scores) / 3) * 0.5)
            
            # Add dimension score to results
            dimension_scores[dimension] = {
                "value": dimension_value,
                "confidence": confidence,
                "contributing_factors": contributing_factors,
                "temporal_pattern": "not_analyzed"  # Will be updated if temporal analysis is done
            }
        
        return dimension_scores
    
    def _apply_feature_weights(self, features: Dict[str, Any], weights: Dict[str, float]) -> float:
        """
        Apply weights to features to calculate a weighted score.
        
        Supports nested feature access using dot notation in weight keys.
        
        Args:
            features: Feature dictionary
            weights: Weight dictionary with feature paths as keys
            
        Returns:
            Weighted score
        """
        score = 0.0
        weight_sum = 0.0
        
        for feature_path, weight in weights.items():
            # Handle nested features using dot notation
            if "." in feature_path:
                parts = feature_path.split(".")
                value = features
                try:
                    for part in parts:
                        if part == "*" and isinstance(value, list):
                            # For list features, average the values
                            subvalues = []
                            for item in value:
                                if isinstance(item, dict) and parts[-1] in item:
                                    subvalues.append(item[parts[-1]])
                            if subvalues:
                                value = sum(subvalues) / len(subvalues)
                            else:
                                value = 0.0
                            break
                        else:
                            value = value[part]
                    
                    # Handle different value types
                    if isinstance(value, list):
                        # If we get here with a list, just use the length
                        value = len(value) / 10.0  # Normalize by assuming 10 is "high"
                    elif isinstance(value, dict):
                        # If we get here with a dict, use average of values
                        if value:
                            value = sum(value.values()) / len(value)
                        else:
                            value = 0.0
                    
                    score += float(value) * weight
                    weight_sum += weight
                except (KeyError, TypeError, ValueError):
                    # Skip if feature not found or not numeric
                    pass
            
            # Handle direct feature access
            elif feature_path in features:
                value = features[feature_path]
                try:
                    # Handle different value types
                    if isinstance(value, list):
                        value = len(value) / 10.0  # Normalize
                    elif isinstance(value, dict):
                        if value:
                            value = sum(value.values()) / len(value)
                        else:
                            value = 0.0
                    
                    score += float(value) * weight
                    weight_sum += weight
                except (TypeError, ValueError):
                    # Skip if not numeric
                    pass
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _calculate_composite_score(
        self, dimension_scores: Dict[str, Dict[str, Any]], platform_norm: Dict[str, Any]
    ) -> float:
        """
        Calculate composite engagement score from dimension scores.
        
        Args:
            dimension_scores: Dictionary of dimension scores
            platform_norm: Platform-specific normalization parameters
            
        Returns:
            Composite engagement score
        """
        # Get dimension weights for this platform
        dimension_weights = platform_norm.get("dimension_weights", {
            "aesthetic_appeal": 0.2,
            "focused_attention": 0.25,
            "emotional_response": 0.2,
            "perceived_usability": 0.2,
            "novelty": 0.15
        })
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for dimension, score_data in dimension_scores.items():
            if dimension in dimension_weights:
                # Weight by both dimension weight and confidence
                weight = dimension_weights[dimension] * score_data["confidence"]
                weighted_sum += score_data["value"] * weight
                weight_sum += weight
        
        # Calculate weighted average
        if weight_sum > 0:
            composite_score = weighted_sum / weight_sum
        else:
            # Fallback to simple average if no valid weights
            valid_scores = [s["value"] for s in dimension_scores.values() if s["value"] > 0]
            composite_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        # Apply platform baseline adjustment
        baseline = platform_norm.get("baseline_engagement", 0.65)
        composite_score = (composite_score * 0.8) + (baseline * 0.2)  # Slight regression to platform mean
        
        return min(1.0, max(0.0, composite_score))  # Ensure 0-1 range
    
    def _analyze_temporal_pattern(
        self, temporal_features: List[Dict[str, Any]], platform: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze temporal patterns in engagement over time.
        
        Args:
            temporal_features: List of feature sets at different time points
            platform: Platform the content is from
            
        Returns:
            Tuple of (pattern_name, temporal_data)
        """
        # Process each time point to get engagement scores
        time_points = []
        dimension_series = {
            "aesthetic_appeal": [],
            "focused_attention": [],
            "emotional_response": [],
            "perceived_usability": [],
            "novelty": []
        }
        composite_series = []
        
        for i, features in enumerate(temporal_features):
            platform_norm = self._platform_norms.get(
                platform, self._platform_norms["youtube"]
            )
            
            # Calculate dimension scores for this time point
            time_dimension_scores = self._calculate_dimension_scores(features, platform, platform_norm)
            
            # Calculate composite score for this time point
            time_composite_score = self._calculate_composite_score(time_dimension_scores, platform_norm)
            
            # Add to time series
            time_points.append(i / (len(temporal_features) - 1) if len(temporal_features) > 1 else 0)
            composite_series.append(time_composite_score)
            
            for dimension, score_data in time_dimension_scores.items():
                if dimension in dimension_series:
                    dimension_series[dimension].append(score_data["value"])
        
        # Identify pattern by comparing to templates
        pattern_name = self._identify_temporal_pattern(composite_series)
        
        # Update dimension temporal patterns
        dimension_patterns = {}
        for dimension, series in dimension_series.items():
            if series:
                dimension_patterns[dimension] = self._identify_temporal_pattern(series)
        
        # Return pattern name and temporal data
        return pattern_name, {
            "time_points": time_points,
            "composite_series": composite_series,
            "dimension_series": dimension_series,
            "dimension_patterns": dimension_patterns
        }
    
    def _identify_temporal_pattern(self, series: List[float]) -> str:
        """
        Identify the temporal pattern that best matches a time series.
        
        Args:
            series: Time series of engagement values
            
        Returns:
            Name of the best matching pattern
        """
        if not series or len(series) < 3:
            return "insufficient_data"
        
        # Normalize series to 0-1 range for comparison
        if max(series) != min(series):
            normalized = [(x - min(series)) / (max(series) - min(series)) for x in series]
        else:
            normalized = [0.5] * len(series)
        
        # Resample to match template length if needed
        if len(normalized) < 6:
            # Simple linear interpolation
            resampled = []
            for i in range(6):
                idx = i * (len(normalized) - 1) / 5
                idx_floor = int(idx)
                idx_ceil = min(idx_floor + 1, len(normalized) - 1)
                frac = idx - idx_floor
                value = normalized[idx_floor] * (1 - frac) + normalized[idx_ceil] * frac
                resampled.append(value)
            normalized = resampled
        elif len(normalized) > 6:
            # Downsample
            indices = [int(i * (len(normalized) - 1) / 5) for i in range(6)]
            normalized = [normalized[i] for i in indices]
        
        # Compare to each template
        best_match = "unknown"
        best_score = float('inf')
        
        for pattern_name, template in self._temporal_patterns.items():
            # Calculate mean squared error
            error = sum((normalized[i] - template[i])**2 for i in range(6)) / 6
            if error < best_score:
                best_score = error
                best_match = pattern_name
        
        # Detect flat pattern (special case)
        std_dev = np.std(normalized)
        if std_dev < 0.05:  # Very little variation
            return "sustained"
        
        return best_match
    
    def _compare_to_benchmarks(
        self, dimension_scores: Dict[str, Dict[str, Any]], composite_score: float,
        platform: str, category: str
    ) -> Dict[str, Any]:
        """
        Compare engagement scores to benchmarks for similar content.
        
        Args:
            dimension_scores: Dictionary of dimension scores
            composite_score: Composite engagement score
            platform: Platform the content is from
            category: Content category
            
        Returns:
            Dictionary with benchmark comparison results
        """
        # Get benchmark data
        platform_benchmarks = self._benchmarks.get(platform, {})
        category_benchmarks = platform_benchmarks.get(
            category, next(iter(platform_benchmarks.values())) if platform_benchmarks else {}
        )
        
        if not category_benchmarks:
            return {
                "status": "no_benchmark_available",
                "percentile": None
            }
        
        # Compare each dimension
        dimension_comparisons = {}
        for dimension, score_data in dimension_scores.items():
            if dimension in category_benchmarks:
                benchmark = category_benchmarks[dimension]
                difference = score_data["value"] - benchmark
                
                # Calculate percentile based on expected distribution
                # This is a simplified approximation
                z_score = difference / 0.15  # Assuming SD of 0.15
                percentile = min(100, max(0, 50 + int(z_score * 30)))
                
                dimension_comparisons[dimension] = {
                    "benchmark": benchmark,
                    "difference": difference,
                    "percentile": percentile
                }
        
        # Compare composite score
        if "composite_score" in category_benchmarks:
            benchmark = category_benchmarks["composite_score"]
            difference = composite_score - benchmark
            z_score = difference / 0.12  # Assuming different SD for composite
            percentile = min(100, max(0, 50 + int(z_score * 30)))
        else:
            benchmark = None
            difference = None
            percentile = None
        
        return {
            "status": "benchmark_available",
            "category": category,
            "composite": {
                "benchmark": benchmark,
                "difference": difference,
                "percentile": percentile
            },
            "dimensions": dimension_comparisons
        }
    
    def _get_platform_specific_metrics(
        self, features: Dict[str, Any], platform: str, 
        composite_score: float, temporal_pattern: str
    ) -> Dict[str, Any]:
        """
        Calculate platform-specific engagement metrics.
        
        Args:
            features: Feature dictionary
            platform: Platform the content is from
            composite_score: Composite engagement score
            temporal_pattern: Identified temporal pattern
            
        Returns:
            Dictionary of platform-specific metrics
        """
        if platform == "youtube":
            # Calculate YouTube-specific metrics
            retention_index = min(1.0, max(0.0, 
                composite_score * 0.7 + 
                (0.3 if temporal_pattern in ["sustained", "increasing"] else 0.1)
            ))
            
            shareability = min(1.0, max(0.0,
                composite_score * 0.6 +
                (0.4 if "emotional_tone" in features.get("audio_features", {}) and
                 features["audio_features"]["emotional_tone"].get("valence", 0) > 0.7 else 0.2)
            ))
            
            return {
                "youtube_retention_index": retention_index,
                "predicted_shareability": shareability
            }
        
        elif platform == "reddit":
            # Calculate Reddit-specific metrics
            upvote_ratio = min(1.0, max(0.5,
                0.5 + (composite_score - 0.5) * 0.8
            ))
            
            comment_engagement = min(1.0, max(0.0,
                composite_score * 0.7 +
                (0.3 if "topics" in features.get("text_features", {}) and
                 len(features["text_features"].get("topics", [])) > 2 else 0.1)
            ))
            
            return {
                "reddit_upvote_ratio": upvote_ratio,
                "comment_engagement_index": comment_engagement
            }
        
        else:
            # Generic platform
            return {
                "generic_engagement_index": composite_score
            }
    
    def _update_feature_weights(self, weight_updates: Dict[str, Any]) -> None:
        """
        Update feature weights based on feedback or learning.
        
        Args:
            weight_updates: Dictionary with weight updates
        """
        if "modality" in weight_updates and "dimension" in weight_updates:
            modality = weight_updates["modality"]
            dimension = weight_updates["dimension"]
            
            if "updates" in weight_updates and isinstance(weight_updates["updates"], dict):
                # Ensure modality and dimension exist in weights
                if modality not in self._feature_weights:
                    self._feature_weights[modality] = {}
                    
                if dimension not in self._feature_weights[modality]:
                    self._feature_weights[modality][dimension] = {}
                
                # Apply updates
                for feature, weight in weight_updates["updates"].items():
                    self._feature_weights[modality][dimension][feature] = weight
                
                self.logger.info(f"Updated weights for {modality}.{dimension}")
                
        # Handle platform norm updates
        if "platform_norms" in weight_updates:
            platform = weight_updates.get("platform")
            if platform and platform in self._platform_norms:
                for key, value in weight_updates["platform_norms"].items():
                    if key in self._platform_norms[platform]:
                        self._platform_norms[platform][key] = value
                
                self.logger.info(f"Updated platform norms for {platform}") 