"""
Feature Extraction Workflows

This module implements feature extraction techniques for different content types,
extracting engagement-relevant features based on established research in
media richness theory, user engagement frameworks, and multimodal analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import re
import json

from ..models.content import ContentType, Platform
from ..models.engagement import EngagementDimension

# Configure logging
logger = logging.getLogger("utils.feature_extraction")


class FeatureExtractor:
    """Base class for feature extractors."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.scaler = MinMaxScaler()
        
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from data.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Extracted features
        """
        raise NotImplementedError("Subclasses must implement extract()")


class TextFeatureExtractor(FeatureExtractor):
    """Extract features from textual content."""
    
    def __init__(self):
        """Initialize the text feature extractor."""
        super().__init__()
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from textual content.
        
        Extracts linguistic complexity, sentiment, readability,
        and topic-based features that correlate with engagement.
        
        Args:
            data: Preprocessed data containing cleaned text
            
        Returns:
            Dictionary of extracted features
        """
        results = {}
        
        text = data.get("cleaned_text", "")
        if not text:
            logger.warning("No text content to extract features from")
            return {"text_features": {}}
        
        # Basic text statistics
        word_count = len(text.split())
        results["word_count"] = word_count
        results["char_count"] = len(text)
        results["avg_word_length"] = len(text) / max(1, word_count)
        
        # Sentence-level features
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        results["sentence_count"] = len(sentences)
        
        if sentences:
            results["avg_sentence_length"] = word_count / len(sentences)
            
            # Calculate sentence complexity variation
            sentence_lengths = [len(s.split()) for s in sentences]
            results["sentence_length_std"] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Calculate readability (simplified Flesch Reading Ease)
        if word_count > 0 and len(sentences) > 0:
            results["readability_score"] = self._calculate_readability(text, sentences)
        
        # Extract sentiment features (simplified)
        results["sentiment_features"] = self._extract_sentiment(text)
        
        # TF-IDF for topic modeling (top terms)
        if word_count >= 20:  # Only do topic modeling if enough text
            try:
                # Fit TF-IDF
                tfidf_matrix = self.tfidf.fit_transform([text])
                
                # Get feature names
                feature_names = self.tfidf.get_feature_names_out()
                
                # Get top terms
                dense = tfidf_matrix.todense()
                episode = dense[0].tolist()[0]
                phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
                sorted_phrase_scores = sorted(phrase_scores, key=lambda x: x[1], reverse=True)
                
                # Extract top 5 terms
                top_terms = []
                for i, score in sorted_phrase_scores[:5]:
                    top_terms.append((feature_names[i], score))
                
                results["top_terms"] = top_terms
                
            except Exception as e:
                logger.error(f"Error in TF-IDF extraction: {e}")
        
        return {"text_features": results}
    
    def _calculate_readability(self, text: str, sentences: List[str]) -> float:
        """
        Calculate readability score (simplified Flesch Reading Ease).
        
        Args:
            text: Full text content
            sentences: List of sentences
            
        Returns:
            Readability score (0-100, higher is easier to read)
        """
        word_count = len(text.split())
        sentence_count = len(sentences)
        
        # Count syllables (simplified approach)
        syllable_count = 0
        for word in text.lower().split():
            word = word.strip(".,;:!?()[]{}\"'")
            if not word:
                continue
                
            # Count vowel groups as syllables
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
                
            # Words should have at least one syllable
            syllable_count += max(1, count)
        
        # Calculate Flesch Reading Ease
        if word_count == 0 or sentence_count == 0:
            return 50.0  # Default mid-range value
            
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count
        
        score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        
        # Ensure score is in the 0-100 range
        return max(0, min(100, score))
    
    def _extract_sentiment(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features from text.
        
        In a production system, this would use a pre-trained model.
        This implementation uses a very simplified lexicon approach.
        
        Args:
            text: Text content
            
        Returns:
            Dictionary of sentiment features
        """
        # Simplified sentiment analysis using word lists
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "awesome", "best", "love", "like", "happy", "positive", "beautiful",
            "perfect", "recommend", "enjoy", "impressive", "useful", "helpful"
        }
        
        negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "poor", "disappointment",
            "disappointing", "hate", "dislike", "negative", "useless", "waste",
            "problem", "issue", "difficult", "frustrating", "annoying", "ugly"
        }
        
        # Emotional intensity words
        intensity_words = {
            "very", "extremely", "absolutely", "completely", "totally",
            "utterly", "incredibly", "remarkably", "exceptionally"
        }
        
        # Tokenize text to words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count occurrences
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        intensity_count = sum(1 for word in words if word in intensity_words)
        
        # Calculate sentiment scores
        word_count = len(words)
        if word_count > 0:
            pos_ratio = pos_count / word_count
            neg_ratio = neg_count / word_count
            intensity_ratio = intensity_count / word_count
            
            # Calculate overall sentiment (-1 to 1)
            sentiment_score = (pos_ratio - neg_ratio) * (1 + intensity_ratio)
            
            # Ensure it's in -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        else:
            pos_ratio = neg_ratio = intensity_ratio = sentiment_score = 0.0
        
        return {
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "intensity_ratio": intensity_ratio,
            "sentiment_score": sentiment_score,
            "valence": (sentiment_score + 1) / 2,  # Convert to 0-1 scale
            "emotional_intensity": (pos_ratio + neg_ratio) * (1 + intensity_ratio)
        }


class AudioFeatureExtractor(FeatureExtractor):
    """Extract features from audio content."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from audio analysis.
        
        Args:
            data: Preprocessed data containing audio analysis
            
        Returns:
            Dictionary of extracted features
        """
        results = {}
        
        audio_features = data.get("audio_features", {})
        if not audio_features:
            logger.warning("No audio features to extract from")
            return {"audio_features": {}}
        
        # Extract speech features
        speech_segments = audio_features.get("speech_segments", [])
        if speech_segments:
            results["speech_coverage"] = self._calculate_speech_coverage(speech_segments)
            results["speaker_count"] = self._count_speakers(speech_segments)
        
        # Extract voice characteristics
        voice_chars = audio_features.get("voice_characteristics", {})
        if voice_chars:
            results["speech_rate"] = voice_chars.get("speech_rate", 0)
            results["pitch_variation"] = self._calculate_pitch_variation(voice_chars)
            results["articulation_clarity"] = voice_chars.get("articulation_clarity", 0)
        
        # Extract emotional tone
        emotional_tone = audio_features.get("emotional_tone", {})
        if emotional_tone:
            results["emotional_valence"] = emotional_tone.get("valence", 0.5)
            results["emotional_arousal"] = emotional_tone.get("arousal", 0.5)
            results["enthusiasm"] = emotional_tone.get("enthusiasm", 0.5)
        
        # Extract music features
        music_segments = audio_features.get("music_segments", [])
        if music_segments:
            results["music_coverage"] = self._calculate_music_coverage(music_segments)
            results["avg_tempo"] = self._calculate_avg_tempo(music_segments)
            results["avg_energy"] = self._calculate_avg_energy(music_segments)
        
        # Extract volume dynamics
        volume_dynamics = audio_features.get("volume_dynamics", {})
        if volume_dynamics:
            results["dynamic_range"] = volume_dynamics.get("dynamic_range", 0)
            results["volume_consistency"] = volume_dynamics.get("volume_consistency", 0)
        
        return {"audio_features": results}
    
    def _calculate_speech_coverage(self, speech_segments: List[Dict[str, Any]]) -> float:
        """
        Calculate the proportion of content covered by speech.
        
        Args:
            speech_segments: List of speech segments with start/end times
            
        Returns:
            Speech coverage ratio (0-1)
        """
        if not speech_segments:
            return 0.0
        
        total_speech_duration = 0.0
        for segment in speech_segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            total_speech_duration += max(0, end - start)
        
        # Assume the last segment's end time is the content duration
        content_duration = max([segment.get("end", 0) for segment in speech_segments])
        
        if content_duration > 0:
            return min(1.0, total_speech_duration / content_duration)
        return 0.0
    
    def _count_speakers(self, speech_segments: List[Dict[str, Any]]) -> int:
        """
        Count unique speakers in the audio.
        
        Args:
            speech_segments: List of speech segments with speaker ID
            
        Returns:
            Number of unique speakers
        """
        speaker_ids = set()
        for segment in speech_segments:
            speaker_id = segment.get("speaker_id")
            if speaker_id:
                speaker_ids.add(speaker_id)
        
        return len(speaker_ids)
    
    def _calculate_pitch_variation(self, voice_chars: Dict[str, Any]) -> float:
        """
        Calculate pitch variation from voice characteristics.
        
        Args:
            voice_chars: Voice characteristics dictionary
            
        Returns:
            Pitch variation score (0-1)
        """
        pitch_mean = voice_chars.get("pitch_mean", 0)
        pitch_range = voice_chars.get("pitch_range", 0)
        
        # No variation if no mean pitch
        if pitch_mean <= 0:
            return 0.0
        
        # Calculate normalized variation
        variation = pitch_range / pitch_mean
        
        # Typical range is 0-0.5, normalize to 0-1
        return min(1.0, variation * 2)
    
    def _calculate_music_coverage(self, music_segments: List[Dict[str, Any]]) -> float:
        """
        Calculate the proportion of content covered by music.
        
        Args:
            music_segments: List of music segments with start/end times
            
        Returns:
            Music coverage ratio (0-1)
        """
        if not music_segments:
            return 0.0
        
        total_music_duration = 0.0
        for segment in music_segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            total_music_duration += max(0, end - start)
        
        # Assume the last segment's end time is the content duration
        content_duration = max([segment.get("end", 0) for segment in music_segments])
        
        if content_duration > 0:
            return min(1.0, total_music_duration / content_duration)
        return 0.0
    
    def _calculate_avg_tempo(self, music_segments: List[Dict[str, Any]]) -> float:
        """
        Calculate average music tempo.
        
        Args:
            music_segments: List of music segments with tempo information
            
        Returns:
            Average tempo in BPM, normalized to 0-1
        """
        tempos = []
        durations = []
        
        for segment in music_segments:
            tempo = segment.get("tempo", 0)
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            duration = max(0, end - start)
            
            if tempo > 0 and duration > 0:
                tempos.append(tempo)
                durations.append(duration)
        
        if not tempos:
            return 0.0
        
        # Weighted average by duration
        avg_tempo = sum(t * d for t, d in zip(tempos, durations)) / sum(durations)
        
        # Normalize to 0-1 (typical range 60-180 BPM)
        return min(1.0, max(0.0, (avg_tempo - 60) / 120))
    
    def _calculate_avg_energy(self, music_segments: List[Dict[str, Any]]) -> float:
        """
        Calculate average music energy.
        
        Args:
            music_segments: List of music segments with energy information
            
        Returns:
            Average energy (0-1)
        """
        energies = []
        durations = []
        
        for segment in music_segments:
            energy = segment.get("energy", 0)
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            duration = max(0, end - start)
            
            if energy >= 0 and duration > 0:
                energies.append(energy)
                durations.append(duration)
        
        if not energies:
            return 0.0
        
        # Weighted average by duration
        return sum(e * d for e, d in zip(energies, durations)) / sum(durations)


class VideoFeatureExtractor(FeatureExtractor):
    """Extract features from video content."""
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from video analysis.
        
        Args:
            data: Preprocessed data containing video analysis
            
        Returns:
            Dictionary of extracted features
        """
        results = {}
        
        video_features = data.get("video_features", {})
        if not video_features:
            logger.warning("No video features to extract from")
            return {"video_features": {}}
        
        # Extract resolution quality
        resolution = video_features.get("resolution", "")
        if resolution:
            width, height = self._parse_resolution(resolution)
            results["resolution_quality"] = self._calculate_resolution_quality(width, height)
        
        # Extract frame rate quality
        fps = video_features.get("fps", 0)
        results["frame_rate_quality"] = self._calculate_fps_quality(fps)
        
        # Extract scene transition features
        scene_transitions = video_features.get("scene_transitions", [])
        if scene_transitions:
            results["scene_transition_frequency"] = len(scene_transitions) / max(1, video_features.get("duration", 60))
            results["scene_pacing"] = self._calculate_scene_pacing(scene_transitions)
        
        # Extract visual complexity
        visual_complexity = video_features.get("visual_complexity", {})
        if visual_complexity:
            results["spatial_complexity"] = visual_complexity.get("spatial_complexity", 0)
            results["temporal_complexity"] = visual_complexity.get("temporal_complexity", 0)
            results["information_density"] = visual_complexity.get("information_density", 0)
        
        # Extract motion intensity
        motion_intensity = video_features.get("motion_intensity", {})
        if motion_intensity:
            results["motion_intensity_avg"] = motion_intensity.get("motion_intensity_avg", 0)
            results["motion_consistency"] = motion_intensity.get("motion_consistency", 0)
            results["camera_stability"] = motion_intensity.get("camera_stability", 0)
        
        # Extract color scheme
        color_scheme = video_features.get("color_scheme", {})
        if color_scheme:
            results["color_diversity"] = color_scheme.get("color_diversity", 0)
            results["saturation_avg"] = color_scheme.get("saturation_avg", 0)
            results["brightness_avg"] = color_scheme.get("brightness_avg", 0)
        
        # Extract production quality
        results["production_quality"] = video_features.get("production_quality", 0)
        
        # Extract thumbnail features
        thumbnail_data = video_features.get("thumbnail_data", {})
        if thumbnail_data:
            results["thumbnail_face_presence"] = thumbnail_data.get("face_presence", 0)
            results["thumbnail_visual_salience"] = thumbnail_data.get("visual_salience", 0)
            results["thumbnail_click_prediction"] = thumbnail_data.get("click_prediction", 0)
        
        return {"video_features": results}
    
    def _parse_resolution(self, resolution: str) -> tuple:
        """
        Parse resolution string into width and height.
        
        Args:
            resolution: Resolution string (e.g., "1920x1080")
            
        Returns:
            Tuple of (width, height)
        """
        try:
            width, height = resolution.lower().split('x')
            return int(width), int(height)
        except (ValueError, AttributeError):
            return 0, 0
    
    def _calculate_resolution_quality(self, width: int, height: int) -> float:
        """
        Calculate resolution quality score.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            
        Returns:
            Resolution quality score (0-1)
        """
        if width <= 0 or height <= 0:
            return 0.0
        
        # Calculate pixel count
        pixel_count = width * height
        
        # Normalize based on common resolutions
        if pixel_count >= 8294400:  # 4K (3840x2160)
            return 1.0
        elif pixel_count >= 2073600:  # Full HD (1920x1080)
            return 0.8
        elif pixel_count >= 921600:  # HD (1280x720)
            return 0.6
        elif pixel_count >= 409920:  # SD (854x480)
            return 0.4
        else:
            return 0.2
    
    def _calculate_fps_quality(self, fps: float) -> float:
        """
        Calculate frame rate quality score.
        
        Args:
            fps: Frames per second
            
        Returns:
            Frame rate quality score (0-1)
        """
        if fps <= 0:
            return 0.0
        
        # Normalize based on common frame rates
        if fps >= 60:
            return 1.0
        elif fps >= 30:
            return 0.8
        elif fps >= 24:
            return 0.6
        elif fps >= 15:
            return 0.3
        else:
            return 0.1
    
    def _calculate_scene_pacing(self, scene_transitions: List[float]) -> float:
        """
        Calculate scene pacing score based on transition timing.
        
        Args:
            scene_transitions: List of timestamps for scene transitions
            
        Returns:
            Scene pacing score (0-1)
        """
        if len(scene_transitions) < 2:
            return 0.5  # Neutral pacing for few transitions
        
        # Calculate intervals between transitions
        intervals = [scene_transitions[i] - scene_transitions[i-1] 
                    for i in range(1, len(scene_transitions))]
        
        # Calculate mean and standard deviation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Calculate coefficient of variation (normalized std dev)
        if mean_interval > 0:
            cv = std_interval / mean_interval
        else:
            cv = 0
        
        # Consistency score (inverse of variation)
        consistency = 1.0 / (1.0 + cv)
        
        # Pace score based on mean interval (shorter is faster)
        # 2-10 seconds is typical range
        if mean_interval < 2:
            pace = 1.0  # Very fast
        elif mean_interval < 5:
            pace = 0.8  # Fast
        elif mean_interval < 10:
            pace = 0.6  # Moderate
        elif mean_interval < 20:
            pace = 0.4  # Slow
        else:
            pace = 0.2  # Very slow
        
        # Combined score with higher weight on consistency
        return 0.4 * pace + 0.6 * consistency


class EngagementFeatureExtractor:
    """Extract engagement features from all available data."""
    
    def __init__(self):
        """Initialize the engagement feature extractor."""
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.video_extractor = VideoFeatureExtractor()
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all engagement-related features from preprocessed data.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Dictionary of extracted features
        """
        all_features = {}
        
        # Add metadata
        all_features["content_id"] = data.get("content_id", "unknown")
        all_features["platform"] = data.get("platform", "unknown")
        all_features["content_type"] = data.get("content_type", "unknown")
        all_features["timestamp"] = data.get("timestamp", None)
        
        # Extract metrics features
        metrics = data.get("metrics", {})
        if metrics:
            all_features["metric_features"] = metrics
        
        # Extract content-specific features based on content type
        content_type = data.get("content_type", "").lower()
        
        # Text features (can be present in any content type)
        if "cleaned_text" in data:
            text_features = self.text_extractor.extract(data)
            all_features.update(text_features)
        
        # Audio features
        if "audio_features" in data:
            audio_features = self.audio_extractor.extract(data)
            all_features.update(audio_features)
        
        # Video features
        if "video_features" in data:
            video_features = self.video_extractor.extract(data)
            all_features.update(video_features)
        
        # Temporal features
        if "temporal_segments" in data:
            temporal_features = self._extract_temporal_features(data)
            all_features["temporal_features"] = temporal_features
        
        # Derive engagement dimensions
        all_features["engagement_dimensions"] = self._calculate_engagement_dimensions(all_features)
        
        return all_features
    
    def _extract_temporal_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features related to temporal patterns.
        
        Args:
            data: Preprocessed data with temporal segments
            
        Returns:
            Temporal features dictionary
        """
        results = {}
        
        # Use pre-calculated temporal statistics if available
        if "temporal_statistics" in data:
            results.update(data["temporal_statistics"])
        
        # Analyze temporal segments for patterns
        segments = data.get("temporal_segments", [])
        if segments:
            # Convert to numpy array for analysis
            timestamps = [segment.get("timestamp") for segment in segments]
            values = [segment.get("value", 0) for segment in segments]
            
            if len(values) >= 3:
                # Detect trend
                results["trend"] = self._detect_trend(values)
                
                # Detect pattern type
                results["pattern_type"] = self._detect_pattern_type(values)
                
                # Calculate peak-to-average ratio
                avg_value = np.mean(values)
                if avg_value > 0:
                    peak_value = np.max(values)
                    results["peak_to_avg_ratio"] = peak_value / avg_value
                else:
                    results["peak_to_avg_ratio"] = 1.0
        
        return results
    
    def _detect_trend(self, values: List[float]) -> str:
        """
        Detect overall trend in time series data.
        
        Args:
            values: List of values in temporal sequence
            
        Returns:
            Trend type: "increasing", "decreasing", "stable", "fluctuating"
        """
        if len(values) < 3:
            return "stable"
        
        # Simple linear regression for trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Calculate residuals to measure fluctuation
        _, residuals, _, _, _ = np.polyfit(x, values, 1, full=True)
        
        # Determine trend based on slope and residuals
        avg_value = np.mean(values)
        if avg_value == 0:
            return "stable"
            
        relative_residual = np.sqrt(residuals[0]) / len(values) / avg_value if len(residuals) > 0 else 0
        
        if relative_residual > 0.2:
            return "fluctuating"
        elif slope > 0.05 * avg_value:
            return "increasing"
        elif slope < -0.05 * avg_value:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_pattern_type(self, values: List[float]) -> str:
        """
        Detect the pattern type in time series data.
        
        Args:
            values: List of values in temporal sequence
            
        Returns:
            Pattern type: "sustained", "early_peak", "late_peak", "multi_peak", "u_shaped"
        """
        if len(values) < 5:
            return "sustained"
        
        # Normalize values
        min_val = np.min(values)
        range_val = np.max(values) - min_val
        if range_val == 0:
            return "sustained"
            
        normalized = [(v - min_val) / range_val for v in values]
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(normalized) - 1):
            if normalized[i] > normalized[i-1] and normalized[i] > normalized[i+1]:
                # Only count significant peaks (>0.6)
                if normalized[i] > 0.6:
                    peaks.append(i)
        
        # Check pattern types
        if len(peaks) == 0:
            # No clear peaks, check if sustained
            if np.std(normalized) < 0.2:
                return "sustained"
            
            # Check for U-shape
            mid_point = len(normalized) // 2
            first_quarter = np.mean(normalized[:mid_point//2])
            middle = np.mean(normalized[mid_point//2:mid_point + mid_point//2])
            last_quarter = np.mean(normalized[mid_point + mid_point//2:])
            
            if middle < first_quarter * 0.7 and middle < last_quarter * 0.7:
                return "u_shaped"
            else:
                return "sustained"
                
        elif len(peaks) == 1:
            # Single peak
            peak_pos = peaks[0] / len(normalized)
            
            if peak_pos < 0.3:
                return "early_peak"
            elif peak_pos > 0.7:
                return "late_peak"
            else:
                return "mid_peak"
        else:
            return "multi_peak"
    
    def _calculate_engagement_dimensions(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate engagement dimensions based on features.
        
        Implements calculations based on User Engagement Scale frameworks
        and adapts them to social media content contexts.
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary mapping engagement dimensions to scores
        """
        dimensions = {}
        
        # 1. Focused Attention
        focused_attention_features = []
        
        # Add video features related to attention if available
        video_features = features.get("video_features", {})
        if video_features:
            focused_attention_features.extend([
                ("scene_transition_frequency", video_features.get("scene_transition_frequency", 0.5), 0.2),
                ("motion_intensity_avg", video_features.get("motion_intensity_avg", 0.5), 0.15),
                ("visual_complexity", video_features.get("spatial_complexity", 0.5), 0.15)
            ])
        
        # Add audio features related to attention if available
        audio_features = features.get("audio_features", {})
        if audio_features:
            focused_attention_features.extend([
                ("speech_coverage", audio_features.get("speech_coverage", 0.5), 0.1),
                ("volume_consistency", audio_features.get("volume_consistency", 0.5), 0.05)
            ])
        
        # Add text features related to attention if available
        text_features = features.get("text_features", {})
        if text_features:
            if "readability_score" in text_features:
                # Transform readability to 0-1 (higher is more engaging)
                readability = text_features["readability_score"] / 100
                # Readability sweet spot around 60-80 is most engaging
                readability_engagement = 1.0 - abs(readability - 0.7) * 2.5
                readability_engagement = max(0, min(1, readability_engagement))
                
                focused_attention_features.append(
                    ("readability", readability_engagement, 0.1)
                )
        
        # Add temporal features if available
        temporal_features = features.get("temporal_features", {})
        if temporal_features:
            pattern_type = temporal_features.get("pattern_type", "sustained")
            pattern_score = {
                "sustained": 0.8,
                "early_peak": 0.6,
                "mid_peak": 0.7,
                "late_peak": 0.5,
                "multi_peak": 0.6,
                "u_shaped": 0.4
            }.get(pattern_type, 0.5)
            
            focused_attention_features.append(
                ("temporal_pattern", pattern_score, 0.25)
            )
        
        # Calculate weighted average for focused attention
        dimensions[EngagementDimension.FOCUSED_ATTENTION] = self._weighted_average(focused_attention_features)
        
        # 2. Emotional Response
        emotional_response_features = []
        
        # Add text sentiment if available
        if text_features and "sentiment_features" in text_features:
            sentiment = text_features["sentiment_features"]
            emotional_response_features.extend([
                ("sentiment_valence", sentiment.get("valence", 0.5), 0.2),
                ("emotional_intensity", sentiment.get("emotional_intensity", 0.5), 0.2)
            ])
        
        # Add audio emotional tone if available
        if audio_features:
            emotional_response_features.extend([
                ("voice_enthusiasm", audio_features.get("enthusiasm", 0.5), 0.15),
                ("emotional_valence", audio_features.get("emotional_valence", 0.5), 0.15),
                ("emotional_arousal", audio_features.get("emotional_arousal", 0.5), 0.15)
            ])
        
        # Add video color features if available
        if video_features:
            emotional_response_features.extend([
                ("color_diversity", video_features.get("color_diversity", 0.5), 0.05),
                ("saturation_avg", video_features.get("saturation_avg", 0.5), 0.05),
                ("brightness_avg", video_features.get("brightness_avg", 0.5), 0.05)
            ])
        
        dimensions[EngagementDimension.EMOTIONAL_RESPONSE] = self._weighted_average(emotional_response_features)
        
        # 3. Aesthetic Appeal
        aesthetic_appeal_features = []
        
        # Add video production quality if available
        if video_features:
            aesthetic_appeal_features.extend([
                ("production_quality", video_features.get("production_quality", 0.5), 0.4),
                ("resolution_quality", video_features.get("resolution_quality", 0.5), 0.2),
                ("frame_rate_quality", video_features.get("frame_rate_quality", 0.5), 0.1),
                ("camera_stability", video_features.get("camera_stability", 0.5), 0.1)
            ])
        
        # Add audio quality if available
        if audio_features:
            aesthetic_appeal_features.append(
                ("audio_quality", audio_features.get("audio_quality", 0.5), 0.2)
            )
        
        dimensions[EngagementDimension.AESTHETIC_APPEAL] = self._weighted_average(aesthetic_appeal_features)
        
        # 4. Utility
        # This is typically based on direct user feedback
        # We can approximate from available signals
        utility_features = []
        
        # Add metrics related to utility if available
        metric_features = features.get("metric_features", {})
        if metric_features:
            utility_features.extend([
                ("engagement_rate", metric_features.get("engagement_rate", 0.5), 0.4),
                ("retention_rate", metric_features.get("retention_rate", 0.5), 0.6)
            ])
        
        dimensions[EngagementDimension.UTILITY] = self._weighted_average(utility_features, default=0.5)
        
        # Combine all dimensions into overall engagement score
        dimensions["overall_engagement"] = (
            dimensions.get(EngagementDimension.FOCUSED_ATTENTION, 0) * 0.3 +
            dimensions.get(EngagementDimension.EMOTIONAL_RESPONSE, 0) * 0.3 +
            dimensions.get(EngagementDimension.AESTHETIC_APPEAL, 0) * 0.2 + 
            dimensions.get(EngagementDimension.UTILITY, 0) * 0.2
        )
        
        return dimensions
    
    def _weighted_average(self, features: List[tuple], default: float = 0.5) -> float:
        """
        Calculate weighted average of features.
        
        Args:
            features: List of (name, value, weight) tuples
            default: Default value to return if no features
            
        Returns:
            Weighted average
        """
        if not features:
            return default
            
        total_weight = sum(weight for _, _, weight in features)
        if total_weight == 0:
            return default
            
        weighted_sum = sum(value * weight for _, value, weight in features)
        return weighted_sum / total_weight 