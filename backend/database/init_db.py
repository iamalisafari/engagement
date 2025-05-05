"""
Database Initialization Script

This script initializes the database and optionally adds sample data
for development and testing purposes.
"""

import logging
import sys
from datetime import datetime, timedelta
import random
import hashlib

from sqlalchemy.exc import SQLAlchemyError

from . import config
from ..models.database import (
    Tag,
    Creator,
    Content,
    VideoFeatures,
    AudioFeatures,
    TextFeatures,
    EngagementMetrics,
    AnalysisJob
)
from ..models.database_extensions import (
    User,
    AnalysisPreset,
    TimeSeriesData,
    AnalysisResult,
    APIKey,
    ContentCollection
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_init")


def init_db(add_sample_data: bool = False) -> None:
    """
    Initialize the database by creating all tables.
    
    Args:
        add_sample_data: Whether to add sample data after initialization
    """
    try:
        logger.info("Creating database tables...")
        config.init_db()
        logger.info("Database tables created successfully.")
        
        if add_sample_data:
            logger.info("Adding sample data...")
            add_sample_data_to_db()
            logger.info("Sample data added successfully.")
            
        logger.info("Database initialization complete.")
        
    except SQLAlchemyError as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


def add_sample_data_to_db() -> None:
    """Add sample data to the database for development and testing."""
    # Use a context manager for the database session
    with config.db_session() as db:
        # Create sample tags
        tags = [
            Tag(name="educational"),
            Tag(name="technology"),
            Tag(name="tutorial"),
            Tag(name="research"),
            Tag(name="engagement"),
            Tag(name="social media")
        ]
        db.add_all(tags)
        
        # Create sample creators
        creators = [
            Creator(
                id="youtube_creator_1",
                name="Academic Research Channel",
                platform="YOUTUBE",
                url="https://www.youtube.com/c/academicresearch"
            ),
            Creator(
                id="reddit_creator_1",
                name="ResearchPosts",
                platform="REDDIT",
                url="https://www.reddit.com/user/researchposts"
            )
        ]
        db.add_all(creators)
        
        # Commit to get tag and creator IDs
        db.commit()
        
        # Create sample content
        now = datetime.now()
        
        content_items = [
            Content(
                id="yt_12345abcde",
                content_type="VIDEO",
                title="Understanding User Engagement in Social Media",
                description="This video explores the factors affecting user engagement...",
                creator_id="youtube_creator_1",
                platform="YOUTUBE",
                published_at=now - timedelta(days=30),
                url="https://www.youtube.com/watch?v=12345abcde",
                category="Education",
                language="en",
                duration_seconds=600
            ),
            Content(
                id="yt_67890fghij",
                content_type="VIDEO",
                title="Media Richness Theory Explained",
                description="An explanation of media richness theory and its applications...",
                creator_id="youtube_creator_1",
                platform="YOUTUBE",
                published_at=now - timedelta(days=15),
                url="https://www.youtube.com/watch?v=67890fghij",
                category="Education",
                language="en",
                duration_seconds=540
            ),
            Content(
                id="rd_12345abcde",
                content_type="TEXT",
                title="Research on Social Media Engagement Patterns",
                description="A discussion of recent findings in engagement research...",
                creator_id="reddit_creator_1",
                platform="REDDIT",
                published_at=now - timedelta(days=7),
                url="https://www.reddit.com/r/datascience/comments/12345abcde",
                category="Research",
                language="en"
            )
        ]
        db.add_all(content_items)
        
        # Add tags to content
        content_items[0].tags.extend([tags[0], tags[3], tags[4], tags[5]])
        content_items[1].tags.extend([tags[0], tags[1], tags[2]])
        content_items[2].tags.extend([tags[3], tags[4], tags[5]])
        
        # Commit to get content IDs
        db.commit()
        
        # Add video features
        video_features = [
            VideoFeatures(
                content_id="yt_12345abcde",
                resolution="1080p",
                fps=30.0,
                scene_transitions=[10.5, 25.2, 42.8, 60.1, 75.3, 90.6],
                visual_complexity={
                    "spatial_complexity": 0.72,
                    "temporal_complexity": 0.65,
                    "information_density": 0.68,
                    "edge_density": 0.54,
                    "object_count_avg": 4.3
                },
                motion_intensity={
                    "motion_intensity_avg": 0.45,
                    "motion_consistency": 0.78,
                    "camera_stability": 0.92,
                    "motion_segments": 5,
                    "dynamic_range": 0.65
                },
                color_scheme={
                    "color_diversity": 0.68,
                    "color_harmony": 0.75,
                    "saturation_avg": 0.62,
                    "brightness_avg": 0.58,
                    "contrast_avg": 0.71
                },
                production_quality=0.85,
                thumbnail_data={
                    "visual_salience": 0.82,
                    "text_presence": 0.90,
                    "face_presence": 1.0,
                    "emotion_intensity": 0.75,
                    "color_contrast": 0.68,
                    "click_prediction": 0.72
                }
            ),
            VideoFeatures(
                content_id="yt_67890fghij",
                resolution="1080p",
                fps=30.0,
                scene_transitions=[8.2, 22.5, 35.8, 52.3, 68.9, 82.1],
                visual_complexity={
                    "spatial_complexity": 0.65,
                    "temporal_complexity": 0.58,
                    "information_density": 0.72,
                    "edge_density": 0.48,
                    "object_count_avg": 3.8
                },
                motion_intensity={
                    "motion_intensity_avg": 0.38,
                    "motion_consistency": 0.82,
                    "camera_stability": 0.95,
                    "motion_segments": 4,
                    "dynamic_range": 0.58
                },
                color_scheme={
                    "color_diversity": 0.62,
                    "color_harmony": 0.78,
                    "saturation_avg": 0.55,
                    "brightness_avg": 0.62,
                    "contrast_avg": 0.68
                },
                production_quality=0.82,
                thumbnail_data={
                    "visual_salience": 0.78,
                    "text_presence": 0.85,
                    "face_presence": 1.0,
                    "emotion_intensity": 0.65,
                    "color_contrast": 0.72,
                    "click_prediction": 0.68
                }
            )
        ]
        db.add_all(video_features)
        
        # Add audio features
        audio_features = [
            AudioFeatures(
                content_id="yt_12345abcde",
                sample_rate=44100,
                bit_depth=16,
                speech_segments=[
                    {
                        "start": 0.0,
                        "end": 15.7,
                        "confidence": 0.92,
                        "speaker_id": "speaker_1",
                        "is_music_overlay": False
                    },
                    {
                        "start": 20.3,
                        "end": 45.8,
                        "confidence": 0.88,
                        "speaker_id": "speaker_1",
                        "is_music_overlay": True
                    },
                    {
                        "start": 50.2,
                        "end": 75.5,
                        "confidence": 0.95,
                        "speaker_id": "speaker_2",
                        "is_music_overlay": False
                    }
                ],
                music_segments=[
                    {
                        "start": 0.0,
                        "end": 10.0,
                        "confidence": 0.85,
                        "genre": "electronic",
                        "tempo": 120,
                        "energy": 0.72,
                        "is_background": True
                    },
                    {
                        "start": 20.3,
                        "end": 45.8,
                        "confidence": 0.92,
                        "genre": "electronic",
                        "tempo": 125,
                        "energy": 0.68,
                        "is_background": True
                    }
                ],
                volume_dynamics={
                    "mean_volume": 0.68,
                    "max_volume": 0.92,
                    "min_volume": 0.32,
                    "dynamic_range": 0.60,
                    "volume_consistency": 0.75,
                    "sudden_changes": 4
                },
                voice_characteristics={
                    "pitch_mean": 165.2,
                    "pitch_range": 48.5,
                    "speech_rate": 3.2,
                    "articulation_clarity": 0.82,
                    "voice_consistency": 0.88,
                    "pause_frequency": 0.15,
                    "pause_duration_avg": 0.45
                },
                emotional_tone={
                    "valence": 0.65,
                    "arousal": 0.72,
                    "enthusiasm": 0.68,
                    "confidence": 0.82,
                    "tension": 0.35,
                    "primary_emotion": "interest",
                    "emotion_intensity": 0.74
                },
                audio_quality=0.78
            ),
            AudioFeatures(
                content_id="yt_67890fghij",
                sample_rate=44100,
                bit_depth=16,
                speech_segments=[
                    {
                        "start": 0.0,
                        "end": 18.2,
                        "confidence": 0.95,
                        "speaker_id": "speaker_1",
                        "is_music_overlay": False
                    },
                    {
                        "start": 25.5,
                        "end": 42.8,
                        "confidence": 0.92,
                        "speaker_id": "speaker_1",
                        "is_music_overlay": True
                    }
                ],
                music_segments=[
                    {
                        "start": 0.0,
                        "end": 8.5,
                        "confidence": 0.88,
                        "genre": "ambient",
                        "tempo": 95,
                        "energy": 0.58,
                        "is_background": True
                    }
                ],
                volume_dynamics={
                    "mean_volume": 0.72,
                    "max_volume": 0.88,
                    "min_volume": 0.35,
                    "dynamic_range": 0.53,
                    "volume_consistency": 0.82,
                    "sudden_changes": 2
                },
                voice_characteristics={
                    "pitch_mean": 158.5,
                    "pitch_range": 42.8,
                    "speech_rate": 2.9,
                    "articulation_clarity": 0.85,
                    "voice_consistency": 0.92,
                    "pause_frequency": 0.12,
                    "pause_duration_avg": 0.38
                },
                emotional_tone={
                    "valence": 0.72,
                    "arousal": 0.65,
                    "enthusiasm": 0.78,
                    "confidence": 0.88,
                    "tension": 0.28,
                    "primary_emotion": "interest",
                    "emotion_intensity": 0.68
                },
                audio_quality=0.85
            )
        ]
        db.add_all(audio_features)
        
        # Add text features
        text_features = [
            TextFeatures(
                content_id="yt_12345abcde",
                word_count=845,
                sentiment={
                    "positive": 0.58,
                    "negative": 0.12,
                    "neutral": 0.30,
                    "compound": 0.46,
                    "objectivity": 0.65,
                    "subjectivity": 0.35
                },
                topics=[
                    {
                        "topic": "social media engagement",
                        "relevance": 0.85,
                        "keywords": ["engagement", "social", "media", "interaction", "users"]
                    },
                    {
                        "topic": "content analysis",
                        "relevance": 0.72,
                        "keywords": ["content", "analysis", "metrics", "performance", "quality"]
                    }
                ],
                readability_scores={
                    "flesch_reading_ease": 65.8,
                    "flesch_kincaid_grade": 9.2,
                    "smog_index": 10.1,
                    "coleman_liau_index": 11.3,
                    "automated_readability_index": 9.8,
                    "dale_chall_readability_score": 8.4
                },
                linguistic_complexity=0.62,
                keywords=[
                    {"keyword": "engagement", "relevance": 0.92},
                    {"keyword": "social media", "relevance": 0.88},
                    {"keyword": "analysis", "relevance": 0.78}
                ],
                emotional_content={
                    "joy": 0.45,
                    "trust": 0.62,
                    "fear": 0.12,
                    "surprise": 0.28,
                    "sadness": 0.15,
                    "disgust": 0.08,
                    "anger": 0.05,
                    "anticipation": 0.38,
                    "emotional_intensity": 0.58
                }
            ),
            TextFeatures(
                content_id="yt_67890fghij",
                word_count=725,
                sentiment={
                    "positive": 0.52,
                    "negative": 0.08,
                    "neutral": 0.40,
                    "compound": 0.42,
                    "objectivity": 0.72,
                    "subjectivity": 0.28
                },
                topics=[
                    {
                        "topic": "media richness theory",
                        "relevance": 0.92,
                        "keywords": ["media", "richness", "theory", "communication", "channels"]
                    },
                    {
                        "topic": "information exchange",
                        "relevance": 0.68,
                        "keywords": ["information", "exchange", "data", "understanding", "meaning"]
                    }
                ],
                readability_scores={
                    "flesch_reading_ease": 62.5,
                    "flesch_kincaid_grade": 10.1,
                    "smog_index": 10.8,
                    "coleman_liau_index": 11.6,
                    "automated_readability_index": 10.2,
                    "dale_chall_readability_score": 8.8
                },
                linguistic_complexity=0.68,
                keywords=[
                    {"keyword": "media richness", "relevance": 0.95},
                    {"keyword": "communication", "relevance": 0.85},
                    {"keyword": "theory", "relevance": 0.78}
                ],
                emotional_content={
                    "joy": 0.38,
                    "trust": 0.72,
                    "fear": 0.08,
                    "surprise": 0.22,
                    "sadness": 0.12,
                    "disgust": 0.05,
                    "anger": 0.03,
                    "anticipation": 0.45,
                    "emotional_intensity": 0.48
                }
            ),
            TextFeatures(
                content_id="rd_12345abcde",
                word_count=1250,
                sentiment={
                    "positive": 0.48,
                    "negative": 0.15,
                    "neutral": 0.37,
                    "compound": 0.32,
                    "objectivity": 0.78,
                    "subjectivity": 0.22
                },
                topics=[
                    {
                        "topic": "engagement research",
                        "relevance": 0.88,
                        "keywords": ["engagement", "research", "findings", "metrics", "study"]
                    },
                    {
                        "topic": "social media platforms",
                        "relevance": 0.75,
                        "keywords": ["platforms", "social", "media", "youtube", "reddit"]
                    }
                ],
                readability_scores={
                    "flesch_reading_ease": 48.2,
                    "flesch_kincaid_grade": 12.3,
                    "smog_index": 13.5,
                    "coleman_liau_index": 12.8,
                    "automated_readability_index": 13.1,
                    "dale_chall_readability_score": 10.2
                },
                linguistic_complexity=0.78,
                keywords=[
                    {"keyword": "engagement patterns", "relevance": 0.92},
                    {"keyword": "research findings", "relevance": 0.88},
                    {"keyword": "social platforms", "relevance": 0.82}
                ],
                emotional_content={
                    "joy": 0.28,
                    "trust": 0.65,
                    "fear": 0.15,
                    "surprise": 0.35,
                    "sadness": 0.12,
                    "disgust": 0.06,
                    "anger": 0.04,
                    "anticipation": 0.52,
                    "emotional_intensity": 0.45
                }
            )
        ]
        db.add_all(text_features)
        
        # Add engagement metrics
        engagement_metrics = [
            EngagementMetrics(
                content_id="yt_12345abcde",
                composite_score=0.76,
                dimensions={
                    "aesthetic_appeal": {
                        "value": 0.68,
                        "confidence": 0.85,
                        "contributing_factors": {
                            "visual_quality": 0.75,
                            "color_harmony": 0.62,
                            "production_value": 0.83
                        },
                        "temporal_pattern": "sustained"
                    },
                    "focused_attention": {
                        "value": 0.82,
                        "confidence": 0.95,
                        "contributing_factors": {
                            "scene_transitions": 0.65,
                            "audio_tempo": 0.78,
                            "narrative_coherence": 0.88
                        },
                        "temporal_pattern": "sustained"
                    },
                    "emotional_response": {
                        "value": 0.71,
                        "confidence": 0.92,
                        "contributing_factors": {
                            "emotional_tone": 0.65,
                            "visual_sentiment": 0.78,
                            "narrative_tension": 0.63
                        },
                        "temporal_pattern": "peak_and_valley"
                    }
                },
                platform_specific={
                    "youtube_retention_index": 0.72,
                    "predicted_shareability": 0.65
                },
                temporal_pattern="sustained",
                analysis_version="1.0.3"
            ),
            EngagementMetrics(
                content_id="yt_67890fghij",
                composite_score=0.72,
                dimensions={
                    "aesthetic_appeal": {
                        "value": 0.65,
                        "confidence": 0.82,
                        "contributing_factors": {
                            "visual_quality": 0.72,
                            "color_harmony": 0.65,
                            "production_value": 0.78
                        },
                        "temporal_pattern": "sustained"
                    },
                    "focused_attention": {
                        "value": 0.78,
                        "confidence": 0.92,
                        "contributing_factors": {
                            "scene_transitions": 0.62,
                            "audio_tempo": 0.72,
                            "narrative_coherence": 0.92
                        },
                        "temporal_pattern": "declining"
                    },
                    "emotional_response": {
                        "value": 0.68,
                        "confidence": 0.88,
                        "contributing_factors": {
                            "emotional_tone": 0.72,
                            "visual_sentiment": 0.65,
                            "narrative_tension": 0.58
                        },
                        "temporal_pattern": "sustained"
                    }
                },
                platform_specific={
                    "youtube_retention_index": 0.68,
                    "predicted_shareability": 0.62
                },
                temporal_pattern="declining",
                analysis_version="1.0.3"
            ),
            EngagementMetrics(
                content_id="rd_12345abcde",
                composite_score=0.68,
                dimensions={
                    "aesthetic_appeal": {
                        "value": 0.58,
                        "confidence": 0.75,
                        "contributing_factors": {
                            "formatting_quality": 0.62,
                            "text_structure": 0.65,
                            "visual_elements": 0.48
                        },
                        "temporal_pattern": "not_applicable"
                    },
                    "focused_attention": {
                        "value": 0.72,
                        "confidence": 0.85,
                        "contributing_factors": {
                            "topic_relevance": 0.82,
                            "content_depth": 0.78,
                            "narrative_coherence": 0.65
                        },
                        "temporal_pattern": "not_applicable"
                    },
                    "emotional_response": {
                        "value": 0.62,
                        "confidence": 0.78,
                        "contributing_factors": {
                            "emotional_content": 0.58,
                            "narrative_tension": 0.52,
                            "personal_relevance": 0.78
                        },
                        "temporal_pattern": "not_applicable"
                    }
                },
                platform_specific={
                    "reddit_upvote_ratio": 0.82,
                    "comment_engagement_index": 0.75
                },
                temporal_pattern="not_applicable",
                analysis_version="1.0.3"
            )
        ]
        db.add_all(engagement_metrics)
        
        # Add sample users
        users = [
            User(
                username="researcher1",
                email="researcher1@example.com",
                password_hash=hashlib.sha256("password123".encode()).hexdigest(),
                first_name="John",
                last_name="Researcher",
                is_admin=False
            ),
            User(
                username="admin_user",
                email="admin@example.com",
                password_hash=hashlib.sha256("admin123".encode()).hexdigest(),
                first_name="Admin",
                last_name="User",
                is_admin=True
            )
        ]
        db.add_all(users)
        db.commit()
        
        # Add analysis presets
        presets = [
            AnalysisPreset(
                user_id=1,  # researcher1
                name="YouTube Educational Content",
                description="Preset for analyzing educational video content on YouTube",
                configuration={
                    "dimension_weights": {
                        "focused_attention": 0.35,
                        "emotional_response": 0.25,
                        "aesthetic_appeal": 0.20,
                        "perceived_usability": 0.20
                    },
                    "feature_extractors": {
                        "video": True,
                        "audio": True,
                        "text": True
                    },
                    "platform_specific": {
                        "retention_threshold": 0.65,
                        "min_engagement_rate": 0.02
                    }
                },
                is_default=True
            ),
            AnalysisPreset(
                user_id=1,  # researcher1
                name="Reddit Text Analysis",
                description="Preset for analyzing text posts on Reddit",
                configuration={
                    "dimension_weights": {
                        "focused_attention": 0.30,
                        "emotional_response": 0.30,
                        "perceived_usability": 0.40
                    },
                    "feature_extractors": {
                        "video": False,
                        "audio": False,
                        "text": True
                    },
                    "platform_specific": {
                        "comment_depth": 3,
                        "min_upvote_ratio": 0.7
                    }
                },
                is_default=False
            )
        ]
        db.add_all(presets)
        db.commit()
        
        # Add sample time series data for view counts
        time_series_data = []
        
        # For YouTube video 1 (30 days of hourly data for demonstration)
        base_date = now - timedelta(days=30)
        base_views = 100
        for day in range(30):
            for hour in range(24):
                current_time = base_date + timedelta(days=day, hours=hour)
                # Apply some patterns to the view data
                day_factor = min(1.0, day / 10)  # Gradual increase over first 10 days
                hour_factor = 0.5 + 0.5 * abs(12 - hour) / 12  # More views during midday
                
                # Add some randomness
                random_factor = 0.8 + 0.4 * random.random()
                
                # Calculate views for this hour
                views = int(base_views * day_factor * hour_factor * random_factor)
                
                time_series_data.append(
                    TimeSeriesData(
                        content_id="yt_12345abcde",
                        metric_type="views",
                        timestamp=current_time,
                        value=views,
                        source="youtube_api"
                    )
                )
        
        # Add like data (fewer data points)
        for day in range(30):
            current_time = base_date + timedelta(days=day)
            # Likes are typically a fraction of views
            likes = int(base_views * day / 10 * (0.02 + 0.01 * random.random()))
            
            time_series_data.append(
                TimeSeriesData(
                    content_id="yt_12345abcde",
                    metric_type="likes",
                    timestamp=current_time,
                    value=likes,
                    source="youtube_api"
                )
            )
        
        # Add time series data for the second YouTube video (less data for variety)
        base_date = now - timedelta(days=15)
        base_views = 75
        for day in range(15):
            for hour in range(0, 24, 4):  # Every 4 hours
                current_time = base_date + timedelta(days=day, hours=hour)
                
                # Different pattern - initial spike then decline
                day_factor = 1.0 - min(0.7, day / 10)
                hour_factor = 0.5 + 0.5 * abs(12 - hour) / 12
                
                # Add randomness
                random_factor = 0.8 + 0.4 * random.random()
                
                # Calculate views for this hour
                views = int(base_views * day_factor * hour_factor * random_factor)
                
                time_series_data.append(
                    TimeSeriesData(
                        content_id="yt_67890fghij",
                        metric_type="views",
                        timestamp=current_time,
                        value=views,
                        source="youtube_api"
                    )
                )
        
        db.add_all(time_series_data)
        
        # Add content collection
        collection = ContentCollection(
            user_id=1,  # researcher1
            name="Educational Content Study",
            description="Collection of educational content for engagement research",
            query_parameters={
                "platforms": ["youtube"],
                "categories": ["Education"],
                "min_duration": 300,
                "max_duration": 1200
            }
        )
        db.add(collection)
        db.commit()
        
        # Add content to collection
        collection.content_items.extend([content_items[0], content_items[1]])
        
        # Add API keys
        api_keys = [
            APIKey(
                platform="youtube",
                key_name="YouTube Data API Key",
                key_value="SAMPLE_YT_API_KEY_123456",
                access_level="read",
                rate_limit=100,
                is_active=True
            ),
            APIKey(
                platform="reddit",
                key_name="Reddit API Client",
                key_value="SAMPLE_REDDIT_API_KEY_123456",
                access_level="read",
                rate_limit=60,
                is_active=True
            )
        ]
        db.add_all(api_keys)
        
        # Add analysis results with human feedback
        analysis_results = [
            AnalysisResult(
                content_id="yt_12345abcde",
                user_id=1,  # researcher1
                automated_metrics={
                    "composite_score": 0.76,
                    "focused_attention": 0.82,
                    "emotional_response": 0.71
                },
                human_feedback={
                    "overall_rating": 4.5,
                    "attention_quality": 5,
                    "emotional_impact": 4,
                    "comments": "Excellent content engagement, especially in the middle section."
                },
                confidence_scores={
                    "focused_attention": 0.95,
                    "emotional_response": 0.92,
                    "aesthetic_appeal": 0.85
                },
                final_score=0.78,  # Slightly adjusted based on human feedback
                notes="Human feedback increased the emotional impact score slightly."
            )
        ]
        db.add_all(analysis_results)
        
        # Commit all sample data
        db.commit()


if __name__ == "__main__":
    # Execute as script
    add_sample_data = "--sample-data" in sys.argv
    init_db(add_sample_data) 