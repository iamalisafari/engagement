"""
Data Preprocessing Pipelines

This module implements preprocessing functions for social media data
based on best practices in data science and engagement research methodologies.
It provides standardized pipelines for cleaning, normalizing, and preparing 
data from different social media platforms for analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta

from ..models.content import Content, ContentType, Platform

# Configure logging
logger = logging.getLogger("utils.preprocessing")


def normalize_engagement_metrics(
    metrics: Dict[str, Union[int, float]],
    platform: Platform,
    content_type: ContentType,
    follower_count: Optional[int] = None
) -> Dict[str, float]:
    """
    Normalize raw engagement metrics based on platform-specific characteristics.
    
    This implements normalization techniques from engagement research literature,
    accounting for differences in how metrics should be interpreted across platforms.
    
    Args:
        metrics: Raw engagement metrics (likes, comments, shares, etc.)
        platform: Social media platform
        content_type: Type of content (video, text, etc.)
        follower_count: Creator's follower count (for relative metrics)
        
    Returns:
        Dictionary of normalized metrics
    """
    normalized = {}
    
    # Basic validation
    if not metrics:
        logger.warning("Empty metrics dictionary provided")
        return {}
    
    # Calculate normalization factors based on platform
    if platform == Platform.YOUTUBE:
        # YouTube-specific normalization based on view count
        view_count = metrics.get("views", 0)
        if view_count > 0:
            normalized["engagement_rate"] = (
                (metrics.get("likes", 0) + metrics.get("comments", 0) * 2 + 
                 metrics.get("shares", 0) * 3) / view_count
            )
            normalized["like_rate"] = metrics.get("likes", 0) / view_count
            normalized["comment_rate"] = metrics.get("comments", 0) / view_count
            normalized["share_rate"] = metrics.get("shares", 0) / view_count
            
            # Calculate watch time metrics if available
            if "avg_watch_time" in metrics and "duration" in metrics:
                normalized["retention_rate"] = min(
                    1.0, metrics["avg_watch_time"] / metrics["duration"]
                )
        else:
            logger.warning("YouTube metrics missing view count")
            normalized = {k: 0.0 for k in ["engagement_rate", "like_rate", "comment_rate", "share_rate"]}
    
    elif platform == Platform.REDDIT:
        # Reddit-specific normalization
        post_score = metrics.get("score", 0)  # Net upvotes
        comment_count = metrics.get("comments", 0)
        
        # Basic engagement rate
        normalized["engagement_rate"] = (post_score + comment_count * 2) / max(1, post_score)
        normalized["comment_rate"] = comment_count / max(1, post_score)
        
        # Calculate controversy score (unique to Reddit)
        if "upvote_ratio" in metrics:
            upvote_ratio = metrics["upvote_ratio"]
            total_votes = post_score / max(0.01, (2 * upvote_ratio - 1))
            normalized["controversy_score"] = 1 - abs(2 * upvote_ratio - 1)
    
    elif platform == Platform.TWITTER:
        # Twitter/X-specific normalization
        if follower_count and follower_count > 0:
            normalized["engagement_rate"] = (
                (metrics.get("likes", 0) + metrics.get("replies", 0) * 2 + 
                 metrics.get("retweets", 0) * 3) / follower_count
            )
            normalized["amplification_rate"] = metrics.get("retweets", 0) / follower_count
        else:
            # Fall back to impression-based if available
            impressions = metrics.get("impressions", 0)
            if impressions > 0:
                normalized["engagement_rate"] = (
                    (metrics.get("likes", 0) + metrics.get("replies", 0) * 2 + 
                     metrics.get("retweets", 0) * 3) / impressions
                )
            else:
                logger.warning("Twitter metrics missing follower count and impressions")
                normalized = {k: 0.0 for k in ["engagement_rate", "amplification_rate"]}
    
    # Add platform-agnostic composite metrics
    normalized["raw_engagement_count"] = sum(
        metrics.get(k, 0) for k in ["likes", "comments", "shares", "replies", "retweets"]
    )
    
    # Apply logarithmic transformation for viral content
    if normalized.get("engagement_rate", 0) > 0:
        normalized["log_engagement"] = np.log1p(normalized["engagement_rate"])
    
    return normalized


def clean_text_content(
    text: str,
    platform: Platform,
    remove_urls: bool = True,
    remove_mentions: bool = True
) -> str:
    """
    Clean text content from social media platforms.
    
    Implements cleaning functions tailored to each platform's text 
    format and characteristics.
    
    Args:
        text: Raw text content
        platform: Social media platform
        remove_urls: Whether to remove URLs
        remove_mentions: Whether to remove mentions/tags
        
    Returns:
        Cleaned text
    """
    import re
    
    if not text:
        return ""
    
    # Common cleaning operations
    if remove_urls:
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
    if remove_mentions:
        # Remove @mentions (format varies by platform)
        if platform in [Platform.TWITTER, Platform.INSTAGRAM]:
            text = re.sub(r'@\w+', ' ', text)
        elif platform == Platform.REDDIT:
            text = re.sub(r'u/\w+', ' ', text)
    
    # Platform-specific cleaning
    if platform == Platform.YOUTUBE:
        # Remove time stamps (common in YouTube comments/descriptions)
        text = re.sub(r'\d+:\d+(?::\d+)?', ' ', text)
        
    elif platform == Platform.REDDIT:
        # Remove subreddit references
        text = re.sub(r'r/\w+', ' ', text)
        # Remove markdown formatting
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'[*_~]{1,3}(.*?)[*_~]{1,3}', r'\1', text)  # Emphasis
        
    # Generic cleaning operations
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    return text


def segment_time_series(
    data: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    segment_size: str = "1D",
    aggregation: str = "mean"
) -> pd.DataFrame:
    """
    Segment time series data for temporal analysis.
    
    This function resamples time series data to consistent intervals,
    handles missing values, and prepares it for pattern analysis.
    
    Args:
        data: Time series data with timestamps
        timestamp_col: Column containing timestamps
        value_col: Column containing values to segment
        segment_size: Size of each segment (pandas offset string)
        aggregation: Aggregation method for values in each segment
        
    Returns:
        Segmented time series DataFrame
    """
    # Ensure timestamp column is datetime
    if data[timestamp_col].dtype != 'datetime64[ns]':
        data = data.copy()
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    
    # Set timestamp as index for resampling
    data = data.set_index(timestamp_col)
    
    # Resample based on specified segment size
    if aggregation == "mean":
        resampled = data[value_col].resample(segment_size).mean()
    elif aggregation == "sum":
        resampled = data[value_col].resample(segment_size).sum()
    elif aggregation == "median":
        resampled = data[value_col].resample(segment_size).median()
    elif aggregation == "max":
        resampled = data[value_col].resample(segment_size).max()
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    # Handle missing values using forward fill (then backward if still missing)
    resampled = resampled.fillna(method='ffill').fillna(method='bfill')
    
    # Convert back to DataFrame
    result = resampled.reset_index()
    result.columns = [timestamp_col, value_col]
    
    return result


def preprocess_engagement_data(
    content: Content,
    raw_metrics: Dict[str, Any],
    temporal_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Preprocess engagement data for a content item.
    
    This is the main preprocessing pipeline that applies appropriate
    transformations based on content type and available data.
    
    Args:
        content: Content metadata object
        raw_metrics: Raw engagement metrics
        temporal_data: Temporal engagement data (optional)
        
    Returns:
        Preprocessed data ready for feature extraction
    """
    processed = {
        "content_id": content.id,
        "platform": content.platform.value,
        "content_type": content.content_type.value,
        "timestamp": content.published_at,
    }
    
    # Normalize engagement metrics
    processed["metrics"] = normalize_engagement_metrics(
        metrics=raw_metrics,
        platform=content.platform,
        content_type=content.content_type,
        follower_count=content.creator.follower_count if hasattr(content, "creator") else None
    )
    
    # Clean textual content if available
    if hasattr(content, "text") and content.text:
        processed["cleaned_text"] = clean_text_content(
            text=content.text,
            platform=content.platform
        )
    
    # Process temporal data if available
    if temporal_data is not None and not temporal_data.empty:
        # Segment the time series data
        processed["temporal_segments"] = segment_time_series(
            data=temporal_data,
            timestamp_col="timestamp",
            value_col="engagement_value",
            segment_size="1H"  # Default 1-hour segments
        ).to_dict(orient="records")
        
        # Extract basic temporal statistics
        try:
            temporal_stats = calculate_temporal_statistics(temporal_data)
            processed["temporal_statistics"] = temporal_stats
        except Exception as e:
            logger.error(f"Error calculating temporal statistics: {e}")
    
    return processed


def calculate_temporal_statistics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate basic statistics for temporal engagement data.
    
    Args:
        data: Temporal engagement data
        
    Returns:
        Dictionary of statistical measures
    """
    # Ensure data has engagement_value column
    if "engagement_value" not in data.columns:
        raise ValueError("Data must contain 'engagement_value' column")
    
    values = data["engagement_value"].values
    
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std_dev": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "skewness": float(pd.Series(values).skew()),
        "kurtosis": float(pd.Series(values).kurt()),
    } 