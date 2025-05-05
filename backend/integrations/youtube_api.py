"""
YouTube API Integration

This module provides functionality for retrieving video data from the YouTube API,
including video metadata, statistics, captions, and comments.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.discovery import Resource
from pydantic import BaseModel, ValidationError

from ..models.content import Content, ContentMetadata, ContentType, Platform


class YouTubeApiConfig(BaseModel):
    """Configuration for YouTube API access."""
    api_key: str
    max_results: int = 50
    quota_limit_per_day: int = 10000
    current_quota_usage: int = 0


class YouTubeVideoMetadata(BaseModel):
    """Metadata for a YouTube video."""
    video_id: str
    title: str
    description: Optional[str] = None
    channel_id: str
    channel_title: str
    published_at: datetime
    tags: List[str] = []
    category_id: Optional[str] = None
    live_broadcast_content: Optional[str] = None
    default_language: Optional[str] = None
    localized: Dict[str, str] = {}
    default_audio_language: Optional[str] = None


class YouTubeVideoStatistics(BaseModel):
    """Statistics for a YouTube video."""
    view_count: int = 0
    like_count: int = 0
    dislike_count: int = 0
    favorite_count: int = 0
    comment_count: int = 0


class YouTubeVideoComment(BaseModel):
    """A comment on a YouTube video."""
    comment_id: str
    text: str
    author_display_name: str
    author_profile_image_url: Optional[str] = None
    author_channel_id: Optional[str] = None
    like_count: int = 0
    published_at: datetime
    updated_at: Optional[datetime] = None
    reply_count: int = 0


class YouTubeVideoDetail(BaseModel):
    """Comprehensive details about a YouTube video."""
    metadata: YouTubeVideoMetadata
    statistics: YouTubeVideoStatistics
    content_details: Dict[str, Any] = {}
    status: Dict[str, Any] = {}
    player: Dict[str, Any] = {}
    topic_details: Optional[Dict[str, Any]] = None


class YouTubeApiClient:
    """
    Client for interacting with the YouTube Data API.
    
    This class provides methods for retrieving video data, including
    metadata, statistics, captions, and comments.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the YouTube API client.
        
        Args:
            api_key: API key for YouTube Data API. If not provided,
                    will attempt to read from YOUTUBE_API_KEY environment variable.
        """
        self.logger = logging.getLogger("integrations.youtube")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        if not self.api_key:
            self.logger.warning("YouTube API key not provided. API calls will fail.")
        
        self.config = YouTubeApiConfig(
            api_key=self.api_key or "invalid",
            max_results=50,
            quota_limit_per_day=10000,
            current_quota_usage=0
        )
        
        # Initialize API client
        self.youtube = None
        if self.api_key:
            try:
                api_service_name = "youtube"
                api_version = "v3"
                self.youtube = googleapiclient.discovery.build(
                    api_service_name, api_version, developerKey=self.api_key
                )
                self.logger.info("YouTube API client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize YouTube API client: {e}")
    
    def get_video_details(self, video_id: str) -> Optional[YouTubeVideoDetail]:
        """
        Get comprehensive details about a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            YouTubeVideoDetail object or None if retrieval fails
        """
        if not self.youtube:
            self.logger.error("YouTube API client not initialized")
            return None
        
        try:
            # Request video details with all parts
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics,status,player,topicDetails",
                id=video_id
            )
            response = request.execute()
            self.config.current_quota_usage += 1  # Simple quota tracking
            
            # Check if video exists
            if not response.get("items"):
                self.logger.warning(f"Video not found: {video_id}")
                return None
            
            video_data = response["items"][0]
            
            # Extract metadata from snippet
            snippet = video_data.get("snippet", {})
            metadata = YouTubeVideoMetadata(
                video_id=video_id,
                title=snippet.get("title", "Unknown Title"),
                description=snippet.get("description"),
                channel_id=snippet.get("channelId", ""),
                channel_title=snippet.get("channelTitle", "Unknown Channel"),
                published_at=datetime.fromisoformat(snippet.get("publishedAt", "").replace("Z", "+00:00")),
                tags=snippet.get("tags", []),
                category_id=snippet.get("categoryId"),
                live_broadcast_content=snippet.get("liveBroadcastContent"),
                default_language=snippet.get("defaultLanguage"),
                localized=snippet.get("localized", {}),
                default_audio_language=snippet.get("defaultAudioLanguage")
            )
            
            # Extract statistics
            stats = video_data.get("statistics", {})
            statistics = YouTubeVideoStatistics(
                view_count=int(stats.get("viewCount", 0)),
                like_count=int(stats.get("likeCount", 0)),
                dislike_count=int(stats.get("dislikeCount", 0)),
                favorite_count=int(stats.get("favoriteCount", 0)),
                comment_count=int(stats.get("commentCount", 0))
            )
            
            # Create full video detail object
            video_detail = YouTubeVideoDetail(
                metadata=metadata,
                statistics=statistics,
                content_details=video_data.get("contentDetails", {}),
                status=video_data.get("status", {}),
                player=video_data.get("player", {}),
                topic_details=video_data.get("topicDetails")
            )
            
            return video_detail
            
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"YouTube API error: {e}")
            return None
        except ValidationError as e:
            self.logger.error(f"Data validation error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
    
    def get_video_comments(
        self, video_id: str, max_results: int = 100, 
        include_replies: bool = False
    ) -> List[YouTubeVideoComment]:
        """
        Get comments for a video.
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to retrieve
            include_replies: Whether to include replies to comments
            
        Returns:
            List of YouTubeVideoComment objects
        """
        if not self.youtube:
            self.logger.error("YouTube API client not initialized")
            return []
        
        try:
            comments = []
            next_page_token = None
            
            while len(comments) < max_results:
                # Request comments
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token,
                    order="relevance"
                )
                response = request.execute()
                self.config.current_quota_usage += 1  # Simple quota tracking
                
                # Process comment threads
                for item in response.get("items", []):
                    comment_data = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                    
                    if comment_data:
                        try:
                            published_at = datetime.fromisoformat(
                                comment_data.get("publishedAt", "").replace("Z", "+00:00")
                            )
                            updated_at = None
                            if comment_data.get("updatedAt"):
                                updated_at = datetime.fromisoformat(
                                    comment_data.get("updatedAt", "").replace("Z", "+00:00")
                                )
                            
                            comment = YouTubeVideoComment(
                                comment_id=item.get("id", ""),
                                text=comment_data.get("textDisplay", ""),
                                author_display_name=comment_data.get("authorDisplayName", ""),
                                author_profile_image_url=comment_data.get("authorProfileImageUrl"),
                                author_channel_id=comment_data.get("authorChannelId", {}).get("value"),
                                like_count=int(comment_data.get("likeCount", 0)),
                                published_at=published_at,
                                updated_at=updated_at,
                                reply_count=item.get("snippet", {}).get("totalReplyCount", 0)
                            )
                            comments.append(comment)
                        except ValidationError as e:
                            self.logger.warning(f"Invalid comment data: {e}")
                    
                    # If we're including replies and this comment has replies, fetch them
                    if (include_replies and 
                        item.get("snippet", {}).get("totalReplyCount", 0) > 0 and
                        len(comments) < max_results):
                        
                        replies = self._get_comment_replies(item.get("id", ""), max_results - len(comments))
                        comments.extend(replies)
                
                # Check if there are more pages
                next_page_token = response.get("nextPageToken")
                if not next_page_token or len(comments) >= max_results:
                    break
            
            return comments[:max_results]
            
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"YouTube API error when fetching comments: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error when fetching comments: {e}")
            return []
    
    def _get_comment_replies(self, comment_thread_id: str, max_results: int = 20) -> List[YouTubeVideoComment]:
        """
        Get replies to a specific comment thread.
        
        Args:
            comment_thread_id: ID of the comment thread
            max_results: Maximum number of replies to retrieve
            
        Returns:
            List of YouTubeVideoComment objects
        """
        if not self.youtube:
            return []
        
        try:
            replies = []
            next_page_token = None
            
            while len(replies) < max_results:
                # Request replies
                request = self.youtube.comments().list(
                    part="snippet",
                    parentId=comment_thread_id,
                    maxResults=min(100, max_results - len(replies)),
                    pageToken=next_page_token
                )
                response = request.execute()
                self.config.current_quota_usage += 1  # Simple quota tracking
                
                # Process replies
                for item in response.get("items", []):
                    reply_data = item.get("snippet", {})
                    
                    if reply_data:
                        try:
                            published_at = datetime.fromisoformat(
                                reply_data.get("publishedAt", "").replace("Z", "+00:00")
                            )
                            updated_at = None
                            if reply_data.get("updatedAt"):
                                updated_at = datetime.fromisoformat(
                                    reply_data.get("updatedAt", "").replace("Z", "+00:00")
                                )
                            
                            reply = YouTubeVideoComment(
                                comment_id=item.get("id", ""),
                                text=reply_data.get("textDisplay", ""),
                                author_display_name=reply_data.get("authorDisplayName", ""),
                                author_profile_image_url=reply_data.get("authorProfileImageUrl"),
                                author_channel_id=reply_data.get("authorChannelId", {}).get("value"),
                                like_count=int(reply_data.get("likeCount", 0)),
                                published_at=published_at,
                                updated_at=updated_at,
                                reply_count=0  # Replies don't have replies
                            )
                            replies.append(reply)
                        except ValidationError as e:
                            self.logger.warning(f"Invalid reply data: {e}")
                
                # Check if there are more pages
                next_page_token = response.get("nextPageToken")
                if not next_page_token or len(replies) >= max_results:
                    break
            
            return replies[:max_results]
            
        except googleapiclient.errors.HttpError as e:
            self.logger.error(f"YouTube API error when fetching replies: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error when fetching replies: {e}")
            return []
    
    def extract_video_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract the video ID from a YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID or None if not a valid YouTube URL
        """
        import re
        
        # Regex patterns for different YouTube URL formats
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})",  # Standard and short URLs
            r"youtube\.com\/embed\/([a-zA-Z0-9_-]{11})",  # Embedded URLs
            r"youtube\.com\/v\/([a-zA-Z0-9_-]{11})"       # Old-style URLs
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def to_content_model(self, video_id: str, include_comments: bool = True) -> Optional[Content]:
        """
        Convert YouTube video data to a Content model.
        
        Args:
            video_id: YouTube video ID
            include_comments: Whether to include comments data
            
        Returns:
            Content model or None if retrieval fails
        """
        # Get video details
        video_detail = self.get_video_details(video_id)
        if not video_detail:
            return None
        
        # Get comments if requested
        comments = []
        if include_comments:
            comments = self.get_video_comments(video_id, max_results=100)
        
        try:
            # Create Content model
            content = Content(
                id=f"yt_{video_id}",
                content_type=ContentType.VIDEO,
                metadata=ContentMetadata(
                    title=video_detail.metadata.title,
                    description=video_detail.metadata.description,
                    creator_id=video_detail.metadata.channel_id,
                    creator_name=video_detail.metadata.channel_title,
                    platform=Platform.YOUTUBE,
                    published_at=video_detail.metadata.published_at,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    tags=video_detail.metadata.tags,
                    language=video_detail.metadata.default_language or "en",
                    # Extract duration in seconds from content details
                    duration_seconds=self._parse_iso_duration(
                        video_detail.content_details.get("duration", "PT0S")
                    )
                ),
                # We don't set features here as they're generated by analysis agents
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return content
            
        except ValidationError as e:
            self.logger.error(f"Error creating Content model: {e}")
            return None
    
    def _parse_iso_duration(self, iso_duration: str) -> int:
        """
        Parse ISO 8601 duration format to seconds.
        
        Args:
            iso_duration: Duration string in ISO 8601 format (e.g., PT1H30M15S)
            
        Returns:
            Duration in seconds
        """
        import re
        import isodate
        
        try:
            duration = isodate.parse_duration(iso_duration)
            return int(duration.total_seconds())
        except:
            # Fallback parser for simple cases
            seconds = 0
            hours_match = re.search(r'(\d+)H', iso_duration)
            minutes_match = re.search(r'(\d+)M', iso_duration)
            seconds_match = re.search(r'(\d+)S', iso_duration)
            
            if hours_match:
                seconds += int(hours_match.group(1)) * 3600
            if minutes_match:
                seconds += int(minutes_match.group(1)) * 60
            if seconds_match:
                seconds += int(seconds_match.group(1))
                
            return seconds 