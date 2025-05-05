"""
Reddit API Integration

This module provides functionality for retrieving post data from the Reddit API,
including post metadata, text content, and comments.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import praw
from praw.models import Comment, Submission, Subreddit
from praw.exceptions import PRAWException
from pydantic import BaseModel, ValidationError, HttpUrl

from ..models.content import Content, ContentMetadata, ContentType, Platform


class RedditApiConfig(BaseModel):
    """Configuration for Reddit API access."""
    client_id: str
    client_secret: str
    user_agent: str
    username: Optional[str] = None
    password: Optional[str] = None
    rate_limit_per_minute: int = 60


class RedditPostMetadata(BaseModel):
    """Metadata for a Reddit post."""
    post_id: str
    title: str
    author: str
    subreddit: str
    created_utc: datetime
    permalink: str
    url: HttpUrl
    is_self: bool  # True for text posts, False for link posts
    is_video: bool
    post_hint: Optional[str] = None  # Type of post (link, rich:video, image, etc.)
    post_type: str  # "text", "link", "image", "video", "poll", etc.


class RedditPostStatistics(BaseModel):
    """Statistics for a Reddit post."""
    score: int = 0
    upvote_ratio: float = 0.0
    num_comments: int = 0
    gilded: int = 0
    view_count: Optional[int] = None


class RedditPostComment(BaseModel):
    """A comment on a Reddit post."""
    comment_id: str
    text: str
    author: str
    score: int = 0
    created_utc: datetime
    edited: bool = False
    is_submitter: bool = False
    parent_id: str  # 't3_' prefix for post, 't1_' for comments
    depth: int = 0
    replies_count: int = 0


class RedditPostDetail(BaseModel):
    """Comprehensive details about a Reddit post."""
    metadata: RedditPostMetadata
    statistics: RedditPostStatistics
    text_content: Optional[str] = None
    media_metadata: Optional[Dict[str, Any]] = None
    poll_data: Optional[Dict[str, Any]] = None
    flair: Optional[Dict[str, str]] = None


class RedditApiClient:
    """
    Client for interacting with the Reddit API.
    
    This class provides methods for retrieving post data, including
    metadata, statistics, text content, and comments.
    """
    
    def __init__(
        self, 
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize the Reddit API client.
        
        Args:
            client_id: Client ID for Reddit API. If not provided,
                       will attempt to read from REDDIT_CLIENT_ID environment variable.
            client_secret: Client secret for Reddit API. If not provided,
                          will attempt to read from REDDIT_CLIENT_SECRET environment variable.
            user_agent: User agent for Reddit API. If not provided,
                       will use a default value or REDDIT_USER_AGENT environment variable.
            username: Reddit username for authenticated requests (optional).
            password: Reddit password for authenticated requests (optional).
        """
        self.logger = logging.getLogger("integrations.reddit")
        
        # Get credentials from parameters or environment
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        
        default_user_agent = "python:social-media-engagement-analysis:v0.1.0 (by /u/research_bot)"
        self.user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT", default_user_agent)
        
        self.username = username or os.environ.get("REDDIT_USERNAME")
        self.password = password or os.environ.get("REDDIT_PASSWORD")
        
        if not self.client_id or not self.client_secret:
            self.logger.warning("Reddit API credentials not provided. API calls will fail.")
        
        self.config = RedditApiConfig(
            client_id=self.client_id or "invalid",
            client_secret=self.client_secret or "invalid",
            user_agent=self.user_agent,
            username=self.username,
            password=self.password
        )
        
        # Initialize API client
        self.reddit = None
        if self.client_id and self.client_secret:
            try:
                if self.username and self.password:
                    # Initialize with user authentication
                    self.reddit = praw.Reddit(
                        client_id=self.client_id,
                        client_secret=self.client_secret,
                        user_agent=self.user_agent,
                        username=self.username,
                        password=self.password
                    )
                    self.logger.info("Reddit API client initialized with user authentication")
                else:
                    # Initialize with application-only authentication
                    self.reddit = praw.Reddit(
                        client_id=self.client_id,
                        client_secret=self.client_secret,
                        user_agent=self.user_agent
                    )
                    self.logger.info("Reddit API client initialized with application-only authentication")
            except Exception as e:
                self.logger.error(f"Failed to initialize Reddit API client: {e}")
    
    def get_post_details(self, post_id: str) -> Optional[RedditPostDetail]:
        """
        Get comprehensive details about a post.
        
        Args:
            post_id: Reddit post ID (without the 't3_' prefix)
            
        Returns:
            RedditPostDetail object or None if retrieval fails
        """
        if not self.reddit:
            self.logger.error("Reddit API client not initialized")
            return None
        
        try:
            # Ensure the post_id is in the correct format
            if post_id.startswith('t3_'):
                post_id = post_id[3:]
            
            # Retrieve the submission
            submission = self.reddit.submission(id=post_id)
            
            # Determine post type
            post_type = "text"
            if not submission.is_self:
                if submission.is_video:
                    post_type = "video"
                elif hasattr(submission, "post_hint"):
                    if submission.post_hint == "image":
                        post_type = "image"
                    elif submission.post_hint in ["link", "rich:video"]:
                        post_type = "link"
                elif hasattr(submission, "poll_data"):
                    post_type = "poll"
                else:
                    post_type = "link"
            
            # Create post metadata
            metadata = RedditPostMetadata(
                post_id=submission.id,
                title=submission.title,
                author=submission.author.name if submission.author else "[deleted]",
                subreddit=submission.subreddit.display_name,
                created_utc=datetime.fromtimestamp(submission.created_utc),
                permalink=f"https://www.reddit.com{submission.permalink}",
                url=submission.url,
                is_self=submission.is_self,
                is_video=submission.is_video,
                post_hint=getattr(submission, "post_hint", None),
                post_type=post_type
            )
            
            # Create post statistics
            statistics = RedditPostStatistics(
                score=submission.score,
                upvote_ratio=submission.upvote_ratio,
                num_comments=submission.num_comments,
                gilded=submission.gilded,
                view_count=getattr(submission, "view_count", None)
            )
            
            # Create full post detail object
            post_detail = RedditPostDetail(
                metadata=metadata,
                statistics=statistics,
                text_content=submission.selftext if submission.is_self else None,
                media_metadata=getattr(submission, "media_metadata", None),
                poll_data=getattr(submission, "poll_data", None),
                flair={
                    "text": submission.link_flair_text or "",
                    "css_class": submission.link_flair_css_class or ""
                } if submission.link_flair_text else None
            )
            
            return post_detail
            
        except PRAWException as e:
            self.logger.error(f"Reddit API error: {e}")
            return None
        except ValidationError as e:
            self.logger.error(f"Data validation error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
    
    def get_post_comments(
        self, post_id: str, max_results: int = 100, 
        sort_by: str = "top", include_replies: bool = True,
        max_comment_depth: int = 3
    ) -> List[RedditPostComment]:
        """
        Get comments for a post.
        
        Args:
            post_id: Reddit post ID (without the 't3_' prefix)
            max_results: Maximum number of comments to retrieve
            sort_by: How to sort comments ("top", "best", "new", "controversial", "old")
            include_replies: Whether to include replies to comments
            max_comment_depth: Maximum depth of comment replies to retrieve
            
        Returns:
            List of RedditPostComment objects
        """
        if not self.reddit:
            self.logger.error("Reddit API client not initialized")
            return []
        
        try:
            # Ensure the post_id is in the correct format
            if post_id.startswith('t3_'):
                post_id = post_id[3:]
            
            # Retrieve the submission and sort comments
            submission = self.reddit.submission(id=post_id)
            
            if sort_by == "top":
                submission.comment_sort = "top"
            elif sort_by == "best":
                submission.comment_sort = "confidence"
            elif sort_by == "new":
                submission.comment_sort = "new"
            elif sort_by == "controversial":
                submission.comment_sort = "controversial"
            elif sort_by == "old":
                submission.comment_sort = "old"
            
            # Load all comments up to the specified depth
            submission.comments.replace_more(limit=None)
            
            # Process comments
            comments = []
            self._process_comments(
                submission.comments,
                comments,
                max_results,
                include_replies,
                max_comment_depth,
                0,
                f"t3_{post_id}"
            )
            
            return comments[:max_results]
            
        except PRAWException as e:
            self.logger.error(f"Reddit API error when fetching comments: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error when fetching comments: {e}")
            return []
    
    def _process_comments(
        self, comment_forest, result_list: List[RedditPostComment], 
        max_results: int, include_replies: bool, max_depth: int, 
        current_depth: int, parent_id: str
    ) -> None:
        """
        Process a comment forest and extract comment data recursively.
        
        Args:
            comment_forest: PRAW comment forest
            result_list: List to append comments to
            max_results: Maximum number of comments to retrieve
            include_replies: Whether to include replies
            max_depth: Maximum depth to traverse
            current_depth: Current depth in the comment tree
            parent_id: ID of the parent (post or comment)
        """
        if len(result_list) >= max_results or current_depth > max_depth:
            return
        
        for comment in comment_forest:
            if len(result_list) >= max_results:
                return
                
            if isinstance(comment, Comment):
                try:
                    # Extract comment data
                    comment_data = RedditPostComment(
                        comment_id=comment.id,
                        text=comment.body,
                        author=comment.author.name if comment.author else "[deleted]",
                        score=comment.score,
                        created_utc=datetime.fromtimestamp(comment.created_utc),
                        edited=bool(comment.edited),
                        is_submitter=comment.is_submitter,
                        parent_id=parent_id,
                        depth=current_depth,
                        replies_count=len(comment.replies)
                    )
                    result_list.append(comment_data)
                    
                    # Process replies if needed
                    if include_replies and comment.replies and current_depth < max_depth:
                        self._process_comments(
                            comment.replies,
                            result_list,
                            max_results,
                            include_replies,
                            max_depth,
                            current_depth + 1,
                            f"t1_{comment.id}"
                        )
                except Exception as e:
                    self.logger.warning(f"Error processing comment: {e}")
    
    def extract_post_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract the post ID from a Reddit URL.
        
        Args:
            url: Reddit post URL
            
        Returns:
            Post ID or None if not a valid Reddit URL
        """
        import re
        
        # Regex patterns for different Reddit URL formats
        patterns = [
            r"reddit\.com/r/\w+/comments/([a-zA-Z0-9]+)",  # Standard URL
            r"redd\.it/([a-zA-Z0-9]+)",                   # Short URL
            r"reddit\.com/comments/([a-zA-Z0-9]+)"        # Direct comment URL
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def to_content_model(self, post_id: str, include_comments: bool = True) -> Optional[Content]:
        """
        Convert Reddit post data to a Content model.
        
        Args:
            post_id: Reddit post ID
            include_comments: Whether to include comments data
            
        Returns:
            Content model or None if retrieval fails
        """
        # Get post details
        post_detail = self.get_post_details(post_id)
        if not post_detail:
            return None
        
        # Get comments if requested
        comments = []
        if include_comments:
            comments = self.get_post_comments(post_id, max_results=100)
        
        try:
            # Determine content type
            content_type = ContentType.TEXT
            if post_detail.metadata.post_type == "image":
                content_type = ContentType.IMAGE
            elif post_detail.metadata.post_type == "video":
                content_type = ContentType.VIDEO
            elif post_detail.metadata.post_type == "link" and "youtube.com" in post_detail.metadata.url:
                content_type = ContentType.VIDEO
            
            # Create Content model
            content = Content(
                id=f"rd_{post_id}",
                content_type=content_type,
                metadata=ContentMetadata(
                    title=post_detail.metadata.title,
                    description=post_detail.text_content or "",
                    creator_id=f"reddit_user_{post_detail.metadata.author}",
                    creator_name=post_detail.metadata.author,
                    platform=Platform.REDDIT,
                    published_at=post_detail.metadata.created_utc,
                    url=str(post_detail.metadata.permalink),
                    tags=[post_detail.metadata.subreddit],
                    category=post_detail.flair.get("text", "") if post_detail.flair else "",
                    language="en"  # Default to English
                ),
                # We don't set features here as they're generated by analysis agents
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return content
            
        except ValidationError as e:
            self.logger.error(f"Error creating Content model: {e}")
            return None 