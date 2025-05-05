#!/usr/bin/env python3
"""
API Integration Test Tool

This script tests the YouTube and Reddit API integrations 
to ensure they are working correctly.
"""

import argparse
import json
import logging
import os
import sys
from pprint import pprint

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.integrations.youtube_api import YouTubeApiClient
from backend.integrations.reddit_api import RedditApiClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_integrations")


def test_youtube_api(video_url: str = None, video_id: str = None):
    """
    Test the YouTube API integration.
    
    Args:
        video_url: YouTube video URL to test with
        video_id: YouTube video ID to test with
    """
    logger.info("Testing YouTube API integration...")
    
    # Initialize the YouTube API client
    youtube_client = YouTubeApiClient()
    
    if not youtube_client.api_key:
        logger.error("YouTube API key not found. Set the YOUTUBE_API_KEY environment variable.")
        return False
    
    # Determine video ID
    if video_url:
        video_id = youtube_client.extract_video_id_from_url(video_url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {video_url}")
            return False
    elif not video_id:
        # Use a default video ID for testing
        video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    
    logger.info(f"Testing with video ID: {video_id}")
    
    # Get video details
    video_detail = youtube_client.get_video_details(video_id)
    if not video_detail:
        logger.error("Failed to retrieve video details")
        return False
    
    logger.info(f"Successfully retrieved details for video: {video_detail.metadata.title}")
    logger.info(f"Channel: {video_detail.metadata.channel_title}")
    logger.info(f"Views: {video_detail.statistics.view_count}")
    logger.info(f"Likes: {video_detail.statistics.like_count}")
    
    # Get comments
    comments = youtube_client.get_video_comments(video_id, max_results=5)
    logger.info(f"Retrieved {len(comments)} comments")
    if comments:
        logger.info("Sample comment:")
        logger.info(f"Author: {comments[0].author_display_name}")
        logger.info(f"Text: {comments[0].text[:100]}...")
    
    # Convert to content model
    content_model = youtube_client.to_content_model(video_id, include_comments=False)
    if not content_model:
        logger.error("Failed to convert to content model")
        return False
    
    logger.info("Successfully converted to content model")
    logger.info(f"Content ID: {content_model.id}")
    
    logger.info("YouTube API integration test completed successfully")
    return True


def test_reddit_api(post_url: str = None, post_id: str = None):
    """
    Test the Reddit API integration.
    
    Args:
        post_url: Reddit post URL to test with
        post_id: Reddit post ID to test with
    """
    logger.info("Testing Reddit API integration...")
    
    # Initialize the Reddit API client
    reddit_client = RedditApiClient()
    
    if not reddit_client.client_id or not reddit_client.client_secret:
        logger.error("Reddit API credentials not found. Set the REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        return False
    
    # Determine post ID
    if post_url:
        post_id = reddit_client.extract_post_id_from_url(post_url)
        if not post_id:
            logger.error(f"Could not extract post ID from URL: {post_url}")
            return False
    elif not post_id:
        # Use a default post ID for testing
        post_id = "12nm6y7"  # A popular post from r/AskReddit
    
    logger.info(f"Testing with post ID: {post_id}")
    
    # Get post details
    post_detail = reddit_client.get_post_details(post_id)
    if not post_detail:
        logger.error("Failed to retrieve post details")
        return False
    
    logger.info(f"Successfully retrieved details for post: {post_detail.metadata.title}")
    logger.info(f"Author: {post_detail.metadata.author}")
    logger.info(f"Subreddit: {post_detail.metadata.subreddit}")
    logger.info(f"Score: {post_detail.statistics.score}")
    logger.info(f"Comments: {post_detail.statistics.num_comments}")
    
    # Get comments
    comments = reddit_client.get_post_comments(post_id, max_results=5)
    logger.info(f"Retrieved {len(comments)} comments")
    if comments:
        logger.info("Sample comment:")
        logger.info(f"Author: {comments[0].author}")
        logger.info(f"Text: {comments[0].text[:100]}...")
    
    # Convert to content model
    content_model = reddit_client.to_content_model(post_id, include_comments=False)
    if not content_model:
        logger.error("Failed to convert to content model")
        return False
    
    logger.info("Successfully converted to content model")
    logger.info(f"Content ID: {content_model.id}")
    
    logger.info("Reddit API integration test completed successfully")
    return True


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test API integrations")
    
    parser.add_argument(
        '--youtube',
        action='store_true',
        help='Test YouTube API integration'
    )
    parser.add_argument(
        '--reddit',
        action='store_true',
        help='Test Reddit API integration'
    )
    parser.add_argument(
        '--youtube-url',
        type=str,
        help='YouTube video URL to test with'
    )
    parser.add_argument(
        '--youtube-id',
        type=str,
        help='YouTube video ID to test with'
    )
    parser.add_argument(
        '--reddit-url',
        type=str,
        help='Reddit post URL to test with'
    )
    parser.add_argument(
        '--reddit-id',
        type=str,
        help='Reddit post ID to test with'
    )
    
    args = parser.parse_args()
    
    # If no specific test is requested, run all tests
    run_all = not (args.youtube or args.reddit)
    
    success = True
    
    if args.youtube or run_all:
        youtube_success = test_youtube_api(args.youtube_url, args.youtube_id)
        success = success and youtube_success
    
    if args.reddit or run_all:
        reddit_success = test_reddit_api(args.reddit_url, args.reddit_id)
        success = success and reddit_success
    
    if success:
        logger.info("All API integration tests completed successfully")
        sys.exit(0)
    else:
        logger.error("One or more API integration tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 