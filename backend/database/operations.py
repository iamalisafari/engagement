"""
Database Operations

This module provides functions for common database operations using the
SQLAlchemy ORM, serving as a repository layer for data access.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc, asc

from ..models.database import (
    Content, Creator, Tag, VideoFeatures, AudioFeatures, 
    TextFeatures, EngagementMetrics, AnalysisJob
)
from ..models.database_extensions import (
    User, AnalysisPreset, TimeSeriesData, AnalysisResult,
    APIKey, ContentCollection
)

logger = logging.getLogger("database.operations")


# User operations
def create_user(db: Session, username: str, email: str, password_hash: str,
               first_name: Optional[str] = None, last_name: Optional[str] = None,
               is_admin: bool = False) -> Optional[User]:
    """
    Create a new user in the database.
    
    Args:
        db: Database session
        username: Unique username
        email: User email address
        password_hash: Hashed password
        first_name: User's first name
        last_name: User's last name
        is_admin: Whether the user has admin privileges
        
    Returns:
        Created user or None if failed
    """
    try:
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            is_admin=is_admin
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating user: {e}")
        return None


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """
    Get a user by ID.
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        User object or None if not found
    """
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """
    Get a user by username.
    
    Args:
        db: Database session
        username: Username to search for
        
    Returns:
        User object or None if not found
    """
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Get a user by email address.
    
    Args:
        db: Database session
        email: Email address to search for
        
    Returns:
        User object or None if not found
    """
    return db.query(User).filter(User.email == email).first()


def update_user(db: Session, user_id: int, 
               data: Dict[str, Any]) -> Optional[User]:
    """
    Update user information.
    
    Args:
        db: Database session
        user_id: User ID
        data: Dictionary of fields to update
        
    Returns:
        Updated user or None if failed
    """
    try:
        user = get_user_by_id(db, user_id)
        if not user:
            return None
            
        for key, value in data.items():
            if hasattr(user, key):
                setattr(user, key, value)
                
        user.updated_at = datetime.now()
        db.commit()
        db.refresh(user)
        return user
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating user: {e}")
        return None


# Analysis preset operations
def create_analysis_preset(db: Session, user_id: int, name: str,
                         description: Optional[str], configuration: Dict[str, Any],
                         is_default: bool = False) -> Optional[AnalysisPreset]:
    """
    Create a new analysis preset.
    
    Args:
        db: Database session
        user_id: User ID
        name: Preset name
        description: Description of what the preset does
        configuration: JSON-serializable configuration
        is_default: Whether this is the user's default preset
        
    Returns:
        Created preset or None if failed
    """
    try:
        # If setting as default, unset any existing defaults
        if is_default:
            existing_defaults = db.query(AnalysisPreset).filter(
                AnalysisPreset.user_id == user_id,
                AnalysisPreset.is_default == True
            ).all()
            
            for preset in existing_defaults:
                preset.is_default = False
        
        preset = AnalysisPreset(
            user_id=user_id,
            name=name,
            description=description,
            configuration=configuration,
            is_default=is_default
        )
        db.add(preset)
        db.commit()
        db.refresh(preset)
        return preset
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating analysis preset: {e}")
        return None


def get_analysis_presets(db: Session, user_id: int) -> List[AnalysisPreset]:
    """
    Get all analysis presets for a user.
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        List of presets
    """
    return db.query(AnalysisPreset).filter(
        AnalysisPreset.user_id == user_id
    ).order_by(AnalysisPreset.name).all()


def get_default_preset(db: Session, user_id: int) -> Optional[AnalysisPreset]:
    """
    Get the default preset for a user.
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        Default preset or None if not found
    """
    return db.query(AnalysisPreset).filter(
        AnalysisPreset.user_id == user_id,
        AnalysisPreset.is_default == True
    ).first()


# Time series data operations
def add_time_series_datapoints(db: Session, datapoints: List[Dict[str, Any]]) -> bool:
    """
    Add multiple time series datapoints in bulk.
    
    Args:
        db: Database session
        datapoints: List of datapoint dicts with content_id, metric_type, timestamp, value
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data_objects = [TimeSeriesData(**datapoint) for datapoint in datapoints]
        db.add_all(data_objects)
        db.commit()
        return True
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error adding time series datapoints: {e}")
        return False


def get_time_series_data(db: Session, content_id: str, metric_type: str,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[TimeSeriesData]:
    """
    Get time series data for a content item.
    
    Args:
        db: Database session
        content_id: Content ID
        metric_type: Type of metric to retrieve
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        List of time series datapoints
    """
    query = db.query(TimeSeriesData).filter(
        TimeSeriesData.content_id == content_id,
        TimeSeriesData.metric_type == metric_type
    )
    
    if start_time:
        query = query.filter(TimeSeriesData.timestamp >= start_time)
    
    if end_time:
        query = query.filter(TimeSeriesData.timestamp <= end_time)
    
    return query.order_by(TimeSeriesData.timestamp).all()


# Analysis results operations
def save_analysis_result(db: Session, content_id: str, 
                        automated_metrics: Dict[str, Any],
                        user_id: Optional[int] = None,
                        job_id: Optional[int] = None,
                        human_feedback: Optional[Dict[str, Any]] = None,
                        confidence_scores: Optional[Dict[str, float]] = None,
                        final_score: Optional[float] = None,
                        notes: Optional[str] = None) -> Optional[AnalysisResult]:
    """
    Save analysis results to the database.
    
    Args:
        db: Database session
        content_id: Content ID
        automated_metrics: Metrics from automated analysis
        user_id: Optional user ID who performed/validated the analysis
        job_id: Optional analysis job ID
        human_feedback: Optional human feedback
        confidence_scores: Optional confidence scores
        final_score: Optional final engagement score
        notes: Optional notes
        
    Returns:
        Saved result or None if failed
    """
    try:
        result = AnalysisResult(
            content_id=content_id,
            user_id=user_id,
            job_id=job_id,
            automated_metrics=automated_metrics,
            human_feedback=human_feedback,
            confidence_scores=confidence_scores,
            final_score=final_score,
            notes=notes
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        return result
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error saving analysis result: {e}")
        return None


def get_analysis_results(db: Session, content_id: str) -> List[AnalysisResult]:
    """
    Get all analysis results for a content item.
    
    Args:
        db: Database session
        content_id: Content ID
        
    Returns:
        List of analysis results
    """
    return db.query(AnalysisResult).filter(
        AnalysisResult.content_id == content_id
    ).order_by(desc(AnalysisResult.created_at)).all()


# API key operations
def create_api_key(db: Session, platform: str, key_name: str, 
                 key_value: str, access_level: str = "read",
                 rate_limit: Optional[int] = None) -> Optional[APIKey]:
    """
    Create a new API key.
    
    Args:
        db: Database session
        platform: Platform name
        key_name: Name/description for the key
        key_value: Actual key value
        access_level: Access level (read, write, full)
        rate_limit: Optional rate limit
        
    Returns:
        Created API key or None if failed
    """
    try:
        api_key = APIKey(
            platform=platform,
            key_name=key_name,
            key_value=key_value,
            access_level=access_level,
            rate_limit=rate_limit,
            is_active=True
        )
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        return api_key
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating API key: {e}")
        return None


def get_active_api_key(db: Session, platform: str) -> Optional[APIKey]:
    """
    Get an active API key for a platform.
    
    Args:
        db: Database session
        platform: Platform name
        
    Returns:
        Active API key or None if not found
    """
    return db.query(APIKey).filter(
        APIKey.platform == platform,
        APIKey.is_active == True
    ).first()


def update_api_key_usage(db: Session, key_id: int) -> bool:
    """
    Update the last used timestamp for an API key.
    
    Args:
        db: Database session
        key_id: API key ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
        if api_key:
            api_key.last_used_at = datetime.now()
            db.commit()
            return True
        return False
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating API key usage: {e}")
        return False


# Content collection operations
def create_content_collection(db: Session, user_id: int, name: str,
                            description: Optional[str] = None,
                            query_parameters: Optional[Dict[str, Any]] = None) -> Optional[ContentCollection]:
    """
    Create a new content collection.
    
    Args:
        db: Database session
        user_id: User ID
        name: Collection name
        description: Collection description
        query_parameters: Parameters used to generate this collection
        
    Returns:
        Created collection or None if failed
    """
    try:
        collection = ContentCollection(
            user_id=user_id,
            name=name,
            description=description,
            query_parameters=query_parameters
        )
        db.add(collection)
        db.commit()
        db.refresh(collection)
        return collection
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating content collection: {e}")
        return None


def add_content_to_collection(db: Session, collection_id: int, 
                             content_ids: List[str]) -> bool:
    """
    Add content items to a collection.
    
    Args:
        db: Database session
        collection_id: Collection ID
        content_ids: List of content IDs to add
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the collection
        collection = db.query(ContentCollection).filter(
            ContentCollection.id == collection_id
        ).first()
        
        if not collection:
            return False
        
        # Get existing content items
        existing_content_ids = [c.id for c in collection.content_items]
        
        # Add new content items
        for content_id in content_ids:
            if content_id not in existing_content_ids:
                content = db.query(Content).filter(Content.id == content_id).first()
                if content:
                    collection.content_items.append(content)
        
        db.commit()
        return True
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error adding content to collection: {e}")
        return False


def get_user_collections(db: Session, user_id: int) -> List[ContentCollection]:
    """
    Get all collections for a user.
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        List of collections
    """
    return db.query(ContentCollection).filter(
        ContentCollection.user_id == user_id
    ).order_by(ContentCollection.name).all()


def get_collection_contents(db: Session, collection_id: int) -> List[Content]:
    """
    Get all content items in a collection.
    
    Args:
        db: Database session
        collection_id: Collection ID
        
    Returns:
        List of content items
    """
    collection = db.query(ContentCollection).filter(
        ContentCollection.id == collection_id
    ).first()
    
    if collection:
        return collection.content_items
    return [] 