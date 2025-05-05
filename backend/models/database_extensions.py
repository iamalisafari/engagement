"""
Database Models Extensions

This module defines additional SQLAlchemy ORM models for the social media
engagement analysis system, extending the core models in database.py.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, 
    String, Table, Text, create_engine, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from .database import Base


class User(Base):
    """User model for researchers and analysts."""
    __tablename__ = 'user'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    analysis_presets = relationship("AnalysisPreset", back_populates="user")
    analysis_results = relationship("AnalysisResult", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}')>"


class AnalysisPreset(Base):
    """Preset configurations for analysis settings."""
    __tablename__ = 'analysis_preset'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    configuration = Column(JSON, nullable=False)  # Serialized analysis settings
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="analysis_presets")
    
    def __repr__(self):
        return f"<AnalysisPreset(name='{self.name}')>"


class TimeSeriesData(Base):
    """Time series data for temporal engagement analysis."""
    __tablename__ = 'time_series_data'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(50), ForeignKey('content.id'), nullable=False)
    metric_type = Column(String(50), nullable=False)  # e.g., 'views', 'likes', 'comments'
    timestamp = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    source = Column(String(50))  # Where the data point came from
    
    # Relationships
    content = relationship("Content")
    
    __table_args__ = (
        # Composite index for efficient time-based queries
        Index('idx_timeseries_content_metric_timestamp', 
              'content_id', 'metric_type', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<TimeSeriesData(content_id='{self.content_id}', metric='{self.metric_type}', timestamp='{self.timestamp}')>"


class AnalysisResult(Base):
    """Analysis results with human feedback integration."""
    __tablename__ = 'analysis_result'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(50), ForeignKey('content.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('user.id'))
    job_id = Column(Integer, ForeignKey('analysis_job.id'))
    automated_metrics = Column(JSON)  # Metrics from automated analysis
    human_feedback = Column(JSON)  # Structured human feedback
    confidence_scores = Column(JSON)  # Confidence in different aspects of analysis
    final_score = Column(Float)  # Final engagement score after HITL integration
    notes = Column(Text)  # Analyst notes
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    content = relationship("Content")
    user = relationship("User", back_populates="analysis_results")
    job = relationship("AnalysisJob")
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, content_id='{self.content_id}')>"


class APIKey(Base):
    """API access keys for platform integrations."""
    __tablename__ = 'api_key'
    
    id = Column(Integer, primary_key=True)
    platform = Column(String(50), nullable=False)
    key_name = Column(String(100), nullable=False)
    key_value = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    access_level = Column(String(20), default="read")  # read, write, full
    rate_limit = Column(Integer)  # Calls per minute limit
    created_at = Column(DateTime, default=datetime.now)
    last_used_at = Column(DateTime)
    
    def __repr__(self):
        return f"<APIKey(platform='{self.platform}', key_name='{self.key_name}')>"


class ContentCollection(Base):
    """Collection of content items for comparative analysis."""
    __tablename__ = 'content_collection'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    query_parameters = Column(JSON)  # Parameters used to create this collection
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Many-to-many relationship with content
    content_items = relationship(
        "Content",
        secondary=Table(
            'content_collection_item',
            Base.metadata,
            Column('collection_id', Integer, ForeignKey('content_collection.id')),
            Column('content_id', String(50), ForeignKey('content.id'))
        )
    )
    
    # Relationship with user
    user = relationship("User")
    
    def __repr__(self):
        return f"<ContentCollection(name='{self.name}')>" 