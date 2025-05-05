"""
Database Models

This module defines SQLAlchemy ORM models for storing content analysis
and engagement metrics in a relational database.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, 
    String, Table, Text, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Create declarative base
Base = declarative_base()

# Define association tables for many-to-many relationships
content_tag_association = Table(
    'content_tag',
    Base.metadata,
    Column('content_id', String(50), ForeignKey('content.id')),
    Column('tag_id', Integer, ForeignKey('tag.id'))
)


class Tag(Base):
    """Tag model for content categorization."""
    __tablename__ = 'tag'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    
    def __repr__(self):
        return f"<Tag(name='{self.name}')>"


class Creator(Base):
    """Creator model for content authors/channels."""
    __tablename__ = 'creator'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    platform = Column(String(20), nullable=False)
    url = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    contents = relationship("Content", back_populates="creator")
    
    def __repr__(self):
        return f"<Creator(id='{self.id}', name='{self.name}')>"


class Content(Base):
    """Content model for storing media content metadata."""
    __tablename__ = 'content'
    
    id = Column(String(50), primary_key=True)
    content_type = Column(String(20), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    creator_id = Column(String(50), ForeignKey('creator.id'), nullable=False)
    platform = Column(String(20), nullable=False)
    published_at = Column(DateTime)
    url = Column(String(255), nullable=False)
    category = Column(String(50))
    language = Column(String(10), default='en')
    duration_seconds = Column(Integer)
    raw_data_location = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    creator = relationship("Creator", back_populates="contents")
    tags = relationship("Tag", secondary=content_tag_association)
    video_features = relationship("VideoFeatures", uselist=False, back_populates="content")
    audio_features = relationship("AudioFeatures", uselist=False, back_populates="content")
    text_features = relationship("TextFeatures", uselist=False, back_populates="content")
    engagement_metrics = relationship("EngagementMetrics", back_populates="content")
    
    def __repr__(self):
        return f"<Content(id='{self.id}', title='{self.title}')>"


class VideoFeatures(Base):
    """Video features extracted from content."""
    __tablename__ = 'video_features'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(50), ForeignKey('content.id'), nullable=False)
    resolution = Column(String(20))
    fps = Column(Float)
    scene_transitions = Column(JSON)  # List of timestamps
    visual_complexity = Column(JSON)  # Dict of complexity metrics
    motion_intensity = Column(JSON)  # Dict of motion metrics
    color_scheme = Column(JSON)  # Dict of color metrics
    production_quality = Column(Float)
    thumbnail_data = Column(JSON)  # Dict of thumbnail metrics
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    content = relationship("Content", back_populates="video_features")
    
    def __repr__(self):
        return f"<VideoFeatures(content_id='{self.content_id}')>"


class AudioFeatures(Base):
    """Audio features extracted from content."""
    __tablename__ = 'audio_features'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(50), ForeignKey('content.id'), nullable=False)
    sample_rate = Column(Integer)
    bit_depth = Column(Integer)
    speech_segments = Column(JSON)  # List of speech segments
    music_segments = Column(JSON)  # List of music segments
    volume_dynamics = Column(JSON)  # Dict of volume metrics
    voice_characteristics = Column(JSON)  # Dict of voice metrics
    emotional_tone = Column(JSON)  # Dict of emotional tone metrics
    audio_quality = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    content = relationship("Content", back_populates="audio_features")
    
    def __repr__(self):
        return f"<AudioFeatures(content_id='{self.content_id}')>"


class TextFeatures(Base):
    """Text features extracted from content."""
    __tablename__ = 'text_features'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(50), ForeignKey('content.id'), nullable=False)
    word_count = Column(Integer)
    sentiment = Column(JSON)  # Dict of sentiment metrics
    topics = Column(JSON)  # List of topic distributions
    readability_scores = Column(JSON)  # Dict of readability metrics
    linguistic_complexity = Column(Float)
    keywords = Column(JSON)  # List of keyword data
    emotional_content = Column(JSON)  # Dict of emotional metrics
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    content = relationship("Content", back_populates="text_features")
    
    def __repr__(self):
        return f"<TextFeatures(content_id='{self.content_id}')>"


class EngagementMetrics(Base):
    """Engagement metrics for content analysis results."""
    __tablename__ = 'engagement_metrics'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(50), ForeignKey('content.id'), nullable=False)
    composite_score = Column(Float, nullable=False)
    dimensions = Column(JSON, nullable=False)  # Dict of dimension scores
    platform_specific = Column(JSON)  # Dict of platform-specific metrics
    demographic_breakdown = Column(JSON)  # Dict of demographic scores
    comparative_metrics = Column(JSON)  # Dict of comparative metrics
    temporal_pattern = Column(String(20), nullable=False)
    analysis_version = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    content = relationship("Content", back_populates="engagement_metrics")
    
    def __repr__(self):
        return f"<EngagementMetrics(content_id='{self.content_id}', score={self.composite_score})>"


class AnalysisJob(Base):
    """Analysis job tracking model."""
    __tablename__ = 'analysis_job'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(String(50), ForeignKey('content.id'), nullable=False)
    status = Column(String(20), nullable=False)  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    error_message = Column(Text)
    requested_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    def __repr__(self):
        return f"<AnalysisJob(id={self.id}, content_id='{self.content_id}', status='{self.status}')>"


# Database connection setup function
def get_db_engine(connection_string: str):
    """Create and return a database engine."""
    return create_engine(connection_string)


def get_db_session(engine):
    """Create and return a database session."""
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(engine):
    """Initialize the database schema."""
    Base.metadata.create_all(engine) 