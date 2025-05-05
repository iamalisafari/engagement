"""
Database Configuration

This module provides SQLAlchemy configuration and session management
for connecting to the database.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from ..models.database import Base

# Get database URL from environment with fallback to SQLite
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite:///./social_media_engagement.db"
)

# Create engine
# For SQLite, connect_args needed for multi-threading support
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL, 
    connect_args=connect_args,
    echo=os.environ.get("SQL_ECHO", "").lower() == "true"
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    
    This should be called once at application startup.
    """
    # Create all tables
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.
    
    This is a generator function that yields a SQLAlchemy session
    and ensures it's closed after use.
    
    Yields:
        A SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    This provides a context manager that automatically closes
    the session when exiting the context.
    
    Yields:
        A SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_db() -> None:
    """
    Reset the database by dropping and recreating all tables.
    
    Warning: This will delete all data!
    Only use in development/testing environments.
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine) 