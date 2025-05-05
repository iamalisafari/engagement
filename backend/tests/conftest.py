"""
Test configuration for pytest.

This module provides fixtures and configuration for the test suite.
"""

import os
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Dict, Generator, List

from ..api.main import app
from ..database import config
from ..api.auth import (
    get_current_active_user, create_access_token,
    User, Token
)


# Test database setup
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "sqlite:///./test_db.sqlite")
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Override get_db dependency for testing
def override_get_db() -> Generator[Session, None, None]:
    """Override the get_db dependency to use the test database."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override user authentication for testing
async def override_get_current_user() -> User:
    """Override get_current_user for testing to return a test user."""
    return User(
        username="test_user",
        email="test@example.com",
        full_name="Test User",
        disabled=False,
        role="researcher"
    )


async def override_get_admin_user() -> User:
    """Override get_current_user for testing to return an admin user."""
    return User(
        username="admin",
        email="admin@example.com",
        full_name="Admin User",
        disabled=False,
        role="admin"
    )


# Setup test client with overridden dependencies
@pytest.fixture
def client() -> TestClient:
    """Create a test client with overridden dependencies."""
    # Override database dependency
    app.dependency_overrides[config.get_db] = override_get_db
    
    # Create test client
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up
    app.dependency_overrides = {}


@pytest.fixture
def authenticated_client() -> TestClient:
    """Create a test client authenticated as a researcher."""
    # Override database and authentication dependencies
    app.dependency_overrides[config.get_db] = override_get_db
    app.dependency_overrides[get_current_active_user] = override_get_current_user
    
    # Create test client
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up
    app.dependency_overrides = {}


@pytest.fixture
def admin_client() -> TestClient:
    """Create a test client authenticated as an admin."""
    # Override database and authentication dependencies
    app.dependency_overrides[config.get_db] = override_get_db
    app.dependency_overrides[get_current_active_user] = override_get_admin_user
    
    # Create test client
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up
    app.dependency_overrides = {}


@pytest.fixture
def test_db() -> Generator[Session, None, None]:
    """Create a test database session."""
    # Create all tables
    # In a real implementation, this would use alembic migrations
    # For testing, we'll create tables directly
    # from ..database.models import Base
    # Base.metadata.create_all(bind=engine)
    
    # Create session
    db = TestSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        
    # Clean up database
    # Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_preset() -> Dict:
    """Sample analysis preset for testing."""
    return {
        "name": "Test Preset",
        "description": "A preset for testing",
        "depth": "standard",
        "features_enabled": {
            "video_analysis": True,
            "audio_analysis": True,
            "text_analysis": True,
            "temporal_analysis": True,
            "engagement_scoring": True,
            "comparative_analysis": False,
            "hitl_validation": False
        },
        "platform_specific_settings": {}
    }


@pytest.fixture
def sample_content_request() -> Dict:
    """Sample content analysis request for testing."""
    return {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "platform": "youtube",
        "preset_id": None,
        "custom_settings": None
    }


@pytest.fixture
def sample_batch_request(sample_content_request) -> Dict:
    """Sample batch analysis request for testing."""
    return {
        "items": [
            sample_content_request,
            {
                "url": "https://www.reddit.com/r/science/comments/abcdef",
                "platform": "reddit"
            }
        ],
        "preset_id": None,
        "custom_settings": None,
        "priority": 1
    }


@pytest.fixture
def sample_agent_config() -> Dict:
    """Sample agent configuration for testing."""
    return {
        "enabled": True,
        "priority": 2,
        "parameters": {
            "model_size": "medium",
            "batch_size": 16,
            "features_enabled": ["all"],
            "device": "cpu"
        }
    } 