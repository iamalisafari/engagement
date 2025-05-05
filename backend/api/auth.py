"""
Authentication and Authorization Module

This module provides authentication, authorization, and rate limiting
functionality for the API.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import config

# Security configuration
SECRET_KEY = "CHANGE_THIS_TO_A_SECURE_SECRET_KEY_IN_PRODUCTION"  # In production, use env vars
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100  # requests per window


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Models
class User(BaseModel):
    """User model for authentication."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    role: str = "user"  # user, admin, researcher


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    role: Optional[str] = None


# Mock user database - in production, use a real database
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("adminpassword"),
        "disabled": False,
        "role": "admin"
    },
    "researcher": {
        "username": "researcher",
        "full_name": "Academic Researcher",
        "email": "researcher@example.com",
        "hashed_password": pwd_context.hash("researcherpassword"),
        "disabled": False,
        "role": "researcher"
    },
    "user": {
        "username": "user",
        "full_name": "Regular User",
        "email": "user@example.com",
        "hashed_password": pwd_context.hash("userpassword"),
        "disabled": False,
        "role": "user"
    }
}


# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, window_seconds: int = RATE_LIMIT_WINDOW, max_requests: int = RATE_LIMIT_MAX_REQUESTS):
        """Initialize rate limiter."""
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests: Dict[str, List[float]] = {}
        
    def is_rate_limited(self, key: str) -> tuple[bool, int, int]:
        """
        Check if a key is rate limited.
        
        Args:
            key: The key to check (typically username or IP)
            
        Returns:
            Tuple of (is_limited, current_requests, reset_time)
        """
        now = time.time()
        
        # Initialize if key not in requests
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove expired timestamps
        self.requests[key] = [ts for ts in self.requests[key] if now - ts < self.window_seconds]
        
        # Check if rate limited
        is_limited = len(self.requests[key]) >= self.max_requests
        
        # Add current timestamp if not limited
        if not is_limited:
            self.requests[key].append(now)
        
        # Calculate reset time
        if len(self.requests[key]) > 0:
            oldest = min(self.requests[key])
            reset_time = int(oldest + self.window_seconds - now)
        else:
            reset_time = 0
            
        return is_limited, len(self.requests[key]), reset_time


# Initialize rate limiter
rate_limiter = RateLimiter()


# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)


def get_user(db: Session, username: str) -> Optional[UserInDB]:
    """Get a user from the database."""
    # In production, query from a real database
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(db: Session, username: str, password: str) -> Union[UserInDB, bool]:
    """Authenticate a user with username and password."""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Dependency functions
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(config.get_db)) -> User:
    """Get the current user based on the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def check_admin_role(current_user: User = Depends(get_current_active_user)) -> User:
    """Check if the current user has admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


def check_researcher_role(current_user: User = Depends(get_current_active_user)) -> User:
    """Check if the current user has researcher role."""
    if current_user.role not in ["admin", "researcher"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


# Rate limiting middleware
async def rate_limit_middleware(request: Request, current_user: User = Depends(get_current_active_user)) -> None:
    """
    Rate limiting middleware.
    
    Limits the number of requests a user can make within a time window.
    Admin users are exempt from rate limiting.
    """
    # Admin users are exempt from rate limiting
    if current_user.role == "admin":
        return
    
    # Use username as the rate limit key
    key = current_user.username
    
    # Check if rate limited
    is_limited, current_requests, reset_time = rate_limiter.is_rate_limited(key)
    
    # Add rate limit headers
    request.state.rate_limit_remaining = rate_limiter.max_requests - current_requests
    request.state.rate_limit_reset = reset_time
    
    # Raise exception if rate limited
    if is_limited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {reset_time} seconds.",
            headers={
                "X-Rate-Limit-Limit": str(rate_limiter.max_requests),
                "X-Rate-Limit-Remaining": "0",
                "X-Rate-Limit-Reset": str(reset_time)
            }
        )
        

# Add rate limit headers to response
async def add_rate_limit_headers(request: Request, response: Any) -> Any:
    """Add rate limit headers to the response."""
    if hasattr(request.state, "rate_limit_remaining"):
        response.headers["X-Rate-Limit-Limit"] = str(rate_limiter.max_requests)
        response.headers["X-Rate-Limit-Remaining"] = str(request.state.rate_limit_remaining)
        response.headers["X-Rate-Limit-Reset"] = str(request.state.rate_limit_reset)
    return response 