"""
Authentication API Routes

API endpoints for user authentication, token management, and registration.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from ..database import config
from .auth import (
    User, Token, authenticate_user, create_access_token,
    get_current_active_user, check_admin_role, ACCESS_TOKEN_EXPIRE_MINUTES,
    get_password_hash, fake_users_db
)


router = APIRouter(prefix="/auth", tags=["authentication"])


# Additional models
class UserCreate(BaseModel):
    """Request model for user registration."""
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class UserResponse(BaseModel):
    """Response model for user information."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    disabled: bool = False


# Routes
@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(config.get_db)
):
    """
    Get an access token with username and password.
    
    This endpoint implements OAuth2 password flow for obtaining JWT tokens.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get the current user's information."""
    return current_user


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user: UserCreate,
    db: Session = Depends(config.get_db),
    admin: User = Depends(check_admin_role)
):
    """
    Register a new user. Only admin users can register new users.
    
    In a production environment, this would be replaced with a proper
    user registration system with email verification.
    """
    # Check if username already exists
    if user.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    new_user = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "disabled": False,
        "role": "user"  # Default role
    }
    
    # In production, save to database
    # For demo, we'll use the fake database
    fake_users_db[user.username] = new_user
    
    # Return user information without password
    return UserResponse(
        username=new_user["username"],
        email=new_user["email"],
        full_name=new_user["full_name"],
        role=new_user["role"],
        disabled=new_user["disabled"]
    )


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 10,
    admin: User = Depends(check_admin_role),
    db: Session = Depends(config.get_db)
):
    """
    List all users. Only admin users can access this endpoint.
    """
    # In production, query from database with pagination
    # For demo, we'll use the fake database
    users = list(fake_users_db.values())
    
    # Apply pagination
    paginated_users = users[skip:skip + limit]
    
    # Convert to response model
    return [
        UserResponse(
            username=user["username"],
            email=user["email"],
            full_name=user["full_name"],
            role=user["role"],
            disabled=user["disabled"]
        )
        for user in paginated_users
    ]


@router.put("/users/{username}/role", response_model=UserResponse)
async def update_user_role(
    username: str,
    role: str,
    admin: User = Depends(check_admin_role),
    db: Session = Depends(config.get_db)
):
    """
    Update a user's role. Only admin users can update roles.
    """
    # Validate role
    valid_roles = ["user", "researcher", "admin"]
    if role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}"
        )
    
    # Check if user exists
    if username not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update role
    fake_users_db[username]["role"] = role
    
    # Return updated user
    return UserResponse(
        username=fake_users_db[username]["username"],
        email=fake_users_db[username]["email"],
        full_name=fake_users_db[username]["full_name"],
        role=fake_users_db[username]["role"],
        disabled=fake_users_db[username]["disabled"]
    )


@router.put("/users/{username}/disable", response_model=UserResponse)
async def disable_user(
    username: str,
    disabled: bool = True,
    admin: User = Depends(check_admin_role),
    db: Session = Depends(config.get_db)
):
    """
    Disable or enable a user. Only admin users can modify this.
    """
    # Check if user exists
    if username not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent disabling own account
    if username == admin.username and disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disable your own account"
        )
    
    # Update disabled status
    fake_users_db[username]["disabled"] = disabled
    
    # Return updated user
    return UserResponse(
        username=fake_users_db[username]["username"],
        email=fake_users_db[username]["email"],
        full_name=fake_users_db[username]["full_name"],
        role=fake_users_db[username]["role"],
        disabled=fake_users_db[username]["disabled"]
    ) 