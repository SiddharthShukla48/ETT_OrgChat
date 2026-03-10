from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_active_user, require_admin, require_hr
from app.core.user_service import get_users, get_user, update_user, delete_user, create_user, get_user_by_username, deactivate_user, activate_user
from app.schemas.user import User as UserSchema, UserUpdate, UserCreate
from app.models.user import User

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserSchema)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user


@router.post("/", response_model=UserSchema)
async def create_user_endpoint(
    user: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Create a new user (Admin only)."""
    print(f"DEBUG: Received user data: {user}")
    print(f"DEBUG: User model dump: {user.model_dump()}")
    
    # Check if user already exists
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Create new user
    return create_user(db=db, user=user)


@router.get("/", response_model=List[UserSchema])
async def read_users(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: User = Depends(require_hr)
):
    """Get list of users (HR and Admin only)."""
    users = get_users(db, skip=skip, limit=limit)
    
    # If current user is admin, exclude themselves from the list
    if current_user.role.value == "admin":
        users = [user for user in users if user.id != current_user.id]
    
    return users


@router.get("/{user_id}", response_model=UserSchema)
async def read_user(
    user_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(require_hr)
):
    """Get user by ID (HR and Admin only)."""
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.put("/{user_id}", response_model=UserSchema)
async def update_user_endpoint(
    user_id: str,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Update user (Admin only)."""
    db_user = update_user(db, user_id=user_id, user_update=user_update)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.patch("/{user_id}/deactivate")
async def deactivate_user_endpoint(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Deactivate user (Admin only) - reversible."""
    success = deactivate_user(db, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deactivated successfully"}


@router.patch("/{user_id}/activate")
async def activate_user_endpoint(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Activate user (Admin only)."""
    success = activate_user(db, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User activated successfully"}


@router.delete("/{user_id}")
async def delete_user_endpoint(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Permanently delete user (Admin only)."""
    success = delete_user(db, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted permanently"}
