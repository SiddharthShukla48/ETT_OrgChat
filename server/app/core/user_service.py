from sqlalchemy.orm import Session
from typing import Optional, List
import uuid
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserUpdate
from app.core.auth import get_password_hash, verify_password


def get_user(db: Session, user_id: str) -> Optional[User]:
    """Get user by ID."""
    # Ensure user_id is a string
    if isinstance(user_id, uuid.UUID):
        user_id = str(user_id)
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Get list of users with pagination."""
    return db.query(User).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate) -> User:
    """Create a new user."""
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        password_hash=hashed_password,
        role=user.role,
        is_active=user.is_active
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: str, user_update: UserUpdate) -> Optional[User]:
    """Update user information."""
    # Ensure user_id is a string
    if isinstance(user_id, uuid.UUID):
        user_id = str(user_id)
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        update_data = user_update.dict(exclude_unset=True)
        if "password" in update_data:
            update_data["password_hash"] = get_password_hash(update_data.pop("password"))
        
        for field, value in update_data.items():
            setattr(db_user, field, value)
        
        db.commit()
        db.refresh(db_user)
    return db_user


def deactivate_user(db: Session, user_id: str) -> bool:
    """Deactivate user (set is_active to False) - reversible."""
    # Ensure user_id is a string
    if isinstance(user_id, uuid.UUID):
        user_id = str(user_id)
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db_user.is_active = False
        db.commit()
        return True
    return False


def activate_user(db: Session, user_id: str) -> bool:
    """Activate user (set is_active to True)."""
    # Ensure user_id is a string
    if isinstance(user_id, uuid.UUID):
        user_id = str(user_id)
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db_user.is_active = True
        db.commit()
        return True
    return False


def delete_user(db: Session, user_id: str) -> bool:
    """Hard delete user (permanently remove from database)."""
    # Ensure user_id is a string
    if isinstance(user_id, uuid.UUID):
        user_id = str(user_id)
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
        return True
    return False


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
