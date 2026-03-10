from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID
from app.models.user import UserRole


# Base User schema
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    role: UserRole = UserRole.EMPLOYEE
    is_active: bool = True


# Schema for creating a user
class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


# Schema for updating a user
class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    password: Optional[str] = Field(None, min_length=6)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


# Schema for returning user data (without password)
class User(UserBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str
