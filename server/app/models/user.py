from sqlalchemy import Column, String, Boolean, DateTime, Enum
from sqlalchemy.sql import func
from app.core.database import Base
import uuid
import enum


class UserRole(str, enum.Enum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    HR = "hr"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.EMPLOYEE)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
