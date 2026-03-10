from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/iic_auth"
    
    # JWT
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Application
    app_name: str = "IIC Authentication API"
    debug: bool = True
    
    class Config:
        env_file = ".env"


settings = Settings()
