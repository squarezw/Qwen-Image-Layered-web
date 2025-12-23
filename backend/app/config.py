"""
Configuration settings for the Qwen Image Layered Backend API
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database
    database_url: str = "postgresql://qwen_user:password@postgres:5432/qwen_layered"

    # Storage
    file_storage_path: str = "/storage"
    file_retention_days: int = 7
    max_file_size_mb: int = 50

    # Models
    model_cache_dir: str = "/models"
    cuda_visible_devices: str = "0"

    # API
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost"]
    api_title: str = "Qwen Image Layered API"
    api_version: str = "1.0.0"

    # Redis (optional)
    redis_url: str = "redis://redis:6379/0"

    # Application
    environment: str = "development"
    debug: bool = False

    # Model settings
    decomposition_model: str = "Qwen/Qwen-Image-Layered"
    editing_model: str = "Qwen/Qwen-Image-Edit-2509"
    rmbg_model: str = "briaai/RMBG-2.0"

    # Default inference parameters
    default_resolution: int = 640
    max_inference_steps: int = 50
    max_layers: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
