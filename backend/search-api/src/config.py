"""Configuration management for the fashion search engine."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_chat_model: str = Field(default="gpt-4.1-mini")  # Fixed model name
    openai_max_retries: int = Field(default=3)
    openai_timeout: float = Field(default=30.0)
    
    # Data Configuration
    data_dir: Path = Field(default=Path("data"))
    dataset_file: str = Field(default=Path("raw/meta_Amazon_Fashion.jsonl"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    
    # Vector Search Configuration
    faiss_index_path: Optional[Path] = Field(default=None)
    embedding_cache_path: Optional[Path] = Field(default=None)
    max_products: Optional[int] = Field(default=None)  # None = use all
    
    # API Configuration
    api_host: str = Field(default="127.0.0.1")  # Changed from 0.0.0.0 for local development
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    
    # Search Configuration
    default_top_k: int = Field(default=10)
    max_top_k: int = Field(default=50)
    similarity_threshold: float = Field(default=0.3)
    
    # Cache Configuration
    enable_query_cache: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)  # 1 hour
    max_cache_size: int = Field(default=1000)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings() 