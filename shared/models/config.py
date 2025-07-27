"""Configuration management for the Fashion Search Engine."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    
    # Data Configuration
    data_sample_size: int = Field(default=50000, description="Number of products to process")
    data_batch_size: int = Field(default=100, description="Batch size for API calls")
    use_sample_data: bool = Field(default=False, description="Use 500-product sample for testing")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # Development Settings
    log_level: str = Field(default="INFO", description="Logging level")
    development_mode: bool = Field(default=True, description="Development mode")
    
    # Paths
    raw_data_path: Path = Field(default=Path("data/raw/meta_Amazon_Fashion.jsonl"), description="Raw data file path")
    sample_data_path: Path = Field(default=Path("data/sample_500_products.jsonl"), description="Sample data file path")
    processed_data_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    embeddings_dir: Path = Field(default=Path("data/embeddings"), description="Embeddings directory")
    
    @validator("openai_api_key")
    def validate_openai_key(cls, v):
        """Validate OpenAI API key is provided."""
        if not v or v == "your_openai_api_key_here":
            raise ValueError("Please set a valid OPENAI_API_KEY in your .env file")
        return v
    
    @validator("processed_data_dir", "embeddings_dir")
    def ensure_directories_exist(cls, v):
        """Ensure data directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @property
    def data_path(self) -> Path:
        """Get the data path based on sample mode."""
        return self.sample_data_path if self.use_sample_data else self.raw_data_path
    
    @property
    def effective_sample_size(self) -> int:
        """Get effective sample size based on mode."""
        return 500 if self.use_sample_data else self.data_sample_size
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False