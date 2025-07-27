"""Configuration management for the Fashion Search Engine."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Find the project root directory by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=str(find_project_root() / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    llm_model: str = Field(default="gpt-4.1-mini", description="LLM model for query processing")
    
    # Data Configuration (Stratified Sampling Strategy)
    stratified_sample_size: int = Field(default=150000, description="Stratified sample size as per final_exploration.md")
    data_batch_size: int = Field(default=100, description="Batch size for API calls")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # Development Settings
    log_level: str = Field(default="INFO", description="Logging level")
    development_mode: bool = Field(default=True, description="Development mode")
    
    # Paths (relative to project root)
    raw_data_path: Path = Field(default=Path("data/raw/meta_Amazon_Fashion.jsonl"), description="Raw data file path")
    processed_data_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    embeddings_dir: Path = Field(default=Path("data/embeddings"), description="Embeddings directory")
    
    def __init__(self, **kwargs):
        """Initialize settings and resolve paths relative to project root."""
        super().__init__(**kwargs)
        
        # Make all paths absolute relative to project root
        project_root = find_project_root()
        
        if not self.raw_data_path.is_absolute():
            object.__setattr__(self, 'raw_data_path', project_root / self.raw_data_path)
        if not self.processed_data_dir.is_absolute():
            object.__setattr__(self, 'processed_data_dir', project_root / self.processed_data_dir)
        if not self.embeddings_dir.is_absolute():
            object.__setattr__(self, 'embeddings_dir', project_root / self.embeddings_dir)
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v):
        """Validate OpenAI API key is provided."""
        if not v or v == "your_openai_api_key_here":
            raise ValueError("Please set a valid OPENAI_API_KEY in your .env file")
        return v