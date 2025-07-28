"""Configuration models for the fashion search system."""

import os
from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # === API Configuration ===
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # === Data Processing Configuration ===
    # Optimized for take-home assessment
    stratified_sample_size: int = Field(default=50000, description="Sample size for stratified sampling")
    embedding_batch_size: int = Field(default=1000, description="Batch size for embedding generation")
    llm_concurrent_limit: int = Field(default=10, description="Maximum concurrent LLM requests")
    sequential_processing: bool = Field(default=False, description="Use sequential processing to avoid rate limits")
    
    # === Model Configuration ===
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model for query enhancement")
    
    # === OpenAI Configuration ===
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_max_retries: int = Field(default=3, description="Maximum retries for OpenAI API calls")
    openai_timeout: float = Field(default=30.0, description="Timeout for OpenAI API calls")
    
    # === Data Paths ===
    raw_data_path: str = Field(default="data/raw", description="Path to raw data")
    processed_data_path: str = Field(default="data/processed", description="Path to processed data")
    embeddings_path: str = Field(default="data/embeddings", description="Path to embeddings")
    sample_data_path: str = Field(default="data/sample", description="Path to sample data for quick testing")
    
    # === Search Configuration ===
    default_top_k: int = Field(default=10, description="Default number of search results")
    max_top_k: int = Field(default=50, description="Maximum number of search results")
    default_min_similarity: float = Field(default=0.0, description="Default minimum similarity threshold")
    
    # === Performance Configuration ===
    faiss_index_type: Literal["flat", "ivf"] = Field(default="ivf", description="FAISS index type")
    faiss_nlist: Optional[int] = Field(default=None, description="Number of clusters for IVF index (auto if None)")
    faiss_nprobe: Optional[int] = Field(default=None, description="Number of clusters to search (auto if None)")
    
    # === Logging Configuration ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    # === Quality Control ===
    min_title_length: int = Field(default=10, description="Minimum title length for quality filtering")
    max_description_length: int = Field(default=5000, description="Maximum description length")
    require_price: bool = Field(default=False, description="Require price for inclusion")
    require_rating: bool = Field(default=False, description="Require rating for inclusion")
    
    # === Development Configuration ===
    debug: bool = Field(default=False, description="Enable debug mode")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"