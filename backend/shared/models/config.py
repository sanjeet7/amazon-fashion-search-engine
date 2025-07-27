"""Configuration models for services."""

from pathlib import Path
from typing import Optional
from pydantic import Field

from .base import BaseModel, LogLevel


class DataPipelineConfig(BaseModel):
    """Configuration for the data pipeline service."""
    
    # Data paths
    raw_data_path: Path = Field(default=Path("data/raw/meta_Amazon_Fashion.jsonl"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    embeddings_cache_dir: Path = Field(default=Path("data/processed/embeddings_cache"))
    
    # Processing parameters
    sample_size: int = Field(default=50000, description="Number of products to sample")
    min_quality_score: float = Field(default=0.5, description="Minimum quality score")
    quality_weight: float = Field(default=0.4, description="Weight for quality in sampling")
    diversity_weight: float = Field(default=0.3, description="Weight for diversity in sampling")
    rating_weight: float = Field(default=0.3, description="Weight for ratings in sampling")
    
    # OpenAI configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    
    # Embedding configuration
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    embedding_batch_size: int = Field(default=100, description="Batch size for embedding API calls")
    embedding_rate_limit_delay: float = Field(default=0.1, description="Delay between API calls (seconds)")
    embedding_cost_per_1k_tokens: float = Field(default=0.00002, description="Cost per 1K tokens")
    
    # Retry configuration
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries (seconds)")
    
    # Performance settings
    max_workers: int = Field(default=4, description="Maximum worker threads")
    chunk_size: int = Field(default=1000, description="Chunk size for processing")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_file: Optional[Path] = Field(default=None, description="Log file path")


class SearchAPIConfig(BaseModel):
    """Configuration for the search API service."""
    
    # API settings
    host: str = Field(default="127.0.0.1", env="SEARCH_API_HOST")
    port: int = Field(default=8000, env="SEARCH_API_PORT")
    workers: int = Field(default=1, env="SEARCH_API_WORKERS")
    reload: bool = Field(default=True, description="Enable auto-reload in development")
    
    # Data paths
    processed_data_dir: Path = Field(default=Path("data/processed"))
    embeddings_cache_dir: Path = Field(default=Path("data/processed/embeddings_cache"))
    faiss_index_path: Optional[Path] = Field(default=None)
    
    # Search configuration
    default_top_k: int = Field(default=10, description="Default number of results")
    max_top_k: int = Field(default=50, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity threshold")
    
    # Ranking weights (data-driven)
    semantic_weight: float = Field(default=0.6, description="Weight for semantic similarity")
    rating_weight: float = Field(default=0.25, description="Weight for rating signal")
    review_weight: float = Field(default=0.15, description="Weight for review count signal")
    filter_boost: float = Field(default=0.1, description="Boost for matching filters")
    
    # FAISS configuration
    faiss_index_type: str = Field(default="IndexIVFFlat", description="FAISS index type")
    n_clusters: int = Field(default=1000, description="Number of clusters for IVF index")
    search_k_factor: int = Field(default=10, description="Factor for expanding search")
    
    # OpenAI configuration (for query enhancement)
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    chat_model: str = Field(default="gpt-4-turbo", description="Chat model for query enhancement")
    openai_timeout: float = Field(default=30.0, description="OpenAI API timeout")
    
    # Caching
    enable_query_cache: bool = Field(default=True, description="Enable query result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum cache entries")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    
    # CORS settings
    cors_origins: list = Field(default=["http://localhost:3000"], description="Allowed CORS origins")
    cors_methods: list = Field(default=["GET", "POST"], description="Allowed CORS methods")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    
    model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    batch_size: int = Field(default=100, description="Batch size for API calls")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries (seconds)")
    cache_embeddings: bool = Field(default=True, description="Whether to cache embeddings")
    max_tokens_per_request: int = Field(default=8192, description="Maximum tokens per API request")
    
    # Cost tracking
    track_costs: bool = Field(default=True, description="Track embedding generation costs")
    cost_per_1k_tokens: float = Field(default=0.00002, description="Cost per 1K tokens")


class FrontendConfig(BaseModel):
    """Configuration for the frontend service."""
    
    # API connection
    api_url: str = Field(default="http://localhost:8000", env="NEXT_PUBLIC_API_URL")
    api_timeout: int = Field(default=30000, description="API timeout in milliseconds")
    
    # Development settings
    port: int = Field(default=3000, description="Development server port")
    
    # Feature flags
    enable_query_enhancement: bool = Field(default=True, description="Enable query enhancement")
    enable_similarity_scores: bool = Field(default=True, description="Show similarity scores")
    enable_explanations: bool = Field(default=False, description="Show result explanations")
    
    # UI configuration
    results_per_page: int = Field(default=20, description="Results per page")
    max_query_length: int = Field(default=500, description="Maximum query length")
    debounce_delay_ms: int = Field(default=300, description="Search debounce delay") 