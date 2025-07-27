"""Pydantic models for API requests and responses."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Query intent classification."""
    SPECIFIC_ITEM = "specific_item"
    STYLE_SEARCH = "style_search"
    OCCASION_BASED = "occasion_based"
    SEASONAL = "seasonal"
    GENERAL = "general"
    UNCLEAR = "unclear"


class Season(str, Enum):
    """Season classification."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all_season"


class Occasion(str, Enum):
    """Occasion classification."""
    WORK = "work"
    CASUAL = "casual"
    FORMAL = "formal"
    PARTY = "party"
    ATHLETIC = "athletic"
    WEDDING = "wedding"
    DATE = "date"
    TRAVEL = "travel"
    OTHER = "other"


# API Request/Response Models

class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Natural language search query", min_length=1, max_length=500)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=50)
    use_query_enhancement: bool = Field(default=True, description="Whether to enhance query with LLM")
    price_min: Optional[float] = Field(default=None, description="Minimum price filter", ge=0)
    price_max: Optional[float] = Field(default=None, description="Maximum price filter", ge=0)
    min_rating: Optional[float] = Field(default=None, description="Minimum average rating", ge=1, le=5)
    category_filter: Optional[str] = Field(default=None, description="Category to filter by")
    explain_results: bool = Field(default=False, description="Include explanation for recommendations")


class ProductResult(BaseModel):
    """Individual product result."""
    parent_asin: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    main_category: Optional[str] = Field(default=None, description="Main product category")
    price: Optional[float] = Field(default=None, description="Product price in USD")
    average_rating: Optional[float] = Field(default=None, description="Average customer rating")
    rating_number: Optional[int] = Field(default=None, description="Number of ratings")
    similarity_score: float = Field(..., description="Similarity to query (0-1)", ge=0, le=1)
    features: List[str] = Field(default=[], description="Product features")
    description: List[str] = Field(default=[], description="Product description")
    store: Optional[str] = Field(default=None, description="Store name")
    categories: List[str] = Field(default=[], description="Product categories")
    images: List[str] = Field(default=[], description="Product image URLs")
    videos: List[str] = Field(default=[], description="Product video URLs")
    primary_image: Optional[str] = Field(default=None, description="Primary product image URL")
    explanation: Optional[str] = Field(default=None, description="Why this product was recommended")


class QueryAnalysis(BaseModel):
    """Query analysis results."""
    original_query: str = Field(..., description="Original user query")
    enhanced_query: Optional[str] = Field(default=None, description="LLM-enhanced query")
    intent: QueryIntent = Field(..., description="Detected query intent")
    season: Optional[Season] = Field(default=None, description="Detected season preference")
    occasion: Optional[Occasion] = Field(default=None, description="Detected occasion")
    extracted_attributes: Dict[str, Any] = Field(default={}, description="Extracted product attributes")


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[ProductResult] = Field(..., description="Search results")
    query_analysis: QueryAnalysis = Field(..., description="Query analysis details")
    total_results: int = Field(..., description="Total number of products searched")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    used_cache: bool = Field(default=False, description="Whether cached results were used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    total_products: int = Field(..., description="Number of products in index")
    embedding_model: str = Field(..., description="Embedding model being used")
    cache_size: int = Field(default=0, description="Current cache size")


class StatsResponse(BaseModel):
    """Statistics response."""
    dataset_stats: Dict[str, Any] = Field(..., description="Dataset statistics")
    search_stats: Dict[str, Any] = Field(default={}, description="Search performance statistics")
    api_stats: Dict[str, Any] = Field(default={}, description="API usage statistics")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


# Data Processing Models

class DatasetInfo(BaseModel):
    """Dataset structure information."""
    file_size_gb: float
    total_records: int
    avg_record_size_bytes: int
    estimated_memory_requirement_gb: float
    processing_feasibility: str


class SamplingConfig(BaseModel):
    """Configuration for stratified sampling."""
    target_sample_size: int = Field(default=150000, description="Target sample size based on analysis")
    quality_weight: float = Field(default=0.4, description="Weight for quality criteria")
    diversity_weight: float = Field(default=0.3, description="Weight for brand/category diversity")
    rating_weight: float = Field(default=0.3, description="Weight for rating availability")
    min_quality_score: float = Field(default=0.8, description="Minimum quality score threshold")


# Embedding Models

class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    batch_size: int = Field(default=100, description="Batch size for API calls")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries (seconds)")
    cache_embeddings: bool = Field(default=True, description="Whether to cache embeddings")
    max_tokens_per_request: int = Field(default=8192, description="Maximum tokens per API request")


class TokenAnalysis(BaseModel):
    """Token analysis results for cost estimation."""
    total_tokens: int
    total_cost: float
    average_tokens_per_product: float
    cost_per_product: float
    batch_count: int
    model_used: str


class EmbeddingResult(BaseModel):
    """Embedding generation result."""
    product_id: str
    embedding: List[float]
    text_content: str
    token_count: int
    processing_time: float
    model_used: str


# Search Models

class FilterCriteria(BaseModel):
    """Structured filter criteria extracted from queries."""
    brand_store: Optional[str] = None
    semantic_category: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    quality_tier: Optional[str] = None  # 'high', 'medium', 'emerging'
    season: Optional[Season] = None
    occasion: Optional[Occasion] = None


class SearchConfig(BaseModel):
    """Configuration for search operations."""
    faiss_index_type: str = Field(default="IndexIVFFlat", description="FAISS index type")
    n_clusters: int = Field(default=1000, description="Number of clusters for IVF index")
    search_k_factor: int = Field(default=10, description="Factor for expanding search during clustering")
    default_top_k: int = Field(default=10, description="Default number of results")
    max_top_k: int = Field(default=50, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity threshold")
    
    # Ranking weights (data-driven from analysis)
    semantic_weight: float = Field(default=0.6, description="Weight for semantic similarity")
    rating_weight: float = Field(default=0.25, description="Weight for rating signal (CV: 0.252)")
    review_weight: float = Field(default=0.15, description="Weight for review count signal (CV: 0.579)")
    filter_boost: float = Field(default=0.1, description="Boost for matching filters")


class RankingSignals(BaseModel):
    """Business ranking signals for a product."""
    semantic_score: float
    rating_score: float = 0.0
    review_score: float = 0.0
    filter_boost: float = 0.0
    final_score: float = 0.0 