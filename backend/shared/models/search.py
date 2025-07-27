"""Search request and response models."""

from typing import List, Optional, Dict, Any
from pydantic import Field

from .base import BaseModel, QueryIntent, Season, Occasion
from .product import ProductResult


class FilterCriteria(BaseModel):
    """Structured filter criteria extracted from queries."""
    
    brand_store: Optional[str] = Field(default=None, description="Brand or store filter")
    semantic_category: Optional[str] = Field(default=None, description="Semantic category filter")
    price_min: Optional[float] = Field(default=None, description="Minimum price filter", ge=0)
    price_max: Optional[float] = Field(default=None, description="Maximum price filter", ge=0)
    quality_tier: Optional[str] = Field(default=None, description="Quality tier (high, medium, low)")
    season: Optional[Season] = Field(default=None, description="Season preference")
    occasion: Optional[Occasion] = Field(default=None, description="Occasion preference")
    
    
class QueryAnalysis(BaseModel):
    """Query analysis results from LLM processing."""
    
    original_query: str = Field(..., description="Original user query")
    enhanced_query: Optional[str] = Field(default=None, description="LLM-enhanced query")
    intent: QueryIntent = Field(..., description="Detected query intent")
    season: Optional[Season] = Field(default=None, description="Detected season preference")
    occasion: Optional[Occasion] = Field(default=None, description="Detected occasion")
    extracted_attributes: Dict[str, Any] = Field(default={}, description="Extracted product attributes")
    confidence_score: Optional[float] = Field(default=None, description="Analysis confidence (0-1)")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")


class RankingSignals(BaseModel):
    """Business ranking signals for a product."""
    
    semantic_score: float = Field(..., description="Semantic similarity score")
    rating_score: float = Field(default=0.0, description="Rating signal score")
    review_score: float = Field(default=0.0, description="Review count signal score")
    filter_boost: float = Field(default=0.0, description="Filter matching boost")
    final_score: float = Field(default=0.0, description="Final combined score")


class SearchRequest(BaseModel):
    """Search request model with all parameters."""
    
    # Core search parameters
    query: str = Field(..., description="Natural language search query", min_length=1, max_length=500)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=50)
    
    # Search behavior options
    use_query_enhancement: bool = Field(default=True, description="Whether to enhance query with LLM")
    explain_results: bool = Field(default=False, description="Include explanation for recommendations")
    
    # Filtering options
    price_min: Optional[float] = Field(default=None, description="Minimum price filter", ge=0)
    price_max: Optional[float] = Field(default=None, description="Maximum price filter", ge=0)
    min_rating: Optional[float] = Field(default=None, description="Minimum average rating", ge=1, le=5)
    category_filter: Optional[str] = Field(default=None, description="Category to filter by")
    
    # Advanced options
    similarity_threshold: Optional[float] = Field(
        default=None, 
        description="Minimum similarity threshold", 
        ge=0, 
        le=1
    )
    include_low_quality: bool = Field(default=False, description="Include low quality products")


class SearchResponse(BaseModel):
    """Search response with results and metadata."""
    
    # Search results
    results: List[ProductResult] = Field(..., description="Search results")
    query_analysis: QueryAnalysis = Field(..., description="Query analysis details")
    
    # Response metadata  
    total_results: int = Field(..., description="Total number of products searched")
    returned_results: int = Field(..., description="Number of results returned")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    
    # Performance metadata
    used_cache: bool = Field(default=False, description="Whether cached results were used")
    search_time_ms: Optional[float] = Field(default=None, description="Search operation time")
    ranking_time_ms: Optional[float] = Field(default=None, description="Ranking operation time")
    
    # Debug information (only included if requested)
    debug_info: Optional[Dict[str, Any]] = Field(default=None, description="Debug information")


class SampleProductsRequest(BaseModel):
    """Request for sample products."""
    
    limit: int = Field(default=10, description="Number of products to return", ge=1, le=100)
    with_images_only: bool = Field(default=True, description="Only return products with images")
    category_filter: Optional[str] = Field(default=None, description="Filter by category")
    min_rating: Optional[float] = Field(default=None, description="Minimum rating filter")


class SampleProductsResponse(BaseModel):
    """Response with sample products."""
    
    products: List[ProductResult] = Field(..., description="Sample products")
    total_available: int = Field(..., description="Total products available")
    returned: int = Field(..., description="Number returned")
    has_images: int = Field(..., description="Number with images")
    has_videos: int = Field(..., description="Number with videos")
    filters_applied: Dict[str, Any] = Field(default={}, description="Filters that were applied")


class SearchStatsResponse(BaseModel):
    """Search performance and usage statistics."""
    
    # Search performance
    total_searches: int = Field(..., description="Total searches performed")
    average_response_time_ms: float = Field(..., description="Average response time")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    
    # Popular queries
    popular_queries: List[str] = Field(default=[], description="Most popular search queries")
    recent_queries: List[str] = Field(default=[], description="Recent search queries")
    
    # Error statistics
    error_rate: float = Field(..., description="Error rate (0-1)")
    common_errors: List[str] = Field(default=[], description="Common error types")
    
    # Dataset statistics
    total_products_indexed: int = Field(..., description="Total products in search index")
    index_size_mb: float = Field(..., description="Search index size in MB")
    last_updated: str = Field(..., description="Last index update timestamp") 