"""Search API models."""

from typing import List, Optional, Dict, Any
from pydantic import Field

from .base import BaseModel
from .product import ProductResult


class SearchRequest(BaseModel):
    """Search request model."""
    
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=50)
    min_similarity: float = Field(default=0.0, description="Minimum similarity threshold", ge=0, le=1)
    
    # Optional filters
    price_min: Optional[float] = Field(default=None, description="Minimum price filter")
    price_max: Optional[float] = Field(default=None, description="Maximum price filter") 
    category: Optional[str] = Field(default=None, description="Category filter")
    min_rating: Optional[float] = Field(default=None, description="Minimum rating filter")


class SearchResult(BaseModel):
    """Individual search result."""
    
    product: ProductResult = Field(..., description="Product information")
    rank: int = Field(..., description="Result rank (1-based)")


class SearchResponse(BaseModel):
    """Search response model."""
    
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search time in milliseconds")
    
    # Query analysis (if available)
    enhanced_query: Optional[str] = Field(default=None, description="AI-enhanced query")
    detected_intent: Optional[str] = Field(default=None, description="Detected search intent")
    filters_applied: Optional[Dict[str, Any]] = Field(default=None, description="Applied filters")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    # System information
    total_products: int = Field(..., description="Total indexed products")
    index_ready: bool = Field(..., description="Search index ready")
    embeddings_loaded: bool = Field(..., description="Embeddings loaded")
    
    # Performance metrics
    avg_search_time_ms: Optional[float] = Field(default=None, description="Average search time")
    total_searches: int = Field(default=0, description="Total searches performed")