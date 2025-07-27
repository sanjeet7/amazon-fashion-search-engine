"""Product data models."""

from typing import List, Optional, Dict, Any
from pydantic import Field

from .base import BaseModel


class Product(BaseModel):
    """Core product model."""
    
    # Core identifiers
    parent_asin: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    
    # Category and classification
    main_category: Optional[str] = Field(default=None, description="Main product category")
    categories: List[str] = Field(default_factory=list, description="All product categories")
    
    # Pricing and ratings
    price: Optional[float] = Field(default=None, description="Product price in USD")
    average_rating: Optional[float] = Field(default=None, description="Average customer rating")
    rating_number: Optional[int] = Field(default=None, description="Number of ratings")
    
    # Content and media
    features: List[str] = Field(default_factory=list, description="Product features")
    description: List[str] = Field(default_factory=list, description="Product description")
    images: List[str] = Field(default_factory=list, description="Product image URLs")
    
    # Store and metadata
    store: Optional[str] = Field(default=None, description="Store name")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional product details")
    
    # Processing metadata
    text_for_embedding: Optional[str] = Field(default=None, description="Processed text for embedding")


class ProductResult(BaseModel):
    """Product search result with metadata."""
    
    # Core product data
    parent_asin: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    main_category: Optional[str] = Field(default=None, description="Main product category")
    price: Optional[float] = Field(default=None, description="Product price")
    average_rating: Optional[float] = Field(default=None, description="Average rating")
    rating_number: Optional[int] = Field(default=None, description="Number of ratings")
    
    # Search metadata
    similarity_score: float = Field(..., description="Similarity to query (0-1)", ge=0, le=1)
    
    # Additional data
    features: List[str] = Field(default_factory=list, description="Product features")
    description: List[str] = Field(default_factory=list, description="Product description")
    store: Optional[str] = Field(default=None, description="Store name")
    categories: List[str] = Field(default_factory=list, description="Product categories")
    images: List[str] = Field(default_factory=list, description="Product image URLs")


class ProductMetadata(BaseModel):
    """Product processing metadata."""
    
    parent_asin: str = Field(..., description="Product identifier")
    has_images: bool = Field(..., description="Has product images")
    has_ratings: bool = Field(..., description="Has customer ratings")
    has_price: bool = Field(..., description="Has price information")
    feature_count: int = Field(..., description="Number of features")
    token_count: int = Field(..., description="Token count for embedding")
    quality_score: float = Field(..., description="Quality score (0-1)", ge=0, le=1)