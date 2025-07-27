"""Product models for the fashion search system."""

from typing import List, Optional, Dict, Any
from pydantic import Field

from .base import BaseModel


class ProductMetadata(BaseModel):
    """Product metadata for analytics."""
    
    total_tokens: int = Field(..., description="Total tokens in processed text")
    processing_time: float = Field(..., description="Processing time in seconds")
    embedding_model: str = Field(..., description="Model used for embeddings")
    quality_score: Optional[float] = Field(default=None, description="Quality score (0-1)")


class Product(BaseModel):
    """Core product model."""
    
    parent_asin: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    main_category: Optional[str] = Field(default=None, description="Primary category")
    categories: List[str] = Field(default_factory=list, description="All product categories")
    price: Optional[float] = Field(default=None, description="Product price", ge=0)
    average_rating: Optional[float] = Field(default=None, description="Average customer rating", ge=0, le=5)
    rating_number: Optional[int] = Field(default=None, description="Number of ratings", ge=0)
    features: List[str] = Field(default_factory=list, description="Product features")
    description: List[str] = Field(default_factory=list, description="Product descriptions")
    images: List[str] = Field(default_factory=list, description="Product image URLs")
    store: Optional[str] = Field(default=None, description="Store/brand name")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional product details")
    
    # Generated fields
    text_for_embedding: Optional[str] = Field(default=None, description="Processed text for embedding generation")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Standardized filter values")
    quality_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Quality indicators for sampling")


class ProductResult(BaseModel):
    """Product result with similarity score for search responses."""
    
    parent_asin: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    main_category: Optional[str] = Field(default=None, description="Primary category")
    price: Optional[float] = Field(default=None, description="Product price", ge=0)
    average_rating: Optional[float] = Field(default=None, description="Average customer rating", ge=0, le=5)
    rating_number: Optional[int] = Field(default=None, description="Number of ratings", ge=0)
    similarity_score: float = Field(..., description="Similarity score from search", ge=0, le=1)
    
    # Optional additional fields
    features: List[str] = Field(default_factory=list, description="Product features")
    description: List[str] = Field(default_factory=list, description="Product descriptions")
    store: Optional[str] = Field(default=None, description="Store/brand name")
    categories: List[str] = Field(default_factory=list, description="All product categories")
    images: List[str] = Field(default_factory=list, description="Product image URLs")
    
    # Filter metadata for debugging
    matched_filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters that matched this result")