"""Product data models for the fashion search engine."""

from typing import List, Optional
from pydantic import Field

from .base import BaseModel


class Product(BaseModel):
    """Core product model with all available fields."""
    
    # Core identifiers
    parent_asin: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    
    # Category and classification
    main_category: Optional[str] = Field(default=None, description="Main product category")
    categories: List[str] = Field(default=[], description="All product categories")
    
    # Pricing and ratings
    price: Optional[float] = Field(default=None, description="Product price in USD")
    average_rating: Optional[float] = Field(default=None, description="Average customer rating")
    rating_number: Optional[int] = Field(default=None, description="Number of ratings")
    
    # Content and media
    features: List[str] = Field(default=[], description="Product features")
    description: List[str] = Field(default=[], description="Product description")
    images: List[str] = Field(default=[], description="Product image URLs")
    videos: List[str] = Field(default=[], description="Product video URLs")
    primary_image: Optional[str] = Field(default=None, description="Primary product image URL")
    
    # Store and metadata
    store: Optional[str] = Field(default=None, description="Store name")
    details: Optional[dict] = Field(default=None, description="Additional product details")
    
    # Processing metadata
    quality_score: Optional[float] = Field(default=None, description="Quality score (0-1)")
    text_for_embedding: Optional[str] = Field(default=None, description="Processed text for embedding")


class ProductResult(BaseModel):
    """Product result with search metadata."""
    
    # Core product data
    parent_asin: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    main_category: Optional[str] = Field(default=None, description="Main product category")
    price: Optional[float] = Field(default=None, description="Product price in USD")
    average_rating: Optional[float] = Field(default=None, description="Average customer rating")
    rating_number: Optional[int] = Field(default=None, description="Number of ratings")
    
    # Search-specific metadata
    similarity_score: float = Field(..., description="Similarity to query (0-1)", ge=0, le=1)
    features: List[str] = Field(default=[], description="Product features")
    description: List[str] = Field(default=[], description="Product description")
    store: Optional[str] = Field(default=None, description="Store name")
    categories: List[str] = Field(default=[], description="Product categories")
    
    # Media
    images: List[str] = Field(default=[], description="Product image URLs")
    videos: List[str] = Field(default=[], description="Product video URLs")
    primary_image: Optional[str] = Field(default=None, description="Primary product image URL")
    
    # Search explanation
    explanation: Optional[str] = Field(default=None, description="Why this product was recommended")


class ProductMetadata(BaseModel):
    """Metadata about product processing and quality."""
    
    parent_asin: str = Field(..., description="Product identifier")
    quality_score: float = Field(..., description="Quality score (0-1)")
    has_images: bool = Field(..., description="Has product images")
    has_videos: bool = Field(..., description="Has product videos")
    has_ratings: bool = Field(..., description="Has customer ratings")
    has_price: bool = Field(..., description="Has price information")
    feature_count: int = Field(..., description="Number of features")
    description_length: int = Field(..., description="Length of description text")
    embedding_text_length: int = Field(..., description="Length of text used for embedding")
    categories_count: int = Field(..., description="Number of categories")


class DatasetStats(BaseModel):
    """Statistics about the product dataset."""
    
    total_products: int = Field(..., description="Total number of products")
    products_with_images: int = Field(..., description="Products with images")
    products_with_ratings: int = Field(..., description="Products with ratings") 
    products_with_price: int = Field(..., description="Products with price")
    
    # Quality distribution
    high_quality_products: int = Field(..., description="High quality products (score > 0.8)")
    medium_quality_products: int = Field(..., description="Medium quality products (0.5-0.8)")
    low_quality_products: int = Field(..., description="Low quality products (< 0.5)")
    
    # Category statistics
    unique_categories: int = Field(..., description="Number of unique categories")
    top_categories: List[str] = Field(default=[], description="Most common categories")
    
    # Price and rating statistics
    price_stats: dict = Field(default={}, description="Price statistics (min, max, mean, std)")
    rating_stats: dict = Field(default={}, description="Rating statistics")
    
    # Processing statistics
    total_tokens: Optional[int] = Field(default=None, description="Total tokens for embeddings")
    average_tokens_per_product: Optional[float] = Field(default=None, description="Average tokens per product")
    estimated_embedding_cost: Optional[float] = Field(default=None, description="Estimated cost for embeddings") 