"""Shared data models for the Fashion Search Engine."""

from .base import BaseModel
from .config import Settings
from .product import Product, ProductResult, ProductMetadata
from .search import SearchRequest, SearchResponse, SearchResult

__all__ = [
    "BaseModel",
    "Settings", 
    "Product",
    "ProductResult",
    "ProductMetadata",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]