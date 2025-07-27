"""Shared utilities for the fashion search system."""

from .logging import setup_logger
from .text_processing import (
    clean_text, 
    prepare_embedding_text, 
    extract_keywords,
    calculate_text_quality_score,
    normalize_brand_name,
    is_fashion_related
)

__all__ = [
    "setup_logger",
    "clean_text",
    "prepare_embedding_text",
    "extract_keywords",
    "calculate_text_quality_score",
    "normalize_brand_name",
    "is_fashion_related"
] 