"""Data Pipeline Service for Amazon Fashion Search Engine.

This service handles:
- Raw dataset analysis and validation
- Quality-based stratified sampling
- Text preparation and feature extraction
- OpenAI embedding generation with batching
- FAISS index creation and optimization
- Data caching and persistence
"""

__version__ = "1.0.0" 