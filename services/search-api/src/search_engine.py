"""Search engine with vector similarity and intelligent ranking."""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import faiss
from openai import AsyncOpenAI

from shared.models import Settings, ProductResult, SearchRequest
from shared.utils import extract_search_filters_with_llm, enhance_query_with_context, calculate_tokens, validate_extracted_filters


logger = logging.getLogger(__name__)


class SearchEngine:
    """Advanced search engine with vector similarity and intelligent ranking."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Search components
        self.index: Optional[faiss.Index] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.product_ids: Optional[List[str]] = None
        
        # Performance tracking
        self.total_searches = 0
        self.total_search_time = 0.0
    
    def initialize(self, embeddings: np.ndarray, products_df: pd.DataFrame, product_ids: List[str]) -> None:
        """Initialize the search engine with embeddings and product data."""
        
        self.logger.info("Initializing search engine...")
        
        self.embeddings = embeddings.astype(np.float32)
        self.products_df = products_df
        self.product_ids = product_ids
        
        # Try to load pre-built FAISS index first
        self._load_or_build_faiss_index()
        
        # Log available filter columns
        filter_columns = [col for col in products_df.columns if col.startswith('filter_')]
        self.logger.info(f"Available filter columns: {filter_columns}")
        
        self.logger.info(f"Search engine initialized with {len(products_df)} products")
    
    def _load_or_build_faiss_index(self) -> None:
        """Load pre-built FAISS index or build one if it doesn't exist."""
        
        # Try to load pre-built index from embedding generator
        try:
            from services.data_pipeline.src.embedding_generator import EmbeddingGenerator
            embedding_gen = EmbeddingGenerator(self.settings)
            
            if embedding_gen.faiss_index_exists():
                self.logger.info("Loading pre-built FAISS index...")
                self.index = embedding_gen.load_faiss_index()
                self.logger.info("✅ Loaded pre-built FAISS index successfully")
                return
        except Exception as e:
            self.logger.warning(f"Could not load pre-built FAISS index: {e}")
        
        # Fallback: build index from embeddings
        self.logger.info("Building FAISS index from embeddings...")
        self._build_faiss_index()
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for efficient similarity search."""
        
        dimension = self.embeddings.shape[1]
        
        # Use IVF (Inverted File) index for better performance with larger datasets
        nlist = min(100, max(1, len(self.embeddings) // 50))  # Adaptive nlist
        quantizer = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Train and add embeddings
        self.index.train(self.embeddings)
        self.index.add(self.embeddings)
        
        # Set search parameters
        self.index.nprobe = min(10, nlist)  # Number of clusters to search
        
        self.logger.info(f"Built FAISS index with {self.index.ntotal} vectors, {nlist} clusters")
    
    async def search(self, request: SearchRequest) -> Tuple[List[ProductResult], Dict[str, Any]]:
        """Perform intelligent search with filtering and ranking."""
        
        start_time = time.time()
        self.total_searches += 1
        
        try:
            # Step 1: Process query with LLM enhancement and filter extraction
            enhanced_query, extracted_filters = await self._process_query(request.query)
            
            # Step 2: Generate query embedding
            query_embedding = await self._get_query_embedding(enhanced_query)
            
            # Step 3: Vector similarity search
            similarities, indices = self._vector_search(query_embedding, request.top_k * 3)  # Get more candidates for filtering
            
            # Step 4: Convert to ProductResult objects
            candidate_products = self._get_candidate_products(similarities, indices)
            
            # Step 5: Apply filters using standardized values
            filtered_products = self._apply_standardized_filters(candidate_products, request, extracted_filters)
            
            # Step 6: Apply intelligent ranking
            final_products = self._rank_products(filtered_products, request.query)
            
            # Step 7: Limit to requested number
            final_products = final_products[:request.top_k]
            
            search_time = time.time() - start_time
            self.total_search_time += search_time
            
            # Search metadata
            metadata = {
                'enhanced_query': enhanced_query,
                'extracted_filters': extracted_filters,
                'search_time_ms': search_time * 1000,
                'candidates_found': len(candidate_products),
                'after_filtering': len(filtered_products)
            }
            
            self.logger.info(f"Search completed in {search_time:.3f}s, found {len(final_products)} results")
            
            return final_products, metadata
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    async def _process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process query with LLM enhancement and filter extraction."""
        
        # Extract filters using GPT-4.1-mini with standardized values
        extracted_filters = await extract_search_filters_with_llm(query, self.client)
        
        # Enhance query with fashion context
        enhanced_query = await enhance_query_with_context(query, self.client)
        
        return enhanced_query, extracted_filters
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for search query."""
        
        try:
            response = await self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=[query]
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise
    
    def _vector_search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform vector similarity search."""
        
        if self.index is None:
            raise ValueError("Search index not initialized")
        
        similarities, indices = self.index.search(query_embedding, k)
        
        return similarities[0], indices[0]
    
    def _get_candidate_products(self, similarities: np.ndarray, indices: np.ndarray) -> List[ProductResult]:
        """Convert search results to ProductResult objects."""
        
        if self.products_df is None:
            raise ValueError("Products data not loaded")
        
        products = []
        
        for sim, idx in zip(similarities, indices):
            if idx == -1:  # Invalid index
                continue
            
            product_row = self.products_df.iloc[idx]
            
            # Get filter metadata for this product
            filter_metadata = {}
            for col in self.products_df.columns:
                if col.startswith('filter_') and pd.notna(product_row[col]):
                    filter_name = col.replace('filter_', '')
                    filter_metadata[filter_name] = product_row[col]
            
            # Handle nan values properly for Pydantic validation
            def safe_get(value):
                """Convert nan values to None for Pydantic."""
                if pd.isna(value):
                    return None
                return value
            
            product = ProductResult(
                parent_asin=product_row['parent_asin'],
                title=product_row['title'],
                main_category=safe_get(product_row.get('main_category')),
                price=safe_get(product_row.get('price')),
                average_rating=safe_get(product_row.get('average_rating')),
                rating_number=safe_get(product_row.get('rating_number')),
                similarity_score=min(max(float(sim), 0.0), 1.0),  # Clamp to [0, 1] range
                features=self._safe_get_list(product_row.get('features')),
                description=self._safe_get_list(product_row.get('description')),
                store=safe_get(product_row.get('store')),
                categories=self._safe_get_list(product_row.get('categories')),
                images=self._safe_get_list(product_row.get('images')),
                matched_filters=filter_metadata
            )
            
            products.append(product)
        
        return products
    
    def _safe_get_list(self, value) -> List:
        """Safely convert value to list, handling pandas arrays and nan values."""
        try:
            if value is None or (hasattr(value, '__len__') and len(value) == 0):
                return []
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            # pd.isna() might fail on complex objects
            if value is None:
                return []
        
        if isinstance(value, (list, np.ndarray)):
            return list(value)
        
        # If it's a single value, wrap in list
        return [value]
    
    def _apply_standardized_filters(self, products: List[ProductResult], request: SearchRequest, extracted_filters: Dict[str, Any]) -> List[ProductResult]:
        """Apply search filters using standardized values from ingestion."""
        
        filtered = []
        
        for product in products:
            # Similarity threshold
            if product.similarity_score < request.min_similarity:
                continue
            
            # Apply request-based filters
            if not self._check_request_filters(product, request):
                continue
            
            # Apply LLM-extracted filters using standardized values
            if not self._check_extracted_filters(product, extracted_filters):
                continue
            
            filtered.append(product)
        
        return filtered
    
    def _check_request_filters(self, product: ProductResult, request: SearchRequest) -> bool:
        """Check request-based filters."""
        
        # Price filters
        if request.price_min is not None and (product.price is None or product.price < request.price_min):
            return False
        if request.price_max is not None and (product.price is None or product.price > request.price_max):
            return False
        
        # Category filter (case-insensitive matching)
        if request.category:
            category_match = False
            search_category = request.category.lower()
            
            # Check main category
            if product.main_category and search_category in product.main_category.lower():
                category_match = True
            
            # Check all categories
            for cat in product.categories:
                if search_category in cat.lower():
                    category_match = True
                    break
            
            if not category_match:
                return False
        
        # Rating filter
        if request.min_rating is not None and (product.average_rating is None or product.average_rating < request.min_rating):
            return False
        
        return True
    
    def _check_extracted_filters(self, product: ProductResult, extracted_filters: Dict[str, Any]) -> bool:
        """Check LLM-extracted filters using standardized metadata."""
        
        if not extracted_filters or not product.matched_filters:
            return True  # No filters to apply
        
        # Check each extracted filter against product's standardized filter metadata
        for filter_name, filter_value in extracted_filters.items():
            if filter_name == 'price_range':
                # Handle price range filter
                if 'min' in filter_value and (product.price is None or product.price < filter_value['min']):
                    return False
                if 'max' in filter_value and (product.price is None or product.price > filter_value['max']):
                    return False
                continue
            
            # Check standardized filter values
            product_filter_value = product.matched_filters.get(filter_name)
            if product_filter_value is None:
                continue  # Product doesn't have this filter, but don't exclude (inclusive approach)
            
            # Exact match for standardized values
            if isinstance(filter_value, str) and isinstance(product_filter_value, str):
                if filter_value.lower() != product_filter_value.lower():
                    return False
            elif filter_value != product_filter_value:
                return False
        
        return True
    
    def _rank_products(self, products: List[ProductResult], query: str) -> List[ProductResult]:
        """Apply intelligent ranking based on multiple signals."""
        
        for product in products:
            # Start with similarity score (primary signal)
            score = product.similarity_score
            
            # Business signal boosts (as per final_exploration.md strategy)
            if product.average_rating is not None:
                # Rating boost: scale around 3.0 (neutral)
                rating_boost = (product.average_rating - 3.0) * 0.15  
                score += max(0, rating_boost)
            
            # Review count boost (popularity signal with logarithmic scaling)
            if product.rating_number is not None and product.rating_number > 0:
                import math
                log_reviews = math.log(product.rating_number + 1)
                review_boost = min(0.1, log_reviews / 50)  # Cap at 0.1, scale to reasonable range
                score += review_boost
            
            # Completeness boost (products with more data)
            completeness_boost = 0
            if product.price is not None:
                completeness_boost += 0.02
            if product.features:
                completeness_boost += 0.02
            if product.images:
                completeness_boost += 0.01
            
            score += completeness_boost
            
            # Filter match boost (products matching extracted filters get slight boost)
            if product.matched_filters:
                filter_match_boost = len(product.matched_filters) * 0.01
                score += min(0.05, filter_match_boost)  # Cap at 0.05
            
            # Store the final ranking score (clamped to [0, 1] for validation)
            product.similarity_score = min(max(score, 0.0), 1.0)
        
        # Sort by final score
        return sorted(products, key=lambda p: p.similarity_score, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        
        return {
            'total_searches': self.total_searches,
            'avg_search_time_ms': (self.total_search_time / max(1, self.total_searches)) * 1000,
            'index_size': self.index.ntotal if self.index else 0,
            'total_products': len(self.products_df) if self.products_df is not None else 0,
            'embeddings_loaded': self.embeddings is not None,
            'index_ready': self.index is not None
        }
    
    def is_ready(self) -> bool:
        """Check if search engine is ready for queries."""
        return all([
            self.index is not None,
            self.products_df is not None,
            self.embeddings is not None,
            self.product_ids is not None
        ])