"""
Refactored Search Engine

Clean, modular search engine that orchestrates specialized components:
- VectorSearchManager for FAISS operations
- LLMProcessor for query enhancement and filter extraction
- FilterManager for product filtering with graceful degradation
- RankingManager for intelligent product ranking

This replaces the monolithic SearchEngine with a clean, maintainable architecture.
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from shared.models import Settings, ProductResult, SearchRequest
from .vector_search import VectorSearchManager
from .llm_integration import LLMProcessor
from .filtering import FilterManager
from .ranking import RankingManager

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Refactored search engine with modular architecture.
    
    This orchestrates specialized components for a clean separation of concerns:
    - Vector search operations
    - LLM integration
    - Product filtering
    - Ranking algorithms
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized components
        self.vector_search = VectorSearchManager(settings)
        self.llm_processor = LLMProcessor(settings)
        self.filter_manager = FilterManager()
        self.ranking_manager = RankingManager()
        
        # Data storage
        self.products_df: Optional[pd.DataFrame] = None
        self.product_ids: Optional[List[str]] = None
        
        # Performance tracking
        self.total_searches = 0
        self.total_search_time = 0.0
    
    def initialize(self, embeddings: np.ndarray, products_df: pd.DataFrame, product_ids: List[str]) -> None:
        """Initialize the search engine with embeddings and product data."""
        
        self.logger.info("Initializing modular search engine...")
        
        # Store product data
        self.products_df = products_df
        self.product_ids = product_ids
        
        # Initialize vector search component
        self.vector_search.initialize(embeddings)
        
        # Log initialization details
        filter_columns = [col for col in products_df.columns if col.startswith('filter_')]
        self.logger.info(f"Available filter columns: {filter_columns}")
        self.logger.info(f"Search engine initialized with {len(products_df)} products")
    
    async def search(self, request: SearchRequest) -> Tuple[List[ProductResult], Dict[str, Any]]:
        """
        Perform intelligent search with modular processing pipeline.
        
        Pipeline:
        1. LLM query processing (enhancement + filter extraction)
        2. Vector similarity search
        3. Candidate product conversion
        4. Intelligent filtering with graceful degradation
        5. Ranking (heuristic or LLM-based)
        6. Result limitation and metadata collection
        """
        
        start_time = time.time()
        self.total_searches += 1
        
        try:
            # Step 1: Process query with LLM enhancement and filter extraction
            self.logger.debug(f"Processing query: {request.query}")
            enhanced_query, extracted_filters = await self.llm_processor.process_search_query(request.query)
            
            # Step 2: Generate query embedding
            query_embedding = await self.llm_processor.generate_query_embedding(enhanced_query)
            
            # Step 3: Vector similarity search
            candidate_count = min(request.top_k * 3, 100)  # Get more candidates for filtering
            similarities, indices = self.vector_search.search(query_embedding, candidate_count)
            
            # Step 4: Convert to ProductResult objects
            candidate_products = self._convert_to_product_results(similarities, indices)
            
            # Step 5: Apply intelligent filtering with graceful degradation
            filtered_products = self.filter_manager.apply_filters(
                candidate_products, request, extracted_filters
            )
            
            # Step 6: Apply intelligent ranking
            if request.reranking_method == "llm" and len(filtered_products) <= 20:
                # Use LLM reranking for small result sets
                final_products = await self.llm_processor.rerank_with_llm(
                    filtered_products, request.query, max_products=10
                )
                ranking_method = "llm"
            else:
                # Use heuristic ranking
                final_products = self.ranking_manager.rank_products(
                    filtered_products, request.query, method="heuristic"
                )
                ranking_method = "heuristic"
            
            # Step 7: Limit to requested number
            final_products = final_products[:request.top_k]
            
            search_time = time.time() - start_time
            self.total_search_time += search_time
            
            # Collect comprehensive metadata
            metadata = {
                'enhanced_query': enhanced_query,
                'extracted_filters': extracted_filters,
                'search_time_ms': search_time * 1000,
                'candidates_found': len(candidate_products),
                'after_filtering': len(filtered_products),
                'ranking_method': ranking_method,
                'vector_search_time_ms': self.vector_search.total_search_time * 1000 / max(1, self.vector_search.search_count)
            }
            
            self.logger.info(
                f"Search completed in {search_time:.3f}s: "
                f"{len(candidate_products)} candidates → {len(filtered_products)} filtered → {len(final_products)} final"
            )
            
            return final_products, metadata
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}", exc_info=True)
            raise
    
    def _convert_to_product_results(self, similarities: np.ndarray, indices: np.ndarray) -> List[ProductResult]:
        """Convert search results to ProductResult objects."""
        
        if self.products_df is None:
            raise ValueError("Products data not loaded")
        
        products = []
        
        for sim, idx in zip(similarities, indices):
            if idx == -1:  # Invalid index
                continue
            
            try:
                product_row = self.products_df.iloc[idx]
                
                # Get filter metadata for this product
                filter_metadata = {}
                for col in self.products_df.columns:
                    if col.startswith('filter_') and pd.notna(product_row[col]):
                        filter_name = col.replace('filter_', '')
                        filter_metadata[filter_name] = product_row[col]
                
                # Create ProductResult with proper null handling
                product = ProductResult(
                    parent_asin=product_row['parent_asin'],
                    title=self._safe_get_string(product_row.get('title')),
                    main_category=self._safe_get_string(product_row.get('main_category')),
                    price=self._safe_get_float(product_row.get('price')),
                    average_rating=self._safe_get_float(product_row.get('average_rating')),
                    rating_number=self._safe_get_int(product_row.get('rating_number')),
                    similarity_score=float(np.clip(sim, 0.0, 1.0)),  # Ensure valid range
                    features=self._safe_get_list(product_row.get('features')),
                    description=self._safe_get_list(product_row.get('description')),
                    store=self._safe_get_string(product_row.get('store')),
                    categories=self._safe_get_list(product_row.get('categories')),
                    images=self._safe_get_list(product_row.get('images')),
                    matched_filters=filter_metadata
                )
                
                products.append(product)
                
            except Exception as e:
                self.logger.warning(f"Error processing product at index {idx}: {e}")
                continue
        
        return products
    
    def _safe_get_string(self, value) -> Optional[str]:
        """Safely convert value to string, handling pandas NaN."""
        if value is None or pd.isna(value):
            return None
        return str(value)
    
    def _safe_get_float(self, value) -> Optional[float]:
        """Safely convert value to float, handling pandas NaN."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_get_int(self, value) -> Optional[int]:
        """Safely convert value to int, handling pandas NaN."""
        if value is None or pd.isna(value):
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_get_list(self, value) -> List:
        """Safely convert value to list, handling pandas arrays and NaN."""
        if value is None or pd.isna(value):
            return []
        
        if isinstance(value, (list, np.ndarray)):
            return list(value)
        
        # If it's a single value, wrap in list
        return [value]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive search engine statistics."""
        
        avg_search_time = 0.0
        if self.total_searches > 0:
            avg_search_time = (self.total_search_time / self.total_searches) * 1000
        
        # Combine stats from all components
        stats = {
            'total_searches': self.total_searches,
            'avg_search_time_ms': avg_search_time,
            'total_products': len(self.products_df) if self.products_df is not None else 0,
            'embeddings_loaded': self.vector_search.embeddings is not None,
            'index_ready': self.vector_search.index is not None,
        }
        
        # Add component-specific stats
        stats.update({
            'vector_search': self.vector_search.get_stats(),
            'llm_processor': self.llm_processor.get_stats(),
            'filter_manager': self.filter_manager.get_stats(),
            'ranking_manager': self.ranking_manager.get_stats()
        })
        
        return stats
    
    def is_ready(self) -> bool:
        """Check if search engine is ready for queries."""
        return all([
            self.vector_search.is_ready(),
            self.products_df is not None,
            self.product_ids is not None
        ])
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save the FAISS index to disk."""
        if path is None:
            path = str(Path(self.settings.embeddings_path) / "faiss_index.index")
        
        self.vector_search.save_index(path)