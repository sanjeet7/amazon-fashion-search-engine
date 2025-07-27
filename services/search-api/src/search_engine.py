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
from shared.utils import extract_search_filters_with_llm, enhance_query_with_context, calculate_tokens


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
        
        # Build FAISS index
        self._build_faiss_index()
        
        self.logger.info(f"Search engine initialized with {len(products_df)} products")
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for efficient vector search."""
        
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")
        
        # Use IndexFlatIP for exact cosine similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        
        self.logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    async def search(self, request: SearchRequest) -> Tuple[List[ProductResult], Dict[str, Any]]:
        """Perform semantic search with intelligent query processing."""
        
        start_time = time.time()
        
        try:
            # Step 1: Enhance query and extract filters
            enhanced_query, extracted_filters = await self._process_query(request.query)
            
            # Step 2: Generate query embedding
            query_embedding = await self._get_query_embedding(enhanced_query)
            
            # Step 3: Vector search
            similarities, indices = self._vector_search(query_embedding, request.top_k * 2)  # Get more for filtering
            
            # Step 4: Convert to products and apply filters
            candidate_products = self._get_candidate_products(similarities, indices)
            
            # Step 5: Apply filters
            filtered_products = self._apply_filters(candidate_products, request, extracted_filters)
            
            # Step 6: Intelligent ranking
            ranked_products = self._rank_products(filtered_products, request.query)
            
            # Step 7: Limit results
            final_products = ranked_products[:request.top_k]
            
            search_time = time.time() - start_time
            self.total_searches += 1
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
        
        # For prototype, use simple rule-based processing
        # In production, this would use GPT-4 for intelligent processing
        enhanced_query = enhance_query_with_context(query)
        extracted_filters = extract_search_filters_with_llm(query)
        
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
            
            product = ProductResult(
                parent_asin=product_row['parent_asin'],
                title=product_row['title'],
                main_category=product_row.get('main_category'),
                price=product_row.get('price'),
                average_rating=product_row.get('average_rating'),
                rating_number=product_row.get('rating_number'),
                similarity_score=float(sim),
                features=product_row.get('features', []),
                description=product_row.get('description', []),
                store=product_row.get('store'),
                categories=product_row.get('categories', []),
                images=product_row.get('images', [])
            )
            
            products.append(product)
        
        return products
    
    def _apply_filters(self, products: List[ProductResult], request: SearchRequest, extracted_filters: Dict[str, Any]) -> List[ProductResult]:
        """Apply search filters to product results."""
        
        filtered = []
        
        for product in products:
            # Similarity threshold
            if product.similarity_score < request.min_similarity:
                continue
            
            # Price filters
            if request.price_min is not None and (product.price is None or product.price < request.price_min):
                continue
            if request.price_max is not None and (product.price is None or product.price > request.price_max):
                continue
            
            # Extracted price filter (overrides request filter)
            if 'price_max' in extracted_filters and (product.price is None or product.price > extracted_filters['price_max']):
                continue
            
            # Category filter
            if request.category and product.main_category and request.category.lower() not in product.main_category.lower():
                continue
            
            # Extracted category filter
            if 'category' in extracted_filters:
                category_match = False
                search_category = extracted_filters['category'].lower()
                
                # Check main category
                if product.main_category and search_category in product.main_category.lower():
                    category_match = True
                
                # Check all categories
                for cat in product.categories:
                    if search_category in cat.lower():
                        category_match = True
                        break
                
                if not category_match:
                    continue
            
            # Rating filter
            if request.min_rating is not None and (product.average_rating is None or product.average_rating < request.min_rating):
                continue
            
            filtered.append(product)
        
        return filtered
    
    def _rank_products(self, products: List[ProductResult], query: str) -> List[ProductResult]:
        """Apply intelligent ranking based on multiple signals."""
        
        # For this prototype, we'll use a simple weighted ranking
        # In production, this could be much more sophisticated
        
        for product in products:
            # Start with similarity score
            score = product.similarity_score
            
            # Boost products with ratings
            if product.average_rating is not None:
                rating_boost = (product.average_rating - 3.0) * 0.1  # Boost high-rated products
                score += max(0, rating_boost)
            
            # Boost products with more reviews (popularity signal)
            if product.rating_number is not None and product.rating_number > 0:
                review_boost = min(0.1, product.rating_number / 1000)  # Cap boost at 0.1
                score += review_boost
            
            # Boost products with price information
            if product.price is not None:
                score += 0.05
            
            # Store the final ranking score
            product.similarity_score = score
        
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