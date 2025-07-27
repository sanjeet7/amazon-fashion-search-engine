"""Core search engine for Amazon Fashion products using FAISS and hybrid ranking."""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from openai import AsyncOpenAI

from shared.models import (
    SearchAPIConfig, SearchRequest, SearchResponse, 
    SampleProductsRequest, SampleProductsResponse,
    Product, ProductResult, FilterCriteria
)
from shared.utils import setup_logger, prepare_embedding_text


class SearchEngine:
    """Core search engine with vector similarity and hybrid ranking."""
    
    def __init__(self, config: SearchAPIConfig, logger: logging.Logger):
        """Initialize search engine.
        
        Args:
            config: Search API configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Initialize OpenAI client for query enhancement
        self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        
        # Data storage
        self.products_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.product_id_to_idx: Dict[str, int] = {}
        
        # Performance metrics
        self.search_stats = {
            'total_searches': 0,
            'avg_response_time_ms': 0.0,
            'cache_hits': 0
        }
        
        self.logger.info("Initialized SearchEngine")

    async def initialize(self) -> None:
        """Initialize the search engine by loading processed data and index."""
        self.logger.info("Initializing search engine components...")
        
        try:
            # Load processed data
            await self._load_processed_data()
            
            # Load embeddings
            await self._load_embeddings()
            
            # Load FAISS index
            await self._load_index()
            
            # Build product ID mapping
            self._build_product_mapping()
            
            self.logger.info("Search engine initialization complete")
            self.logger.info(f"Loaded {len(self.products_df):,} products")
            
        except Exception as e:
            self.logger.error(f"Search engine initialization failed: {e}")
            raise

    async def _load_processed_data(self) -> None:
        """Load processed product data."""
        data_file = self.config.processed_data_dir / "processed_sample.parquet"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data not found at {data_file}")
        
        self.products_df = pd.read_parquet(data_file)
        self.logger.info(f"Loaded product data: {len(self.products_df):,} products")

    async def _load_embeddings(self) -> None:
        """Load embeddings from cache."""
        embeddings_file = self.config.embeddings_cache_dir / "embeddings.npz"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings not found at {embeddings_file}")
        
        data = np.load(embeddings_file, allow_pickle=True)
        self.embeddings = data['embeddings']
        
        self.logger.info(f"Loaded embeddings: {self.embeddings.shape}")

    async def _load_index(self) -> None:
        """Load FAISS index."""
        index_file = self.config.embeddings_cache_dir / "faiss_index.bin"
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        self.logger.info(f"Loaded FAISS index: {self.index.ntotal:,} vectors")

    def _build_product_mapping(self) -> None:
        """Build mapping from product ID to dataframe index."""
        self.product_id_to_idx = {
            row['parent_asin']: idx 
            for idx, row in self.products_df.iterrows()
        }
        self.logger.info(f"Built product mapping: {len(self.product_id_to_idx):,} products")

    async def search(
        self, 
        request: SearchRequest, 
        query_analysis: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """Perform semantic search with hybrid ranking.
        
        Args:
            request: Search request
            query_analysis: Optional query analysis from query processor
            
        Returns:
            Search response with ranked results
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(request.query)
            
            # Perform vector search
            candidate_indices, similarity_scores = await self._vector_search(
                query_embedding, request.top_k * 3  # Get more candidates for reranking
            )
            
            # Apply filters
            filtered_results = await self._apply_filters(
                candidate_indices, similarity_scores, request.filters
            )
            
            # Hybrid ranking
            ranked_results = await self._hybrid_ranking(
                filtered_results, request, query_analysis
            )
            
            # Format results
            final_results = ranked_results[:request.top_k]
            
            # Update stats
            response_time = time.time() - start_time
            self._update_search_stats(response_time)
            
            return SearchResponse(
                results=final_results,
                total_results=len(ranked_results),
                response_time_ms=response_time * 1000,
                query_analysis=query_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    async def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        try:
            response = await self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=query,
                encoding_format="float"
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding.reshape(1, -1))
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Query embedding generation failed: {e}")
            raise

    async def _vector_search(
        self, 
        query_embedding: np.ndarray, 
        k: int
    ) -> Tuple[List[int], List[float]]:
        """Perform vector similarity search.
        
        Args:
            query_embedding: Query vector
            k: Number of results to retrieve
            
        Returns:
            Tuple of (indices, similarity_scores)
        """
        # Reshape for FAISS
        query_vector = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_vector, k)
        
        # Convert to lists and filter valid results
        valid_results = [(idx, score) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        
        if not valid_results:
            return [], []
        
        indices_list, scores_list = zip(*valid_results)
        return list(indices_list), list(scores_list)

    async def _apply_filters(
        self, 
        indices: List[int], 
        scores: List[float], 
        filters: Optional[FilterCriteria]
    ) -> List[Tuple[int, float]]:
        """Apply filters to search results.
        
        Args:
            indices: Product indices
            scores: Similarity scores
            filters: Filter criteria
            
        Returns:
            List of (index, score) tuples after filtering
        """
        if not filters:
            return list(zip(indices, scores))
        
        filtered_results = []
        
        for idx, score in zip(indices, scores):
            product = self.products_df.iloc[idx]
            
            # Apply filters
            if self._passes_filters(product, filters):
                filtered_results.append((idx, score))
        
        return filtered_results

    def _passes_filters(self, product: pd.Series, filters: FilterCriteria) -> bool:
        """Check if product passes filter criteria.
        
        Args:
            product: Product data
            filters: Filter criteria
            
        Returns:
            True if product passes all filters
        """
        # Price filters
        if filters.price_min is not None:
            price = product.get('price')
            if price is None or price < filters.price_min:
                return False
        
        if filters.price_max is not None:
            price = product.get('price')
            if price is None or price > filters.price_max:
                return False
        
        # Rating filter
        if filters.min_rating is not None:
            rating = product.get('average_rating')
            if rating is None or rating < filters.min_rating:
                return False
        
        # Category filter
        if filters.category:
            categories = product.get('categories', [])
            if not any(filters.category.lower() in str(cat).lower() for cat in categories):
                return False
        
        # Brand filter
        if filters.brand:
            store = product.get('store', '')
            if filters.brand.lower() not in str(store).lower():
                return False
        
        return True

    async def _hybrid_ranking(
        self, 
        results: List[Tuple[int, float]], 
        request: SearchRequest,
        query_analysis: Optional[Dict[str, Any]]
    ) -> List[ProductResult]:
        """Apply hybrid ranking combining semantic similarity with business signals.
        
        Args:
            results: List of (index, similarity_score) tuples
            request: Search request
            query_analysis: Query analysis results
            
        Returns:
            List of ranked ProductResult objects
        """
        ranked_products = []
        
        for idx, similarity_score in results:
            product = self.products_df.iloc[idx]
            
            # Calculate business signals
            rating_signal = self._calculate_rating_signal(product)
            review_signal = self._calculate_review_signal(product)
            
            # Calculate final score (weights from analysis)
            final_score = (
                similarity_score * 0.6 +           # Semantic similarity (60%)
                rating_signal * 0.25 +             # Rating signal (25%)
                review_signal * 0.15               # Review count signal (15%)
            )
            
            # Create product result
            product_result = self._create_product_result(product, similarity_score, final_score)
            ranked_products.append(product_result)
        
        # Sort by final score
        ranked_products.sort(key=lambda x: x.final_score, reverse=True)
        
        return ranked_products

    def _calculate_rating_signal(self, product: pd.Series) -> float:
        """Calculate rating-based ranking signal.
        
        Args:
            product: Product data
            
        Returns:
            Normalized rating signal (0-1)
        """
        rating = product.get('average_rating')
        if rating is None:
            return 0.0
        
        # Normalize to 0-1 scale (assuming 5-star rating system)
        return min(rating / 5.0, 1.0)

    def _calculate_review_signal(self, product: pd.Series) -> float:
        """Calculate review count-based ranking signal.
        
        Args:
            product: Product data
            
        Returns:
            Normalized review signal (0-1)
        """
        review_count = product.get('rating_number', 0)
        if review_count <= 0:
            return 0.0
        
        # Log transformation to handle wide range
        import math
        log_reviews = math.log(1 + review_count)
        
        # Normalize based on analysis (max ~10K reviews)
        max_log_reviews = math.log(10000)
        return min(log_reviews / max_log_reviews, 1.0)

    def _create_product_result(
        self, 
        product: pd.Series, 
        similarity_score: float, 
        final_score: float
    ) -> ProductResult:
        """Create ProductResult from product data.
        
        Args:
            product: Product pandas Series
            similarity_score: Semantic similarity score
            final_score: Final hybrid ranking score
            
        Returns:
            ProductResult object
        """
        # Extract and clean data
        title = str(product.get('title', '')).strip()
        price = product.get('price')
        rating = product.get('average_rating')
        review_count = product.get('rating_number')
        store = str(product.get('store', '')).strip()
        
        # Handle features
        features = product.get('features', [])
        if isinstance(features, list):
            features_list = [str(f).strip() for f in features if f]
        else:
            features_list = []
        
        # Handle images
        images = product.get('processed_images', [])
        primary_image = product.get('primary_image')
        
        # Handle videos
        videos = product.get('processed_videos', [])
        
        return ProductResult(
            parent_asin=str(product.get('parent_asin', '')),
            title=title,
            price=float(price) if price is not None else None,
            average_rating=float(rating) if rating is not None else None,
            rating_number=int(review_count) if review_count is not None else None,
            similarity_score=float(similarity_score),
            final_score=float(final_score),
            store=store,
            features=features_list,
            images=images if isinstance(images, list) else [],
            videos=videos if isinstance(videos, list) else [],
            primary_image=str(primary_image) if primary_image else None
        )

    async def get_sample_products(self, request: SampleProductsRequest) -> SampleProductsResponse:
        """Get sample products for frontend development.
        
        Args:
            request: Sample products request
            
        Returns:
            Sample products response
        """
        # Start with all products
        df = self.products_df.copy()
        
        # Apply filters
        if request.with_images_only:
            df = df[df['processed_images'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
        
        if request.category_filter:
            df = df[df['semantic_category'] == request.category_filter]
        
        if request.min_rating is not None:
            df = df[df['average_rating'] >= request.min_rating]
        
        # Sample products
        sample_size = min(request.limit, len(df))
        if sample_size == 0:
            return SampleProductsResponse(products=[], total_available=0)
        
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        # Convert to Product objects
        products = []
        for _, product in sampled_df.iterrows():
            products.append(self._create_product_from_series(product))
        
        return SampleProductsResponse(
            products=products,
            total_available=len(df)
        )

    def _create_product_from_series(self, product: pd.Series) -> Product:
        """Create Product object from pandas Series.
        
        Args:
            product: Product pandas Series
            
        Returns:
            Product object
        """
        return Product(
            parent_asin=str(product.get('parent_asin', '')),
            title=str(product.get('title', '')).strip(),
            price=float(product.get('price')) if product.get('price') is not None else None,
            average_rating=float(product.get('average_rating')) if product.get('average_rating') is not None else None,
            rating_number=int(product.get('rating_number')) if product.get('rating_number') is not None else None,
            store=str(product.get('store', '')).strip(),
            features=product.get('features', []) if isinstance(product.get('features'), list) else [],
            images=product.get('processed_images', []) if isinstance(product.get('processed_images'), list) else [],
            videos=product.get('processed_videos', []) if isinstance(product.get('processed_videos'), list) else [],
            categories=product.get('categories', []) if isinstance(product.get('categories'), list) else []
        )

    def _update_search_stats(self, response_time: float) -> None:
        """Update search performance statistics.
        
        Args:
            response_time: Response time in seconds
        """
        self.search_stats['total_searches'] += 1
        
        # Update rolling average
        current_avg = self.search_stats['avg_response_time_ms']
        new_avg = ((current_avg * (self.search_stats['total_searches'] - 1)) + 
                  (response_time * 1000)) / self.search_stats['total_searches']
        self.search_stats['avg_response_time_ms'] = new_avg

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self.openai_client, 'close'):
            await self.openai_client.close()
        
        self.logger.info("Search engine cleanup complete") 