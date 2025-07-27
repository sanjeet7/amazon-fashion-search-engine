"""Hybrid semantic search with structured filtering for Amazon Fashion products.

This module implements the complete search pipeline including:
- FAISS vector similarity search
- Structured filter processing (Brand, Category, Quality, Price)
- Adaptive ranking with business signals
- Multi-intent query processing with GPT-4.1
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import re

import numpy as np
import pandas as pd
import faiss
from openai import OpenAI

from .config import settings
from .embedding import EmbeddingManager, create_embedding_manager
from .models import (
    ProductResult, QueryAnalysis, SearchResponse, QueryIntent, Season, Occasion,
    FilterCriteria, SearchConfig, RankingSignals
)

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid semantic and structured search engine for fashion products."""
    
    def __init__(
        self, 
        config: Optional[SearchConfig] = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        """Initialize search engine.
        
        Args:
            config: Search configuration
            embedding_manager: Embedding manager instance
        """
        self.config = config or SearchConfig()
        self.embedding_manager = embedding_manager or create_embedding_manager()
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Search components
        self.faiss_index: Optional[faiss.Index] = None
        self.product_df: Optional[pd.DataFrame] = None
        self.product_embeddings: Optional[np.ndarray] = None
        self.product_ids: List[str] = []
        
        # Filter mappings for normalization
        self.brand_normalizer = {}
        self.category_patterns = {
            'dress': ['dress', 'gown'],
            'tops': ['shirt', 'top', 'blouse', 'sweater', 'jacket'],
            'bottoms': ['pants', 'jeans', 'shorts', 'skirt'],
            'shoes': ['shoe', 'boot', 'sneaker', 'heel', 'sandal'],
            'accessories': ['bag', 'belt', 'hat', 'jewelry', 'watch']
        }
        
        logger.info("Initialized HybridSearchEngine")

    def build_index(
        self, 
        embeddings: np.ndarray, 
        product_df: pd.DataFrame,
        product_ids: List[str]
    ) -> None:
        """Build FAISS index for vector search.
        
        Args:
            embeddings: Product embeddings matrix
            product_df: Product metadata DataFrame
            product_ids: List of product IDs matching embedding order
        """
        logger.info(f"Building FAISS index for {len(embeddings):,} products...")
        
        # Store data
        self.product_embeddings = embeddings.astype(np.float32)
        self.product_df = product_df.copy()
        self.product_ids = product_ids.copy()
        
        # Build index based on configuration
        embedding_dim = embeddings.shape[1]
        
        if self.config.faiss_index_type == "IndexFlatIP":
            # Flat index with inner product (fastest, exact)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        elif self.config.faiss_index_type == "IndexIVFFlat":
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, 
                embedding_dim, 
                min(self.config.n_clusters, len(embeddings) // 10)
            )
        else:
            # Default to flat index
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.product_embeddings)
        
        # Add embeddings to index
        if hasattr(self.faiss_index, 'train'):
            self.faiss_index.train(self.product_embeddings)
        
        self.faiss_index.add(self.product_embeddings)
        
        # Build brand normalizer
        self._build_brand_normalizer()
        
        logger.info(f"FAISS index built successfully: {self.faiss_index.ntotal} vectors")

    def _build_brand_normalizer(self) -> None:
        """Build brand name normalizer for consistent filtering."""
        if self.product_df is None:
            return
        
        brands = self.product_df['store'].dropna().unique()
        
        for brand in brands:
            # Normalize brand name
            normalized = brand.lower().strip()
            
            # Remove common suffixes
            suffixes = [' inc', ' inc.', ' llc', ' llc.', ' ltd', ' ltd.', ' corp', ' corp.']
            for suffix in suffixes:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)].strip()
            
            self.brand_normalizer[normalized] = brand
        
        logger.info(f"Built brand normalizer with {len(self.brand_normalizer)} entries")

    def _normalize_brand_name(self, brand_query: str) -> Optional[str]:
        """Normalize brand name for filtering.
        
        Args:
            brand_query: Brand name from query
            
        Returns:
            Normalized brand name or None
        """
        if not brand_query:
            return None
        
        normalized_query = brand_query.lower().strip()
        
        # Direct match
        if normalized_query in self.brand_normalizer:
            return self.brand_normalizer[normalized_query]
        
        # Fuzzy match
        for normalized_brand, original_brand in self.brand_normalizer.items():
            if normalized_query in normalized_brand or normalized_brand in normalized_query:
                return original_brand
        
        return None

    def extract_semantic_category(self, text: str) -> Optional[str]:
        """Extract semantic category from text.
        
        Args:
            text: Input text
            
        Returns:
            Semantic category or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        for category, patterns in self.category_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return category
        
        return None

    def enhance_query_with_llm(self, query: str) -> Tuple[str, QueryAnalysis]:
        """Enhance query using GPT-4.1 for better semantic search.
        
        Args:
            query: Original user query
            
        Returns:
            Tuple of (enhanced_query, query_analysis)
        """
        try:
            # Prompt for query enhancement and analysis
            prompt = f"""Analyze this fashion search query and extract structured information:

Query: "{query}"

Please provide:
1. Enhanced query for semantic search (more descriptive, fashion-focused)
2. Extracted filters: brand, category, price range, season, occasion
3. Query intent classification

Respond in JSON format:
{{
    "enhanced_query": "descriptive text for embedding search",
    "filters": {{
        "brand": "brand name or null",
        "category": "dress/tops/bottoms/shoes/accessories or null", 
        "price_min": number or null,
        "price_max": number or null,
        "season": "spring/summer/fall/winter or null",
        "occasion": "work/casual/formal/party/athletic/wedding/date/travel or null"
    }},
    "intent": "specific_item/style_search/occasion_based/seasonal/general",
    "confidence": 0.0-1.0
}}"""

            response = self.openai_client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": "You are a fashion search expert. Extract relevant information from user queries to improve fashion product search."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
                
                enhanced_query = analysis_data.get('enhanced_query', query)
                filters_data = analysis_data.get('filters', {})
                
                # Create query analysis
                query_analysis = QueryAnalysis(
                    original_query=query,
                    enhanced_query=enhanced_query,
                    intent=QueryIntent(analysis_data.get('intent', 'general')),
                    season=Season(filters_data.get('season')) if filters_data.get('season') else None,
                    occasion=Occasion(filters_data.get('occasion')) if filters_data.get('occasion') else None,
                    extracted_attributes=filters_data
                )
                
                return enhanced_query, query_analysis
            else:
                # Fallback if JSON parsing fails
                return query, QueryAnalysis(
                    original_query=query,
                    enhanced_query=query,
                    intent=QueryIntent.GENERAL,
                    extracted_attributes={}
                )
                
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return query, QueryAnalysis(
                original_query=query,
                enhanced_query=query,
                intent=QueryIntent.GENERAL,
                extracted_attributes={}
            )

    def extract_filter_criteria(self, query_analysis: QueryAnalysis) -> FilterCriteria:
        """Extract structured filter criteria from query analysis.
        
        Args:
            query_analysis: Query analysis results
            
        Returns:
            Filter criteria
        """
        attributes = query_analysis.extracted_attributes
        
        # Extract and normalize brand
        brand_raw = attributes.get('brand')
        brand_normalized = self._normalize_brand_name(brand_raw) if brand_raw else None
        
        # Extract semantic category
        category = attributes.get('category')
        if not category and query_analysis.enhanced_query:
            category = self.extract_semantic_category(query_analysis.enhanced_query)
        
        # Map quality tiers - we'll derive this from rating/review requirements
        quality_tier = None
        if 'quality' in query_analysis.original_query.lower():
            if any(term in query_analysis.original_query.lower() for term in ['high quality', 'best', 'top rated']):
                quality_tier = 'high'
            elif any(term in query_analysis.original_query.lower() for term in ['good', 'decent']):
                quality_tier = 'medium'
        
        return FilterCriteria(
            brand_store=brand_normalized,
            semantic_category=category,
            price_min=attributes.get('price_min'),
            price_max=attributes.get('price_max'),
            quality_tier=quality_tier,
            season=query_analysis.season,
            occasion=query_analysis.occasion
        )

    def calculate_business_signals(self, product_row: pd.Series) -> RankingSignals:
        """Calculate business ranking signals for a product.
        
        Based on analysis findings:
        - Rating CV: 0.252 (moderate discriminative power)
        - Log Review Count CV: 0.579 (high discriminative power)
        
        Args:
            product_row: Product data row
            
        Returns:
            Ranking signals
        """
        signals = RankingSignals(semantic_score=0.0)
        
        # Rating signal (normalized 0-1)
        rating = product_row.get('average_rating')
        if pd.notna(rating) and rating > 0:
            signals.rating_score = (rating - 1.0) / 4.0  # Normalize 1-5 to 0-1
        
        # Review count signal (log-normalized)
        review_count = product_row.get('rating_number')
        if pd.notna(review_count) and review_count > 0:
            # Log transform and normalize (based on analysis: max reviews ~50K)
            log_reviews = np.log1p(review_count)
            signals.review_score = min(log_reviews / 11.0, 1.0)  # log(50000) ≈ 11
        
        return signals

    def apply_filters(
        self, 
        candidates_df: pd.DataFrame, 
        filter_criteria: FilterCriteria
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply structured filters to candidate products.
        
        Uses inclusive filtering strategy - boosts matching products
        rather than eliminating due to missing data.
        
        Args:
            candidates_df: Candidate products DataFrame
            filter_criteria: Filter criteria to apply
            
        Returns:
            Tuple of (filtered_df, filter_boost_scores)
        """
        filter_boosts = pd.Series(0.0, index=candidates_df.index)
        
        # Brand/Store filter
        if filter_criteria.brand_store:
            brand_matches = candidates_df['store'].str.contains(
                filter_criteria.brand_store, case=False, na=False
            )
            filter_boosts += brand_matches * 0.3
        
        # Semantic category filter
        if filter_criteria.semantic_category and '_semantic_category' in candidates_df.columns:
            category_matches = candidates_df['_semantic_category'] == filter_criteria.semantic_category
            filter_boosts += category_matches * 0.25
        
        # Price range filter
        if filter_criteria.price_min is not None or filter_criteria.price_max is not None:
            price_mask = pd.Series(True, index=candidates_df.index)
            
            if filter_criteria.price_min is not None:
                price_mask &= (candidates_df['price'] >= filter_criteria.price_min) | candidates_df['price'].isna()
            
            if filter_criteria.price_max is not None:
                price_mask &= (candidates_df['price'] <= filter_criteria.price_max) | candidates_df['price'].isna()
            
            filter_boosts += price_mask * 0.2
        
        # Quality tier filter
        if filter_criteria.quality_tier:
            quality_boost = 0.0
            
            if filter_criteria.quality_tier == 'high':
                quality_mask = (
                    (candidates_df['average_rating'] >= 4.0) & 
                    (candidates_df['rating_number'] >= 10)
                )
                quality_boost = 0.25
            elif filter_criteria.quality_tier == 'medium':
                quality_mask = (
                    (candidates_df['average_rating'] >= 3.0) & 
                    (candidates_df['average_rating'] < 4.0) & 
                    (candidates_df['rating_number'] >= 5)
                )
                quality_boost = 0.15
            else:
                quality_mask = pd.Series(False, index=candidates_df.index)
            
            filter_boosts += quality_mask * quality_boost
        
        return candidates_df, filter_boosts

    def adaptive_ranking(
        self, 
        candidates_df: pd.DataFrame,
        similarity_scores: np.ndarray,
        filter_boosts: pd.Series
    ) -> pd.DataFrame:
        """Apply adaptive ranking combining semantic and business signals.
        
        Args:
            candidates_df: Candidate products
            similarity_scores: Semantic similarity scores
            filter_boosts: Filter boost scores
            
        Returns:
            Ranked DataFrame with scores
        """
        ranked_products = []
        
        for idx, (_, product_row) in enumerate(candidates_df.iterrows()):
            # Calculate business signals
            business_signals = self.calculate_business_signals(product_row)
            business_signals.semantic_score = float(similarity_scores[idx])
            business_signals.filter_boost = float(filter_boosts.iloc[idx])
            
            # Adaptive weighted combination
            final_score = (
                business_signals.semantic_score * self.config.semantic_weight +
                business_signals.rating_score * self.config.rating_weight +
                business_signals.review_score * self.config.review_weight +
                business_signals.filter_boost * self.config.filter_boost
            )
            
            business_signals.final_score = final_score
            
            # Create product result
            product_result = {
                'parent_asin': product_row.get('parent_asin', ''),
                'title': product_row.get('title', ''),
                'main_category': product_row.get('main_category'),
                'price': product_row.get('price'),
                'average_rating': product_row.get('average_rating'),
                'rating_number': product_row.get('rating_number'),
                'similarity_score': business_signals.semantic_score,
                'final_score': final_score,
                'features': product_row.get('features', []) if isinstance(product_row.get('features'), list) else [],
                'description': product_row.get('description', []) if isinstance(product_row.get('description'), list) else [],
                'store': product_row.get('store'),
                'categories': product_row.get('categories', []) if isinstance(product_row.get('categories'), list) else [],
                'images': product_row.get('_processed_images', product_row.get('images', [])),
                'videos': product_row.get('_processed_videos', product_row.get('videos', [])),
                'primary_image': product_row.get('_primary_image'),
                'ranking_signals': business_signals.dict()
            }
            
            ranked_products.append(product_result)
        
        # Convert to DataFrame and sort by final score
        ranked_df = pd.DataFrame(ranked_products)
        ranked_df = ranked_df.sort_values('final_score', ascending=False)
        
        return ranked_df

    def search(
        self, 
        query: str, 
        top_k: int = None,
        use_query_enhancement: bool = True
    ) -> SearchResponse:
        """Perform hybrid semantic and structured search.
        
        Args:
            query: User search query
            top_k: Number of results to return
            use_query_enhancement: Whether to enhance query with LLM
            
        Returns:
            Search response with ranked results
        """
        start_time = time.time()
        top_k = top_k or self.config.default_top_k
        top_k = min(top_k, self.config.max_top_k)
        
        logger.info(f"Starting search for query: '{query}' (top_k={top_k})")
        
        # Step 1: Query enhancement and analysis
        if use_query_enhancement:
            enhanced_query, query_analysis = self.enhance_query_with_llm(query)
        else:
            enhanced_query = query
            query_analysis = QueryAnalysis(
                original_query=query,
                enhanced_query=enhanced_query,
                intent=QueryIntent.GENERAL,
                extracted_attributes={}
            )
        
        # Step 2: Generate query embedding
        query_embedding_results = self.embedding_manager.generate_embeddings(
            [enhanced_query], 
            ['query']
        )
        query_embedding = np.array([query_embedding_results[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Step 3: Vector similarity search
        search_k = min(top_k * self.config.search_k_factor, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(query_embedding, search_k)
        
        # Filter by similarity threshold
        valid_indices = similarities[0] >= self.config.similarity_threshold
        filtered_similarities = similarities[0][valid_indices]
        filtered_indices = indices[0][valid_indices]
        
        if len(filtered_indices) == 0:
            logger.warning(f"No products found above similarity threshold {self.config.similarity_threshold}")
            return SearchResponse(
                results=[],
                query_analysis=query_analysis,
                total_results=0,
                response_time_ms=(time.time() - start_time) * 1000
            )
        
        # Step 4: Get candidate products
        candidates_df = self.product_df.iloc[filtered_indices].copy()
        
        # Step 5: Extract and apply filters
        filter_criteria = self.extract_filter_criteria(query_analysis)
        candidates_df, filter_boosts = self.apply_filters(candidates_df, filter_criteria)
        
        # Step 6: Adaptive ranking
        ranked_df = self.adaptive_ranking(candidates_df, filtered_similarities, filter_boosts)
        
        # Step 7: Create response
        results = []
        for _, row in ranked_df.head(top_k).iterrows():
            product_result = ProductResult(
                parent_asin=row['parent_asin'],
                title=row['title'],
                main_category=row.get('main_category'),
                price=row.get('price'),
                average_rating=row.get('average_rating'),
                rating_number=row.get('rating_number'),
                similarity_score=row['similarity_score'],
                features=row.get('features', []),
                description=row.get('description', []),
                store=row.get('store'),
                categories=row.get('categories', []),
                images=row.get('images', []),
                videos=row.get('videos', []),
                primary_image=row.get('primary_image')
            )
            results.append(product_result)
        
        response_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Search completed: {len(results)} results in {response_time_ms:.1f}ms")
        
        return SearchResponse(
            results=results,
            query_analysis=query_analysis,
            total_results=len(filtered_indices),
            response_time_ms=response_time_ms
        )

    def get_index_stats(self) -> Dict[str, Any]:
        """Get search index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.faiss_index:
            return {"error": "Index not built"}
        
        return {
            "total_products": self.faiss_index.ntotal,
            "embedding_dimension": self.faiss_index.d,
            "index_type": type(self.faiss_index).__name__,
            "index_trained": getattr(self.faiss_index, 'is_trained', True),
            "product_df_shape": self.product_df.shape if self.product_df is not None else None,
            "brand_normalizer_size": len(self.brand_normalizer)
        }


def create_search_engine(
    config: Optional[SearchConfig] = None,
    embedding_manager: Optional[EmbeddingManager] = None
) -> HybridSearchEngine:
    """Factory function to create search engine.
    
    Args:
        config: Optional search configuration
        embedding_manager: Optional embedding manager
        
    Returns:
        Configured search engine instance
    """
    return HybridSearchEngine(config, embedding_manager) 