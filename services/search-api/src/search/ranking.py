"""
Ranking Manager Module

Handles product ranking and scoring including:
- Heuristic ranking with business signals
- LLM-powered semantic reranking
- Score combination and optimization
"""

import logging
import math
from typing import List, Dict, Any
from shared.models import ProductResult

logger = logging.getLogger(__name__)


class RankingManager:
    """Manages product ranking and scoring operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.ranking_operations = 0
        self.heuristic_rankings = 0
        self.llm_rankings = 0
    
    def rank_products(
        self, 
        products: List[ProductResult], 
        query: str, 
        method: str = "heuristic"
    ) -> List[ProductResult]:
        """
        Rank products using the specified method.
        
        Args:
            products: List of products to rank
            query: Original search query for context
            method: Ranking method ("heuristic" or "llm")
            
        Returns:
            Ranked list of products
        """
        
        if not products:
            return products
        
        self.ranking_operations += 1
        
        if method == "heuristic":
            return self._heuristic_ranking(products, query)
        elif method == "llm":
            # Note: LLM ranking is handled by LLMProcessor.rerank_with_llm()
            # This is a fallback to heuristic if LLM ranking is not available
            self.logger.debug("LLM ranking requested but not available, using heuristic")
            return self._heuristic_ranking(products, query)
        else:
            self.logger.warning(f"Unknown ranking method: {method}, using heuristic")
            return self._heuristic_ranking(products, query)
    
    def _heuristic_ranking(self, products: List[ProductResult], query: str) -> List[ProductResult]:
        """
        Fast heuristic-based ranking using business signals and query relevance.
        
        Ranking factors:
        1. Semantic similarity (primary)
        2. Business signals (rating, popularity)
        3. Data completeness
        4. Query-specific relevance
        """
        
        self.heuristic_rankings += 1
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for product in products:
            # Start with semantic similarity as base score
            score = product.similarity_score
            
            # Business signal boosts
            score += self._calculate_business_score(product)
            
            # Query relevance boost
            score += self._calculate_query_relevance_score(product, query_lower, query_words)
            
            # Data completeness boost
            score += self._calculate_completeness_score(product)
            
            # Filter match boost
            score += self._calculate_filter_match_score(product)
            
            # Store the final ranking score (clamped to [0, 1] for validation)
            product.rank_score = min(max(score, 0.0), 1.0)
        
        # Sort by final score (highest first)
        return sorted(products, key=lambda p: p.rank_score, reverse=True)
    
    def _calculate_business_score(self, product: ProductResult) -> float:
        """Calculate business signal contribution to ranking."""
        
        business_score = 0.0
        
        # Rating boost: scale around 3.0 (neutral rating)
        if product.average_rating is not None:
            # Boost products with 4+ stars, slight penalty for below 3
            rating_boost = (product.average_rating - 3.0) * 0.15
            business_score += max(0, rating_boost)  # No negative boost
        
        # Review count boost (popularity signal with logarithmic scaling)
        if product.rating_number is not None and product.rating_number > 0:
            # Logarithmic scaling to prevent dominance of super-popular items
            log_reviews = math.log(product.rating_number + 1)
            review_boost = min(0.1, log_reviews / 50)  # Cap at 0.1
            business_score += review_boost
        
        return business_score
    
    def _calculate_query_relevance_score(
        self, 
        product: ProductResult, 
        query_lower: str, 
        query_words: set
    ) -> float:
        """Calculate query-specific relevance boost."""
        
        relevance_score = 0.0
        
        # Title relevance boost
        if product.title:
            title_lower = product.title.lower()
            
            # Exact phrase match in title (strong signal)
            if query_lower in title_lower:
                relevance_score += 0.05
            
            # Word overlap in title
            title_words = set(title_lower.split())
            word_matches = len(query_words.intersection(title_words))
            if word_matches > 0:
                # Scale by percentage of query words matched
                word_overlap_score = (word_matches / len(query_words)) * 0.03
                relevance_score += word_overlap_score
        
        # Category relevance boost
        if product.main_category:
            category_lower = product.main_category.lower()
            if any(word in category_lower for word in query_words):
                relevance_score += 0.02
        
        # Feature relevance boost
        if product.features:
            features_text = ' '.join(product.features).lower()
            feature_matches = sum(1 for word in query_words if word in features_text)
            if feature_matches > 0:
                relevance_score += min(0.02, feature_matches * 0.005)
        
        return relevance_score
    
    def _calculate_completeness_score(self, product: ProductResult) -> float:
        """Calculate data completeness contribution."""
        
        completeness_score = 0.0
        
        # Boost products with more complete data
        if product.price is not None:
            completeness_score += 0.01
        
        if product.features and len(product.features) > 0:
            completeness_score += 0.01
        
        if product.images and len(product.images) > 0:
            completeness_score += 0.005
        
        if product.average_rating is not None:
            completeness_score += 0.005
        
        if product.store:
            completeness_score += 0.005
        
        return completeness_score
    
    def _calculate_filter_match_score(self, product: ProductResult) -> float:
        """Calculate filter match contribution."""
        
        filter_score = 0.0
        
        # Boost products that match extracted filters
        if product.matched_filters:
            # Small boost per matched filter
            filter_match_boost = len(product.matched_filters) * 0.01
            filter_score += min(0.05, filter_match_boost)  # Cap at 0.05
        
        return filter_score
    
    def apply_diversity_boost(self, products: List[ProductResult], diversity_factor: float = 0.1) -> List[ProductResult]:
        """
        Apply diversity boost to prevent too many similar products at the top.
        
        Args:
            products: Ranked list of products
            diversity_factor: How much to boost diverse products (0.0 - 1.0)
            
        Returns:
            Re-ranked list with diversity boost applied
        """
        
        if not products or diversity_factor <= 0:
            return products
        
        # Track categories we've seen
        seen_categories = set()
        seen_stores = set()
        
        for i, product in enumerate(products):
            diversity_boost = 0.0
            
            # Boost products from new categories
            if product.main_category and product.main_category not in seen_categories:
                diversity_boost += diversity_factor * 0.5
                seen_categories.add(product.main_category)
            
            # Boost products from new stores
            if product.store and product.store not in seen_stores:
                diversity_boost += diversity_factor * 0.3
                seen_stores.add(product.store)
            
            # Apply position penalty (later results get less boost)
            position_penalty = max(0, 1.0 - (i / len(products)))
            diversity_boost *= position_penalty
            
            # Apply the boost
            if hasattr(product, 'rank_score'):
                product.rank_score += diversity_boost
            else:
                product.rank_score = product.similarity_score + diversity_boost
        
        # Re-sort with diversity boost applied
        return sorted(products, key=lambda p: getattr(p, 'rank_score', p.similarity_score), reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ranking operation statistics."""
        
        total_ops = self.ranking_operations
        
        return {
            'total_ranking_operations': total_ops,
            'heuristic_rankings': self.heuristic_rankings,
            'llm_rankings': self.llm_rankings,
            'heuristic_usage_rate': self.heuristic_rankings / max(1, total_ops)
        }