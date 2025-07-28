"""
LLM Integration Module

Handles all interactions with OpenAI's API including:
- Query enhancement and expansion
- Filter extraction from natural language
- Embedding generation for search queries
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import faiss
from openai import AsyncOpenAI

from shared.models import Settings
from shared.utils import extract_search_filters_with_llm, enhance_query_with_context

logger = logging.getLogger(__name__)


class LLMProcessor:
    """Handles LLM operations for search enhancement."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.query_count = 0
        self.embedding_count = 0
        self.filter_extraction_count = 0
    
    async def process_search_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a search query with LLM enhancement and filter extraction.
        
        Returns:
            Tuple of (enhanced_query, extracted_filters)
        """
        
        self.logger.debug(f"Processing query: {query}")
        
        try:
            # Extract filters from natural language query
            extracted_filters = await self._extract_filters(query)
            
            # Enhance query for better semantic search
            enhanced_query = await self._enhance_query(query)
            
            self.query_count += 1
            
            return enhanced_query, extracted_filters
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            # Fallback to original query with empty filters
            return query, {}
    
    async def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract structured filters from natural language query."""
        
        try:
            self.filter_extraction_count += 1
            return await extract_search_filters_with_llm(query, self.client)
        except Exception as e:
            self.logger.warning(f"Filter extraction failed: {e}")
            return {}
    
    async def _enhance_query(self, query: str) -> str:
        """Enhance query with fashion context and synonyms."""
        
        try:
            return await enhance_query_with_context(query, self.client)
        except Exception as e:
            self.logger.warning(f"Query enhancement failed: {e}")
            return query  # Fallback to original query
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        
        try:
            self.logger.debug(f"Generating embedding for query: {query[:50]}...")
            
            response = await self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=[query]
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            self.embedding_count += 1
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise
    
    async def rerank_with_llm(self, products: list, query: str, max_products: int = 10) -> list:
        """
        Use LLM to rerank search results for better semantic relevance.
        
        Args:
            products: List of ProductResult objects
            query: Original search query
            max_products: Maximum number of products to rerank
            
        Returns:
            Reranked list of products
        """
        
        if not products or len(products) <= 1:
            return products
        
        # Limit to reasonable number for LLM processing
        products_to_rank = products[:max_products]
        
        try:
            # Prepare product summaries for LLM
            product_summaries = []
            for i, product in enumerate(products_to_rank):
                summary = f"Product {i+1}: {product.title}"
                if product.main_category:
                    summary += f" (Category: {product.main_category})"
                if product.price:
                    summary += f" - ${product.price}"
                if product.average_rating:
                    summary += f" - {product.average_rating}★"
                
                product_summaries.append(summary)
            
            # Create LLM prompt for reranking
            system_prompt = """You are a fashion search expert. Rerank these products based on relevance to the user's query.

Consider:
- Semantic relevance to the query
- Product title match quality
- Category appropriateness
- Style and occasion fit
- Price range match

Return ONLY a JSON array of product numbers (1-based) in order of relevance, most relevant first.
Example: [3, 1, 7, 2, 5, 4, 6]"""

            user_prompt = f"""Query: "{query}"

Products:
{chr(10).join(product_summaries)}

Rerank by relevance:"""

            response = await self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse LLM response
            import json
            ranking = json.loads(response.choices[0].message.content)
            
            # Apply LLM ranking
            reranked_products = []
            for rank_idx, product_idx in enumerate(ranking):
                if 1 <= product_idx <= len(products_to_rank):
                    product = products_to_rank[product_idx - 1]  # Convert to 0-based
                    product.rank_score = len(products_to_rank) - rank_idx  # Higher score for better rank
                    reranked_products.append(product)
            
            # Add any products not included in LLM ranking at the end
            included_indices = {idx - 1 for idx in ranking if 1 <= idx <= len(products_to_rank)}
            for i, product in enumerate(products_to_rank):
                if i not in included_indices:
                    product.rank_score = 0
                    reranked_products.append(product)
            
            # Add remaining products beyond max_products
            reranked_products.extend(products[max_products:])
            
            return reranked_products
            
        except Exception as e:
            self.logger.warning(f"LLM reranking failed, falling back to original order: {e}")
            return products
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM processing statistics."""
        
        return {
            'total_queries_processed': self.query_count,
            'embeddings_generated': self.embedding_count,
            'filter_extractions': self.filter_extraction_count,
            'embedding_model': self.settings.embedding_model,
            'llm_model': self.settings.llm_model
        }