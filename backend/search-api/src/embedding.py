"""Text embedding generation and management for Amazon Fashion products.

This module handles OpenAI text embedding generation with precise token counting,
cost optimization, batch processing, and caching strategies derived from analysis.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import hashlib

import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI, AsyncOpenAI
import pickle

from .config import settings
from .models import EmbeddingConfig, TokenAnalysis, EmbeddingResult

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embedding generation and caching for fashion products."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedding manager.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Initialize tiktoken for precise token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken: {e}")
            raise RuntimeError(f"Tokenizer initialization failed: {e}")
        
        # Setup cache
        self.cache_dir = settings.processed_data_dir / "embeddings_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_cache = {}
        self._load_cache()
        
        # OpenAI pricing (as of January 2025)
        self.pricing = {
            'text-embedding-3-small': 0.00002 / 1000,  # $0.02 per 1M tokens
            'text-embedding-3-large': 0.00013 / 1000   # $0.13 per 1M tokens
        }
        
        logger.info(f"Initialized EmbeddingManager with model: {self.config.model}")

    def _load_cache(self) -> None:
        """Load embeddings cache from disk."""
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        if cache_file.exists() and self.config.cache_embeddings:
            try:
                with open(cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Loaded embedding cache: {len(self._embedding_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._embedding_cache = {}

    def _save_cache(self) -> None:
        """Save embeddings cache to disk."""
        if not self.config.cache_embeddings:
            return
            
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            logger.info(f"Saved embedding cache: {len(self._embedding_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination.
        
        Args:
            text: Input text
            model: Model name
            
        Returns:
            Cache key string
        """
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough approximation
            return len(text.split()) * 1.3

    def estimate_cost(self, total_tokens: int, model: str = None) -> float:
        """Estimate cost for token count.
        
        Args:
            total_tokens: Total token count
            model: Model name (defaults to config model)
            
        Returns:
            Estimated cost in USD
        """
        model = model or self.config.model
        price_per_token = self.pricing.get(model, self.pricing['text-embedding-3-small'])
        return total_tokens * price_per_token

    def analyze_token_usage(self, texts: List[str], model: str = None) -> TokenAnalysis:
        """Analyze token usage and costs for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            model: Model name (defaults to config model)
            
        Returns:
            Token analysis results
        """
        model = model or self.config.model
        
        token_counts = [self.count_tokens(text) for text in texts]
        total_tokens = sum(token_counts)
        total_cost = self.estimate_cost(total_tokens, model)
        
        return TokenAnalysis(
            total_tokens=total_tokens,
            total_cost=total_cost,
            average_tokens_per_product=total_tokens / len(texts) if texts else 0,
            cost_per_product=total_cost / len(texts) if texts else 0,
            batch_count=len(texts) // self.config.batch_size + (1 if len(texts) % self.config.batch_size else 0),
            model_used=model
        )

    def _create_batches(self, texts: List[str], product_ids: List[str]) -> List[Tuple[List[str], List[str]]]:
        """Create batches respecting token limits.
        
        Args:
            texts: List of texts
            product_ids: List of product IDs
            
        Returns:
            List of (text_batch, id_batch) tuples
        """
        batches = []
        current_texts = []
        current_ids = []
        current_tokens = 0
        
        for text, product_id in zip(texts, product_ids):
            text_tokens = self.count_tokens(text)
            
            # Check if adding this text would exceed limits
            if (len(current_texts) >= self.config.batch_size or 
                current_tokens + text_tokens > self.config.max_tokens_per_request):
                
                if current_texts:  # Save current batch
                    batches.append((current_texts.copy(), current_ids.copy()))
                    current_texts = []
                    current_ids = []
                    current_tokens = 0
            
            current_texts.append(text)
            current_ids.append(product_id)
            current_tokens += text_tokens
        
        # Add final batch
        if current_texts:
            batches.append((current_texts, current_ids))
        
        return batches

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts.
        
        Args:
            texts: Batch of texts
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If API call fails after retries
        """
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                    encoding_format="float"
                )
                
                embeddings = [data.embedding for data in response.data]
                return embeddings
                
            except Exception as e:
                logger.warning(f"Embedding API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to get embeddings after {self.config.max_retries} attempts: {e}")

    async def _get_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts asynchronously.
        
        Args:
            texts: Batch of texts
            
        Returns:
            List of embedding vectors
        """
        for attempt in range(self.config.max_retries):
            try:
                response = await self.async_client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                    encoding_format="float"
                )
                
                embeddings = [data.embedding for data in response.data]
                return embeddings
                
            except Exception as e:
                logger.warning(f"Async embedding API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise RuntimeError(f"Failed to get embeddings after {self.config.max_retries} attempts: {e}")

    def generate_embeddings(
        self, 
        texts: List[str], 
        product_ids: List[str],
        use_cache: bool = True
    ) -> List[EmbeddingResult]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            product_ids: List of product IDs
            use_cache: Whether to use/update cache
            
        Returns:
            List of embedding results
            
        Raises:
            ValueError: If texts and product_ids length mismatch
        """
        if len(texts) != len(product_ids):
            raise ValueError("texts and product_ids must have the same length")
        
        logger.info(f"Generating embeddings for {len(texts):,} products...")
        
        results = []
        texts_to_process = []
        ids_to_process = []
        cached_results = []
        
        # Check cache first
        for text, product_id in zip(texts, product_ids):
            cache_key = self._get_cache_key(text, self.config.model)
            
            if use_cache and cache_key in self._embedding_cache:
                cached_data = self._embedding_cache[cache_key]
                cached_results.append(EmbeddingResult(
                    product_id=product_id,
                    embedding=cached_data['embedding'],
                    text_content=text,
                    token_count=cached_data['token_count'],
                    processing_time=0.0,  # Cached
                    model_used=self.config.model
                ))
            else:
                texts_to_process.append(text)
                ids_to_process.append(product_id)
        
        logger.info(f"Found {len(cached_results)} cached embeddings, processing {len(texts_to_process)} new ones")
        
        # Process non-cached texts
        if texts_to_process:
            batches = self._create_batches(texts_to_process, ids_to_process)
            logger.info(f"Processing {len(batches)} batches...")
            
            for batch_idx, (text_batch, id_batch) in enumerate(batches):
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(text_batch)} items)")
                
                start_time = time.time()
                embeddings = self._get_embeddings_batch(text_batch)
                processing_time = time.time() - start_time
                
                # Store results and update cache
                for text, product_id, embedding in zip(text_batch, id_batch, embeddings):
                    token_count = self.count_tokens(text)
                    
                    result = EmbeddingResult(
                        product_id=product_id,
                        embedding=embedding,
                        text_content=text,
                        token_count=token_count,
                        processing_time=processing_time / len(text_batch),
                        model_used=self.config.model
                    )
                    results.append(result)
                    
                    # Update cache
                    if use_cache:
                        cache_key = self._get_cache_key(text, self.config.model)
                        self._embedding_cache[cache_key] = {
                            'embedding': embedding,
                            'token_count': token_count,
                            'timestamp': time.time()
                        }
                
                # Add delay between batches to respect rate limits
                if batch_idx < len(batches) - 1:
                    time.sleep(0.1)
        
        # Combine cached and new results
        all_results = cached_results + results
        
        # Sort by original order
        result_dict = {r.product_id: r for r in all_results}
        ordered_results = [result_dict[pid] for pid in product_ids]
        
        # Save cache
        if use_cache and results:  # Only save if we added new embeddings
            self._save_cache()
        
        # Log statistics
        total_tokens = sum(r.token_count for r in ordered_results)
        total_cost = self.estimate_cost(total_tokens)
        
        logger.info(f"Embedding generation complete:")
        logger.info(f"  Total products: {len(ordered_results):,}")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Estimated cost: ${total_cost:.4f}")
        logger.info(f"  Average tokens per product: {total_tokens / len(ordered_results):.1f}")
        logger.info(f"  Cache hits: {len(cached_results):,}")
        
        return ordered_results

    async def generate_embeddings_async(
        self,
        texts: List[str],
        product_ids: List[str],
        use_cache: bool = True
    ) -> List[EmbeddingResult]:
        """Generate embeddings asynchronously.
        
        Args:
            texts: List of texts to embed
            product_ids: List of product IDs
            use_cache: Whether to use/update cache
            
        Returns:
            List of embedding results
        """
        # Similar to sync version but with async API calls
        # Implementation would follow same pattern as generate_embeddings
        # but using _get_embeddings_batch_async
        pass  # Simplified for now

    def create_embedding_matrix(self, results: List[EmbeddingResult]) -> np.ndarray:
        """Create embedding matrix from results.
        
        Args:
            results: List of embedding results
            
        Returns:
            NumPy array of shape (n_products, embedding_dim)
        """
        embeddings = np.array([r.embedding for r in results])
        logger.info(f"Created embedding matrix: {embeddings.shape}")
        return embeddings

    def save_embeddings(
        self, 
        results: List[EmbeddingResult], 
        filename: str = "embeddings.npz"
    ) -> Path:
        """Save embeddings to disk in compressed format.
        
        Args:
            results: List of embedding results
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.cache_dir / filename
        
        # Prepare data
        embeddings = self.create_embedding_matrix(results)
        product_ids = [r.product_id for r in results]
        text_contents = [r.text_content for r in results]
        token_counts = [r.token_count for r in results]
        
        # Save as compressed NumPy archive
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            product_ids=product_ids,
            text_contents=text_contents,
            token_counts=token_counts,
            model_used=self.config.model,
            generation_timestamp=time.time()
        )
        
        logger.info(f"Saved embeddings to {output_path}")
        return output_path

    def load_embeddings(self, filename: str = "embeddings.npz") -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Load embeddings from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Tuple of (embeddings_matrix, product_ids, metadata)
        """
        input_path = self.cache_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Embeddings file not found at {input_path}")
        
        data = np.load(input_path, allow_pickle=True)
        
        embeddings = data['embeddings']
        product_ids = data['product_ids'].tolist()
        
        metadata = {
            'model_used': str(data['model_used']),
            'generation_timestamp': float(data['generation_timestamp']),
            'text_contents': data['text_contents'].tolist(),
            'token_counts': data['token_counts'].tolist()
        }
        
        logger.info(f"Loaded embeddings from {input_path}: {embeddings.shape}")
        return embeddings, product_ids, metadata


def create_embedding_manager(config: Optional[EmbeddingConfig] = None) -> EmbeddingManager:
    """Factory function to create EmbeddingManager instance.
    
    Args:
        config: Optional embedding configuration
        
    Returns:
        Configured EmbeddingManager instance
    """
    return EmbeddingManager(config) 