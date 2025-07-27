"""OpenAI embedding generation with batch processing and cost tracking."""

import asyncio
import time
from typing import List, Dict, Any, Tuple
import logging

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
import tiktoken

# Removed shared imports - will use local config classes


class DataPipelineConfig:
    """Simple configuration for data pipeline."""
    def __init__(self):
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed") 
        self.embeddings_cache_dir = Path("data/embeddings")
        self.dataset_filename = "meta_Amazon_Fashion.jsonl"
        self.sample_size = 150000
        self.openai_api_key = ""


class EmbeddingResult:
    """Result container for embedding generation."""
    
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.product_ids: List[str] = []
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.batch_count: int = 0
        self.processing_time: float = 0.0


class EmbeddingGenerator:
    """OpenAI embedding generator with batch processing and cost optimization."""
    
    def __init__(self, config: DataPipelineConfig, logger: logging.Logger):
        """Initialize embedding generator.
        
        Args:
            config: Data pipeline configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Initialize OpenAI client with explicit configuration
        import os
        if config.openai_api_key:
            os.environ['OPENAI_API_KEY'] = config.openai_api_key
        
        # Use the latest OpenAI client initialization
        self.client = AsyncOpenAI(
            api_key=config.openai_api_key,
            max_retries=3
        )
        
        # Initialize tokenizer for cost calculation
        self.tokenizer = tiktoken.encoding_for_model(self.config.embedding_model)
        
        # Embedding cache directory
        self.cache_dir = config.embeddings_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized EmbeddingGenerator with model: {self.config.embedding_model}")

    async def generate_batch_embeddings(self, df: pd.DataFrame) -> EmbeddingResult:
        """Generate embeddings for all products in batches.
        
        Args:
            df: DataFrame with products and embedding_text column
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        result = EmbeddingResult()
        
        # Prepare text data
        texts = df['embedding_text'].tolist()
        product_ids = df['parent_asin'].tolist()
        
        self.logger.info(f"Generating embeddings for {len(texts):,} products")
        self.logger.info(f"Batch size: {self.config.embedding_batch_size}")
        
        # Calculate estimated cost
        total_tokens = self._count_tokens(texts)
        estimated_cost = self._calculate_cost(total_tokens)
        self.logger.info(f"Estimated tokens: {total_tokens:,}")
        self.logger.info(f"Estimated cost: ${estimated_cost:.2f}")
        
        # Process in batches
        batch_size = self.config.embedding_batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            batch_texts = texts[i:i + batch_size]
            batch_ids = product_ids[i:i + batch_size]
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} items)")
            
            try:
                # Generate embeddings for batch
                batch_embeddings, batch_tokens = await self._generate_embeddings_batch(batch_texts)
                
                # Store results
                result.embeddings.extend(batch_embeddings)
                result.product_ids.extend(batch_ids)
                result.total_tokens += batch_tokens
                result.batch_count += 1
                
                # Rate limiting
                if batch_num < total_batches:
                    await asyncio.sleep(self.config.embedding_rate_limit_delay)
                    
            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed: {e}")
                raise
        
        # Calculate final metrics
        result.total_cost = self._calculate_cost(result.total_tokens)
        result.processing_time = time.time() - start_time
        
        self.logger.info(f"Embedding generation complete:")
        self.logger.info(f"  Total embeddings: {len(result.embeddings):,}")
        self.logger.info(f"  Total tokens: {result.total_tokens:,}")
        self.logger.info(f"  Total cost: ${result.total_cost:.2f}")
        self.logger.info(f"  Processing time: {result.processing_time:.1f}s")
        self.logger.info(f"  Average tokens per product: {result.total_tokens / len(result.embeddings):.1f}")
        
        return result

    async def _generate_embeddings_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        """Generate embeddings for a single batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tuple of (embeddings, token_count)
        """
        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            
            # Count tokens (approximate from response usage if available)
            # Fix for OpenAI v1.40+ API response format
            token_count = response.usage.total_tokens if hasattr(response, 'usage') else 0
            if token_count == 0:
                # Fallback to manual counting
                token_count = sum(self._count_tokens([text]) for text in texts)
            
            return embeddings, token_count
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise

    def _count_tokens(self, texts: List[str]) -> int:
        """Count tokens for cost estimation.
        
        Args:
            texts: List of texts
            
        Returns:
            Total token count
        """
        try:
            total_tokens = 0
            for text in texts:
                if text:
                    tokens = self.tokenizer.encode(str(text))
                    total_tokens += len(tokens)
            return total_tokens
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}")
            # Fallback estimation: ~4 chars per token
            return sum(len(str(text)) // 4 for text in texts if text)

    def _calculate_cost(self, token_count: int) -> float:
        """Calculate cost for token count.
        
        Args:
            token_count: Number of tokens
            
        Returns:
            Cost in USD
        """
        return (token_count / 1000) * self.config.embedding_cost_per_1k_tokens

    async def save_embeddings(self, result: EmbeddingResult) -> None:
        """Save embeddings to disk.
        
        Args:
            result: EmbeddingResult to save
        """
        # Convert embeddings to numpy array
        embeddings_array = np.array(result.embeddings)
        
        # Save embeddings
        embeddings_file = self.cache_dir / "embeddings.npz"
        np.savez_compressed(
            embeddings_file,
            embeddings=embeddings_array,
            product_ids=result.product_ids,
            metadata={
                'total_tokens': result.total_tokens,
                'total_cost': result.total_cost,
                'batch_count': result.batch_count,
                'processing_time': result.processing_time,
                'model_name': self.config.embedding_model
            }
        )
        
        self.logger.info(f"Saved embeddings to {embeddings_file}")

    async def load_embeddings(self) -> EmbeddingResult:
        """Load embeddings from disk.
        
        Returns:
            EmbeddingResult loaded from disk
        """
        embeddings_file = self.cache_dir / "embeddings.npz"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings not found at {embeddings_file}")
        
        data = np.load(embeddings_file, allow_pickle=True)
        
        result = EmbeddingResult()
        result.embeddings = [emb for emb in data['embeddings']]
        result.product_ids = data['product_ids'].tolist()
        
        # Load metadata if available
        if 'metadata' in data:
            metadata = data['metadata'].item()
            result.total_tokens = metadata.get('total_tokens', 0)
            result.total_cost = metadata.get('total_cost', 0.0)
            result.batch_count = metadata.get('batch_count', 0)
            result.processing_time = metadata.get('processing_time', 0.0)
        
        self.logger.info(f"Loaded embeddings from {embeddings_file}: {len(result.embeddings):,} embeddings")
        return result 