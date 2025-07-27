"""Embedding generation using OpenAI API."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import openai
from openai import AsyncOpenAI

from shared.models import Settings
from shared.utils import calculate_tokens


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using OpenAI API with batching and caching."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Pricing for text-embedding-3-small (per 1K tokens)
        self.price_per_1k_tokens = 0.00002
    
    async def generate_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate embeddings for a list of texts with batching."""
        
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        batch_size = self.settings.data_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._generate_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)
            
            # Progress logging
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(texts):
                self.logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Calculate final cost
        estimated_cost = (self.total_tokens / 1000) * self.price_per_1k_tokens
        
        metadata = {
            'total_tokens': self.total_tokens,
            'estimated_cost': estimated_cost,
            'model': self.settings.embedding_model,
            'embedding_dim': embeddings_array.shape[1] if len(embeddings_array) > 0 else 0,
            'total_embeddings': len(embeddings_array)
        }
        
        self.logger.info(f"Generated {len(embeddings_array)} embeddings")
        self.logger.info(f"Total tokens: {self.total_tokens:,}")
        self.logger.info(f"Estimated cost: ${estimated_cost:.4f}")
        
        return embeddings_array, metadata
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        
        try:
            # Calculate tokens for this batch
            batch_tokens = sum(calculate_tokens(text, self.settings.embedding_model) for text in texts)
            self.total_tokens += batch_tokens
            
            # Call OpenAI API
            response = await self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            # Add small delay to respect rate limits
            await asyncio.sleep(0.1)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings for batch: {e}")
            # Return zero embeddings as fallback
            embedding_dim = 1536  # Default for text-embedding-3-small
            return [[0.0] * embedding_dim for _ in texts]
    
    def save_embeddings(self, embeddings: np.ndarray, product_ids: List[str], metadata: Dict[str, Any]) -> None:
        """Save embeddings and metadata to disk."""
        
        embeddings_dir = self.settings.embeddings_dir
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy array
        embeddings_file = embeddings_dir / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        # Save product IDs
        ids_file = embeddings_dir / "product_ids.json"
        with open(ids_file, 'w') as f:
            json.dump(product_ids, f)
        
        # Save metadata
        metadata_file = embeddings_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved embeddings to {embeddings_dir}")
    
    def load_embeddings(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Load embeddings and metadata from disk."""
        
        embeddings_dir = self.settings.embeddings_dir
        
        # Load embeddings
        embeddings_file = embeddings_dir / "embeddings.npy"
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")
        
        embeddings = np.load(embeddings_file)
        
        # Load product IDs
        ids_file = embeddings_dir / "product_ids.json"
        if not ids_file.exists():
            raise FileNotFoundError(f"Product IDs not found: {ids_file}")
        
        with open(ids_file, 'r') as f:
            product_ids = json.load(f)
        
        # Load metadata
        metadata_file = embeddings_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        self.logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings, product_ids, metadata
    
    def embeddings_exist(self) -> bool:
        """Check if embeddings already exist."""
        embeddings_file = self.settings.embeddings_dir / "embeddings.npy"
        ids_file = self.settings.embeddings_dir / "product_ids.json"
        return embeddings_file.exists() and ids_file.exists()