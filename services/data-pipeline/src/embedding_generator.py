"""Embedding generation using OpenAI API."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from openai import AsyncOpenAI

from shared.models import Settings
from shared.utils import calculate_tokens


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles embedding generation and FAISS index creation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.logger = logging.getLogger(__name__)
    
    async def generate_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate embeddings for a list of texts."""
        
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Calculate total tokens for cost estimation
        total_tokens = sum(calculate_tokens(text, self.settings.embedding_model) for text in texts)
        estimated_cost = self._calculate_cost(total_tokens)
        
        self.logger.info(f"Estimated tokens: {total_tokens:,}")
        self.logger.info(f"Estimated cost: ${estimated_cost:.4f}")
        
        # Generate embeddings in batches
        embeddings = []
        batch_size = self.settings.data_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._generate_batch(batch)
            embeddings.extend(batch_embeddings)
            
            self.logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        cost_info = {
            'total_tokens': total_tokens,
            'total_cost': estimated_cost,
            'batch_count': (len(texts) + batch_size - 1) // batch_size
        }
        
        self.logger.info(f"Generated {len(embeddings_array)} embeddings with dimension {embeddings_array.shape[1]}")
        
        return embeddings_array, cost_info
    
    async def _generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        
        try:
            response = await self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate estimated cost for embedding generation."""
        
        # OpenAI pricing for text-embedding-3-small: $0.00002 per 1K tokens
        cost_per_1k_tokens = 0.00002
        return (tokens / 1000) * cost_per_1k_tokens
    
    def save_embeddings(self, embeddings: np.ndarray, product_ids: List[str], metadata: Dict[str, Any]) -> None:
        """Save embeddings, FAISS index, and metadata to disk."""
        
        embeddings_dir = self.settings.embeddings_dir
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving embeddings and FAISS index to {embeddings_dir}")
        
        # Save embeddings as numpy array
        embeddings_file = embeddings_dir / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        self.logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Save product IDs
        ids_file = embeddings_dir / "product_ids.json"
        with open(ids_file, 'w') as f:
            json.dump(product_ids, f)
        self.logger.info(f"Saved product IDs to {ids_file}")
        
        # Create and save FAISS index
        index_file = embeddings_dir / "faiss_index.index"
        self._create_and_save_faiss_index(embeddings, index_file)
        
        # Save metadata
        metadata_file = embeddings_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata to {metadata_file}")
    
    def _create_and_save_faiss_index(self, embeddings: np.ndarray, index_file: Path) -> None:
        """Create and save FAISS index for fast similarity search."""
        
        self.logger.info("Creating FAISS index...")
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        n_embeddings = len(embeddings)
        
        # Choose index type based on dataset size
        if n_embeddings < 1000:
            # Small dataset: use flat index for exact search
            index = faiss.IndexFlatIP(dimension)
            self.logger.info(f"Created flat IP index for {n_embeddings} vectors")
        else:
            # Larger dataset: use IVF for faster search
            nlist = min(100, max(1, n_embeddings // 50))  # Adaptive number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            self.logger.info(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings)
            
            # Set search parameters
            index.nprobe = min(10, nlist)
            self.logger.info(f"Created IVF index with {nlist} clusters, nprobe={index.nprobe}")
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Save index to disk
        faiss.write_index(index, str(index_file))
        self.logger.info(f"Saved FAISS index to {index_file}")
        
        # Save index metadata for loading
        index_metadata = {
            'index_type': 'IVF' if n_embeddings >= 1000 else 'Flat',
            'dimension': dimension,
            'n_vectors': n_embeddings,
            'nlist': getattr(index, 'nlist', None),
            'nprobe': getattr(index, 'nprobe', None)
        }
        
        metadata_file = index_file.parent / "faiss_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(index_metadata, f, indent=2)
        self.logger.info(f"Saved FAISS metadata to {metadata_file}")
    
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
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_dir}")
        
        return embeddings, product_ids, metadata
    
    def load_faiss_index(self) -> faiss.Index:
        """Load pre-built FAISS index from disk."""
        
        embeddings_dir = self.settings.embeddings_dir
        index_file = embeddings_dir / "faiss_index.index"
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        
        self.logger.info(f"Loading FAISS index from {index_file}")
        index = faiss.read_index(str(index_file))
        
        # Load and apply metadata
        metadata_file = embeddings_dir / "faiss_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                index_metadata = json.load(f)
            
            # Set nprobe for IVF indices
            if hasattr(index, 'nprobe') and index_metadata.get('nprobe'):
                index.nprobe = index_metadata['nprobe']
            
            self.logger.info(f"Loaded {index_metadata['index_type']} index with {index_metadata['n_vectors']} vectors")
        
        return index
    
    def faiss_index_exists(self) -> bool:
        """Check if FAISS index exists on disk."""
        index_file = self.settings.embeddings_dir / "faiss_index.index"
        return index_file.exists()
    
    def embeddings_exist(self) -> bool:
        """Check if embeddings exist on disk."""
        embeddings_file = self.settings.embeddings_dir / "embeddings.npy"
        metadata_file = self.settings.embeddings_dir / "metadata.json"
        ids_file = self.settings.embeddings_dir / "product_ids.json"
        
        return all(f.exists() for f in [embeddings_file, metadata_file, ids_file])