"""
Vector Search Manager

Handles FAISS index operations and vector similarity search.
Separated from the main search engine for better modularity.
"""

import logging
import time
from typing import Optional, Tuple
import numpy as np
import faiss
from pathlib import Path

from shared.models import Settings

logger = logging.getLogger(__name__)


class VectorSearchManager:
    """Manages FAISS vector search operations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # FAISS components
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        
        # Performance tracking
        self.search_count = 0
        self.total_search_time = 0.0
    
    def initialize(self, embeddings: np.ndarray) -> None:
        """Initialize the vector search manager with embeddings."""
        
        self.logger.info("Initializing vector search manager...")
        
        self.embeddings = embeddings.astype(np.float32)
        
        # Try to load pre-built FAISS index first
        self._load_or_build_faiss_index()
        
        self.logger.info(f"Vector search initialized with {len(embeddings)} embeddings")
    
    def _load_or_build_faiss_index(self) -> None:
        """Load pre-built FAISS index or build one if it doesn't exist."""
        
        # Try to load pre-built index
        try:
            index_path = Path(self.settings.embeddings_path) / "faiss_index.index"
            if index_path.exists():
                self.logger.info("Loading pre-built FAISS index...")
                self.index = faiss.read_index(str(index_path))
                self.logger.info("✅ Loaded pre-built FAISS index successfully")
                return
        except Exception as e:
            self.logger.warning(f"Could not load pre-built FAISS index: {e}")
        
        # Fallback: build index from embeddings
        self.logger.info("Building FAISS index from embeddings...")
        self._build_faiss_index()
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for efficient similarity search."""
        
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")
        
        dimension = self.embeddings.shape[1]
        num_embeddings = len(self.embeddings)
        
        # Choose index type based on dataset size and settings
        if self.settings.faiss_index_type == "flat" or num_embeddings < 1000:
            # Use flat index for small datasets
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # Use IVF index for larger datasets
            nlist = self.settings.faiss_nlist or min(100, max(1, num_embeddings // 50))
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Set search parameters
            nprobe = self.settings.faiss_nprobe or min(10, nlist)
            self.index.nprobe = nprobe
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Train and add embeddings
        if hasattr(self.index, 'train'):
            self.index.train(self.embeddings)
        self.index.add(self.embeddings)
        
        index_type = type(self.index).__name__
        self.logger.info(f"Built {index_type} FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform vector similarity search."""
        
        if self.index is None:
            raise ValueError("Search index not initialized")
        
        start_time = time.time()
        
        # Ensure query embedding is normalized and correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        # Perform search
        similarities, indices = self.index.search(query_embedding, k)
        
        # Update performance tracking
        search_time = time.time() - start_time
        self.search_count += 1
        self.total_search_time += search_time
        
        return similarities[0], indices[0]
    
    def save_index(self, path: str) -> None:
        """Save the FAISS index to disk."""
        
        if self.index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, path)
        self.logger.info(f"Saved FAISS index to {path}")
    
    def get_stats(self) -> dict:
        """Get vector search performance statistics."""
        
        avg_search_time = 0.0
        if self.search_count > 0:
            avg_search_time = (self.total_search_time / self.search_count) * 1000
        
        return {
            'index_type': type(self.index).__name__ if self.index else None,
            'index_size': self.index.ntotal if self.index else 0,
            'embeddings_loaded': self.embeddings is not None,
            'search_count': self.search_count,
            'avg_search_time_ms': avg_search_time
        }
    
    def is_ready(self) -> bool:
        """Check if vector search is ready for queries."""
        return self.index is not None and self.embeddings is not None