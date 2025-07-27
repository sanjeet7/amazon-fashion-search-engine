"""FAISS index builder for vector similarity search."""

import time
import pickle
from typing import List, Dict, Any
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

# Removed shared imports - will use local config classes


class DataPipelineConfig:
    """Simple configuration for data pipeline."""
    def __init__(self):
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed") 
        self.embeddings_cache_dir = Path("data/embeddings")
        self.dataset_filename = "meta_Amazon_Fashion.jsonl"
        self.sample_size = 150000


class IndexResult:
    """Result container for index building."""
    
    def __init__(self):
        self.index: faiss.Index = None
        self.total_vectors: int = 0
        self.dimension: int = 0
        self.index_size_mb: float = 0.0
        self.build_time: float = 0.0
        self.index_type: str = ""


class IndexBuilder:
    """FAISS index builder for efficient vector similarity search."""
    
    def __init__(self, config: DataPipelineConfig, logger: logging.Logger):
        """Initialize index builder.
        
        Args:
            config: Data pipeline configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.cache_dir = config.embeddings_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialized IndexBuilder")

    async def build_index(
        self, 
        embeddings: List[np.ndarray], 
        df: pd.DataFrame, 
        product_ids: List[str]
    ) -> IndexResult:
        """Build FAISS index from embeddings.
        
        Args:
            embeddings: List of embedding vectors
            df: Product dataframe
            product_ids: List of product IDs
            
        Returns:
            IndexResult with built index and metadata
        """
        start_time = time.time()
        result = IndexResult()
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype(np.float32)
        
        result.total_vectors = len(embeddings_array)
        result.dimension = embeddings_array.shape[1]
        
        self.logger.info(f"Building FAISS index for {result.total_vectors:,} vectors (dim={result.dimension})")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Choose index type based on dataset size
        if result.total_vectors < 10000:
            # Use flat index for small datasets
            index = faiss.IndexFlatIP(result.dimension)
            result.index_type = "IndexFlatIP"
            self.logger.info("Using IndexFlatIP (exact search)")
        else:
            # Use IVF index for larger datasets
            nlist = min(int(np.sqrt(result.total_vectors)), 1000)  # Number of clusters
            quantizer = faiss.IndexFlatIP(result.dimension)
            index = faiss.IndexIVFFlat(quantizer, result.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            result.index_type = f"IndexIVFFlat(nlist={nlist})"
            
            # Train the index
            self.logger.info(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings_array)
            self.logger.info("Using IndexIVFFlat (approximate search)")
        
        # Add vectors to index
        self.logger.info("Adding vectors to index...")
        index.add(embeddings_array)
        
        # Set search parameters for IVF index
        if hasattr(index, 'nprobe'):
            index.nprobe = min(10, max(1, int(0.1 * index.nlist)))  # Search 10% of clusters
            self.logger.info(f"Set nprobe to {index.nprobe}")
        
        result.index = index
        result.build_time = time.time() - start_time
        
        # Calculate index size
        result.index_size_mb = self._estimate_index_size(index) / (1024 * 1024)
        
        self.logger.info(f"Index building complete:")
        self.logger.info(f"  Index type: {result.index_type}")
        self.logger.info(f"  Total vectors: {result.total_vectors:,}")
        self.logger.info(f"  Dimension: {result.dimension}")
        self.logger.info(f"  Index size: {result.index_size_mb:.1f} MB")
        self.logger.info(f"  Build time: {result.build_time:.1f}s")
        
        return result

    def _estimate_index_size(self, index: faiss.Index) -> int:
        """Estimate index size in bytes.
        
        Args:
            index: FAISS index
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Serialize index to estimate size
            index_bytes = faiss.serialize_index(index)
            return len(index_bytes)
        except Exception:
            # Fallback estimation
            if hasattr(index, 'ntotal') and hasattr(index, 'd'):
                return index.ntotal * index.d * 4  # 4 bytes per float32
            return 0

    async def save_index(self, result: IndexResult) -> None:
        """Save FAISS index to disk.
        
        Args:
            result: IndexResult to save
        """
        # Save FAISS index
        index_file = self.cache_dir / "faiss_index.bin"
        faiss.write_index(result.index, str(index_file))
        
        # Save metadata
        metadata_file = self.cache_dir / "index_metadata.pkl"
        metadata = {
            'total_vectors': result.total_vectors,
            'dimension': result.dimension,
            'index_size_mb': result.index_size_mb,
            'build_time': result.build_time,
            'index_type': result.index_type
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"Saved index to {index_file}")
        self.logger.info(f"Saved metadata to {metadata_file}")

    async def load_index(self) -> IndexResult:
        """Load FAISS index from disk.
        
        Returns:
            IndexResult loaded from disk
        """
        index_file = self.cache_dir / "faiss_index.bin"
        metadata_file = self.cache_dir / "index_metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found at {index_file}")
        
        # Load FAISS index
        index = faiss.read_index(str(index_file))
        
        # Load metadata
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
        
        # Create result
        result = IndexResult()
        result.index = index
        result.total_vectors = metadata.get('total_vectors', index.ntotal)
        result.dimension = metadata.get('dimension', index.d)
        result.index_size_mb = metadata.get('index_size_mb', 0.0)
        result.build_time = metadata.get('build_time', 0.0)
        result.index_type = metadata.get('index_type', 'Unknown')
        
        self.logger.info(f"Loaded index from {index_file}: {result.total_vectors:,} vectors")
        return result

    def test_index_performance(self, index: faiss.Index, test_queries: np.ndarray, k: int = 10) -> Dict[str, float]:
        """Test index search performance.
        
        Args:
            index: FAISS index to test
            test_queries: Test query vectors
            k: Number of results to retrieve
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Testing index performance with {len(test_queries)} queries...")
        
        start_time = time.time()
        
        # Perform searches
        scores, indices = index.search(test_queries, k)
        
        search_time = time.time() - start_time
        
        metrics = {
            'total_search_time': search_time,
            'avg_search_time_ms': (search_time / len(test_queries)) * 1000,
            'queries_per_second': len(test_queries) / search_time,
            'avg_score': float(np.mean(scores[scores > 0]))  # Average non-zero scores
        }
        
        self.logger.info(f"Performance test results:")
        self.logger.info(f"  Average search time: {metrics['avg_search_time_ms']:.1f}ms")
        self.logger.info(f"  Queries per second: {metrics['queries_per_second']:.1f}")
        self.logger.info(f"  Average similarity score: {metrics['avg_score']:.3f}")
        
        return metrics 