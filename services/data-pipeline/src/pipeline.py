"""Data pipeline orchestration for Amazon Fashion search."""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from shared.models import Settings
from shared.utils import setup_logger

from .data_processor import DataProcessor
from .embedding_generator import EmbeddingGenerator


class DataPipeline:
    """Orchestrates the complete data processing pipeline."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logger("data-pipeline")
        
        # Initialize components
        self.data_processor = DataProcessor(settings)
        self.embedding_generator = EmbeddingGenerator(settings)
    
    async def run(self, force_rebuild: bool = False, data_source: Optional[Path] = None, test_sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete data pipeline.
        
        Args:
            force_rebuild: Force rebuild even if processed data exists
            data_source: Optional data source path (auto-detected if None)
            test_sample_size: Optional sample size for testing (e.g., 500)
        """
        
        start_time = time.time()
        
        try:
            self.logger.info("Starting data pipeline...")
            
            # Check if we can skip processing
            if not force_rebuild and self._can_skip_processing():
                self.logger.info("Processed data exists and embeddings are current. Skipping processing.")
                df, embedding_texts = self.data_processor.load_processed_data()
                embeddings, product_ids, metadata = self.embedding_generator.load_embeddings()
                
                return {
                    'total_products': len(df),
                    'total_embeddings': len(embeddings),
                    'estimated_cost': metadata.get('estimated_cost', 0),
                    'execution_time': time.time() - start_time,
                    'source': 'existing',
                    'statistics': self.data_processor.get_statistics(df)
                }
            
            # Auto-detect data source if not provided
            if data_source is None:
                data_source, source_description = self.data_processor.detect_data_source()
                self.logger.info(f"Auto-detected data source: {source_description}")
            
            # Step 1: Load raw data
            self.logger.info("Step 1: Loading raw data...")
            raw_products = self.data_processor.load_raw_data(data_source, test_sample_size)
            
            # Step 2: Process products  
            self.logger.info("Step 2: Processing products...")
            df, embedding_texts = self.data_processor.process_products(raw_products)
            
            if len(df) == 0:
                raise ValueError("No valid products found after processing")
            
            # Step 3: Generate embeddings
            self.logger.info("Step 3: Generating embeddings...")
            embeddings, cost_info = await self.embedding_generator.generate_embeddings(embedding_texts)
            
            # Step 4: Save everything
            self.logger.info("Step 4: Saving processed data and embeddings...")
            self.data_processor.save_processed_data(df, embedding_texts)
            
            product_ids = df['parent_asin'].tolist()
            metadata = {
                'total_tokens': cost_info['total_tokens'],
                'estimated_cost': cost_info['total_cost'],
                'model': self.settings.embedding_model,
                'embedding_dim': embeddings.shape[1],
                'total_embeddings': len(embeddings),
                'data_source': str(data_source),
                'test_mode': test_sample_size is not None,
                'sample_size': test_sample_size if test_sample_size else len(df)
            }
            
            self.embedding_generator.save_embeddings(embeddings, product_ids, metadata)
            self.logger.info("✅ Saved embeddings and created FAISS index")
            
            # Step 5: Calculate statistics
            self.logger.info("Step 5: Calculating statistics...")
            statistics = self.data_processor.get_statistics(df)
            
            execution_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            
            return {
                'total_products': len(df),
                'total_embeddings': len(embeddings),
                'estimated_cost': cost_info['total_cost'],
                'execution_time': execution_time,
                'source': 'generated',
                'statistics': statistics
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        
        status = {
            'processed_data_exists': self._processed_data_exists(),
            'embeddings_exist': self._embeddings_exist(),
            'pipeline_ready': self._can_skip_processing()
        }
        
        # Add data source information
        try:
            data_source, source_description = self.data_processor.detect_data_source()
            status['available_data_source'] = str(data_source)
            status['data_source_type'] = source_description
        except FileNotFoundError:
            status['available_data_source'] = None
            status['data_source_type'] = 'No data source found'
        
        # Add processed data info if available
        if status['processed_data_exists']:
            try:
                df, _ = self.data_processor.load_processed_data()
                status['processed_products'] = len(df)
                status['last_processing'] = 'available'
            except Exception:
                status['processed_products'] = 0
                status['last_processing'] = 'error loading'
        
        # Add embeddings info if available
        if status['embeddings_exist']:
            try:
                _, _, metadata = self.embedding_generator.load_embeddings()
                status['embedding_count'] = metadata.get('total_embeddings', 0)
                status['embedding_cost'] = metadata.get('estimated_cost', 0)
                status['embedding_model'] = metadata.get('model', 'unknown')
            except Exception:
                status['embedding_count'] = 0
                status['embedding_cost'] = 0
                status['embedding_model'] = 'error loading'
        
        return status
    
    def _can_skip_processing(self) -> bool:
        """Check if processing can be skipped."""
        return self._processed_data_exists() and self._embeddings_exist()
    
    def _processed_data_exists(self) -> bool:
        """Check if processed data exists."""
        parquet_file = self.settings.processed_data_dir / "processed_products.parquet"
        texts_file = self.settings.processed_data_dir / "embedding_texts.json"
        
        return parquet_file.exists() and texts_file.exists()
    
    def _embeddings_exist(self) -> bool:
        """Check if embeddings exist."""
        embeddings_file = self.settings.embeddings_dir / "embeddings.npy"
        metadata_file = self.settings.embeddings_dir / "metadata.json"
        ids_file = self.settings.embeddings_dir / "product_ids.json"
        faiss_index_file = self.settings.embeddings_dir / "faiss_index.index"
        
        return all(f.exists() for f in [embeddings_file, metadata_file, ids_file, faiss_index_file])