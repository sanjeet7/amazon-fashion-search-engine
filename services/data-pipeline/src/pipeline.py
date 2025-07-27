"""Main data pipeline orchestrator."""

import asyncio
import logging
import time
from typing import Dict, Any

from shared.models import Settings
from shared.utils import setup_logger

from .data_processor import DataProcessor
from .embedding_generator import EmbeddingGenerator


class DataPipeline:
    """Main data pipeline orchestrator."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logger("data-pipeline", settings.log_level)
        
        # Initialize components
        self.data_processor = DataProcessor(settings)
        self.embedding_generator = EmbeddingGenerator(settings)
    
    async def run(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Run the complete data pipeline."""
        
        start_time = time.time()
        self.logger.info("Starting data pipeline...")
        
        try:
            # Check if we should use existing data
            if not force_rebuild and self._processed_data_exists():
                self.logger.info("Using existing processed data")
                df, embedding_texts = self.data_processor.load_processed_data()
                
                if not force_rebuild and self.embedding_generator.embeddings_exist():
                    self.logger.info("Using existing embeddings")
                    embeddings, product_ids, metadata = self.embedding_generator.load_embeddings()
                    
                    return {
                        'status': 'success',
                        'source': 'existing',
                        'total_products': len(df),
                        'total_embeddings': len(embeddings),
                        'estimated_cost': metadata.get('estimated_cost', 0.0),
                        'execution_time': time.time() - start_time
                    }
            
            # Step 1: Load and process raw data
            self.logger.info("Step 1: Loading raw data...")
            raw_products = self.data_processor.load_raw_data()
            
            self.logger.info("Step 2: Processing products...")
            df, embedding_texts = self.data_processor.process_products(raw_products)
            
            # Save processed data
            self.data_processor.save_processed_data(df, embedding_texts)
            
            # Step 3: Generate embeddings (if needed)
            if force_rebuild or not self.embedding_generator.embeddings_exist():
                self.logger.info("Step 3: Generating embeddings...")
                embeddings, embedding_metadata = await self.embedding_generator.generate_embeddings(embedding_texts)
                
                # Save embeddings
                product_ids = df['parent_asin'].tolist()
                self.embedding_generator.save_embeddings(embeddings, product_ids, embedding_metadata)
            else:
                self.logger.info("Step 3: Loading existing embeddings...")
                embeddings, product_ids, embedding_metadata = self.embedding_generator.load_embeddings()
            
            # Step 4: Calculate statistics
            self.logger.info("Step 4: Calculating statistics...")
            stats = self.data_processor.calculate_statistics(df, embedding_texts)
            
            execution_time = time.time() - start_time
            
            # Final results
            results = {
                'status': 'success',
                'source': 'generated',
                'total_products': len(df),
                'total_embeddings': len(embeddings),
                'estimated_cost': embedding_metadata.get('estimated_cost', 0.0),
                'execution_time': execution_time,
                'statistics': stats
            }
            
            self.logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            self.logger.info(f"Processed {results['total_products']} products")
            self.logger.info(f"Generated {results['total_embeddings']} embeddings")
            self.logger.info(f"Estimated cost: ${results['estimated_cost']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _processed_data_exists(self) -> bool:
        """Check if processed data already exists."""
        data_file = self.settings.processed_data_dir / "processed_products.parquet"
        texts_file = self.settings.processed_data_dir / "embedding_texts.json"
        return data_file.exists() and texts_file.exists()
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status and information."""
        
        status = {
            'processed_data_exists': self._processed_data_exists(),
            'embeddings_exist': self.embedding_generator.embeddings_exist(),
            'data_source': str(self.settings.data_path),
            'effective_sample_size': self.settings.effective_sample_size,
            'use_sample_data': self.settings.use_sample_data
        }
        
        # Add statistics if data exists
        if status['processed_data_exists']:
            try:
                df, embedding_texts = self.data_processor.load_processed_data()
                status['total_products'] = len(df)
                status['total_embedding_texts'] = len(embedding_texts)
            except Exception as e:
                self.logger.warning(f"Could not load processed data for status: {e}")
        
        if status['embeddings_exist']:
            try:
                embeddings, product_ids, metadata = self.embedding_generator.load_embeddings()
                status['total_embeddings'] = len(embeddings)
                status['embedding_metadata'] = metadata
            except Exception as e:
                self.logger.warning(f"Could not load embeddings for status: {e}")
        
        return status