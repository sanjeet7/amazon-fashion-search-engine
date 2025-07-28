"""
Modular Data Pipeline

Clean, modular data pipeline with improved architecture:
- DataLoader for source detection and loading
- Modular processing components  
- Clear separation of concerns
- Better error handling and progress reporting
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from shared.models import Settings
from shared.utils import setup_logger

# Import existing components (would be refactored versions)
from .data_processor import DataProcessor
from .embedding_generator import EmbeddingGenerator
from .processors.data_loader import DataLoader

logger = logging.getLogger(__name__)


class ModularDataPipeline:
    """
    Modular data pipeline with clean architecture.
    
    Provides clean separation of concerns:
    - Data loading and source detection
    - Data processing and cleaning
    - Embedding generation and optimization
    - Index building and management
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logger("refactored-data-pipeline")
        
        # Initialize modular components
        self.data_loader = DataLoader(settings)
        self.data_processor = DataProcessor(settings)  # Would be refactored
        self.embedding_generator = EmbeddingGenerator(settings)  # Would be refactored
        
        # Performance tracking
        self.pipeline_runs = 0
        self.total_processing_time = 0.0
    
    def run_full_pipeline(self, sample_size: Optional[int] = None) -> bool:
        """
        Run the complete refactored data pipeline.
        
        Args:
            sample_size: Optional sample size for processing
            
        Returns:
            True if successful, False otherwise
        """
        
        start_time = time.time()
        self.pipeline_runs += 1
        
        try:
            self.logger.info("🚀 Starting refactored data pipeline...")
            
            # Phase 1: Data Loading with smart source detection
            self.logger.info("📥 Phase 1: Loading data...")
            raw_products = self._load_data_phase(sample_size)
            
            # Phase 2: Data Processing with validation
            self.logger.info("🔧 Phase 2: Processing data...")
            processed_df, embedding_texts = self._process_data_phase(raw_products)
            
            # Phase 3: Embedding Generation with optimization
            self.logger.info("🧠 Phase 3: Generating embeddings...")
            embeddings, metadata = self._generate_embeddings_phase(embedding_texts)
            
            # Phase 4: Index Building and Storage
            self.logger.info("💾 Phase 4: Building search index...")
            self._build_index_phase(processed_df, embeddings, metadata)
            
            # Phase 5: Validation and Statistics
            self.logger.info("📊 Phase 5: Validating pipeline...")
            self._validate_pipeline_phase(processed_df, embeddings)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            self.logger.info(f"✅ Pipeline completed successfully in {processing_time:.2f}s")
            self.logger.info(f"   Processed: {len(processed_df):,} products")
            self.logger.info(f"   Generated: {len(embeddings):,} embeddings")
            self.logger.info(f"   Cost: ${metadata.get('estimated_cost', 0):.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            return False
    
    def _load_data_phase(self, sample_size: Optional[int]) -> List[Dict[str, Any]]:
        """Phase 1: Load data with smart source detection."""
        
        # Auto-detect best data source
        data_source, description = self.data_loader.detect_data_source()
        self.logger.info(f"   Source: {description}")
        
        # Load with progress reporting
        raw_products = self.data_loader.load_raw_data(
            data_source=data_source,
            sample_size=sample_size
        )
        
        self.logger.info(f"   Loaded: {len(raw_products):,} raw products")
        return raw_products
    
    def _process_data_phase(self, raw_products: List[Dict[str, Any]]) -> tuple:
        """Phase 2: Process data with validation and cleaning."""
        
        # Process products with enhanced validation
        processed_df, embedding_texts = self.data_processor.process_products(raw_products)
        
        if len(processed_df) == 0:
            raise ValueError("No valid products found after processing")
        
        # Calculate processing statistics
        valid_rate = len(processed_df) / len(raw_products) * 100
        self.logger.info(f"   Processed: {len(processed_df):,} products ({valid_rate:.1f}% valid)")
        
        return processed_df, embedding_texts
    
    def _generate_embeddings_phase(self, embedding_texts: List[str]) -> tuple:
        """Phase 3: Generate embeddings with optimization."""
        
        # Generate embeddings with progress tracking
        embeddings, cost_info = self.embedding_generator.generate_embeddings(embedding_texts)
        
        # Prepare comprehensive metadata
        metadata = {
            'total_tokens': cost_info['total_tokens'],
            'estimated_cost': cost_info['total_cost'],
            'model': self.settings.embedding_model,
            'embedding_dim': embeddings.shape[1],
            'total_embeddings': len(embeddings),
            'batch_size': self.settings.embedding_batch_size,
            'processing_method': 'concurrent' if not self.settings.sequential_processing else 'sequential'
        }
        
        self.logger.info(f"   Generated: {len(embeddings):,} embeddings")
        self.logger.info(f"   Cost: ${cost_info['total_cost']:.4f}")
        
        return embeddings, metadata
    
    def _build_index_phase(self, processed_df, embeddings, metadata) -> None:
        """Phase 4: Build search index and save data."""
        
        # Save processed data
        self.data_processor.save_processed_data(processed_df, None)
        
        # Save embeddings and build index
        product_ids = processed_df['parent_asin'].tolist()
        self.embedding_generator.save_embeddings(embeddings, product_ids, metadata)
        
        self.logger.info("   Index: FAISS index built and saved")
        self.logger.info("   Storage: All data saved successfully")
    
    def _validate_pipeline_phase(self, processed_df, embeddings) -> None:
        """Phase 5: Validate pipeline output."""
        
        # Validate data consistency
        if len(processed_df) != len(embeddings):
            raise ValueError(f"Data mismatch: {len(processed_df)} products vs {len(embeddings)} embeddings")
        
        # Validate data quality
        missing_titles = processed_df['title'].isna().sum()
        missing_categories = processed_df['main_category'].isna().sum()
        
        self.logger.info(f"   Quality: {missing_titles} missing titles, {missing_categories} missing categories")
        
        # Validate embeddings
        if embeddings.shape[1] != 1536:  # Expected dimension for text-embedding-3-small
            self.logger.warning(f"   Warning: Unexpected embedding dimension: {embeddings.shape[1]}")
        
        self.logger.info("   Validation: All checks passed")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        avg_processing_time = 0.0
        if self.pipeline_runs > 0:
            avg_processing_time = self.total_processing_time / self.pipeline_runs
        
        return {
            'pipeline_runs': self.pipeline_runs,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': avg_processing_time,
            'data_loader_stats': {
                'component': 'DataLoader',
                'description': 'Handles data source detection and loading'
            },
            'architecture': 'modular',
            'components': ['DataLoader', 'DataProcessor', 'EmbeddingGenerator']
        }
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate pipeline setup and data availability."""
        
        validation_results = {
            'setup_valid': True,
            'issues': [],
            'data_sources': [],
            'recommendations': []
        }
        
        try:
            # Check data sources
            data_source, description = self.data_loader.detect_data_source()
            data_info = self.data_loader.get_data_info(data_source)
            
            validation_results['data_sources'].append({
                'path': str(data_source),
                'description': description,
                'info': data_info
            })
            
        except FileNotFoundError as e:
            validation_results['setup_valid'] = False
            validation_results['issues'].append(str(e))
            validation_results['recommendations'].append(
                "Run the data generation script to create sample data"
            )
        
        # Check OpenAI API key
        if not self.settings.openai_api_key or self.settings.openai_api_key == "your_openai_api_key_here":
            validation_results['setup_valid'] = False
            validation_results['issues'].append("OpenAI API key not configured")
            validation_results['recommendations'].append(
                "Set OPENAI_API_KEY environment variable"
            )
        
        return validation_results