"""Main data pipeline orchestrator for processing Amazon Fashion dataset."""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any

from backend.shared.models import DataPipelineConfig, DatasetStats
from backend.shared.utils import setup_logger

from .data_processor import DataProcessor
from .embedding_generator import EmbeddingGenerator
from .index_builder import IndexBuilder


class DataPipeline:
    """Main orchestrator for the data processing pipeline."""
    
    def __init__(self, config: DataPipelineConfig):
        """Initialize the data pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = setup_logger("data-pipeline", config.log_level, config.log_file)
        
        # Initialize components
        self.data_processor = DataProcessor(config, self.logger)
        self.embedding_generator = EmbeddingGenerator(config, self.logger)
        self.index_builder = IndexBuilder(config, self.logger)
        
    async def run(
        self, 
        sample_size: Optional[int] = None,
        force_rebuild: bool = False,
        use_existing: bool = True
    ) -> Dict[str, Any]:
        """Run the data processing pipeline with flexible options.
        
        Args:
            sample_size: Number of products to process (None = use config default)
            force_rebuild: Whether to force rebuild even if processed data exists
            use_existing: Whether to use existing processed data if available
            
        Returns:
            Pipeline execution results and statistics
        """
        start_time = time.time()
        self.logger.info("Starting data pipeline execution")
        
        # Use provided sample size or config default
        effective_sample_size = sample_size or self.config.sample_size
        
        try:
            # Step 1: Check if processed data already exists
            print("🔍 Step 1/5: Checking for existing data...")
            if use_existing and not force_rebuild:
                existing_data = await self._check_existing_data()
                if existing_data:
                    print("✅ Using existing processed data")
                    self.logger.info("✅ Using existing processed data")
                    return existing_data
            print("🔄 No existing data found or force rebuild requested")
            
            # Step 2: Analyze raw dataset
            print("📊 Step 2/5: Analyzing raw dataset...")
            self.logger.info("Step 2/5: Analyzing raw dataset")
            dataset_info = await self.data_processor.analyze_dataset()
            print(f"✅ Dataset analyzed: {dataset_info.total_products:,} records")
            self.logger.info(f"Dataset: {dataset_info.total_products:,} records")
            
            # Step 3: Perform stratified sampling (from final_exploration.md strategy)
            print(f"🎯 Step 3/5: Performing stratified sampling (target: {effective_sample_size:,})...")
            self.logger.info(f"Step 3/5: Performing stratified quality-based sampling (target: {effective_sample_size:,})")
            processed_df = await self.data_processor.stratified_quality_sampling(
                sample_size=effective_sample_size
            )
            print(f"✅ Sampled {len(processed_df):,} high-quality products")
            self.logger.info(f"Sampled {len(processed_df):,} high-quality products")
            
            # Step 4: Generate embeddings
            print("🧠 Step 4/5: Generating embeddings...")
            self.logger.info("Step 4/5: Generating embeddings")
            embeddings_result = await self.embedding_generator.generate_batch_embeddings(processed_df)
            print(f"✅ Generated {len(embeddings_result.embeddings):,} embeddings")
            print(f"💰 Total cost: ${embeddings_result.total_cost:.2f}")
            self.logger.info(f"Generated {len(embeddings_result.embeddings):,} embeddings")
            self.logger.info(f"Total cost: ${embeddings_result.total_cost:.2f}")
            
            # Step 5: Build search index
            print("🔍 Step 5/5: Building FAISS search index...")
            self.logger.info("Step 5/5: Building FAISS search index")
            index_result = await self.index_builder.build_index(
                embeddings_result.embeddings,
                processed_df,
                embeddings_result.product_ids
            )
            print(f"✅ Built index with {index_result.total_vectors:,} vectors")
            self.logger.info(f"Built index with {index_result.total_vectors:,} vectors")
            
            # Step 6: Save all processed data
            print("💾 Step 6/6: Saving processed data...")
            self.logger.info("Step 6/6: Saving processed data")
            await self._save_pipeline_results(
                processed_df=processed_df,
                embeddings_result=embeddings_result,
                index_result=index_result
            )
            print("✅ All data saved successfully")
            
            # Calculate final statistics
            execution_time = time.time() - start_time
            
            final_stats = {
                "status": "completed",
                "execution_time": execution_time,
                "sample_size": len(processed_df),
                "total_cost": embeddings_result.total_cost,
                "index_vectors": index_result.total_vectors,
                "dataset_info": dataset_info.model_dump(),
                "quality_distribution": self._calculate_quality_distribution(processed_df)
            }
            
            self.logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            return final_stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    async def _check_existing_data(self) -> Optional[Dict[str, Any]]:
        """Check if processed data already exists and is valid.
        
        Returns:
            Existing data stats if valid, None otherwise
        """
        try:
            processed_file = self.config.processed_data_dir / "processed_sample.parquet"
            embeddings_file = self.config.embeddings_cache_dir / "embeddings.npz"
            index_file = self.config.processed_data_dir / "faiss_index.bin"
            
            if all(f.exists() for f in [processed_file, embeddings_file, index_file]):
                # Load and validate existing data
                processed_df = await self.data_processor.load_processed_data()
                if len(processed_df) > 0:
                    return {
                        "status": "existing_data",
                        "sample_size": len(processed_df),
                        "message": "Using existing processed data"
                    }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error checking existing data: {e}")
            return None

    async def _save_pipeline_results(
        self,
        processed_df,
        embeddings_result,
        index_result
    ) -> None:
        """Save all pipeline results to disk."""
        try:
            # Save processed data
            await self.data_processor.save_processed_data(processed_df)
            
            # Save embeddings
            await self.embedding_generator.save_embeddings(
                embeddings_result.embeddings,
                embeddings_result.product_ids,
                embeddings_result.metadata
            )
            
            # Save index
            await self.index_builder.save_index(index_result.index, index_result.metadata)
            
            self.logger.info("All pipeline results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline results: {e}")
            raise

    def _calculate_quality_distribution(self, df) -> Dict[str, int]:
        """Calculate quality tier distribution of processed data."""
        try:
            if 'quality_score' in df.columns:
                high_quality = len(df[df['quality_score'] >= 0.8])
                medium_quality = len(df[(df['quality_score'] >= 0.6) & (df['quality_score'] < 0.8)])
                low_quality = len(df[df['quality_score'] < 0.6])
                
                return {
                    "high_quality": high_quality,
                    "medium_quality": medium_quality,
                    "low_quality": low_quality
                }
            else:
                return {"total": len(df)}
        except Exception:
            return {"total": len(df)}


# Legacy method for backward compatibility
async def run_full_pipeline(
    config: DataPipelineConfig,
    force_rebuild: bool = False,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """Legacy method - use DataPipeline.run() instead."""
    pipeline = DataPipeline(config)
    return await pipeline.run(
        sample_size=sample_size,
        force_rebuild=force_rebuild,
        use_existing=True
    )


async def main():
    """Main entry point for the data pipeline service."""
    import argparse
    import os
    from shared.models import DataPipelineConfig
    
    parser = argparse.ArgumentParser(description="Amazon Fashion Data Pipeline")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of products to sample")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild even if data exists")
    parser.add_argument("--config-file", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DataPipelineConfig()
    if args.config_file:
        # Could load from file if needed
        pass
    
    # Override with CLI arguments
    if args.sample_size:
        config.sample_size = args.sample_size
    
    # Create and run pipeline
    pipeline = DataPipeline(config)
    
    try:
        results = await pipeline.run(
            sample_size=args.sample_size,
            force_rebuild=args.force_rebuild,
            use_existing=True
        )
        
        print("\n" + "="*60)
        print("🎉 PIPELINE EXECUTION COMPLETED!")
        print("="*60)
        print(f"📊 Products processed: {results['sample_size']:,}")
        print(f"🧠 Embeddings generated: {results['index_vectors']:,}")
        print(f"💰 Total cost: ${results['total_cost']:.2f}")
        print(f"⏱️  Execution time: {results['execution_time']:.1f} seconds")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main())) 