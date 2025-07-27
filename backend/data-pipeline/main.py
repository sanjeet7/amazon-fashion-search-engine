"""Main entry point for the Data Pipeline service.

This service handles data ingestion, processing, embedding generation,
and FAISS index building for the Amazon Fashion dataset.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.shared.models import DataPipelineConfig
from backend.shared.utils import setup_logger
from src.pipeline import DataPipeline


async def main():
    """Main entry point for data pipeline service."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Amazon Fashion Data Pipeline")
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None,
        help="Number of products to process (default: use config)"
    )
    parser.add_argument(
        "--force-rebuild", 
        action="store_true",
        help="Force rebuild even if processed data exists"
    )
    parser.add_argument(
        "--no-use-existing", 
        action="store_true",
        help="Don't use existing processed data"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = DataPipelineConfig(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Setup logging
    logger = setup_logger("data-pipeline", config.log_level)
    logger.info("Starting Data Pipeline Service")
    logger.info(f"Configuration: {config.model_dump()}")
    
    # Add console output for debugging
    print("🚀 Starting Data Pipeline Service")
    print(f"📁 Raw data path: {config.raw_data_path}")
    print(f"📁 Processed data dir: {config.processed_data_dir}")
    print(f"📁 Embeddings cache dir: {config.embeddings_cache_dir}")
    print(f"🔑 OpenAI API key configured: {'Yes' if config.openai_api_key else 'No'}")
    print(f"📊 Sample size: {sample_size:,}")
    print(f"🔄 Force rebuild: {args.force_rebuild}")
    print(f"📂 Use existing: {not args.no_use_existing}")
    print("-" * 60)
    
    # Log pipeline options
    sample_size = args.sample_size or config.sample_size
    logger.info(f"Pipeline options:")
    logger.info(f"  Sample size: {sample_size:,} products")
    logger.info(f"  Force rebuild: {args.force_rebuild}")
    logger.info(f"  Use existing: {not args.no_use_existing}")
    
    try:
        # Initialize pipeline
        pipeline = DataPipeline(config)
        
        # Run the pipeline
        logger.info("Starting data pipeline execution...")
        start_time = time.time()
        
        results = await pipeline.run(
            sample_size=args.sample_size,
            force_rebuild=args.force_rebuild,
            use_existing=not args.no_use_existing
        )
        
        execution_time = time.time() - start_time
        
        # Print final status
        print("\n" + "="*60)
        print("🎉 DATA PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print(f"📊 Products processed: {results['sample_size']:,}")
        print(f"💰 Total cost: ${results['total_cost']:.2f}")
        print(f"📁 Processed data: {config.processed_data_dir}")
        print(f"🔍 Embeddings cache: {config.embeddings_cache_dir}")
        print("="*60)
        print("\n🚀 Ready to start Search API service!")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        logger.error(f"Data pipeline failed: {e}", exc_info=True)
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Data pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 