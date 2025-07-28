#!/usr/bin/env python3
"""
Enhanced Data Pipeline CLI for Amazon Fashion Search Engine.

This script provides a user-friendly interface for data processing with options for:
- Using preloaded data (default)
- Rebuilding with custom sample sizes
- Full dataset processing
- Data validation and overwrite protection
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.models import Settings
from shared.utils import setup_logger
from .src.pipeline import DataPipeline


def setup_cli_logging(level: str = "INFO") -> logging.Logger:
    """Setup CLI-friendly logging with progress indicators."""
    
    logger = setup_logger("data-pipeline-cli", level)
    
    # Add console handler with clean format for CLI
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    # Simple format for CLI
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    
    # Clear existing handlers and add our console handler
    logger.handlers.clear()
    logger.addHandler(console_handler)
    
    return logger


def check_existing_data(settings: Settings) -> dict:
    """Check what data already exists."""
    
    processed_path = Path(settings.processed_data_path) / "processed_products.parquet"
    embeddings_path = Path(settings.embeddings_path) / "embeddings.npy"
    index_path = Path(settings.embeddings_path) / "faiss_index.index"
    
    return {
        'processed_data': processed_path.exists(),
        'embeddings': embeddings_path.exists(),
        'faiss_index': index_path.exists(),
        'processed_path': processed_path,
        'embeddings_path': embeddings_path,
        'index_path': index_path
    }


def get_user_confirmation(message: str) -> bool:
    """Get user confirmation for potentially destructive operations."""
    
    while True:
        response = input(f"{message} (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def estimate_processing_time(sample_size: int, full_dataset: bool = False) -> str:
    """Estimate processing time based on sample size."""
    
    if full_dataset:
        return "2-4 hours (full dataset ~800k products)"
    elif sample_size <= 1000:
        return "2-5 minutes"
    elif sample_size <= 10000:
        return "10-20 minutes" 
    elif sample_size <= 50000:
        return "30-60 minutes"
    else:
        return "1-2 hours"


def main():
    """Main CLI interface for the data pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Amazon Fashion Search Engine - Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use preloaded data (fastest)
  %(prog)s --rebuild                          # Rebuild with 50k sample
  %(prog)s --rebuild --sample-size 1000       # Quick test with 1k products
  %(prog)s --rebuild --full                   # Process complete dataset
  %(prog)s --rebuild --force                  # Overwrite without confirmation

Quick Start (recommended for reviewers):
  1. Just run: %(prog)s
  2. This uses preloaded data and should work immediately
        """
    )
    
    # Primary options
    parser.add_argument(
        "--rebuild", 
        action="store_true",
        help="Rebuild data instead of using preloaded data"
    )
    
    parser.add_argument(
        "--sample-size", 
        type=int, 
        metavar="N",
        help="Custom sample size (default: 50000)"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Process complete dataset (~800k products)"
    )
    
    # Processing options
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force overwrite existing data without confirmation"
    )
    
    parser.add_argument(
        "--sequential", 
        action="store_true",
        help="Use sequential processing (slower, no rate limits)"
    )
    
    # Development options
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate existing data, don't process"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_cli_logging(args.log_level)
    
    try:
        # Load settings
        logger.info("🔧 Loading configuration...")
        settings = Settings()
        
        # Override settings based on CLI args
        if args.sample_size:
            settings.stratified_sample_size = args.sample_size
        if args.sequential:
            settings.sequential_processing = True
        
        # Check existing data
        existing_data = check_existing_data(settings)
        
        # Display current status
        logger.info("📊 Current Data Status:")
        logger.info(f"   Processed Data: {'✅ Found' if existing_data['processed_data'] else '❌ Missing'}")
        logger.info(f"   Embeddings:     {'✅ Found' if existing_data['embeddings'] else '❌ Missing'}")
        logger.info(f"   FAISS Index:    {'✅ Found' if existing_data['faiss_index'] else '❌ Missing'}")
        
        # Validation-only mode
        if args.validate_only:
            if all(existing_data[key] for key in ['processed_data', 'embeddings', 'faiss_index']):
                logger.info("✅ All data is present and ready for use!")
                return 0
            else:
                logger.error("❌ Some data is missing. Run with --rebuild to generate it.")
                return 1
        
        # Default behavior: use preloaded data
        if not args.rebuild:
            if all(existing_data[key] for key in ['processed_data', 'embeddings', 'faiss_index']):
                logger.info("✅ Using preloaded data (recommended for quick testing)")
                logger.info("💡 Data is ready! You can now start the search API:")
                logger.info("   python services/search-api/main.py")
                return 0
            else:
                logger.warning("⚠️  Preloaded data not found. Use --rebuild to generate it.")
                logger.info("💡 Quick start: python services/data-pipeline/main.py --rebuild")
                return 1
        
        # Rebuild mode
        if args.full:
            sample_size = None  # Process full dataset
            logger.info("🔄 Rebuilding with FULL dataset (~800k products)")
        else:
            sample_size = settings.stratified_sample_size
            logger.info(f"🔄 Rebuilding with {sample_size:,} product sample")
        
        # Estimate processing time
        estimated_time = estimate_processing_time(sample_size or 800000, args.full)
        logger.info(f"⏱️  Estimated processing time: {estimated_time}")
        
        # Check for overwrites
        if any(existing_data[key] for key in ['processed_data', 'embeddings', 'faiss_index']):
            if not args.force:
                logger.warning("⚠️  Existing data will be overwritten:")
                if existing_data['processed_data']:
                    logger.warning(f"   - {existing_data['processed_path']}")
                if existing_data['embeddings']:
                    logger.warning(f"   - {existing_data['embeddings_path']}")
                if existing_data['faiss_index']:
                    logger.warning(f"   - {existing_data['index_path']}")
                
                if not get_user_confirmation("Continue with rebuild?"):
                    logger.info("❌ Rebuild cancelled by user")
                    return 0
        
        # Initialize and run pipeline
        logger.info("🚀 Starting data pipeline...")
        start_time = time.time()
        
        pipeline = DataPipeline(settings)
        
        # Run the pipeline
        if args.full:
            result = pipeline.run_full_pipeline()
        else:
            result = pipeline.run_full_pipeline(sample_size=sample_size)
        
        elapsed_time = time.time() - start_time
        
        if result:
            logger.info(f"✅ Pipeline completed successfully in {elapsed_time:.1f}s")
            logger.info("🎉 Data is ready! You can now start the search API:")
            logger.info("   python services/search-api/main.py")
            return 0
        else:
            logger.error("❌ Pipeline failed")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        else:
            logger.info("💡 Use --log-level DEBUG for full error details")
        return 1


if __name__ == "__main__":
    sys.exit(main())