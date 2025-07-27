"""Main entry point for the Data Pipeline service with multiple execution modes."""

import asyncio
import sys
import argparse
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Using local config and logger setup
from .modes import FullPipelineMode, LoadExistingMode, SamplePipelineMode
import os
import logging


class EmbeddingConfig:
    """Simple embedding configuration."""
    def __init__(self):
        self.model = "text-embedding-3-small"
        self.model_name = "text-embedding-3-small"  # For tokenizer
        self.batch_size = 100
        self.max_retries = 3
        self.rate_limit_delay = 0.1
        self.cost_per_1k_tokens = 0.00002  # Cost for text-embedding-3-small


class DataPipelineConfig:
    """Simple configuration for data pipeline."""
    def __init__(self):
        # Use absolute paths from app root
        app_root = Path("/app")
        self.data_dir = app_root / "data"  # Base data directory
        self.raw_data_dir = app_root / "data" / "raw"
        self.processed_data_dir = app_root / "data" / "processed"
        self.embeddings_cache_dir = app_root / "data" / "embeddings"
        self.dataset_filename = "meta_Amazon_Fashion.jsonl"
        self.sample_size = 150000
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.embedding = EmbeddingConfig()


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Set up a logger with the specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Pipeline for Amazon Fashion Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode=full                    # Complete pipeline
  python -m src.main --mode=sample --size=500       # Sample pipeline  
  python -m src.main --mode=load                    # Load existing data
  python -m src.main --mode=status                  # Check data status
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'sample', 'load', 'status'],
        default='sample',
        help='Pipeline execution mode (default: sample)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=500,
        help='Sample size for sample mode (default: 500)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


async def show_data_status(config: DataPipelineConfig, logger):
    """Show status of available data files."""
    logger.info("Checking data file status...")
    
    # Check for different data files
    files_to_check = [
        ('Raw Dataset', config.raw_data_dir / config.dataset_filename),
        ('Processed Data', config.processed_data_dir / 'processed_sample.parquet'),
        ('Sample Data', config.processed_data_dir / 'sample_data.parquet'),
        ('Embeddings', config.embeddings_cache_dir / 'embeddings.npz'),
        ('Sample Embeddings', config.embeddings_cache_dir / 'sample_embeddings.npz'),
        ('FAISS Index', config.embeddings_cache_dir / 'faiss_index.bin'),
        ('Sample Index', config.embeddings_cache_dir / 'sample_index.bin'),
    ]
    
    print("\n" + "="*60)
    print("📊 DATA FILE STATUS")
    print("="*60)
    
    for name, file_path in files_to_check:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"✅ {name:<20} {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {name:<20} {file_path}")
    
    print("="*60)
    
    # Check if we can run each mode
    print("\n🚀 AVAILABLE PIPELINE MODES:")
    
    # Check sample mode
    if (config.raw_data_dir / config.dataset_filename).exists():
        print("✅ sample - Quick testing with small dataset")
    else:
        print("❌ sample - Raw dataset required")
    
    # Check full mode  
    if (config.raw_data_dir / config.dataset_filename).exists():
        print("✅ full   - Complete pipeline with stratified sampling")
    else:
        print("❌ full   - Raw dataset required")
    
    # Check load mode
    load_mode = LoadExistingMode(config, logger)
    if load_mode.validate_prerequisites():
        print("✅ load   - Load existing embeddings and index")
    else:
        print("❌ load   - Pre-built data files required")
    
    print("\n💡 RECOMMENDATIONS:")
    if not (config.raw_data_dir / config.dataset_filename).exists():
        print("  • Add the Amazon Fashion dataset to data/raw/meta_Amazon_Fashion.jsonl")
    
    embeddings_exist = (config.embeddings_cache_dir / 'embeddings.npz').exists()
    if not embeddings_exist:
        print("  • Run 'make data-sample' for quick testing")
        print("  • Run 'make data-full' for complete processing")
    else:
        print("  • Run 'make data-load' to load existing data")
        print("  • Run 'make dev' to start the search system")


async def main():
    """Main entry point for data pipeline service."""
    args = parse_arguments()
    
    # Load configuration
    config = DataPipelineConfig()
    
    # Setup logging
    logger = setup_logger("data-pipeline", args.log_level)
    logger.info(f"Starting Data Pipeline Service - Mode: {args.mode}")
    
    try:
        if args.mode == 'status':
            await show_data_status(config, logger)
            return
        
        # Initialize appropriate mode
        if args.mode == 'full':
            pipeline_mode = FullPipelineMode(config, logger)
            mode_name = "Full Pipeline"
        elif args.mode == 'sample':
            pipeline_mode = SamplePipelineMode(config, logger, args.size)
            mode_name = f"Sample Pipeline ({args.size} products)"
        elif args.mode == 'load':
            pipeline_mode = LoadExistingMode(config, logger)
            mode_name = "Load Existing Data"
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Validate prerequisites
        logger.info(f"Validating prerequisites for {mode_name}...")
        if not pipeline_mode.validate_prerequisites():
            logger.error("Prerequisites validation failed")
            sys.exit(1)
        
        # Show cost estimate for sample mode
        if args.mode == 'sample':
            cost_estimate = pipeline_mode.estimate_cost()
            print(f"\n💰 COST ESTIMATE:")
            print(f"  Sample size: {cost_estimate['sample_size']:,} products")
            print(f"  Estimated tokens: {cost_estimate['estimated_tokens']:,}")
            print(f"  Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")
            print(f"  Cost per product: ${cost_estimate['cost_per_product']:.6f}")
            
            # Ask for confirmation
            response = input("\nProceed with pipeline execution? (y/N): ")
            if response.lower() != 'y':
                print("Pipeline execution cancelled.")
                return
        
        # Execute pipeline
        logger.info(f"Starting {mode_name} execution...")
        start_time = time.time()
        
        results = await pipeline_mode.execute()
        
        execution_time = time.time() - start_time
        
        # Print results summary
        print("\n" + "="*60)
        print(f"🎉 {mode_name.upper()} COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        
        if 'dataset_stats' in results:
            stats = results['dataset_stats']
            print(f"📊 Dataset: {stats.get('total_records', 0):,} records processed")
        
        if 'embedding_stats' in results:
            stats = results['embedding_stats']
            print(f"🧠 Embeddings: {stats.get('total_embeddings', 0):,} generated")
            print(f"💰 Cost: ${stats.get('total_cost', 0):.4f}")
        
        if 'index_stats' in results:
            stats = results['index_stats']
            print(f"🔍 Index: {stats.get('total_vectors', 0):,} vectors ({stats.get('dimension', 0)}D)")
        
        if 'files_created' in results:
            print(f"📁 Files created:")
            for name, path in results['files_created'].items():
                print(f"    {name}: {path}")
        
        print("="*60)
        print("\n🚀 Ready to start Search API service!")
        print("   Run: make dev")
        
    except KeyboardInterrupt:
        logger.info("Data pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        print("\nFor help, run: python -m src.main --help")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 