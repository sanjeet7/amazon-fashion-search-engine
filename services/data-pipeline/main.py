#!/usr/bin/env python3
"""Data Pipeline Service - Main Entry Point."""

print("🚀 Starting Data Pipeline...")
print("📦 Loading dependencies (this may take a moment)...")

import asyncio
import argparse
import sys
from pathlib import Path

print("✅ Basic imports loaded")

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("📚 Loading shared modules...")

from shared.models import Settings
from shared.utils.logging import setup_logger

print("🔧 Loading pipeline components...")

from src.pipeline import DataPipeline

print("✅ All dependencies loaded!")


async def main():
    """Main entry point for data pipeline service."""
    
    parser = argparse.ArgumentParser(description="Amazon Fashion Data Pipeline")
    parser.add_argument(
        "--force-rebuild", 
        action="store_true",
        help="Force rebuild even if processed data exists"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Use small test dataset (500 products) for quick testing"
    )
    parser.add_argument(
        "--data-source", 
        type=str,
        help="Specific data source file to use (overrides auto-detection)"
    )
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Show pipeline status and exit"
    )
    
    args = parser.parse_args()
    print(f"⚙️  Arguments parsed: {args}")
    
    try:
        print("🔑 Loading settings...")
        # Load settings
        settings = Settings()
        print("✅ Settings loaded successfully")
        
        print("🏗️  Initializing pipeline...")
        # Initialize pipeline
        pipeline = DataPipeline(settings)
        print("✅ Pipeline initialized")
        
        # Show status if requested
        if args.status:
            print("📊 Getting pipeline status...")
            status = pipeline.get_status()
            print("\n🔍 Data Pipeline Status:")
            print("=" * 50)
            for key, value in status.items():
                print(f"{key}: {value}")
            return
        
        print("🔍 Detecting data source...")
        # Determine data source
        if args.data_source:
            data_source = Path(args.data_source)
            if not data_source.exists():
                print(f"❌ Specified data source not found: {data_source}")
                sys.exit(1)
            source_description = f"specified file ({data_source.name})"
        else:
            try:
                from src.data_processor import DataProcessor
                processor = DataProcessor(settings)
                data_source, source_description = processor.detect_data_source()
                print(f"✅ Auto-detected: {source_description}")
            except FileNotFoundError as e:
                print(f"❌ {e}")
                print("\n💡 Quick start options:")
                print("   1. Generate test sample: python scripts/generate_stratified_sample.py --test")
                print("   2. Generate stratified sample: python scripts/generate_stratified_sample.py")
                print("   3. Use existing raw data with --data-source flag")
                sys.exit(1)
        
        # Apply test mode if requested (override auto-detection)
        test_sample_size = 500 if args.test else None
        
        # Print startup information
        print("\n🚀 Amazon Fashion Data Pipeline")
        print("=" * 50)
        print(f"📊 Data source: {data_source}")
        print(f"📋 Source type: {source_description}")
        if test_sample_size:
            print(f"🧪 Test mode: Processing first {test_sample_size} products only")
        print(f"🔄 Force rebuild: {args.force_rebuild}")
        print("=" * 50)
        
        print("▶️  Starting pipeline execution...")
        
        # Run pipeline
        results = await pipeline.run(
            force_rebuild=args.force_rebuild,
            data_source=data_source,
            test_sample_size=test_sample_size
        )
        
        # Print results
        print("\n✅ Pipeline Completed Successfully!")
        print("=" * 50)
        print(f"📊 Total products: {results['total_products']:,}")
        print(f"🔗 Total embeddings: {results['total_embeddings']:,}")
        print(f"💰 Estimated cost: ${results['estimated_cost']:.4f}")
        print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
        print(f"📁 Data source: {results['source']}")
        print("=" * 50)
        
        if 'statistics' in results:
            stats = results['statistics']
            print("\n📈 Dataset Statistics:")
            print("-" * 30)
            
            # Core metrics
            print(f"Products with price: {stats.get('products_with_price', 0):,}")
            print(f"Products with rating: {stats.get('products_with_rating', 0):,}")
            print(f"Products with images: {stats.get('products_with_images', 0):,}")
            print(f"Unique categories: {stats.get('unique_categories', 0):,}")
            print(f"Avg tokens per product: {stats.get('avg_tokens_per_product', 0):.1f}")
            
            # Filter coverage
            filter_stats = {k: v for k, v in stats.items() if k.endswith('_coverage')}
            if filter_stats:
                print("\nFilter Coverage:")
                for filter_name, coverage in filter_stats.items():
                    clean_name = filter_name.replace('_coverage', '').replace('_', ' ').title()
                    print(f"  {clean_name}: {coverage}")
        
        if test_sample_size:
            print("\n🧪 Test run completed successfully!")
            print("💡 Remove --test flag to process the full stratified sample")
        else:
            print("\n🎉 Ready to start Search API service!")
            print("💡 Run: python services/search-api/main.py")
        
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("🏃 Starting main function...")
    asyncio.run(main())