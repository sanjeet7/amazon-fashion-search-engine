#!/usr/bin/env python3
"""Data Pipeline Service - Main Entry Point."""

import asyncio
import argparse
import sys
from pathlib import Path

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models import Settings
from shared.utils import setup_logger
from src.pipeline import DataPipeline


async def main():
    """Main entry point for data pipeline service."""
    
    parser = argparse.ArgumentParser(description="Amazon Fashion Data Pipeline")
    parser.add_argument(
        "--force-rebuild", 
        action="store_true",
        help="Force rebuild even if processed data exists"
    )
    parser.add_argument(
        "--sample", 
        action="store_true",
        help="Use 500-product sample for testing"
    )
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Show pipeline status and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Load settings
        settings = Settings()
        
        # Override sample mode if requested
        if args.sample:
            settings.use_sample_data = True
        
        # Initialize pipeline
        pipeline = DataPipeline(settings)
        
        # Show status if requested
        if args.status:
            status = pipeline.get_status()
            print("\n🔍 Data Pipeline Status:")
            print("=" * 50)
            for key, value in status.items():
                print(f"{key}: {value}")
            return
        
        # Print startup information
        print("\n🚀 Amazon Fashion Data Pipeline")
        print("=" * 50)
        print(f"📊 Data source: {settings.data_path}")
        print(f"📦 Sample size: {settings.effective_sample_size:,}")
        print(f"🔄 Force rebuild: {args.force_rebuild}")
        print(f"🧪 Sample mode: {settings.use_sample_data}")
        print("=" * 50)
        
        # Run pipeline
        results = await pipeline.run(force_rebuild=args.force_rebuild)
        
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
            print(f"Products with price: {stats.get('products_with_price', 0):,}")
            print(f"Products with rating: {stats.get('products_with_rating', 0):,}")
            print(f"Products with images: {stats.get('products_with_images', 0):,}")
            print(f"Unique categories: {stats.get('unique_categories', 0):,}")
            print(f"Avg tokens per product: {stats.get('avg_tokens_per_product', 0):.1f}")
        
        print("\n🎉 Ready to start Search API service!")
        
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())