#!/usr/bin/env python3
"""Test script to validate the Amazon Fashion Search Engine."""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from shared.models import Settings, Product, SearchRequest
        from shared.utils import setup_logger, clean_text, extract_features
        print("✅ Shared modules import successfully")
    except Exception as e:
        print(f"❌ Shared module import failed: {e}")
        return False
    
    try:
        # Add data-pipeline to sys.path
        import sys
        data_pipeline_path = Path(__file__).parent.parent / "services" / "data-pipeline" / "src"
        sys.path.insert(0, str(data_pipeline_path))
        
        from data_processor import DataProcessor
        from embedding_generator import EmbeddingGenerator
        from pipeline import DataPipeline
        print("✅ Data pipeline modules import successfully")
    except Exception as e:
        print(f"❌ Data pipeline import failed: {e}")
        return False
    
    try:
        # Add search-api to sys.path
        search_api_path = Path(__file__).parent.parent / "services" / "search-api" / "src"
        sys.path.insert(0, str(search_api_path))
        
        from search_engine import SearchEngine
        print("✅ Search API modules import successfully")
    except Exception as e:
        print(f"❌ Search API import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️ Testing configuration...")
    
    try:
        from shared.models import Settings
        
        # Test with sample data mode
        settings = Settings(use_sample_data=True)
        print(f"✅ Configuration loaded successfully")
        print(f"   - Data path: {settings.data_path}")
        print(f"   - Sample mode: {settings.use_sample_data}")
        print(f"   - Effective sample size: {settings.effective_sample_size}")
        
        return settings
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return None

def test_data_processing(settings):
    """Test data processing functionality."""
    print("\n📊 Testing data processing...")
    
    try:
        # Import from the already added path
        from data_processor import DataProcessor
        
        processor = DataProcessor(settings)
        
        # Test loading sample data
        raw_products = processor.load_raw_data()
        print(f"✅ Loaded {len(raw_products)} products from sample data")
        
        # Test processing
        df, embedding_texts = processor.process_products(raw_products)
        print(f"✅ Processed {len(df)} products successfully")
        print(f"✅ Generated {len(embedding_texts)} embedding texts")
        
        # Test statistics
        stats = processor.calculate_statistics(df, embedding_texts)
        print(f"✅ Calculated statistics: {stats['total_products']} total products")
        
        return df, embedding_texts
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return None, None

def test_text_processing():
    """Test text processing utilities."""
    print("\n🔤 Testing text processing...")
    
    try:
        from shared.utils import clean_text, extract_features, calculate_tokens
        
        # Test text cleaning
        dirty_text = "  This is a <b>test</b> with   extra spaces  "
        clean = clean_text(dirty_text)
        print(f"✅ Text cleaning: '{dirty_text}' → '{clean}'")
        
        # Test feature extraction
        sample_product = {
            "title": "Blue Summer Dress",
            "features": ["Lightweight", "Breathable"],
            "main_category": "Dresses",
            "price": 49.99
        }
        features = extract_features(sample_product)
        print(f"✅ Feature extraction: {features}")
        
        # Test token calculation
        tokens = calculate_tokens(features)
        print(f"✅ Token calculation: {tokens} tokens")
        
        return True
    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        return False

def test_search_models():
    """Test search models and validation."""
    print("\n🔍 Testing search models...")
    
    try:
        from shared.models import SearchRequest, ProductResult
        
        # Test search request validation
        request = SearchRequest(
            query="blue summer dress",
            top_k=5,
            price_max=100.0
        )
        print(f"✅ Search request created: {request.query}")
        
        # Test product result
        result = ProductResult(
            parent_asin="B001",
            title="Test Product",
            similarity_score=0.85
        )
        print(f"✅ Product result created: {result.title}")
        
        return True
    except Exception as e:
        print(f"❌ Search model test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Amazon Fashion Search Engine - System Test")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot continue.")
        return False
    
    # Test 2: Configuration
    settings = test_configuration()
    if not settings:
        print("\n❌ Configuration tests failed. Cannot continue.")
        return False
    
    # Test 3: Text Processing
    if not test_text_processing():
        print("\n❌ Text processing tests failed.")
        return False
    
    # Test 4: Search Models
    if not test_search_models():
        print("\n❌ Search model tests failed.")
        return False
    
    # Test 5: Data Processing
    df, embedding_texts = test_data_processing(settings)
    if df is None:
        print("\n❌ Data processing tests failed.")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("✅ System is ready for use")
    print("✅ Sample data processing works")
    print("✅ All modules import correctly")
    print("✅ Configuration is valid")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set a valid OPENAI_API_KEY in .env")
    print("2. Run: python services/data-pipeline/main.py --sample")
    print("3. Run: python services/search-api/main.py")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)