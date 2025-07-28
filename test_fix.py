#!/usr/bin/env python3
"""Test the fix for filter_metadata issue."""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/workspace')

def test_fixed_pipeline():
    """Test the fixed pipeline with sample data."""
    
    print("🧪 Testing fixed pipeline...")
    
    # Set dummy OpenAI key to avoid API calls
    os.environ["OPENAI_API_KEY"] = "dummy_key_for_testing"
    
    try:
        # Add the services directory to path
        sys.path.insert(0, '/workspace/services/data-pipeline/src')
        
        from data_processor import DataProcessor
        from shared.models import Settings
        
        # Create settings
        settings = Settings()
        processor = DataProcessor(settings)
        
        # Load sample data
        sample_data_path = Path("/workspace/sample_test_data.jsonl")
        raw_products = processor.load_raw_data(data_source=sample_data_path, sample_size=5)
        print(f"✅ Loaded {len(raw_products)} products")
        
        # Process products using the fixed method
        df, embedding_texts = processor.process_products(raw_products)
        print(f"✅ Processed {len(df)} products into DataFrame")
        
        # Check the filter_metadata column
        if 'filter_metadata' in df.columns:
            print(f"✅ filter_metadata column exists")
            print(f"filter_metadata types: {df['filter_metadata'].apply(type).unique()}")
            
            # Check if all are strings (JSON)
            all_strings = df['filter_metadata'].apply(lambda x: isinstance(x, str) or x is None).all()
            if all_strings:
                print("✅ All filter_metadata values are strings or None - fix working!")
            else:
                print("❌ Still have non-string filter_metadata values")
        
        # Test parquet save
        print("\n💾 Testing parquet save...")
        test_file = "test_fixed_pipeline.parquet"
        df.to_parquet(test_file)
        print("✅ Parquet save successful!")
        
        # Load it back to verify
        df_loaded = __import__('pandas').read_parquet(test_file)
        print(f"✅ Parquet load successful! {len(df_loaded)} rows loaded")
        
        # Clean up
        os.remove(test_file)
        
        # Show sample data
        print("\n📋 Sample processed data:")
        for col in ['parent_asin', 'title', 'filter_metadata']:
            if col in df.columns:
                print(f"{col}: {df[col].iloc[0]}")
        
        print(f"\n📊 DataFrame columns: {len(df.columns)}")
        filter_cols = [col for col in df.columns if col.startswith('filter_')]
        print(f"📈 Filter columns created: {len(filter_cols)}")
        print(f"🔹 Filter columns: {filter_cols[:5]}...")  # Show first 5
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_pipeline()