#!/usr/bin/env python3
"""Simple debug to find the exact issue."""

import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/workspace')

def test_pipeline_conversion():
    """Test the exact issue with filter_metadata conversion."""
    
    print("🔍 Testing filter_metadata conversion issue...")
    
    # Import the utilities directly
    from shared.utils.text_processing import extract_product_filters, extract_quality_filters
    
    # Load sample data
    with open("sample_test_data.jsonl", 'r') as f:
        test_records = [json.loads(line.strip()) for line in f]
    
    print(f"✅ Loaded {len(test_records)} test records")
    
    # Process like the actual pipeline does
    processed_data = []
    
    for i, record in enumerate(test_records):
        print(f"\n--- Processing record {i+1}: {record['parent_asin']} ---")
        
        # Extract filters like the actual pipeline
        filter_metadata = extract_product_filters(record)
        quality_metadata = extract_quality_filters(record)
        
        print(f"Filter metadata type: {type(filter_metadata)}")
        print(f"Quality metadata type: {type(quality_metadata)}")
        
        # Create a record like the Product model would
        processed_record = {
            'parent_asin': record['parent_asin'],
            'title': record['title'],
            'main_category': record.get('main_category'),
            'price': record.get('price'),
            'average_rating': record.get('average_rating'),
            'rating_number': record.get('rating_number'),
            'filter_metadata': filter_metadata,
            'quality_metadata': quality_metadata,
        }
        
        processed_data.append(processed_record)
    
    # Try to create DataFrame like the pipeline does
    print("\n📊 Creating DataFrame...")
    df = pd.DataFrame(processed_data)
    print(f"✅ DataFrame created with {len(df)} rows")
    
    # Check data types
    print(f"filter_metadata column types: {df['filter_metadata'].apply(type).unique()}")
    print(f"quality_metadata column types: {df['quality_metadata'].apply(type).unique()}")
    
    # Try to save to parquet
    try:
        print("\n💾 Testing parquet save...")
        df.to_parquet("test_simple.parquet")
        print("✅ Parquet save successful!")
        os.remove("test_simple.parquet")
    except Exception as e:
        print(f"❌ Parquet save failed: {e}")
        print(f"Error type: {type(e)}")
        
        # Let's examine the problematic columns
        print("\n🔬 Analyzing filter_metadata column...")
        for i, val in enumerate(df['filter_metadata']):
            print(f"Row {i}: type={type(val)}, value={val}")
            
        print("\n🔬 Analyzing quality_metadata column...")
        for i, val in enumerate(df['quality_metadata']):
            print(f"Row {i}: type={type(val)}, value={val}")

def test_with_none_values():
    """Test with some None values to see if that causes issues."""
    
    print("\n🧪 Testing with None values...")
    
    # Create test data with some None filter_metadata
    test_data = [
        {'asin': 'A1', 'filter_metadata': {'color': 'blue', 'price': 10.0}},
        {'asin': 'A2', 'filter_metadata': None},
        {'asin': 'A3', 'filter_metadata': {'color': 'red'}},
        {'asin': 'A4', 'filter_metadata': {}},
        {'asin': 'A5', 'filter_metadata': {'price': 50.0, 'brand': 'Test'}},
    ]
    
    df = pd.DataFrame(test_data)
    print(f"✅ DataFrame created with {len(df)} rows")
    print(f"filter_metadata types: {df['filter_metadata'].apply(type).unique()}")
    
    try:
        df.to_parquet("test_none.parquet")
        print("✅ Parquet save with None values successful!")
        os.remove("test_none.parquet")
    except Exception as e:
        print(f"❌ Parquet save with None values failed: {e}")

def test_mixed_types():
    """Test with intentionally mixed types to reproduce the error."""
    
    print("\n🧪 Testing with mixed types...")
    
    # Create test data with mixed types - this should fail
    test_data = [
        {'asin': 'A1', 'filter_metadata': {'color': 'blue'}},
        {'asin': 'A2', 'filter_metadata': ['list', 'instead', 'of', 'dict']},  # This will cause issues
        {'asin': 'A3', 'filter_metadata': {'color': 'red'}},
    ]
    
    df = pd.DataFrame(test_data)
    print(f"✅ DataFrame created with {len(df)} rows")
    print(f"filter_metadata types: {df['filter_metadata'].apply(type).unique()}")
    
    try:
        df.to_parquet("test_mixed.parquet")
        print("✅ Parquet save with mixed types successful!")
        os.remove("test_mixed.parquet")
    except Exception as e:
        print(f"❌ Parquet save with mixed types failed: {e}")
        print("🎯 This is likely the exact error from the pipeline!")

if __name__ == "__main__":
    print("🚀 Starting simple debug...")
    
    test_pipeline_conversion()
    test_with_none_values()
    test_mixed_types()
    
    print("\n✨ Debug complete!")