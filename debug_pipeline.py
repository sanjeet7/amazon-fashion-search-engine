#!/usr/bin/env python3
"""Debug script to identify the filter_metadata issue."""

import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_filter_metadata_extraction():
    """Test the filter metadata extraction with a few sample records."""
    
    print("🔍 Testing filter metadata extraction...")
    
    # Import the functions
    from shared.utils.text_processing import extract_product_filters, extract_quality_filters
    
    # Read a few sample records from the test data
    raw_data_path = Path("sample_test_data.jsonl")
    
    if not raw_data_path.exists():
        print(f"❌ Test data file not found: {raw_data_path}")
        return
    
    # Test with first 10 records
    print("📖 Reading first 10 records...")
    test_records = []
    
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Only test first 10
                break
            try:
                record = json.loads(line.strip())
                test_records.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error at line {i+1}: {e}")
                continue
    
    print(f"✅ Loaded {len(test_records)} test records")
    
    # Test filter extraction on each record
    filter_results = []
    quality_results = []
    
    for i, record in enumerate(test_records):
        print(f"\n--- Record {i+1} ---")
        print(f"ASIN: {record.get('parent_asin', 'unknown')}")
        print(f"Title: {record.get('title', 'unknown')[:50]}...")
        
        try:
            # Test filter extraction
            filters = extract_product_filters(record)
            quality = extract_quality_filters(record)
            
            print(f"Filter metadata type: {type(filters)}")
            print(f"Filter metadata: {filters}")
            print(f"Quality metadata type: {type(quality)}")
            print(f"Quality metadata: {quality}")
            
            filter_results.append(filters)
            quality_results.append(quality)
            
        except Exception as e:
            print(f"❌ Error processing record {i+1}: {e}")
            filter_results.append(None)
            quality_results.append(None)
    
    # Test DataFrame creation
    print("\n🗂️  Testing DataFrame creation...")
    
    try:
        test_data = []
        for i, record in enumerate(test_records):
            test_data.append({
                'parent_asin': record.get('parent_asin', f'test_{i}'),
                'title': record.get('title', f'Test Product {i}'),
                'filter_metadata': filter_results[i],
                'quality_metadata': quality_results[i]
            })
        
        # Create DataFrame
        df = pd.DataFrame(test_data)
        print(f"✅ DataFrame created successfully with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"filter_metadata column types: {df['filter_metadata'].apply(type).unique()}")
        
        # Test saving to parquet
        print("\n💾 Testing parquet save...")
        df.to_parquet("test_debug.parquet")
        print("✅ Parquet save successful")
        
        # Clean up
        os.remove("test_debug.parquet")
        
    except Exception as e:
        print(f"❌ DataFrame/Parquet error: {e}")
        print(f"Error type: {type(e)}")
        
        # Analyze the problematic data
        print("\n🔬 Analyzing filter_metadata values...")
        for i, val in enumerate(filter_results):
            print(f"Record {i+1}: type={type(val)}, value={val}")

def test_small_pipeline():
    """Test the actual pipeline with minimal sample size."""
    
    print("\n🧪 Testing actual pipeline with 5 products...")
    
    try:
        # Set environment to avoid OpenAI calls
        os.environ["OPENAI_API_KEY"] = "dummy_key_for_testing"
        
        from services.data_pipeline.src.data_processor import DataProcessor
        from shared.models import Settings
        
        # Create settings
        settings = Settings()
        processor = DataProcessor(settings)
        
        # Load 5 products from sample data
        raw_products = processor.load_raw_data(data_source=Path("sample_test_data.jsonl"), sample_size=5)
        print(f"✅ Loaded {len(raw_products)} products")
        
        # Process products
        products = []
        for i, raw_product in enumerate(raw_products):
            print(f"\n--- Processing product {i+1} ---")
            cleaned = processor.clean_product(raw_product)
            if cleaned:
                products.append(cleaned)
                print(f"✅ Product processed: {cleaned.parent_asin}")
            else:
                print(f"❌ Product cleaning failed")
        
        print(f"\n✅ Successfully processed {len(products)} products")
        
        # Try to create DataFrame from processed products
        if products:
            print("\n📊 Converting to DataFrame...")
            
            # Convert to dict format
            product_dicts = []
            for product in products:
                product_dict = product.model_dump()
                product_dicts.append(product_dict)
            
            df = pd.DataFrame(product_dicts)
            print(f"✅ DataFrame created with {len(df)} rows, {len(df.columns)} columns")
            
            # Check filter_metadata column specifically
            if 'filter_metadata' in df.columns:
                print(f"filter_metadata types: {df['filter_metadata'].apply(type).unique()}")
                print("Sample filter_metadata values:")
                for i, val in enumerate(df['filter_metadata'].head()):
                    print(f"  Row {i}: {type(val)} = {val}")
            
            # Try parquet save
            print("\n💾 Testing parquet save...")
            df.to_parquet("test_small_pipeline.parquet")
            print("✅ Parquet save successful!")
            
            # Clean up
            os.remove("test_small_pipeline.parquet")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting pipeline debug...")
    
    # Test 1: Filter metadata extraction
    test_filter_metadata_extraction()
    
    # Test 2: Small pipeline run
    test_small_pipeline()
    
    print("\n✨ Debug complete!")