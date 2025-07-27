#!/usr/bin/env python3
"""Simple sample generation for prototype testing."""

import json
from pathlib import Path
import argparse
import random
import time
import sys

def find_raw_data_file():
    """Find the raw data file."""
    data_dir = Path("data/raw")
    print(f"   📁 Checking directory: {data_dir}")
    
    # Look for JSONL files
    jsonl_files = list(data_dir.glob("*.jsonl"))
    print(f"   📋 Found {len(jsonl_files)} JSONL files: {[f.name for f in jsonl_files]}")
    
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    
    # Use the first file found
    data_file = jsonl_files[0]
    file_size = data_file.stat().st_size / (1024*1024)  # MB
    print(f"📂 Using data file: {data_file}")
    print(f"   📏 File size: {file_size:.1f} MB")
    
    return data_file

def create_sample(data_file, size=500):
    """Create a sample by reading only what we need."""
    print(f"\n🎲 Step 2: Creating sample of {size} products...")
    start_time = time.time()
    
    print(f"   🔍 Reading file and collecting valid products...")
    print(f"   🎯 Target: {size} valid products")
    
    valid_products = []
    invalid_count = 0
    lines_read = 0
    
    with open(data_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            lines_read = line_num
            
            # Progress update every 1000 lines
            if line_num % 1000 == 0:
                print(f"   📊 Read {line_num:,} lines, found {len(valid_products)} valid products...")
            
            try:
                product = json.loads(line.strip())
                
                # Validate product
                has_title = product.get('title') and len(str(product.get('title', '')).strip()) > 10
                has_id = product.get('asin') or product.get('parent_asin')
                
                if has_title and has_id:
                    valid_products.append(product)
                    
                    # Stop when we have enough
                    if len(valid_products) >= size:
                        print(f"   🎯 Reached target of {size} valid products!")
                        break
                else:
                    invalid_count += 1
                    
            except json.JSONDecodeError:
                invalid_count += 1
                continue
    
    total_elapsed = time.time() - start_time
    
    print(f"✅ Sample collection complete in {total_elapsed:.2f} seconds")
    print(f"   📊 Lines read: {lines_read:,}")
    print(f"   📊 Valid products found: {len(valid_products)}")
    print(f"   📊 Invalid products skipped: {invalid_count}")
    print(f"   📊 Success rate: {len(valid_products)/lines_read*100:.1f}%")
    print(f"   📊 Reading rate: {lines_read/total_elapsed:.0f} lines/sec")
    
    if len(valid_products) < size:
        print(f"   ⚠️  Warning: Only found {len(valid_products)} valid products (requested {size})")
        print(f"   💡 Try increasing the sample size or check data quality")
    
    return valid_products

def save_sample(sample, output_path):
    """Save sample to JSONL file."""
    print(f"\n💾 Step 3: Saving sample to: {output_path}")
    start_time = time.time()
    
    output_path = Path(output_path)
    print(f"   📁 Creating directory: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"   ✍️  Writing {len(sample)} products to JSONL...")
    write_start = time.time()
    
    with open(output_path, 'w') as f:
        for i, product in enumerate(sample):
            f.write(json.dumps(product) + '\n')
            
            if i % 100 == 0 and i > 0:
                print(f"   📝 Written {i}/{len(sample)} products...")
    
    write_elapsed = time.time() - write_start
    file_size = output_path.stat().st_size / 1024  # KB
    
    print(f"   ✅ File written in {write_elapsed:.3f} seconds")
    print(f"   📏 File size: {file_size:.1f} KB")
    print(f"✅ Saved {len(sample)} products successfully")

def main():
    parser = argparse.ArgumentParser(description="Generate sample data for prototype")
    parser.add_argument("--size", type=int, default=500, help="Sample size")
    parser.add_argument("--output", default="data/prototype_sample.jsonl", help="Output file")
    
    args = parser.parse_args()
    
    print("🚀 Prototype Sample Generator")
    print("=" * 40)
    print(f"🎯 Target: {args.size} products")
    print(f"📁 Output: {args.output}")
    
    try:
        # Find raw data file
        print("\n🔍 Step 1: Finding raw data file...")
        data_file = find_raw_data_file()
        
        # Create sample (reads only what we need!)
        sample = create_sample(data_file, args.size)
        
        # Save sample
        save_sample(sample, args.output)
        
        print(f"\n🎉 Sample generation complete!")
        print(f"   📦 Generated {len(sample)} products")
        print(f"   💾 Saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 