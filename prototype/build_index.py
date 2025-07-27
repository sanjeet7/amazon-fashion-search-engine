#!/usr/bin/env python3
"""Simple index building for prototype testing."""

import json
import numpy as np
import faiss
from pathlib import Path
import argparse
import re
import os
import time
import sys
from openai import OpenAI

def load_sample_data(input_path):
    """Load sample data from JSONL file."""
    print(f"🔍 Step 1: Loading sample data...")
    start_time = time.time()
    
    input_path = Path(input_path)
    print(f"   📂 Input file: {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Sample file not found: {input_path}")
    
    file_size = input_path.stat().st_size / 1024  # KB
    print(f"   📏 File size: {file_size:.1f} KB")
    
    products = []
    invalid_lines = 0
    
    print("   🔄 Reading JSONL file...")
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                product = json.loads(line.strip())
                products.append(product)
                
                if line_num % 50 == 0:
                    print(f"   📊 Loaded {line_num} products...")
                    
            except json.JSONDecodeError:
                invalid_lines += 1
                if invalid_lines <= 5:  # Only show first few errors
                    print(f"   ⚠️  Warning: Skipping malformed line {line_num}")
                continue
    
    elapsed = time.time() - start_time
    print(f"✅ Loaded {len(products)} products in {elapsed:.3f} seconds")
    if invalid_lines > 0:
        print(f"   📊 Skipped {invalid_lines} invalid lines")
    
    return products

def clean_text(text):
    """Simple text cleaning."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters
    text = re.sub(r'[^\w\s\-\.\,\:\;\!\?]', '', text)
    
    return text

def prepare_embedding_text(product):
    """Prepare text for embedding generation."""
    components = []
    
    # Title (most important)
    if product.get('title'):
        components.append(clean_text(product['title']))
    
    # Category info
    if product.get('main_category'):
        components.append(clean_text(product['main_category']))
    
    # Features (first few)
    if product.get('features'):
        features = product['features']
        if isinstance(features, list) and features:
            # Take first 2 features to keep it manageable
            for feature in features[:2]:
                if feature and len(feature.strip()) > 10:
                    components.append(clean_text(feature))
    
    # Brand
    if product.get('brand'):
        components.append(clean_text(product['brand']))
    
    # Combine all components
    combined_text = " ".join(components)
    
    # Truncate if too long (OpenAI has token limits)
    if len(combined_text) > 1000:
        combined_text = combined_text[:1000]
    
    return combined_text

def generate_embeddings(products, api_key):
    """Generate embeddings using OpenAI."""
    print(f"\n🤖 Step 2: Generating embeddings for {len(products)} products...")
    start_time = time.time()
    
    print("   🔗 Connecting to OpenAI API...")
    client = OpenAI(api_key=api_key)
    
    # Prepare texts
    print("   📝 Preparing text for embedding...")
    prep_start = time.time()
    
    texts = []
    product_ids = []
    
    for i, product in enumerate(products):
        if i % 25 == 0 and i > 0:
            print(f"   📊 Prepared {i}/{len(products)} texts...")
            
        text = prepare_embedding_text(product)
        if text:  # Only include products with valid text
            texts.append(text)
            product_ids.append(product.get('asin') or product.get('parent_asin') or f"product_{len(product_ids)}")
    
    prep_elapsed = time.time() - prep_start
    print(f"   ✅ Prepared {len(texts)} texts in {prep_elapsed:.3f} seconds")
    
    # Calculate costs
    total_tokens = sum(len(text.split()) * 1.3 for text in texts)  # Rough estimate
    estimated_cost = (total_tokens / 1000000) * 0.02  # $0.02 per 1M tokens
    print(f"   💰 Estimated cost: ~${estimated_cost:.4f}")
    
    # Generate embeddings in batches
    embeddings = []
    batch_size = 100
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"   🚀 Generating embeddings in {total_batches} batches...")
    embed_start = time.time()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i//batch_size + 1
        
        print(f"   📡 API call {batch_num}/{total_batches} ({len(batch)} texts)...")
        batch_start = time.time()
        
        try:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
            batch_elapsed = time.time() - batch_start
            print(f"   ✅ Batch {batch_num} complete in {batch_elapsed:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error in batch {batch_num}: {e}")
            print(f"   🔄 Adding zero vectors for failed batch...")
            # Add zero vectors for failed batch
            embeddings.extend([[0.0] * 1536] * len(batch))
    
    embed_elapsed = time.time() - embed_start
    total_elapsed = time.time() - start_time
    
    print(f"✅ Generated {len(embeddings)} embeddings in {embed_elapsed:.2f} seconds")
    print(f"   📊 Average: {len(embeddings)/embed_elapsed:.1f} embeddings/sec")
    print(f"   📊 Total step time: {total_elapsed:.2f} seconds")
    
    return np.array(embeddings, dtype=np.float32), texts, product_ids

def build_faiss_index(embeddings):
    """Build FAISS index."""
    print(f"\n🔍 Step 3: Building FAISS index for {len(embeddings)} vectors...")
    start_time = time.time()
    
    dimension = embeddings.shape[1]
    memory_size = embeddings.nbytes / (1024*1024)  # MB
    print(f"   📊 Vector dimension: {dimension}")
    print(f"   📊 Memory usage: {memory_size:.1f} MB")
    
    # Use simple flat index for prototype
    print("   🏗️  Creating FAISS IndexFlatIP (inner product)...")
    index_start = time.time()
    
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    print("   📐 Normalizing embeddings for cosine similarity...")
    norm_start = time.time()
    faiss.normalize_L2(embeddings)
    norm_elapsed = time.time() - norm_start
    print(f"   ✅ Normalization complete in {norm_elapsed:.3f} seconds")
    
    # Add vectors to index
    print(f"   📥 Adding {len(embeddings)} vectors to index...")
    add_start = time.time()
    index.add(embeddings)
    add_elapsed = time.time() - add_start
    
    total_elapsed = time.time() - start_time
    
    print(f"   ✅ Vectors added in {add_elapsed:.3f} seconds")
    print(f"✅ Built index with {index.ntotal} vectors in {total_elapsed:.3f} seconds")
    print(f"   📊 Index size: {index.ntotal} vectors x {dimension} dims")
    
    return index

def save_index_data(index, products, texts, product_ids, output_dir):
    """Save all index data."""
    print(f"\n💾 Step 4: Saving index data...")
    start_time = time.time()
    
    output_dir = Path(output_dir)
    print(f"   📁 Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    print("   💾 Saving FAISS index...")
    index_start = time.time()
    index_path = output_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    index_elapsed = time.time() - index_start
    index_size = index_path.stat().st_size / (1024*1024)  # MB
    print(f"   ✅ FAISS index saved in {index_elapsed:.3f}s: {index_path} ({index_size:.1f} MB)")
    
    # Save product data
    print("   💾 Saving product data...")
    products_start = time.time()
    products_path = output_dir / "products.json"
    with open(products_path, 'w') as f:
        json.dump(products, f, indent=2)
    products_elapsed = time.time() - products_start
    products_size = products_path.stat().st_size / 1024  # KB
    print(f"   ✅ Products saved in {products_elapsed:.3f}s: {products_path} ({products_size:.1f} KB)")
    
    # Save texts and IDs mapping
    print("   💾 Saving text mappings...")
    mapping_start = time.time()
    mapping_path = output_dir / "mapping.json"
    mapping = {
        'texts': texts,
        'product_ids': product_ids,
        'metadata': {
            'num_products': len(products),
            'num_texts': len(texts),
            'embedding_model': 'text-embedding-3-small',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    mapping_elapsed = time.time() - mapping_start
    mapping_size = mapping_path.stat().st_size / 1024  # KB
    print(f"   ✅ Mapping saved in {mapping_elapsed:.3f}s: {mapping_path} ({mapping_size:.1f} KB)")
    
    total_elapsed = time.time() - start_time
    total_size = index_size + (products_size + mapping_size) / 1024  # MB
    
    print(f"✅ All data saved successfully in {total_elapsed:.3f} seconds")
    print(f"   📊 Total index size: {total_size:.1f} MB")
    print(f"   📊 Files created: 3 (index, products, mapping)")

def main():
    print("🚀 Prototype Index Builder", flush=True)
    print("=" * 40, flush=True)
    
    parser = argparse.ArgumentParser(description="Build search index for prototype")
    parser.add_argument("--input", default="data/prototype_sample.jsonl", help="Input sample file")
    parser.add_argument("--output", default="data/prototype_index", help="Output directory")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    print("🔧 Parsing arguments...", flush=True)
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        return 1
    
    try:
        # Load sample data
        products = load_sample_data(args.input)
        
        # Generate embeddings
        embeddings, texts, product_ids = generate_embeddings(products, api_key)
        
        # Build FAISS index
        index = build_faiss_index(embeddings)
        
        # Save everything
        save_index_data(index, products, texts, product_ids, args.output)
        
        print("\n🎉 Index building complete!")
        print(f"   Index contains {len(products)} products")
        print(f"   Ready for search testing!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 