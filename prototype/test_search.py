#!/usr/bin/env python3
"""Simple search testing for prototype."""

import json
import numpy as np
import faiss
from pathlib import Path
import argparse
import os
import time
import sys
from openai import OpenAI

def load_index_data(index_dir):
    """Load FAISS index and product data."""
    print(f"🔍 Step 1: Loading index data...")
    start_time = time.time()
    
    index_dir = Path(index_dir)
    print(f"   📁 Index directory: {index_dir}")
    
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    
    # Load FAISS index
    print("   💾 Loading FAISS index...")
    index_start = time.time()
    index_path = index_dir / "faiss.index"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    index_size = index_path.stat().st_size / (1024*1024)  # MB
    print(f"   📏 Index file size: {index_size:.1f} MB")
    
    index = faiss.read_index(str(index_path))
    index_elapsed = time.time() - index_start
    print(f"   ✅ FAISS index loaded in {index_elapsed:.3f}s ({index.ntotal} vectors)")
    
    # Load products
    print("   💾 Loading product data...")
    products_start = time.time()
    products_path = index_dir / "products.json"
    
    if not products_path.exists():
        raise FileNotFoundError(f"Products file not found: {products_path}")
    
    with open(products_path, 'r') as f:
        products = json.load(f)
    products_elapsed = time.time() - products_start
    print(f"   ✅ Products loaded in {products_elapsed:.3f}s ({len(products)} products)")
    
    # Load mapping
    print("   💾 Loading text mappings...")
    mapping_start = time.time()
    mapping_path = index_dir / "mapping.json"
    
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    mapping_elapsed = time.time() - mapping_start
    
    total_elapsed = time.time() - start_time
    
    print(f"   ✅ Mappings loaded in {mapping_elapsed:.3f}s")
    print(f"✅ All index data loaded in {total_elapsed:.3f} seconds")
    
    # Show metadata if available
    if 'metadata' in mapping:
        meta = mapping['metadata']
        print(f"   📊 Index metadata:")
        print(f"      🤖 Model: {meta.get('embedding_model', 'unknown')}")
        print(f"      📅 Created: {meta.get('created_at', 'unknown')}")
        print(f"      📈 Products: {meta.get('num_products', len(products))}")
        print(f"      📈 Texts: {meta.get('num_texts', len(mapping.get('texts', [])))}")
    
    return index, products, mapping

def generate_query_embedding(query, api_key):
    """Generate embedding for search query."""
    print(f"\n🤖 Step 2: Generating query embedding...")
    start_time = time.time()
    
    print(f"   🔍 Query: '{query}'")
    print(f"   📏 Query length: {len(query)} characters")
    
    print("   🔗 Connecting to OpenAI API...")
    client = OpenAI(api_key=api_key)
    
    try:
        print("   📡 Making API call...")
        api_start = time.time()
        
        response = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        
        api_elapsed = time.time() - api_start
        print(f"   ✅ API call complete in {api_elapsed:.3f} seconds")
        
        print("   🔢 Processing embedding...")
        embedding = np.array([response.data[0].embedding], dtype=np.float32)
        
        print(f"   📊 Embedding shape: {embedding.shape}")
        print(f"   📊 Embedding dimension: {embedding.shape[1]}")
        
        # Normalize for cosine similarity
        print("   📐 Normalizing for cosine similarity...")
        faiss.normalize_L2(embedding)
        
        total_elapsed = time.time() - start_time
        print(f"✅ Query embedding generated in {total_elapsed:.3f} seconds")
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error generating query embedding: {e}")
        return None

def search_products(index, products, mapping, query_embedding, top_k=5):
    """Search for products using the query embedding."""
    print(f"\n🔍 Step 3: Searching for top {top_k} products...")
    start_time = time.time()
    
    print(f"   📊 Index contains {index.ntotal} vectors")
    print(f"   🎯 Searching for {top_k} nearest neighbors...")
    
    # Search the index
    search_start = time.time()
    scores, indices = index.search(query_embedding, top_k)
    search_elapsed = time.time() - search_start
    
    print(f"   ✅ FAISS search complete in {search_elapsed:.3f} seconds")
    print(f"   📊 Found scores: {[f'{s:.4f}' for s in scores[0][:3]]}")
    print(f"   📊 Found indices: {indices[0][:3].tolist()}")
    
    print("   🏗️  Building result objects...")
    results = []
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:  # No more results
            print(f"   ⚠️  No more results at position {i}")
            break
        
        if idx >= len(products):
            print(f"   ⚠️  Invalid index {idx} (max: {len(products)-1})")
            continue
        
        product = products[idx]
        result = {
            'rank': i + 1,
            'score': float(score),
            'asin': product.get('asin') or product.get('parent_asin') or 'Unknown',
            'title': product.get('title', 'No title'),
            'brand': product.get('brand') or product.get('store', 'Unknown'),
            'price': product.get('price', 'N/A'),
            'main_category': product.get('main_category', 'Unknown'),
            'average_rating': product.get('average_rating', 'N/A'),
            'embedding_text': mapping['texts'][idx][:100] + "..." if len(mapping['texts'][idx]) > 100 else mapping['texts'][idx]
        }
        results.append(result)
        
        if i < 3:  # Show details for first 3 results
            print(f"   📋 Result {i+1}: {result['title'][:50]}... (score: {score:.4f})")
    
    total_elapsed = time.time() - start_time
    print(f"✅ Search complete in {total_elapsed:.3f} seconds ({len(results)} results)")
    
    return results

def display_results(query, results):
    """Display search results in a nice format."""
    print(f"\n📋 SEARCH RESULTS")
    print("=" * 70)
    print(f"🔍 Query: '{query}'")
    print(f"📊 Found: {len(results)} results")
    print("=" * 70)
    
    if not results:
        print("   ❌ No results found")
        print("   💡 Try a different search query")
        return
    
    for result in results:
        print(f"\n🥇 RANK {result['rank']} (Score: {result['score']:.4f})")
        print(f"📦 {result['title']}")
        print(f"   🆔 ASIN: {result['asin']}")
        print(f"   🏷️  Brand: {result['brand']}")
        
        # Format price nicely
        if result['price'] != 'N/A':
            try:
                price = float(result['price'])
                print(f"   💰 Price: ${price:.2f}")
            except (ValueError, TypeError):
                print(f"   💰 Price: {result['price']}")
        else:
            print(f"   💰 Price: Not available")
            
        print(f"   📂 Category: {result['main_category']}")
        
        # Format rating nicely
        if result['average_rating'] != 'N/A':
            try:
                rating = float(result['average_rating'])
                stars = "⭐" * int(rating)
                print(f"   ⭐ Rating: {rating:.1f}/5.0 {stars}")
            except (ValueError, TypeError):
                print(f"   ⭐ Rating: {result['average_rating']}")
        else:
            print(f"   ⭐ Rating: Not available")
            
        print(f"   📝 Embedding text: {result['embedding_text']}")
        
        if result['rank'] < len(results):
            print("   " + "─" * 50)

def interactive_search(index, products, mapping, api_key):
    """Interactive search mode."""
    print("\n🎯 Interactive Search Mode")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("Please enter a valid query")
                continue
            
            print(f"🤖 Generating embedding for: '{query}'")
            query_embedding = generate_query_embedding(query, api_key)
            
            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                continue
            
            print("🔍 Searching...")
            results = search_products(index, products, mapping, query_embedding, top_k=5)
            
            display_results(query, results)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test search for prototype")
    parser.add_argument("--index", default="data/prototype_index", help="Index directory")
    parser.add_argument("--query", help="Single query to test")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--interactive", action="store_true", help="Interactive search mode")
    
    args = parser.parse_args()
    
    print("🚀 Prototype Search Tester")
    print("=" * 40)
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        return 1
    
    try:
        # Load index and data
        index, products, mapping = load_index_data(args.index)
        
        if args.interactive:
            # Interactive mode
            interactive_search(index, products, mapping, api_key)
        elif args.query:
            # Single query mode
            print(f"🤖 Generating embedding for: '{args.query}'")
            query_embedding = generate_query_embedding(args.query, api_key)
            
            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return 1
            
            print("🔍 Searching...")
            results = search_products(index, products, mapping, query_embedding, args.top_k)
            
            display_results(args.query, results)
        else:
            print("❌ Please provide --query or use --interactive mode")
            return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 