# Fashion Search Prototype

Simple, working prototype for fashion product search using OpenAI embeddings and FAISS.

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install openai faiss-cpu numpy pandas

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Complete Workflow

**1. Generate Sample Data**
```bash
python prototype/generate_sample.py --size 500
```
- Creates `data/prototype_sample.jsonl` with 500 random products
- Filters for products with valid titles and ASINs

**2. Build Search Index** 
```bash
python prototype/build_index.py
```
- Processes the sample data
- Generates OpenAI embeddings (text-embedding-3-small)
- Creates FAISS index for fast similarity search
- Saves everything to `data/prototype_index/`

**3. Test Search**
```bash
# Single query
python prototype/test_search.py --query "blue summer dress"

# Interactive mode
python prototype/test_search.py --interactive
```

## Output Structure

```
data/
├── prototype_sample.jsonl          # Sample products  
└── prototype_index/
    ├── faiss.index                 # FAISS search index
    ├── products.json               # Product data
    └── mapping.json                # Text/ID mappings
```

## Example Usage

```bash
# Generate 100 products for quick testing
python prototype/generate_sample.py --size 100

# Build index with custom paths
python prototype/build_index.py --input data/my_sample.jsonl --output data/my_index

# Search for specific items
python prototype/test_search.py --query "nike running shoes" --top-k 3
python prototype/test_search.py --query "red dress under $50" 
python prototype/test_search.py --query "winter jacket waterproof"
```

## Features

- ✅ **Simple**: No complex dependencies or configurations
- ✅ **Fast**: FAISS index for efficient similarity search  
- ✅ **Semantic**: OpenAI embeddings understand meaning, not just keywords
- ✅ **Interactive**: Test multiple queries easily
- ✅ **Flexible**: Configurable sample sizes and search parameters

## Cost Estimation

For 500 products:
- Embedding generation: ~$0.02 (text-embedding-3-small)
- Search queries: ~$0.0001 per query

## Next Steps

This prototype demonstrates core functionality. For production:
- Add filtering by price, brand, category
- Implement query enhancement with LLMs  
- Add web API (FastAPI)
- Scale to full dataset
- Optimize index for larger data 