#!/usr/bin/env python3
"""Main entry point for the Search API service."""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import uvicorn

# Add project root to Python path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# Simple configuration
class SearchAPIConfig:
    """Simple configuration for search API."""
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_data_dir = Path("data/processed")
        self.embeddings_cache_dir = Path("data/processed/embeddings_cache")
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Set up a logger with the specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# Initialize FastAPI app
app = FastAPI(
    title="Fashion Search API",
    description="Semantic search API for fashion products",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize config and logger
config = SearchAPIConfig()
logger = setup_logger("search-api")

# Global variables for loaded data
sample_products = []
sample_data_loaded = False


def load_sample_data():
    """Load sample product data from processed parquet file."""
    global sample_products, sample_data_loaded
    
    if sample_data_loaded:
        return sample_products
    
    try:
        import pandas as pd
        
        # Try to load processed data
        processed_file = config.processed_data_dir / "processed_sample.parquet"
        logger.info(f"Looking for processed data at: {processed_file}")
        logger.info(f"File exists: {processed_file.exists()}")
        
        if processed_file.exists():
            df = pd.read_parquet(processed_file)
            # Convert DataFrame to list of dictionaries
            sample_products = df.to_dict('records')
            logger.info(f"Loaded {len(sample_products)} processed products")
            logger.info(f"Sample product keys: {list(sample_products[0].keys()) if sample_products else 'None'}")
            sample_data_loaded = True
        else:
            logger.warning(f"Processed data not found at {processed_file}")
            # Create fallback data
            sample_products = [
                {
                    "parent_asin": "B00000001",
                    "title": "Nike Black Running Shoes - Size M",
                    "brand": "Nike",
                    "price": 89.99,
                    "average_rating": 4.5,
                    "rating_number": 1234,
                    "images": ["https://images.example.com/nike-shoes.jpg"],
                    "description": "High-quality running shoes from Nike. Perfect for everyday wear.",
                    "categories": [["All Categories", "Shoes", "Running Shoes"]]
                }
            ]
            logger.info("Using fallback sample data")
            sample_data_loaded = True
    
    except Exception as e:
        logger.error(f"Failed to load sample data: {e}")
        sample_products = []
    
    return sample_products


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Fashion Search API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if sample data is loaded
        products = load_sample_data()
        
        return {
            "status": "healthy",
            "service": "search-api",
            "data_loaded": len(products) > 0,
            "products_count": len(products),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/sample-products")
async def get_sample_products(limit: int = 20):
    """Get sample products."""
    try:
        products = load_sample_data()
        
        # Limit the results
        limited_products = products[:limit]
        
        return {
            "products": limited_products,
            "total": len(products),
            "limit": limit,
            "count": len(limited_products)
        }
    except Exception as e:
        logger.error(f"Failed to get sample products: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve products")


@app.post("/search")
async def search_products(request: Dict[str, Any]):
    """Search products by query."""
    try:
        query = request.get("query", "")
        limit = request.get("limit", 20)
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        products = load_sample_data()
        
        # Simple text search for demo purposes
        query_lower = query.lower()
        results = []
        
        for product in products:
            title = product.get("title", "").lower()
            brand = product.get("brand", "").lower()
            description = product.get("description", "").lower()
            
            # Simple scoring based on query matches
            score = 0.0
            if query_lower in title:
                score += 1.0
            if query_lower in brand:
                score += 0.8
            if query_lower in description:
                score += 0.5
            
            # Check for partial matches
            query_words = query_lower.split()
            for word in query_words:
                if word in title:
                    score += 0.3
                if word in brand:
                    score += 0.2
                if word in description:
                    score += 0.1
            
            if score > 0:
                result = product.copy()
                result["search_score"] = score
                results.append(result)
        
        # Sort by score and limit
        results.sort(key=lambda x: x["search_score"], reverse=True)
        limited_results = results[:limit]
        
        return {
            "query": query,
            "results": limited_results,
            "total_matches": len(results),
            "limit": limit,
            "count": len(limited_results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.get("/product/{product_id}")
async def get_product(product_id: str):
    """Get a specific product by ID."""
    try:
        products = load_sample_data()
        
        # Find product by parent_asin
        for product in products:
            if product.get("parent_asin") == product_id or product.get("asin") == product_id:
                return product
        
        raise HTTPException(status_code=404, detail="Product not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get product {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve product")


@app.get("/sample-queries")
async def get_sample_queries():
    """Get sample search queries."""
    return [
        "black shoes",
        "nike sneakers", 
        "blue jeans",
        "summer dress",
        "winter jacket",
        "leather bag",
        "running shoes",
        "casual shirt"
    ]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    logger.info("Starting Fashion Search API...")
    load_sample_data()
    logger.info("Fashion Search API ready!")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    ) 