"""FastAPI application for the fashion search API."""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from shared.models import (
    Settings, SearchRequest, SearchResponse, SearchResult, 
    HealthResponse, ProductResult
)
from shared.utils import setup_logger

from .search_engine import SearchEngine


# Global application state
app_state = {
    'search_engine': None,
    'settings': None,
    'start_time': time.time(),
    'initialization_error': None
}


async def initialize_search_service():
    """Initialize the search service with data and embeddings."""
    
    logger = logging.getLogger("search-api")
    
    try:
        logger.info("Initializing search service...")
        
        # Load settings
        settings = Settings()
        app_state['settings'] = settings
        
        # Initialize search engine
        search_engine = SearchEngine(settings)
        
        # Load processed data
        import sys
        from pathlib import Path
        
        # Add the data-pipeline src directory to path
        data_pipeline_src = Path(__file__).parent.parent.parent / "data-pipeline" / "src"
        sys.path.insert(0, str(data_pipeline_src))
        
        from data_processor import DataProcessor
        from embedding_generator import EmbeddingGenerator
        
        data_processor = DataProcessor(settings)
        embedding_generator = EmbeddingGenerator(settings)
        
        # Check if data exists
        if not (settings.processed_data_dir / "processed_products.parquet").exists():
            raise FileNotFoundError(
                "Processed data not found. Please run the data pipeline first:\n"
                "cd services/data-pipeline && python main.py"
            )
        
        if not embedding_generator.embeddings_exist():
            raise FileNotFoundError(
                "Embeddings not found. Please run the data pipeline first:\n"
                "cd services/data-pipeline && python main.py"
            )
        
        # Load data
        logger.info("Loading processed data...")
        products_df, _ = data_processor.load_processed_data()
        
        logger.info("Loading embeddings...")
        embeddings, product_ids, metadata = embedding_generator.load_embeddings()
        
        # Initialize search engine
        search_engine.initialize(embeddings, products_df, product_ids)
        
        app_state['search_engine'] = search_engine
        
        logger.info("Search service initialized successfully")
        
    except Exception as e:
        error_msg = f"Failed to initialize search service: {e}"
        logger.error(error_msg)
        app_state['initialization_error'] = error_msg
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    await initialize_search_service()
    yield
    # Shutdown
    logging.getLogger("search-api").info("Shutting down search service")


# Create FastAPI application
app = FastAPI(
    title="Amazon Fashion Search API",
    description="Semantic search API for fashion products using OpenAI embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logger = setup_logger("search-api", "INFO")


@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with basic information."""
    return {
        "service": "Amazon Fashion Search API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Health check endpoint with detailed system information."""
    
    uptime = time.time() - app_state['start_time']
    
    if app_state['initialization_error']:
        return HealthResponse(
            status="error",
            version="1.0.0",
            uptime_seconds=uptime,
            total_products=0,
            index_ready=False,
            embeddings_loaded=False
        )
    
    search_engine = app_state['search_engine']
    
    if search_engine is None:
        return HealthResponse(
            status="initializing",
            version="1.0.0", 
            uptime_seconds=uptime,
            total_products=0,
            index_ready=False,
            embeddings_loaded=False
        )
    
    stats = search_engine.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        total_products=stats['total_products'],
        index_ready=stats['index_ready'],
        embeddings_loaded=stats['embeddings_loaded'],
        avg_search_time_ms=stats['avg_search_time_ms'],
        total_searches=stats['total_searches']
    )


@app.post("/search", response_model=SearchResponse, summary="Semantic product search")
async def search_products(request: SearchRequest):
    """
    Perform semantic search for fashion products.
    
    This endpoint accepts natural language queries and returns relevant products
    ranked by semantic similarity and business signals.
    
    Example queries:
    - "comfortable summer dresses under $50"
    - "elegant wedding guest outfit"
    - "professional work attire"
    """
    
    if app_state['initialization_error']:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search service not available: {app_state['initialization_error']}"
        )
    
    search_engine = app_state['search_engine']
    
    if search_engine is None or not search_engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not ready. Please wait for initialization to complete."
        )
    
    try:
        # Perform search
        products, metadata = await search_engine.search(request)
        
        # Convert to search results
        results = [
            SearchResult(
                product=product,
                rank=i + 1
            )
            for i, product in enumerate(products)
        ]
        
        # Build response
        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=metadata['search_time_ms'],
            enhanced_query=metadata.get('enhanced_query'),
            detected_intent=None,  # Could be added with LLM analysis
            filters_applied=metadata.get('extracted_filters')
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/stats", summary="Search engine statistics")
async def get_stats():
    """Get search engine performance statistics."""
    
    if app_state['search_engine'] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not initialized"
        )
    
    stats = app_state['search_engine'].get_stats()
    
    return {
        **stats,
        'service_uptime_seconds': time.time() - app_state['start_time'],
        'initialization_error': app_state['initialization_error']
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )