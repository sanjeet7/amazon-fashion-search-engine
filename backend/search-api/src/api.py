"""FastAPI application for the semantic fashion search microservice.

This module provides REST endpoints for:
- Semantic product search with natural language queries
- Health checks and system statistics
- Data ingestion pipeline management
- Index building and maintenance
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from .config import settings
from .models import (
    SearchRequest, SearchResponse, HealthResponse, StatsResponse, 
    ErrorResponse, ProductResult, QueryAnalysis
)
from .data_processing import DataProcessor, create_data_processor
from .embedding import EmbeddingManager, create_embedding_manager  
from .search import HybridSearchEngine, create_search_engine

# Global instances
data_processor: Optional[DataProcessor] = None
embedding_manager: Optional[EmbeddingManager] = None
search_engine: Optional[HybridSearchEngine] = None

# Application state
app_start_time = time.time()
is_initialized = False
logger = logging.getLogger(__name__)


async def initialize_system():
    """Initialize the semantic search system components."""
    global data_processor, embedding_manager, search_engine, is_initialized
    
    try:
        logger.info("Initializing semantic search system...")
        
        # Initialize components
        data_processor = create_data_processor()
        embedding_manager = create_embedding_manager()
        search_engine = create_search_engine(embedding_manager=embedding_manager)
        
        # Check if we have pre-processed data and embeddings
        processed_data_file = settings.processed_data_dir / "processed_sample.parquet"
        embeddings_file = settings.processed_data_dir / "embeddings_cache" / "embeddings.npz"
        
        if processed_data_file.exists() and embeddings_file.exists():
            logger.info("Loading pre-processed data and embeddings...")
            
            # Load processed data
            product_df = data_processor.load_processed_data()
            
            # Load embeddings  
            embeddings, product_ids, metadata = embedding_manager.load_embeddings()
            
            # Build search index
            search_engine.build_index(embeddings, product_df, product_ids)
            
            logger.info("System initialized with existing data")
        else:
            logger.warning("No pre-processed data found. Use /initialize endpoint to process data.")
        
        is_initialized = True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        is_initialized = False
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    await initialize_system()
    yield
    # Shutdown 
    logger.info("Shutting down semantic search service")


# Create FastAPI application
app = FastAPI(
    title="Semantic Fashion Search API",
    description="Advanced semantic search microservice for fashion products using OpenAI embeddings and hybrid ranking",
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


def get_search_engine() -> HybridSearchEngine:
    """Dependency to get search engine instance."""
    if not search_engine:
        raise HTTPException(
            status_code=503, 
            detail="Search engine not initialized. Use /initialize endpoint first."
        )
    return search_engine


def get_data_processor() -> DataProcessor:
    """Dependency to get data processor instance."""
    if not data_processor:
        raise HTTPException(
            status_code=503,
            detail="Data processor not initialized"
        )
    return data_processor


def get_embedding_manager() -> EmbeddingManager:
    """Dependency to get embedding manager instance.""" 
    if not embedding_manager:
        raise HTTPException(
            status_code=503,
            detail="Embedding manager not initialized"
        )
    return embedding_manager


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            detail={"exception_type": type(exc).__name__}
        ).dict()
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Semantic Fashion Search API",
        "version": "1.0.0",
        "status": "operational" if is_initialized else "initializing",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(
    search_engine: HybridSearchEngine = Depends(get_search_engine)
):
    """Health check endpoint with detailed system status."""
    try:
        uptime_seconds = time.time() - app_start_time
        
        # Get system stats
        index_stats = search_engine.get_index_stats()
        total_products = index_stats.get("total_products", 0)
        
        # Check embedding manager cache
        cache_size = len(embedding_manager._embedding_cache) if embedding_manager else 0
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            total_products=total_products,
            embedding_model=settings.openai_embedding_model,
            cache_size=cache_size
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/search", response_model=SearchResponse)
async def search_products(
    request: SearchRequest,
    search_engine: HybridSearchEngine = Depends(get_search_engine)
):
    """Main search endpoint for natural language product queries.
    
    Performs hybrid semantic and structured search with:
    - Natural language query processing
    - GPT-4.1 query enhancement and intent analysis  
    - Vector similarity search with FAISS
    - Adaptive ranking with business signals
    - Structured filtering (brand, category, price, quality)
    """
    try:
        logger.info(f"Search request: '{request.query}' (top_k={request.top_k})")
        
        # Perform search
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            use_query_enhancement=request.use_query_enhancement
        )
        
        # Apply additional filters from request
        if request.price_min is not None or request.price_max is not None or request.min_rating is not None:
            filtered_results = []
            
            for product in results.results:
                # Price filter
                if request.price_min is not None and product.price is not None:
                    if product.price < request.price_min:
                        continue
                
                if request.price_max is not None and product.price is not None:
                    if product.price > request.price_max:
                        continue
                
                # Rating filter  
                if request.min_rating is not None and product.average_rating is not None:
                    if product.average_rating < request.min_rating:
                        continue
                
                # Category filter
                if request.category_filter is not None:
                    if product.main_category != request.category_filter:
                        continue
                
                filtered_results.append(product)
            
            results.results = filtered_results[:request.top_k]
        
        logger.info(f"Search completed: {len(results.results)} results in {results.response_time_ms:.1f}ms")
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search operation failed: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse)
async def get_statistics(
    data_processor: DataProcessor = Depends(get_data_processor),
    search_engine: HybridSearchEngine = Depends(get_search_engine)
):
    """Get comprehensive system and dataset statistics."""
    try:
        # Get dataset statistics
        if data_processor.processed_data_dir.exists():
            processed_file = data_processor.processed_data_dir / "processed_sample.parquet"
            if processed_file.exists():
                df = data_processor.load_processed_data()
                dataset_stats = data_processor.get_dataset_statistics(df)
            else:
                dataset_stats = {"status": "no_processed_data"}
        else:
            dataset_stats = {"status": "no_data_directory"}
        
        # Get search index statistics
        search_stats = search_engine.get_index_stats()
        
        # Get API statistics (simplified for now)
        api_stats = {
            "uptime_seconds": time.time() - app_start_time,
            "embedding_model": settings.openai_embedding_model,
            "chat_model": settings.openai_chat_model
        }
        
        return StatsResponse(
            dataset_stats=dataset_stats,
            search_stats=search_stats,
            api_stats=api_stats
        )
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@app.post("/initialize")
async def initialize_data_pipeline(
    background_tasks: BackgroundTasks,
    sample_size: Optional[int] = 150000,
    force_rebuild: bool = False,
    data_processor: DataProcessor = Depends(get_data_processor),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """Initialize the data processing pipeline.
    
    This endpoint triggers:
    1. Dataset analysis and quality assessment
    2. Stratified quality-based sampling  
    3. Text preparation and embedding generation
    4. FAISS index building
    5. System ready for search operations
    """
    
    async def run_initialization():
        """Background task for data pipeline initialization."""
        global search_engine, is_initialized
        
        try:
            logger.info(f"Starting data pipeline initialization (sample_size={sample_size})")
            
            # Step 1: Assess dataset
            logger.info("Step 1: Assessing dataset structure...")
            dataset_info = data_processor.assess_dataset_structure()
            logger.info(f"Dataset: {dataset_info.total_records:,} records, {dataset_info.file_size_gb}GB")
            
            # Step 2: Stratified sampling
            logger.info("Step 2: Performing stratified quality sampling...")
            processed_df = data_processor.stratified_quality_sampling(sample_size=sample_size)
            
            # Save processed data
            data_processor.save_processed_data(processed_df)
            
            # Step 3: Prepare embedding texts
            logger.info("Step 3: Preparing texts for embedding...")
            embedding_texts = []
            product_ids = []
            
            for _, row in processed_df.iterrows():
                text = data_processor.prepare_embedding_text(row.to_dict())
                embedding_texts.append(text)
                product_ids.append(row.get('parent_asin', f'product_{len(product_ids)}'))
            
            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings...")
            embedding_results = embedding_manager.generate_embeddings(
                embedding_texts, 
                product_ids
            )
            
            # Save embeddings
            embedding_manager.save_embeddings(embedding_results)
            
            # Step 5: Build search index
            logger.info("Step 5: Building FAISS search index...")
            embeddings_matrix = embedding_manager.create_embedding_matrix(embedding_results)
            
            # Initialize search engine if not already done
            if not search_engine:
                search_engine = create_search_engine(embedding_manager=embedding_manager)
            
            search_engine.build_index(embeddings_matrix, processed_df, product_ids)
            
            # Step 6: Finalize
            is_initialized = True
            logger.info("Data pipeline initialization completed successfully!")
            
            # Log final statistics
            stats = data_processor.get_dataset_statistics(processed_df)
            logger.info(f"Final statistics:")
            logger.info(f"  Total products: {stats['total_records']:,}")
            logger.info(f"  Average quality score: {stats['quality_score_stats']['mean']:.3f}")
            logger.info(f"  Products with ratings: {stats['rating_stats']['products_with_ratings']:,}")
            
        except Exception as e:
            logger.error(f"Data pipeline initialization failed: {e}")
            is_initialized = False
            raise
    
    # Check if already initialized and not forcing rebuild
    if is_initialized and not force_rebuild:
        return {
            "message": "System already initialized",
            "status": "ready",
            "force_rebuild": "Set force_rebuild=true to reinitialize"
        }
    
    # Run initialization in background
    background_tasks.add_task(run_initialization)
    
    return {
        "message": "Data pipeline initialization started",
        "status": "processing",
        "sample_size": sample_size,
        "estimated_time": "3-5 minutes",
        "check_status": "/health"
    }


@app.get("/sample-queries")
async def get_sample_queries():
    """Get sample queries for testing the search functionality."""
    return {
        "sample_queries": [
            "Find blue summer dresses under $50",
            "Comfortable running shoes for women",
            "Professional work blouses",
            "Elegant evening gowns for weddings", 
            "Casual weekend outfits",
            "High quality leather handbags",
            "Winter coats for cold weather",
            "Trendy accessories for young adults",
            "Athletic wear for gym workouts",
            "Formal business suits for men"
        ],
        "query_tips": [
            "Use natural language - describe what you're looking for",
            "Include context like occasion, season, or style preferences",
            "Specify price ranges, brands, or quality requirements",
            "The system understands fashion terminology and seasonal preferences"
        ]
    }


@app.get("/sample-products")
async def get_sample_products_with_images(
    limit: int = 10,
    with_images_only: bool = True,
    data_processor: DataProcessor = Depends(get_data_processor)
):
    """Get sample products with images for frontend development.
    
    Args:
        limit: Number of products to return
        with_images_only: Only return products that have images
    """
    try:
        # Check if processed data exists
        processed_data_file = settings.processed_data_dir / "processed_sample.parquet"
        if not processed_data_file.exists():
            raise HTTPException(
                status_code=404,
                detail="No processed data found. Run data processing first."
            )
        
        # Load processed data
        df = data_processor.load_processed_data()
        
        # Filter for products with images if requested
        if with_images_only:
            df_filtered = df[df['_processed_images'].apply(len) > 0]
        else:
            df_filtered = df
        
        # Sample random products
        sample_df = df_filtered.sample(n=min(limit, len(df_filtered))).reset_index(drop=True)
        
        # Convert to API format
        sample_products = []
        for _, row in sample_df.iterrows():
            product = {
                "parent_asin": row.get('parent_asin', ''),
                "title": row.get('title', ''),
                "main_category": row.get('main_category'),
                "price": row.get('price'),
                "average_rating": row.get('average_rating'),
                "rating_number": row.get('rating_number'),
                "store": row.get('store'),
                "features": row.get('features', []) if isinstance(row.get('features'), list) else [],
                "images": row.get('_processed_images', []),
                "videos": row.get('_processed_videos', []),
                "primary_image": row.get('_primary_image'),
                "categories": row.get('categories', []) if isinstance(row.get('categories'), list) else []
            }
            sample_products.append(product)
        
        return {
            "products": sample_products,
            "total_available": len(df_filtered),
            "returned": len(sample_products),
            "has_images": sum(1 for p in sample_products if p['images']),
            "has_videos": sum(1 for p in sample_products if p['videos'])
        }
        
    except Exception as e:
        logger.error(f"Sample products retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sample products: {str(e)}"
        )


def create_app() -> FastAPI:
    """Factory function to create FastAPI application."""
    return app


def run_development_server():
    """Run development server with auto-reload."""
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True,
        log_level=settings.log_level.lower()
    )


def run_production_server():
    """Run production server."""
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
        log_level=settings.log_level.lower()
    ) 