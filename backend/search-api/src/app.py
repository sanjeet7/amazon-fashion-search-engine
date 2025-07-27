"""FastAPI application factory for the Search API service."""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.models import (
    SearchAPIConfig, SearchRequest, SearchResponse, 
    SampleProductsRequest, SampleProductsResponse,
    HealthResponse, SearchStatsResponse, ErrorResponse
)
from shared.utils import setup_logger

from .search_engine import SearchEngine
from .query_processor import QueryProcessor
from .health import HealthMonitor


# Global service instances
search_engine: Optional[SearchEngine] = None
query_processor: Optional[QueryProcessor] = None
health_monitor: Optional[HealthMonitor] = None
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global search_engine, query_processor, health_monitor
    
    # Get config from app state
    config: SearchAPIConfig = app.state.config
    logger = app.state.logger
    
    # Startup
    logger.info("Initializing Search API service...")
    
    try:
        # Initialize core components
        search_engine = SearchEngine(config, logger)
        query_processor = QueryProcessor(config, logger)
        health_monitor = HealthMonitor(config, logger)
        
        # Load processed data and index
        await search_engine.initialize()
        
        logger.info("Search API service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Search API service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Search API service")
    if search_engine:
        await search_engine.cleanup()


def create_app(config: Optional[SearchAPIConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = SearchAPIConfig()
    
    # Setup logging
    logger = setup_logger("search-api", config.log_level, config.log_file)
    
    # Create FastAPI app
    app = FastAPI(
        title="Amazon Fashion Search API",
        description="Advanced semantic search API for fashion products using OpenAI embeddings",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Store config and logger in app state
    app.state.config = config
    app.state.logger = logger
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=config.cors_methods,
        allow_headers=["*"],
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler for better error responses."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred",
                service="search-api",
                timestamp=str(time.time())
            ).dict()
        )
    
    # Dependency injection
    def get_search_engine() -> SearchEngine:
        """Dependency to get search engine instance."""
        if not search_engine:
            raise HTTPException(
                status_code=503,
                detail="Search engine not initialized"
            )
        return search_engine
    
    def get_query_processor() -> QueryProcessor:
        """Dependency to get query processor instance."""
        if not query_processor:
            raise HTTPException(
                status_code=503,
                detail="Query processor not initialized"
            )
        return query_processor
    
    def get_health_monitor() -> HealthMonitor:
        """Dependency to get health monitor instance."""
        if not health_monitor:
            raise HTTPException(
                status_code=503,
                detail="Health monitor not initialized"
            )
        return health_monitor
    
    # API Endpoints
    
    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint with basic service information."""
        return {
            "service": "Amazon Fashion Search API",
            "version": "1.0.0",
            "status": "operational",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check(
        monitor: HealthMonitor = Depends(get_health_monitor),
        engine: SearchEngine = Depends(get_search_engine)
    ):
        """Health check endpoint with detailed system status."""
        try:
            health_data = await monitor.get_health_status(engine)
            return health_data
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unhealthy")
    
    @app.post("/search", response_model=SearchResponse)
    async def search_products(
        request: SearchRequest,
        engine: SearchEngine = Depends(get_search_engine),
        processor: QueryProcessor = Depends(get_query_processor)
    ):
        """Main search endpoint for natural language product queries."""
        try:
            logger.info(f"Search request: '{request.query}' (top_k={request.top_k})")
            
            # Process and enhance query if requested
            query_analysis = None
            if request.use_query_enhancement:
                query_analysis = await processor.analyze_and_enhance_query(request.query)
            
            # Perform search
            search_results = await engine.search(request, query_analysis)
            
            logger.info(f"Search completed: {len(search_results.results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Search operation failed: {str(e)}"
            )
    
    @app.get("/sample-products", response_model=SampleProductsResponse)
    async def get_sample_products(
        limit: int = 10,
        with_images_only: bool = True,
        category_filter: Optional[str] = None,
        min_rating: Optional[float] = None,
        engine: SearchEngine = Depends(get_search_engine)
    ):
        """Get sample products for frontend development and testing."""
        try:
            request = SampleProductsRequest(
                limit=limit,
                with_images_only=with_images_only,
                category_filter=category_filter,
                min_rating=min_rating
            )
            
            results = await engine.get_sample_products(request)
            return results
            
        except Exception as e:
            logger.error(f"Sample products request failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve sample products: {str(e)}"
            )
    
    @app.get("/sample-queries")
    async def get_sample_queries():
        """Get sample queries for testing the search functionality."""
        return {
            "sample_queries": [
                "comfortable summer dresses under $50",
                "elegant wedding guest outfit",
                "professional work attire for men", 
                "vintage leather jacket brown",
                "running shoes for marathon training",
                "casual weekend wear for women",
                "high quality leather handbags",
                "trendy accessories for young adults",
                "athletic wear for gym workouts",
                "formal business suits"
            ],
            "query_tips": [
                "Use natural language - describe what you're looking for",
                "Include context like occasion, season, or style preferences",
                "Specify price ranges, brands, or quality requirements",
                "The system understands fashion terminology and seasonal preferences"
            ]
        }
    
    @app.get("/stats", response_model=SearchStatsResponse)
    async def get_search_statistics(
        monitor: HealthMonitor = Depends(get_health_monitor),
        engine: SearchEngine = Depends(get_search_engine)
    ):
        """Get comprehensive search and performance statistics."""
        try:
            stats = await monitor.get_search_statistics(engine)
            return stats
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve statistics: {str(e)}"
            )
    
    logger.info("FastAPI application created successfully")
    return app 