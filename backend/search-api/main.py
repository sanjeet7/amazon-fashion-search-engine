"""Main entry point for the Search API service."""

import uvicorn
from pathlib import Path

from shared.models import SearchAPIConfig
from shared.utils import setup_logger

from src.app import create_app


def main():
    """Main entry point for the search API service."""
    
    # Load configuration
    config = SearchAPIConfig()
    
    # Setup logging
    logger = setup_logger("search-api", config.log_level, config.log_file)
    
    # Create FastAPI app
    app = create_app(config)
    
    logger.info(f"Starting Search API service on {config.host}:{config.port}")
    logger.info(f"Workers: {config.workers}, Reload: {config.reload}")
    
    # Run the server
    uvicorn.run(
        "src.app:create_app",
        factory=True,
        host=config.host,
        port=config.port,
        workers=config.workers if not config.reload else 1,  # Single worker for reload
        reload=config.reload,
        log_level=config.log_level.value.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main() 