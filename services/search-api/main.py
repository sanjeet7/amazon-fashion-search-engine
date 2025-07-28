#!/usr/bin/env python3
"""
Enhanced Search API CLI for Amazon Fashion Search Engine.

This script provides a user-friendly interface for running the search API with options for:
- Production deployment
- Development mode with auto-reload
- Health checks and validation
- Performance monitoring
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.models import Settings
from shared.utils import setup_logger


def setup_cli_logging(level: str = "INFO") -> logging.Logger:
    """Setup CLI-friendly logging."""
    
    logger = setup_logger("search-api-cli", level)
    
    # Add console handler with clean format for CLI
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    # Simple format for CLI
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    
    # Clear existing handlers and add our console handler
    logger.handlers.clear()
    logger.addHandler(console_handler)
    
    return logger


def check_data_readiness(settings: Settings) -> dict:
    """Check if required data files exist."""
    
    processed_path = Path(settings.processed_data_path) / "processed_products.parquet"
    embeddings_path = Path(settings.embeddings_path) / "embeddings.npy"
    index_path = Path(settings.embeddings_path) / "faiss_index.index"
    
    return {
        'processed_data': processed_path.exists(),
        'embeddings': embeddings_path.exists(),
        'faiss_index': index_path.exists(),
        'all_ready': all([processed_path.exists(), embeddings_path.exists()]),
        'paths': {
            'processed': processed_path,
            'embeddings': embeddings_path,
            'index': index_path
        }
    }


async def health_check(host: str, port: int) -> bool:
    """Perform a health check against the running API."""
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{host}:{port}/health", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('status') == 'healthy'
                return False
                
    except Exception:
        return False


def main():
    """Main CLI interface for the search API."""
    
    parser = argparse.ArgumentParser(
        description="Amazon Fashion Search Engine - Search API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Start API (production mode)
  %(prog)s --reload                           # Development mode with auto-reload
  %(prog)s --host localhost --port 8080       # Custom host/port
  %(prog)s --check                            # Health check only

Quick Start:
  1. Ensure data is ready: python services/data-pipeline/main.py
  2. Start API: %(prog)s
  3. Visit: http://localhost:8000/docs
        """
    )
    
    # Server options
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    # Development options
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Operational options
    parser.add_argument(
        "--check", 
        action="store_true",
        help="Perform health check and exit"
    )
    
    parser.add_argument(
        "--validate-data", 
        action="store_true",
        help="Validate data files before starting"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_cli_logging(args.log_level)
    
    try:
        # Load settings
        logger.info("🔧 Loading configuration...")
        settings = Settings()
        
        # Override settings from CLI
        settings.api_host = args.host
        settings.api_port = args.port
        settings.api_workers = args.workers
        settings.reload = args.reload
        settings.log_level = args.log_level
        
        # Health check mode
        if args.check:
            logger.info(f"🏥 Performing health check on {args.host}:{args.port}...")
            
            if asyncio.run(health_check(args.host, args.port)):
                logger.info("✅ API is healthy and responding")
                return 0
            else:
                logger.error("❌ API is not responding or unhealthy")
                return 1
        
        # Validate data readiness
        logger.info("📊 Checking data readiness...")
        data_status = check_data_readiness(settings)
        
        logger.info("Data Status:")
        logger.info(f"   Processed Data: {'✅ Found' if data_status['processed_data'] else '❌ Missing'}")
        logger.info(f"   Embeddings:     {'✅ Found' if data_status['embeddings'] else '❌ Missing'}")
        logger.info(f"   FAISS Index:    {'✅ Found' if data_status['faiss_index'] else '⚠️  Will be built on startup'}")
        
        if not data_status['all_ready']:
            logger.error("❌ Required data files are missing!")
            logger.info("💡 Please run the data pipeline first:")
            logger.info("   python services/data-pipeline/main.py")
            return 1
        
        if args.validate_data:
            logger.info("✅ All required data is present")
            return 0
        
        # Start the API server
        logger.info("🚀 Starting Search API...")
        logger.info(f"   Host: {args.host}")
        logger.info(f"   Port: {args.port}")
        logger.info(f"   Workers: {args.workers}")
        logger.info(f"   Reload: {args.reload}")
        logger.info(f"   Log Level: {args.log_level}")
        
        if args.reload:
            logger.info("🔄 Development mode: Auto-reload enabled")
        else:
            logger.info("🏭 Production mode")
        
        logger.info(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        logger.info(f"🏥 Health Check: http://{args.host}:{args.port}/health")
        
        # Import and run uvicorn
        import uvicorn
        
        # Configure uvicorn
        config = {
            "app": "services.search_api.src.api:app",
            "host": args.host,
            "port": args.port,
            "log_level": args.log_level.lower(),
            "access_log": True,
        }
        
        if args.reload:
            config.update({
                "reload": True,
                "reload_dirs": [str(project_root / "services"), str(project_root / "shared")]
            })
        else:
            config["workers"] = args.workers
        
        # Run the server
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        logger.info("⚠️  API shutdown requested by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        else:
            logger.info("💡 Use --log-level DEBUG for full error details")
        return 1


if __name__ == "__main__":
    sys.exit(main())