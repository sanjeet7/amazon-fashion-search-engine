#!/usr/bin/env python3
"""Search API Service - Main Entry Point."""

import argparse
import sys
from pathlib import Path
import uvicorn

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models import Settings
from shared.utils import setup_logger


def main():
    """Main entry point for search API service."""
    
    parser = argparse.ArgumentParser(description="Amazon Fashion Search API")
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--workers", 
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    try:
        # Load settings (for validation)
        settings = Settings()
        
        # Setup logging
        logger = setup_logger("search-api", settings.log_level)
        
        print("\n🚀 Amazon Fashion Search API")
        print("=" * 50)
        print(f"🌐 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"👥 Workers: {args.workers}")
        print(f"🔄 Reload: {args.reload}")
        print("=" * 50)
        print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"🔍 Health Check: http://{args.host}:{args.port}/health")
        print("=" * 50)
        
        # Run the server
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=settings.log_level.lower(),
            access_log=True
        )
        
    except Exception as e:
        print(f"\n❌ Failed to start Search API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()