#!/usr/bin/env python3
"""Search API Service - Main Entry Point."""

print("🔍 Starting Search API...")
print("📦 Loading dependencies...")

import argparse
import sys
from pathlib import Path

print("✅ Basic imports loaded")

# Add shared modules to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("📚 Loading shared modules...")

from shared.models import Settings  
from shared.utils.logging import setup_logger

print("🔧 Loading API components...")

import uvicorn

print("✅ All dependencies loaded!")


def main():
    """Main entry point for search API service."""
    
    parser = argparse.ArgumentParser(description="Amazon Fashion Search API")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1", 
        help="Host to bind to (default: 127.0.0.1)"
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
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    print(f"⚙️  Arguments: {args}")
    
    try:
        print("🔑 Loading settings...")
        settings = Settings()
        print("✅ Settings loaded successfully")
        
        # Print startup information
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
        
        print("🚀 Starting server...")
        
        # Start the server
        uvicorn.run(
            "src.api:app",  # Import the app from api.py
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,  # reload doesn't work with multiple workers
            reload=args.reload,
            access_log=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to start Search API: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("🏃 Starting main function...")
    main()