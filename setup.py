#!/usr/bin/env python3
"""
Amazon Fashion Search Engine - Setup Script

Installs the Amazon Fashion Search Engine and its dependencies.
This script sets up both the backend services and can optionally
configure the frontend dependencies.
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or higher is required.")
    sys.exit(1)

# Read the long description from README
README_PATH = Path(__file__).parent / "README.md"
long_description = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

# Read version from pyproject.toml or set default
VERSION = "1.0.0"

# Core dependencies for the search engine
CORE_DEPS = [
    # FastAPI and async support
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    
    # Data processing
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # OpenAI and ML
    "openai>=1.0.0",
    "tiktoken>=0.5.0",
    
    # Vector search
    "faiss-cpu>=1.7.4",
    
    # Data validation and configuration
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    
    # Utilities
    "python-multipart>=0.0.6",  # For FastAPI file uploads
    "python-dotenv>=1.0.0",     # Environment variable support
]

# Development dependencies
DEV_DEPS = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.0.0",
]

# Optional dependencies for enhanced features
EXTRAS = {
    "dev": DEV_DEPS,
    "all": CORE_DEPS + DEV_DEPS,
}

setup(
    name="amazon-fashion-search-engine",
    version=VERSION,
    description="Semantic fashion product search using LLMs and vector similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fashion Search Team",
    author_email="team@fashionsearch.com",
    url="https://github.com/your-org/amazon-fashion-search-engine",
    license="MIT",
    
    # Package discovery
    packages=find_packages(include=["services*", "shared*"]),
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.env*"],
    },
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=CORE_DEPS,
    extras_require=EXTRAS,
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "fashion-search-pipeline=services.data_pipeline.main:main",
            "fashion-search-api=services.search_api.main:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    
    # Keywords
    keywords=[
        "fashion", "search", "semantic-search", "vector-search", 
        "llm", "openai", "embeddings", "fastapi", "machine-learning"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/amazon-fashion-search-engine/issues",
        "Source": "https://github.com/your-org/amazon-fashion-search-engine",
        "Documentation": "https://github.com/your-org/amazon-fashion-search-engine#readme",
    },
    
    # Additional metadata
    zip_safe=False,
)