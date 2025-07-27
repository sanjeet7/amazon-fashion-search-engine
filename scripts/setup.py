#!/usr/bin/env python3
"""Setup script for Amazon Fashion Search Engine."""

import os
import shutil
import sys
from pathlib import Path


def setup_environment():
    """Set up the development environment."""
    
    print("🚀 Setting up Amazon Fashion Search Engine")
    print("=" * 50)
    
    # Check for .env file
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if not env_file.exists() and env_template.exists():
        print("📝 Creating .env file from template...")
        shutil.copy(env_template, env_file)
        print("✅ .env file created")
        print("⚠️  Please edit .env and set your OPENAI_API_KEY")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("❌ No .env.template found")
    
    # Create data directories
    print("\n📁 Creating data directories...")
    
    data_dirs = [
        "data/processed",
        "data/embeddings",
        "logs"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    # Check for sample data
    sample_file = Path("data/sample_500_products.jsonl")
    raw_file = Path("data/raw/meta_Amazon_Fashion.jsonl")
    
    if not sample_file.exists() and raw_file.exists():
        print("\n📊 Creating sample dataset...")
        with open(raw_file, 'r') as infile, open(sample_file, 'w') as outfile:
            for i, line in enumerate(infile):
                if i >= 500:
                    break
                outfile.write(line)
        print("✅ Sample dataset created (500 products)")
    elif sample_file.exists():
        print("✅ Sample dataset already exists")
    
    print("\n🎉 Setup completed!")
    print("=" * 50)
    print("Next steps:")
    print("1. Edit .env and set your OPENAI_API_KEY")
    print("2. Run data pipeline: python services/data-pipeline/main.py --sample")
    print("3. Start search API: python services/search-api/main.py")


if __name__ == "__main__":
    setup_environment()