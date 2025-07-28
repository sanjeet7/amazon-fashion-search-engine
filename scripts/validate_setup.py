#!/usr/bin/env python3
"""
Setup Validation Script

Tests that the Amazon Fashion Search Engine was installed correctly
and all components can be imported and initialized.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    
    print("🔍 Testing module imports...")
    
    try:
        # Test shared modules
        from shared.models import Settings
        print("  ✅ shared.models imported successfully")
        
        from shared.utils import setup_logger
        print("  ✅ shared.utils imported successfully")
        
        # Test search API modules
        from services.search_api.src.search import SearchEngine
        print("  ✅ SearchEngine imported successfully")
        
        from services.search_api.src.search import (
            VectorSearchManager, LLMProcessor, FilterManager, RankingManager
        )
        print("  ✅ All search components imported successfully")
        
        # Test data pipeline modules
        from services.data_pipeline.src.pipeline import DataPipeline
        print("  ✅ DataPipeline imported successfully")
        
        from services.data_pipeline.src.processors import DataLoader
        print("  ✅ DataLoader imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_initialization():
    """Test that components can be initialized."""
    
    print("\n🔧 Testing component initialization...")
    
    try:
        from shared.models import Settings
        from services.search_api.src.search import VectorSearchManager, LLMProcessor
        
        # Create settings (will use defaults if no .env)
        settings = Settings()
        print("  ✅ Settings loaded successfully")
        
        # Test vector search manager
        vector_search = VectorSearchManager(settings)
        print("  ✅ VectorSearchManager initialized")
        
        # Test LLM processor (will fail if no API key, but that's expected)
        try:
            llm_processor = LLMProcessor(settings)
            print("  ✅ LLMProcessor initialized")
        except Exception as e:
            if "api_key" in str(e).lower():
                print("  ⚠️  LLMProcessor needs OPENAI_API_KEY (expected)")
            else:
                raise
        
        return True
        
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

def test_cli_commands():
    """Test that CLI commands are available."""
    
    print("\n🖥️  Testing CLI commands...")
    
    import subprocess
    import shutil
    
    # Check if CLI commands are available
    commands = ["fashion-search-pipeline", "fashion-search-api"]
    
    all_available = True
    for cmd in commands:
        if shutil.which(cmd):
            print(f"  ✅ {cmd} command available")
        else:
            print(f"  ❌ {cmd} command not found")
            all_available = False
    
    if not all_available:
        print("  💡 Run 'pip install -e .' to install CLI commands")
    
    return all_available

def test_environment():
    """Test environment configuration."""
    
    print("\n🔐 Testing environment configuration...")
    
    try:
        from shared.models import Settings
        
        settings = Settings()
        
        # Check OpenAI API key
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
            print("  ✅ OPENAI_API_KEY is configured")
        else:
            print("  ⚠️  OPENAI_API_KEY not configured (set in .env file)")
        
        # Check data paths
        if Path(settings.raw_data_path).exists():
            print(f"  ✅ Raw data path exists: {settings.raw_data_path}")
        else:
            print(f"  ⚠️  Raw data path not found: {settings.raw_data_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Environment test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    
    print("🎯 Amazon Fashion Search Engine - Setup Validation")
    print("=" * 60)
    
    results = {
        "imports": test_imports(),
        "initialization": test_initialization(), 
        "cli_commands": test_cli_commands(),
        "environment": test_environment()
    }
    
    print("\n" + "=" * 60)
    print("📊 Validation Results:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.title()}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Setup is successful.")
        print("\n🚀 Next steps:")
        print("  1. Set OPENAI_API_KEY in .env file")
        print("  2. Run: fashion-search-pipeline --validate-only")
        print("  3. Run: fashion-search-api")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the installation.")
        print("\n🔧 Troubleshooting:")
        print("  1. Run: pip install -e .")
        print("  2. Check Python version (3.8+ required)")
        print("  3. Check that all dependencies installed correctly")
        return 1

if __name__ == "__main__":
    sys.exit(main())