#!/usr/bin/env python3
"""Simple test to validate system structure."""

import json
import sys
from pathlib import Path

def test_structure():
    """Test that all required files exist."""
    print("🧪 Testing project structure...")
    
    required_files = [
        ".env",
        "data/sample_500_products.jsonl",
        "shared/models/__init__.py",
        "shared/utils/__init__.py",
        "services/data-pipeline/src/data_processor.py",
        "services/data-pipeline/src/embedding_generator.py",
        "services/data-pipeline/main.py",
        "services/search-api/src/search_engine.py",
        "services/search-api/src/api.py",
        "services/search-api/main.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def test_sample_data():
    """Test that sample data is valid."""
    print("\n📊 Testing sample data...")
    
    try:
        with open("data/sample_500_products.jsonl", 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 5:
            print(f"❌ Expected at least 5 products, got {len(lines)}")
            return False
        
        # Test first product
        first_product = json.loads(lines[0])
        required_fields = ["parent_asin", "title"]
        
        for field in required_fields:
            if field not in first_product:
                print(f"❌ Missing required field: {field}")
                return False
        
        print(f"✅ Sample data is valid ({len(lines)} products)")
        print(f"✅ First product: {first_product['title']}")
        return True
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False

def test_environment():
    """Test environment configuration."""
    print("\n⚙️ Testing environment...")
    
    try:
        with open(".env", 'r') as f:
            env_content = f.read()
        
        if "OPENAI_API_KEY=" not in env_content:
            print("❌ Missing OPENAI_API_KEY in .env")
            return False
        
        print("✅ Environment file configured")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Amazon Fashion Search Engine - Simple Structure Test")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_structure),
        ("Sample Data", test_sample_data),
        ("Environment", test_environment),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System structure is correct")
        print("✅ Sample data is ready")
        print("✅ Environment is configured")
        print("\nNext steps:")
        print("1. Run: uv run python services/data-pipeline/main.py --sample")
        print("2. Run: uv run python services/search-api/main.py")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)