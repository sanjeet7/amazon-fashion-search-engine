"""
Data Loader Module

Handles raw data loading and source detection for the data pipeline.
Separated from other processing for better modularity.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from shared.models import Settings

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles raw data loading and source detection."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
    
    def detect_data_source(self) -> Tuple[Path, str]:
        """
        Detect the best available data source.
        
        Returns:
            Tuple of (data_path, description)
        """
        
        self.logger.info("Detecting available data sources...")
        
        # Priority order for data sources
        data_sources = [
            # Stratified sample (preferred for balanced data)
            (Path("data/test_sample_500_analysis.json"), "Stratified sample (500 products)"),
            (Path("data/test_sample_500.jsonl"), "Test sample (500 products)"),
            
            # Processed samples
            (Path("data/processed/sample_products.jsonl"), "Processed sample"),
            
            # Raw data (full dataset)
            (Path(self.settings.raw_data_path) / "meta_Amazon_Fashion.jsonl", "Full raw dataset"),
            (Path("data/raw/meta_Amazon_Fashion.jsonl"), "Raw fashion dataset"),
        ]
        
        for data_path, description in data_sources:
            if data_path.exists():
                file_size = data_path.stat().st_size / (1024 * 1024)  # MB
                self.logger.info(f"Found: {description} ({file_size:.1f} MB)")
                return data_path, description
        
        # If nothing found, check for any .jsonl files in data directories
        for data_dir in [Path("data/raw"), Path("data"), Path(".")]:
            if data_dir.exists():
                jsonl_files = list(data_dir.glob("*.jsonl"))
                if jsonl_files:
                    fallback_file = jsonl_files[0]
                    file_size = fallback_file.stat().st_size / (1024 * 1024)
                    self.logger.warning(f"Using fallback file: {fallback_file} ({file_size:.1f} MB)")
                    return fallback_file, f"Fallback file ({fallback_file.name})"
        
        raise FileNotFoundError(
            "No suitable data source found. Please ensure data files are available in:\n"
            "- data/test_sample_500.jsonl (test sample)\n"
            "- data/raw/meta_Amazon_Fashion.jsonl (full dataset)\n"
            "Or run the data generation script first."
        )
    
    def load_raw_data(
        self, 
        data_source: Optional[Path] = None, 
        sample_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load product data from JSONL file.
        
        Args:
            data_source: Optional path to data file (auto-detected if None)
            sample_size: Optional limit for testing
            
        Returns:
            List of product dictionaries
        """
        
        # Auto-detect data source if not provided
        if data_source is None:
            data_source, description = self.detect_data_source()
            self.logger.info(f"Using auto-detected source: {description}")
        
        if not data_source.exists():
            raise FileNotFoundError(f"Data source not found: {data_source}")
        
        self.logger.info(f"Loading data from: {data_source}")
        
        products = []
        
        try:
            with open(data_source, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Handle both JSON and JSONL formats
                        line = line.strip()
                        if not line:
                            continue
                        
                        product = json.loads(line)
                        products.append(product)
                        
                        # Apply sample size limit if specified
                        if sample_size and len(products) >= sample_size:
                            self.logger.info(f"Reached sample size limit: {sample_size}")
                            break
                        
                        # Progress reporting for large files
                        if line_num % 10000 == 0:
                            self.logger.info(f"Loaded {len(products):,} products...")
                    
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing line {line_num}: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error reading data file: {e}")
            raise
        
        if not products:
            raise ValueError(f"No valid products found in {data_source}")
        
        self.logger.info(f"Successfully loaded {len(products):,} products from {data_source}")
        return products
    
    def save_sample_data(self, products: List[Dict[str, Any]], sample_path: Path) -> None:
        """Save a sample of products for quick testing."""
        
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            for product in products:
                f.write(json.dumps(product, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved {len(products)} products to {sample_path}")
    
    def get_data_info(self, data_path: Path) -> Dict[str, Any]:
        """Get information about a data file without loading it."""
        
        if not data_path.exists():
            return {'exists': False}
        
        file_size = data_path.stat().st_size
        line_count = 0
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    line_count += 1
        except Exception as e:
            self.logger.warning(f"Error counting lines in {data_path}: {e}")
        
        return {
            'exists': True,
            'path': str(data_path),
            'size_bytes': file_size,
            'size_mb': file_size / (1024 * 1024),
            'estimated_products': line_count,
            'format': 'JSONL'
        }