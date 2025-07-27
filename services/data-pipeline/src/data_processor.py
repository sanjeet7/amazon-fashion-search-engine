"""Data processing for fashion products."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from shared.models import Product, Settings
from shared.utils import clean_text, extract_features, calculate_tokens, extract_product_filters, extract_quality_filters


logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
    
    def load_raw_data(self, data_source: Optional[Path] = None, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load product data from JSONL file.
        
        Args:
            data_source: Optional path to data file (defaults to raw_data_path)
            sample_size: Optional limit for testing (used for quick testing only)
        """
        
        if data_source is None:
            data_source = self.settings.raw_data_path
        
        self.logger.info(f"Loading data from {data_source}")
        
        if not data_source.exists():
            raise FileNotFoundError(f"Data file not found: {data_source}")
        
        products = []
        
        with open(data_source, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Apply sample size limit if specified (for testing)
                if sample_size is not None and i >= sample_size:
                    break
                
                try:
                    product = json.loads(line.strip())
                    products.append(product)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping invalid JSON at line {i+1}: {e}")
                    continue
        
        source_type = "test sample" if sample_size else "stratified sample" if "stratified" in str(data_source) else "raw data"
        self.logger.info(f"Loaded {len(products)} products from {source_type}")
        return products
    
    def clean_product(self, product_data: Dict[str, Any]) -> Optional[Product]:
        """Clean and validate a single product."""
        
        try:
            # Ensure required fields exist
            if not product_data.get('parent_asin') or not product_data.get('title'):
                return None
            
            # Extract standardized filters
            standardized_filters = extract_product_filters(product_data)
            quality_indicators = extract_quality_filters(product_data)
            
            # Convert to Product model with cleaning
            product = Product(
                parent_asin=product_data['parent_asin'],
                title=clean_text(product_data['title']),
                main_category=clean_text(product_data.get('main_category', '')),
                categories=[clean_text(c) for c in product_data.get('categories', []) if c],
                price=self._safe_float(product_data.get('price')),
                average_rating=self._safe_float(product_data.get('average_rating')),
                rating_number=self._safe_int(product_data.get('rating_number')),
                features=[clean_text(f) for f in product_data.get('features', []) if f],
                description=[clean_text(d) for d in product_data.get('description', []) if d],
                images=self._extract_image_urls(product_data.get('images', [])),
                store=clean_text(product_data.get('store', '')),
                details=product_data.get('details'),
            )
            
            # Generate text for embedding
            product.text_for_embedding = extract_features(product_data)
            
            # Add standardized filter metadata
            product.filter_metadata = standardized_filters
            product.quality_metadata = quality_indicators
            
            return product
            
        except Exception as e:
            self.logger.warning(f"Error cleaning product {product_data.get('parent_asin', 'unknown')}: {e}")
            return None
    
    def process_products(self, raw_products: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
        """Process raw products into clean DataFrame."""
        
        self.logger.info("Processing products...")
        
        cleaned_products = []
        embedding_texts = []
        
        for product_data in raw_products:
            cleaned_product = self.clean_product(product_data)
            if cleaned_product:
                # Convert to dict and flatten filter metadata
                product_dict = cleaned_product.model_dump()
                
                # Add filter metadata as columns for easy filtering
                if cleaned_product.filter_metadata:
                    for key, value in cleaned_product.filter_metadata.items():
                        product_dict[f"filter_{key}"] = value
                
                # Add quality metadata as columns
                if cleaned_product.quality_metadata:
                    for key, value in cleaned_product.quality_metadata.items():
                        product_dict[f"quality_{key}"] = value
                
                cleaned_products.append(product_dict)
                embedding_texts.append(cleaned_product.text_for_embedding)
        
        # Convert to DataFrame
        df = pd.DataFrame(cleaned_products)
        
        self.logger.info(f"Successfully processed {len(df)} products")
        self.logger.info(f"Available filter columns: {[col for col in df.columns if col.startswith('filter_')]}")
        
        return df, embedding_texts
    
    def save_processed_data(self, df: pd.DataFrame, embedding_texts: List[str]) -> None:
        """Save processed data to disk."""
        
        # Ensure output directory exists
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        output_file = self.settings.processed_data_dir / "processed_products.parquet"
        df.to_parquet(output_file, index=False)
        self.logger.info(f"Saved processed data to {output_file}")
        
        # Save embedding texts  
        texts_file = self.settings.processed_data_dir / "embedding_texts.json"
        with open(texts_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_texts, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved embedding texts to {texts_file}")
        
        # Save filter summary for debugging
        filter_columns = [col for col in df.columns if col.startswith('filter_')]
        if filter_columns:
            filter_summary = {}
            for col in filter_columns:
                filter_name = col.replace('filter_', '')
                try:
                    # Handle only hashable types for unique() operation
                    series = df[col].dropna()
                    # Convert complex objects to strings for uniqueness check
                    if len(series) > 0:
                        sample_value = series.iloc[0]
                        if isinstance(sample_value, (dict, list)):
                            # Convert complex types to JSON strings for uniqueness
                            string_values = series.apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else str(x))
                            unique_values = string_values.unique().tolist()
                        else:
                            unique_values = series.unique().tolist()
                    else:
                        unique_values = []
                    
                    filter_summary[filter_name] = {
                        'unique_values': unique_values[:20],  # Limit to first 20 for readability
                        'count': len(unique_values),
                        'coverage': (df[col].notna().sum() / len(df) * 100)
                    }
                except Exception as e:
                    # If there's still an issue, just record basic info
                    filter_summary[filter_name] = {
                        'unique_values': ['Error extracting values'],
                        'count': 'unknown',
                        'coverage': (df[col].notna().sum() / len(df) * 100),
                        'error': str(e)
                    }
            
            summary_file = self.settings.processed_data_dir / "filter_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(filter_summary, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved filter summary to {summary_file}")
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load processed data from disk."""
        
        # Load DataFrame
        parquet_file = self.settings.processed_data_dir / "processed_products.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"Processed data not found: {parquet_file}")
        
        df = pd.read_parquet(parquet_file)
        
        # Load embedding texts
        texts_file = self.settings.processed_data_dir / "embedding_texts.json"
        if not texts_file.exists():
            raise FileNotFoundError(f"Embedding texts not found: {texts_file}")
        
        with open(texts_file, 'r', encoding='utf-8') as f:
            embedding_texts = json.load(f)
        
        self.logger.info(f"Loaded {len(df)} processed products")
        return df, embedding_texts
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        
        stats = {
            'total_products': len(df),
            'products_with_price': df['price'].notna().sum(),
            'products_with_rating': df['average_rating'].notna().sum(),
            'products_with_images': df['images'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum(),
            'unique_categories': df['main_category'].nunique(),
            'avg_tokens_per_product': df['text_for_embedding'].apply(lambda x: len(x.split())).mean() if 'text_for_embedding' in df.columns else 0,
        }
        
        # Add filter statistics
        filter_columns = [col for col in df.columns if col.startswith('filter_')]
        for col in filter_columns:
            filter_name = col.replace('filter_', '')
            coverage = (df[col].notna().sum() / len(df) * 100)
            
            # Handle unhashable types for unique count
            try:
                unique_count = df[col].nunique()
            except TypeError:
                # For unhashable types, convert to strings first
                series = df[col].dropna()
                if len(series) > 0:
                    string_series = series.apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else str(x))
                    unique_count = string_series.nunique()
                else:
                    unique_count = 0
            
            stats[f'{filter_name}_coverage'] = f"{coverage:.1f}%"
            stats[f'{filter_name}_unique_values'] = unique_count
        
        return stats
    
    def detect_data_source(self) -> Tuple[Path, str]:
        """
        Detect the best available data source.
        
        Priority:
        1. Test sample (if exists) - for quick testing
        2. Stratified sample (if exists) - for production
        3. Raw data - fallback
        
        Returns:
            Tuple of (data_path, source_description)
        """
        
        project_root = self.settings.raw_data_path.parent.parent  # Go up from data/raw to project root
        
        # Check for test sample
        test_sample = project_root / "data" / "test_sample_500.jsonl"
        if test_sample.exists():
            return test_sample, "test sample (500 products)"
        
        # Check for stratified samples
        data_dir = project_root / "data"
        if data_dir.exists():
            stratified_files = list(data_dir.glob("stratified_sample_*.jsonl"))
            if stratified_files:
                # Use the largest stratified sample
                largest_sample = max(stratified_files, key=lambda p: p.stat().st_size)
                return largest_sample, f"stratified sample ({largest_sample.stem})"
        
        # Fallback to raw data
        if self.settings.raw_data_path.exists():
            return self.settings.raw_data_path, "raw data (full dataset)"
        
        raise FileNotFoundError("No data source found. Please generate a stratified sample or provide raw data.")
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int."""
        if value is None or value == '':
            return None
        try:
            if isinstance(value, str):
                # Handle comma-separated numbers
                value = value.replace(',', '')
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _extract_image_urls(self, images_data) -> List[str]:
        """Extract image URLs from the images data structure."""
        if not images_data:
            return []
        
        urls = []
        for item in images_data:
            if isinstance(item, dict):
                # Extract high-res first, then large, then thumb as fallback
                for key in ['hi_res', 'large', 'thumb']:
                    url = item.get(key)
                    if url and isinstance(url, str):
                        urls.append(url)
                        break  # Only take the best quality URL per image variant
            elif isinstance(item, str):
                # If it's already a string, use it directly
                urls.append(item)
        
        return urls