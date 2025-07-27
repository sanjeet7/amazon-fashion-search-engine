"""Data processing for fashion products."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from shared.models import Product, Settings
from shared.utils import clean_text, extract_features, calculate_tokens


logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw product data from JSONL file."""
        
        data_path = self.settings.data_path
        self.logger.info(f"Loading data from {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        products = []
        sample_size = self.settings.effective_sample_size
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                
                try:
                    product = json.loads(line.strip())
                    products.append(product)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping invalid JSON at line {i+1}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(products)} products")
        return products
    
    def clean_product(self, product_data: Dict[str, Any]) -> Optional[Product]:
        """Clean and validate a single product."""
        
        try:
            # Ensure required fields exist
            if not product_data.get('parent_asin') or not product_data.get('title'):
                return None
            
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
                images=product_data.get('images', []),
                store=clean_text(product_data.get('store', '')),
                details=product_data.get('details'),
            )
            
            # Generate text for embedding
            product.text_for_embedding = extract_features(product_data)
            
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
                cleaned_products.append(cleaned_product.model_dump())
                embedding_texts.append(cleaned_product.text_for_embedding)
        
        # Convert to DataFrame
        df = pd.DataFrame(cleaned_products)
        
        self.logger.info(f"Successfully processed {len(df)} products")
        
        return df, embedding_texts
    
    def save_processed_data(self, df: pd.DataFrame, embedding_texts: List[str]) -> None:
        """Save processed data to disk."""
        
        # Save DataFrame
        output_file = self.settings.processed_data_dir / "processed_products.parquet"
        df.to_parquet(output_file, index=False)
        self.logger.info(f"Saved processed data to {output_file}")
        
        # Save embedding texts  
        texts_file = self.settings.processed_data_dir / "embedding_texts.json"
        with open(texts_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_texts, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved embedding texts to {texts_file}")
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load previously processed data."""
        
        # Load DataFrame
        data_file = self.settings.processed_data_dir / "processed_products.parquet"
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data not found: {data_file}")
        
        df = pd.read_parquet(data_file)
        
        # Load embedding texts
        texts_file = self.settings.processed_data_dir / "embedding_texts.json" 
        if not texts_file.exists():
            raise FileNotFoundError(f"Embedding texts not found: {texts_file}")
        
        with open(texts_file, 'r', encoding='utf-8') as f:
            embedding_texts = json.load(f)
        
        self.logger.info(f"Loaded {len(df)} processed products")
        return df, embedding_texts
    
    def calculate_statistics(self, df: pd.DataFrame, embedding_texts: List[str]) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        
        stats = {
            'total_products': len(df),
            'products_with_price': df['price'].notna().sum(),
            'products_with_rating': df['average_rating'].notna().sum(),
            'products_with_images': (df['images'].apply(len) > 0).sum(),
            'unique_categories': df['main_category'].nunique(),
            'avg_features_per_product': df['features'].apply(len).mean(),
            'total_tokens': sum(calculate_tokens(text) for text in embedding_texts),
            'avg_tokens_per_product': np.mean([calculate_tokens(text) for text in embedding_texts])
        }
        
        # Price statistics
        if stats['products_with_price'] > 0:
            price_data = df['price'].dropna()
            stats['price_stats'] = {
                'min': float(price_data.min()),
                'max': float(price_data.max()),
                'mean': float(price_data.mean()),
                'median': float(price_data.median())
            }
        
        # Rating statistics  
        if stats['products_with_rating'] > 0:
            rating_data = df['average_rating'].dropna()
            stats['rating_stats'] = {
                'min': float(rating_data.min()),
                'max': float(rating_data.max()),
                'mean': float(rating_data.mean())
            }
        
        return stats
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int."""
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None