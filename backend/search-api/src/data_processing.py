"""Data processing and ingestion pipeline for Amazon Fashion dataset.

This module implements the stratified quality-based sampling strategy identified 
through comprehensive data analysis, ensuring optimal balance between quality, 
cost, and representation.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .config import settings
from .models import DatasetInfo, SamplingConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processing class for Amazon Fashion dataset."""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        """Initialize data processor.
        
        Args:
            dataset_path: Path to the JSONL dataset file
        """
        self.dataset_path = dataset_path or (settings.data_dir / settings.dataset_file)
        self.processed_data_dir = settings.processed_data_dir
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis-derived constants
        self.EXPECTED_TOTAL_RECORDS = 826108
        self.EXPECTED_FILE_SIZE_GB = 1.32
        self.TARGET_SAMPLE_SIZE = 150000  # Based on analysis
        
        logger.info(f"Initialized DataProcessor with dataset: {self.dataset_path}")

    def assess_dataset_structure(self) -> DatasetInfo:
        """Assess dataset structure and resource requirements.
        
        Returns:
            DatasetInfo containing structural analysis results
            
        Raises:
            FileNotFoundError: If dataset file does not exist
            RuntimeError: If dataset analysis fails
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        try:
            # File size analysis
            file_size_bytes = self.dataset_path.stat().st_size
            file_size_gb = file_size_bytes / (1024**3)
            
            # Record count analysis
            logger.info("Analyzing dataset structure...")
            record_count = 0
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record_count += 1
            
            # Calculate resource metrics
            avg_record_size = file_size_bytes / record_count if record_count > 0 else 0
            estimated_memory_gb = file_size_gb * 2.5  # Conservative memory estimate
            
            dataset_info = DatasetInfo(
                file_size_gb=round(file_size_gb, 2),
                total_records=record_count,
                avg_record_size_bytes=int(avg_record_size),
                estimated_memory_requirement_gb=round(estimated_memory_gb, 2),
                processing_feasibility='Suitable for production processing'
            )
            
            logger.info(f"Dataset analysis complete: {record_count:,} records, {file_size_gb:.2f}GB")
            return dataset_info
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {str(e)}")
            raise RuntimeError(f"Failed to analyze dataset: {str(e)}")

    def calculate_quality_score(self, record: Dict[str, Any]) -> float:
        """Calculate quality score for a product record.
        
        Based on analysis findings:
        - Title completeness (critical)
        - Rating availability (business signal)
        - Feature completeness (embedding quality)
        - Store information (filter capability)
        
        Args:
            record: Product record dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Title completeness (30% weight - critical for embeddings)
        if record.get('title') and len(str(record['title']).strip()) > 0:
            score += 0.3
        
        # Rating availability (25% weight - business signals)
        if record.get('average_rating') is not None and record.get('rating_number') is not None:
            rating_num = record.get('rating_number', 0)
            if rating_num > 0:
                score += 0.25
        
        # Features completeness (20% weight - embedding quality)
        features = record.get('features', [])
        if features and len(features) > 0:
            score += 0.2
        
        # Store information (15% weight - filter capability)
        if record.get('store') and len(str(record['store']).strip()) > 0:
            score += 0.15
        
        # Category information (10% weight - diversity)
        categories = record.get('categories', [])
        if categories and len(categories) > 0:
            score += 0.1
        
        return score

    def extract_semantic_category(self, title: str) -> Optional[str]:
        """Extract semantic category from product title.
        
        Based on analysis findings for filter consistency.
        
        Args:
            title: Product title
            
        Returns:
            Semantic category or None
        """
        if not title:
            return None
            
        title_lower = title.lower()
        
        # Fashion category patterns (validated coverage: 70.6%)
        if any(term in title_lower for term in ['dress', 'gown']):
            return 'dress'
        elif any(term in title_lower for term in ['shirt', 'top', 'blouse', 'sweater', 'jacket']):
            return 'tops'
        elif any(term in title_lower for term in ['pants', 'jeans', 'shorts', 'skirt']):
            return 'bottoms'
        elif any(term in title_lower for term in ['shoe', 'boot', 'sneaker', 'heel', 'sandal']):
            return 'shoes'
        elif any(term in title_lower for term in ['bag', 'belt', 'hat', 'jewelry', 'watch']):
            return 'accessories'
        
        return None

    def stratified_quality_sampling(
        self, 
        sample_size: int = None,
        config: SamplingConfig = None
    ) -> pd.DataFrame:
        """Perform stratified quality-based sampling.
        
        Implements the sampling strategy derived from analysis:
        - Quality-based selection (products with ratings, complete titles, etc.)
        - Brand diversity preservation
        - Category representation balance
        
        Args:
            sample_size: Target sample size (defaults to analysis-derived 150K)
            config: Sampling configuration
            
        Returns:
            DataFrame with sampled records
            
        Raises:
            ValueError: If sample size exceeds dataset size
            RuntimeError: If sampling process fails
        """
        sample_size = sample_size or self.TARGET_SAMPLE_SIZE
        config = config or SamplingConfig()
        
        logger.info(f"Starting stratified quality-based sampling for {sample_size:,} records...")
        
        try:
            # Step 1: Load and score all records
            records = []
            quality_scores = []
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        record = json.loads(line.strip())
                        quality_score = self.calculate_quality_score(record)
                        
                        # Enrich record with derived fields
                        record['_quality_score'] = quality_score
                        record['_semantic_category'] = self.extract_semantic_category(
                            record.get('title', '')
                        )
                        record['_line_number'] = line_num
                        
                        # Extract and validate media URLs
                        media_data = self.extract_media_urls(record)
                        record['_processed_images'] = media_data['images']
                        record['_processed_videos'] = media_data['videos']
                        record['_primary_image'] = media_data['primary_image']
                        
                        records.append(record)
                        quality_scores.append(quality_score)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON at line {line_num}")
                        continue
            
            df = pd.DataFrame(records)
            logger.info(f"Loaded {len(df):,} records for sampling")
            
            if sample_size > len(df):
                raise ValueError(f"Sample size {sample_size} exceeds dataset size {len(df)}")
            
            # Step 2: Quality-based stratification
            # High quality: top 40% by quality score
            # Medium quality: middle 40% 
            # Diverse representation: bottom 20% (to ensure real-world representation)
            
            df_sorted = df.sort_values('_quality_score', ascending=False)
            
            high_quality_size = int(sample_size * 0.6)  # 60% high quality
            medium_quality_size = int(sample_size * 0.3)  # 30% medium quality  
            diverse_representation_size = sample_size - high_quality_size - medium_quality_size  # 10% diverse
            
            # Sample from each stratum
            total_records = len(df_sorted)
            
            # High quality stratum (top 30% of dataset)
            high_quality_end = int(total_records * 0.3)
            high_quality_sample = df_sorted.iloc[:high_quality_end].sample(
                n=min(high_quality_size, high_quality_end), 
                random_state=42
            )
            
            # Medium quality stratum (middle 40% of dataset)
            medium_quality_start = high_quality_end
            medium_quality_end = int(total_records * 0.7)
            medium_quality_sample = df_sorted.iloc[medium_quality_start:medium_quality_end].sample(
                n=min(medium_quality_size, medium_quality_end - medium_quality_start),
                random_state=42
            )
            
            # Diverse representation stratum (bottom 30% of dataset)
            diverse_sample = df_sorted.iloc[medium_quality_end:].sample(
                n=min(diverse_representation_size, total_records - medium_quality_end),
                random_state=42
            )
            
            # Combine samples
            sampled_df = pd.concat([
                high_quality_sample,
                medium_quality_sample, 
                diverse_sample
            ]).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            logger.info(f"Stratified sampling complete:")
            logger.info(f"  High quality: {len(high_quality_sample):,} records")
            logger.info(f"  Medium quality: {len(medium_quality_sample):,} records") 
            logger.info(f"  Diverse representation: {len(diverse_sample):,} records")
            logger.info(f"  Total sample: {len(sampled_df):,} records")
            logger.info(f"  Average quality score: {sampled_df['_quality_score'].mean():.3f}")
            
            return sampled_df
            
        except Exception as e:
            logger.error(f"Stratified sampling failed: {str(e)}")
            raise RuntimeError(f"Failed to perform sampling: {str(e)}")

    def extract_media_urls(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate image/video URLs from product record.
        
        Args:
            record: Product record
            
        Returns:
            Dictionary with processed media URLs
        """
        media_data = {
            'images': [],
            'videos': [],
            'primary_image': None
        }
        
        # Extract images
        images = record.get('images', [])
        if images and isinstance(images, list):
            valid_images = []
            for img in images:
                if img and isinstance(img, str) and img.strip():
                    # Basic URL validation
                    img_url = str(img).strip()
                    if img_url.startswith(('http://', 'https://')):
                        valid_images.append(img_url)
            
            media_data['images'] = valid_images
            # Set primary image as first valid image
            if valid_images:
                media_data['primary_image'] = valid_images[0]
        
        # Extract videos
        videos = record.get('videos', [])
        if videos and isinstance(videos, list):
            valid_videos = []
            for video in videos:
                if video and isinstance(video, str) and video.strip():
                    # Basic URL validation
                    video_url = str(video).strip()
                    if video_url.startswith(('http://', 'https://')):
                        valid_videos.append(video_url)
            
            media_data['videos'] = valid_videos
        
        return media_data

    def prepare_embedding_text(self, record: Dict[str, Any]) -> str:
        """Prepare text for embedding generation.
        
        Implements the Title + Features + Selective Details strategy 
        derived from cost-quality analysis (~89 tokens per product).
        
        Args:
            record: Product record
            
        Returns:
            Combined text for embedding
        """
        text_parts = []
        
        # Title (Critical - Primary semantic content)
        title = record.get('title', '')
        if title:
            text_parts.append(str(title).strip())
        
        # Features (High priority - Structured attributes)  
        features = record.get('features', [])
        if features and isinstance(features, list):
            features_text = ' '.join(str(f) for f in features if f)
            if features_text:
                text_parts.append(features_text)
        
        # Selective Details (Medium priority - Fashion-relevant only)
        details = record.get('details', {})
        if details and isinstance(details, dict):
            # Filter for fashion-relevant details, exclude operational
            fashion_keys = [
                'material', 'fabric', 'color', 'pattern', 'style', 
                'fit', 'length', 'sleeve', 'collar', 'care'
            ]
            fashion_details = []
            for key, value in details.items():
                if any(fk in str(key).lower() for fk in fashion_keys):
                    fashion_details.append(f"{key}: {value}")
            
            if fashion_details:
                text_parts.append(' '.join(fashion_details))
        
        return ' '.join(text_parts)

    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_sample.parquet") -> Path:
        """Save processed data to disk.
        
        Args:
            df: Processed DataFrame
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.processed_data_dir / filename
        df.to_parquet(output_path, compression='snappy')
        logger.info(f"Saved processed data to {output_path}")
        return output_path

    def load_processed_data(self, filename: str = "processed_sample.parquet") -> pd.DataFrame:
        """Load processed data from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded DataFrame
        """
        input_path = self.processed_data_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at {input_path}")
        
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded processed data from {input_path}: {len(df):,} records")
        return df

    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics.
        
        Args:
            df: Dataset DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_records': len(df),
            'quality_score_stats': {
                'mean': df['_quality_score'].mean(),
                'median': df['_quality_score'].median(),
                'std': df['_quality_score'].std(),
                'min': df['_quality_score'].min(),
                'max': df['_quality_score'].max()
            },
            'field_completeness': {
                'title': (df['title'].notna() & (df['title'] != '')).sum(),
                'features': df['features'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum(),
                'average_rating': df['average_rating'].notna().sum(),
                'rating_number': df['rating_number'].notna().sum(),
                'store': (df['store'].notna() & (df['store'] != '')).sum(),
                'price': df['price'].notna().sum()
            },
            'semantic_categories': df['_semantic_category'].value_counts().to_dict(),
            'rating_stats': {
                'mean_rating': df['average_rating'].mean(),
                'median_reviews': df['rating_number'].median(),
                'products_with_ratings': df['average_rating'].notna().sum()
            },
            'store_diversity': {
                'unique_stores': df['store'].nunique(),
                'top_stores': df['store'].value_counts().head(10).to_dict()
            },
            'media_availability': {
                'products_with_images': (df['_processed_images'].apply(len) > 0).sum(),
                'products_with_videos': (df['_processed_videos'].apply(len) > 0).sum(),
                'products_with_primary_image': df['_primary_image'].notna().sum(),
                'avg_images_per_product': df['_processed_images'].apply(len).mean(),
                'avg_videos_per_product': df['_processed_videos'].apply(len).mean()
            }
        }
        
        return stats


def create_data_processor(dataset_path: Optional[Path] = None) -> DataProcessor:
    """Factory function to create DataProcessor instance.
    
    Args:
        dataset_path: Optional custom dataset path
        
    Returns:
        Configured DataProcessor instance
    """
    return DataProcessor(dataset_path) 