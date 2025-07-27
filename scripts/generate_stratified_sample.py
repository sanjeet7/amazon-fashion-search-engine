#!/usr/bin/env python3
"""
Stratified Sample Generation for Amazon Fashion Dataset

Implements the sampling strategy from final_exploration.md:
- Quality-based stratification (premium, standard, basic tiers)
- Brand diversity preservation
- Category coverage maintenance
- 150,000 products for production, configurable for testing
"""

print("🧪 Starting Stratified Sample Generation...")
print("📦 Loading dependencies (pandas import may take 30-60 seconds)...")

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

print("✅ Basic imports loaded")
print("🐼 Loading pandas (this is the slow part)...")

import pandas as pd

print("✅ Pandas loaded!")
print("🔢 Loading numpy...")

import numpy as np

print("✅ Numpy loaded!")
print("📚 Loading other utilities...")

from collections import defaultdict, Counter

print("✅ Standard libraries loaded")
print("🔧 Loading shared modules...")

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🔧 Loading Settings...")
from shared.models import Settings

print("🔧 Loading quality filters...")
from shared.utils import extract_quality_filters

print("🔧 Loading product filters...")
from shared.utils import extract_product_filters

print("🔧 Loading logger...")
from shared.utils import setup_logger

print("✅ All dependencies loaded! Ready to process data.")


def load_raw_dataset(raw_data_path: Path) -> List[Dict[str, Any]]:
    """Load the complete raw dataset."""
    products = []
    
    print(f"📂 Loading raw dataset from {raw_data_path}")
    
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                product = json.loads(line.strip())
                products.append(product)
                
                if i % 50000 == 0:
                    print(f"   Loaded {i:,} products...")
                    
            except json.JSONDecodeError:
                continue
    
    print(f"✅ Loaded {len(products):,} total products")
    return products


def analyze_dataset_quality(products: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze dataset and assign quality indicators for stratification."""
    
    print("🔍 Analyzing dataset quality for stratification...")
    
    analysis_data = []
    
    for i, product in enumerate(products):
        if i % 25000 == 0:
            print(f"   Analyzed {i:,} products...")
        
        # Extract quality indicators
        quality_info = extract_quality_filters(product)
        filter_info = extract_product_filters(product)
        
        # Basic product info
        analysis_record = {
            'parent_asin': product.get('parent_asin', ''),
            'title': product.get('title', ''),
            'brand': filter_info.get('brand', ''),
            'category': filter_info.get('category', ''),
            'price': filter_info.get('price'),
            'rating': filter_info.get('rating'),
            'rating_count': quality_info.get('review_count', 0),
            
            # Quality stratification fields
            'quality_score': quality_info.get('quality_score', 0),
            'quality_tier': quality_info.get('quality_tier', 'basic'),
            'rating_tier': quality_info.get('rating_tier', 'unknown'),
            'review_tier': quality_info.get('review_tier', 'low'),
            'content_tier': quality_info.get('content_tier', 'minimal'),
            'completeness_score': quality_info.get('completeness_score', 0),
            
            # Additional stratification factors
            'has_price': filter_info.get('price') is not None,
            'has_brand': bool(filter_info.get('brand')),
            'has_category': bool(filter_info.get('category')),
            'has_rating': quality_info.get('has_rating', False),
        }
        
        analysis_data.append(analysis_record)
    
    print("🔄 Converting to DataFrame (may take a moment)...")
    df = pd.DataFrame(analysis_data)
    
    print(f"✅ Dataset analysis complete: {len(df):,} products")
    return df


def print_stratification_summary(df: pd.DataFrame):
    """Print summary of stratification dimensions."""
    
    print("\n📊 STRATIFICATION ANALYSIS")
    print("=" * 50)
    
    # Quality tier distribution
    quality_dist = df['quality_tier'].value_counts()
    print(f"Quality Tiers:")
    for tier, count in quality_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {tier:>10}: {count:>8,} ({percentage:>5.1f}%)")
    
    # Brand diversity
    brand_count = df['brand'].nunique()
    print(f"\nBrand Diversity: {brand_count:,} unique brands")
    
    # Category coverage
    category_count = df['category'].nunique()
    print(f"Category Coverage: {category_count:,} unique categories")
    
    # Business signals coverage
    print(f"\nBusiness Signals Coverage:")
    print(f"  Has Rating: {df['has_rating'].sum():,} ({(df['has_rating'].sum() / len(df) * 100):.1f}%)")
    print(f"  Has Price:  {df['has_price'].sum():,} ({(df['has_price'].sum() / len(df) * 100):.1f}%)")
    print(f"  Has Brand:  {df['has_brand'].sum():,} ({(df['has_brand'].sum() / len(df) * 100):.1f}%)")
    
    # Rating tier distribution for products with ratings
    rating_df = df[df['has_rating']]
    if len(rating_df) > 0:
        print(f"\nRating Distribution (of {len(rating_df):,} rated products):")
        rating_dist = rating_df['rating_tier'].value_counts()
        for tier, count in rating_dist.items():
            percentage = (count / len(rating_df)) * 100
            print(f"  {tier:>10}: {count:>8,} ({percentage:>5.1f}%)")


def stratified_sample_selection(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
    """
    Select stratified sample based on final_exploration.md strategy.
    
    Strategy:
    1. Ensure quality representation across all tiers
    2. Maintain brand diversity proportionally  
    3. Preserve category coverage
    4. Prioritize business signal availability
    """
    
    print(f"\n🎯 Generating stratified sample of {target_size:,} products")
    print("Strategy: Quality-based with brand/category diversity")
    
    # Step 1: Quality tier stratification (base allocation)
    quality_targets = {
        'premium': int(target_size * 0.35),   # 35% - high-quality products with strong signals
        'standard': int(target_size * 0.45),  # 45% - medium-quality products
        'basic': int(target_size * 0.20)      # 20% - basic products for diversity
    }
    
    print(f"\nQuality Tier Targets:")
    for tier, target in quality_targets.items():
        print(f"  {tier:>10}: {target:,} products")
    
    selected_samples = []
    
    # Step 2: Sample from each quality tier
    for tier, target_count in quality_targets.items():
        print(f"📊 Processing {tier} tier...")
        tier_df = df[df['quality_tier'] == tier].copy()
        
        if len(tier_df) == 0:
            print(f"  ⚠️  No products in {tier} tier")
            continue
        
        if len(tier_df) <= target_count:
            # Take all products in this tier
            tier_sample = tier_df
            print(f"  {tier:>10}: {len(tier_sample):,} (all available)")
        else:
            # Stratified sampling within tier for brand/category diversity
            tier_sample = sample_with_diversity(tier_df, target_count)
            print(f"  {tier:>10}: {len(tier_sample):,} (sampled)")
        
        selected_samples.append(tier_sample)
    
    # Combine all samples
    print("🔗 Combining samples...")
    final_sample = pd.concat(selected_samples, ignore_index=True)
    
    # Step 3: Fill any shortfall with best remaining products
    remaining_needed = target_size - len(final_sample)
    if remaining_needed > 0:
        print(f"📈 Need {remaining_needed:,} more products, selecting top-quality...")
        used_asins = set(final_sample['parent_asin'])
        remaining_df = df[~df['parent_asin'].isin(used_asins)]
        
        # Sort by quality score and business signal strength
        remaining_df = remaining_df.sort_values(['quality_score', 'has_rating', 'has_price'], ascending=False)
        
        additional_sample = remaining_df.head(remaining_needed)
        final_sample = pd.concat([final_sample, additional_sample], ignore_index=True)
        
        print(f"  Additional: {len(additional_sample):,} top-quality products")
    
    print(f"\n✅ Final sample size: {len(final_sample):,} products")
    return final_sample


def sample_with_diversity(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    """Sample products while maintaining brand and category diversity."""
    
    # Priority factors for sampling
    df = df.copy()
    
    # Calculate diversity score
    brand_counts = df['brand'].value_counts()
    category_counts = df['category'].value_counts()
    
    print(f"    🎯 Calculating diversity scores for {len(df):,} products...")
    
    def diversity_score(row):
        score = row['quality_score']
        
        # Boost underrepresented brands
        brand = row['brand']
        if brand and brand in brand_counts:
            brand_rarity = 1.0 / (brand_counts[brand] + 1)
            score += brand_rarity * 2
        
        # Boost underrepresented categories  
        category = row['category']
        if category and category in category_counts:
            category_rarity = 1.0 / (category_counts[category] + 1)
            score += category_rarity * 1.5
        
        # Boost products with business signals
        if row['has_rating']:
            score += 1.0
        if row['has_price']:
            score += 0.5
        
        return score
    
    df['diversity_score'] = df.apply(diversity_score, axis=1)
    
    # Sample based on diversity score (weighted random sampling)
    df = df.sort_values('diversity_score', ascending=False)
    
    # Take top products with some randomization to avoid pure deterministic selection
    if len(df) > target_count * 2:
        print(f"    🎲 Weighted sampling from top {target_count * 2:,} candidates...")
        # From top 2x candidates, sample with weighted probability
        top_candidates = df.head(target_count * 2)
        weights = top_candidates['diversity_score'] / top_candidates['diversity_score'].sum()
        
        sampled_indices = np.random.choice(
            top_candidates.index, 
            size=target_count, 
            replace=False, 
            p=weights
        )
        
        return df.loc[sampled_indices]
    else:
        print(f"    📊 Taking top {target_count:,} products...")
        return df.head(target_count)


def save_stratified_sample(sample_df: pd.DataFrame, output_path: Path, original_products: List[Dict[str, Any]]):
    """Save the stratified sample as JSONL."""
    
    print(f"\n💾 Saving stratified sample to {output_path}")
    
    # Create lookup for original product data
    print("🔍 Creating product lookup...")
    product_lookup = {p.get('parent_asin'): p for p in original_products}
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("💾 Writing JSONL file...")
    saved_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in sample_df.iterrows():
            asin = row['parent_asin']
            if asin in product_lookup:
                original_product = product_lookup[asin]
                f.write(json.dumps(original_product, ensure_ascii=False) + '\n')
                saved_count += 1
    
    print(f"✅ Saved {saved_count:,} products to {output_path}")
    
    # Save sample analysis
    print("📊 Generating sample analysis...")
    analysis_path = output_path.parent / f"{output_path.stem}_analysis.json"
    sample_analysis = analyze_sample_quality(sample_df)
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(sample_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved sample analysis to {analysis_path}")


def analyze_sample_quality(sample_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the quality of the generated sample."""
    
    analysis = {
        'total_products': int(len(sample_df)),
        'quality_distribution': {k: int(v) for k, v in sample_df['quality_tier'].value_counts().to_dict().items()},
        'brand_diversity': int(sample_df['brand'].nunique()),
        'category_diversity': int(sample_df['category'].nunique()),
        'business_signal_coverage': {
            'has_rating': int(sample_df['has_rating'].sum()),
            'has_price': int(sample_df['has_price'].sum()), 
            'has_brand': int(sample_df['has_brand'].sum()),
            'rating_coverage_pct': float((sample_df['has_rating'].sum() / len(sample_df)) * 100),
            'price_coverage_pct': float((sample_df['has_price'].sum() / len(sample_df)) * 100),
        },
        'quality_metrics': {
            'avg_quality_score': float(sample_df['quality_score'].mean()),
            'avg_completeness_score': float(sample_df['completeness_score'].mean()),
        }
    }
    
    return analysis


def main():
    """Generate stratified sample based on command line arguments."""
    
    parser = argparse.ArgumentParser(description="Generate Stratified Sample for Amazon Fashion Dataset")
    parser.add_argument(
        "--size", 
        type=int, 
        default=150000,
        help="Sample size (default: 150,000 as per final_exploration.md)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file path (default: data/stratified_sample_{size}.jsonl)"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Generate 500-product test sample"
    )
    parser.add_argument(
        "--analyze-only", 
        action="store_true",
        help="Only analyze dataset without generating sample"
    )
    
    args = parser.parse_args()
    print(f"⚙️  Arguments: {args}")
    
    # Setup
    print("🔧 Loading settings...")
    settings = Settings()
    logger = setup_logger("stratified_sampling")
    
    # Determine sample size
    if args.test:
        sample_size = 500
        print("🧪 GENERATING TEST SAMPLE (500 products)")
    else:
        sample_size = args.size
        print(f"🎯 GENERATING STRATIFIED SAMPLE ({sample_size:,} products)")
    
    print("=" * 60)
    
    # Load raw dataset
    if not settings.raw_data_path.exists():
        print(f"❌ Raw data file not found: {settings.raw_data_path}")
        sys.exit(1)
    
    products = load_raw_dataset(settings.raw_data_path)
    
    # Analyze dataset
    df = analyze_dataset_quality(products)
    print_stratification_summary(df)
    
    if args.analyze_only:
        print("\n✅ Analysis complete (no sample generated)")
        return
    
    # Generate stratified sample
    sample_df = stratified_sample_selection(df, sample_size)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.test:
            output_path = Path("data/test_sample_500.jsonl")
        else:
            output_path = Path(f"data/stratified_sample_{sample_size}.jsonl")
    
    # Save sample
    save_stratified_sample(sample_df, output_path, products)
    
    # Final summary
    print("\n🎉 STRATIFIED SAMPLE GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Sample size: {len(sample_df):,} products")
    print(f"📁 Output file: {output_path}")
    print(f"📈 Quality coverage: {sample_df['quality_tier'].value_counts().to_dict()}")
    print(f"🏷️  Brand diversity: {sample_df['brand'].nunique():,} brands")
    print(f"📂 Category diversity: {sample_df['category'].nunique():,} categories")
    
    if args.test:
        print("\n🧪 Test sample ready for pipeline testing!")
    else:
        print("\n🚀 Production sample ready for full pipeline!")


if __name__ == "__main__":
    print("🏃 Starting main function...")
    main() 