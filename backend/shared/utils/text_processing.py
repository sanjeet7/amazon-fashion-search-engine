"""Text processing utilities for fashion product data and search queries."""

import re
import string
from typing import List, Set, Dict, Any
import logging


def clean_text(text: str) -> str:
    """Clean and normalize text for processing.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text


def prepare_embedding_text(record: Dict[str, Any]) -> str:
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
        cleaned_title = clean_text(str(title))
        if cleaned_title:
            text_parts.append(cleaned_title)
    
    # Features (High priority - Structured attributes)  
    features = record.get('features', [])
    if features and isinstance(features, list):
        feature_texts = []
        for feature in features:
            if feature:
                cleaned_feature = clean_text(str(feature))
                if cleaned_feature and len(cleaned_feature) > 3:  # Skip very short features
                    feature_texts.append(cleaned_feature)
        
        if feature_texts:
            # Limit to most important features to control token count
            important_features = feature_texts[:5]  # Top 5 features
            text_parts.append(' '.join(important_features))
    
    # Selective Details (Medium priority - Fashion-relevant only)
    details = record.get('details', {})
    if details and isinstance(details, dict):
        # Filter for fashion-relevant details, exclude operational
        fashion_keys = [
            'material', 'fabric', 'color', 'pattern', 'style', 
            'fit', 'length', 'sleeve', 'collar', 'care', 'size'
        ]
        fashion_details = []
        for key, value in details.items():
            if value and any(fk in str(key).lower() for fk in fashion_keys):
                detail_text = f"{key}: {value}"
                cleaned_detail = clean_text(detail_text)
                if cleaned_detail:
                    fashion_details.append(cleaned_detail)
        
        if fashion_details:
            # Limit details to control token count
            text_parts.append(' '.join(fashion_details[:3]))  # Top 3 details
    
    # Store information for brand/filtering
    store = record.get('store', '')
    if store:
        cleaned_store = clean_text(str(store))
        if cleaned_store:
            text_parts.append(f"Brand: {cleaned_store}")
    
    combined_text = ' '.join(text_parts)
    
    # Final cleanup and length check
    combined_text = clean_text(combined_text)
    
    # Trim to reasonable length (target ~400 characters for ~100 tokens)
    if len(combined_text) > 400:
        combined_text = combined_text[:400].rsplit(' ', 1)[0] + '...'
    
    return combined_text


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text for search enhancement.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Clean text
    text = clean_text(text.lower())
    
    # Remove punctuation and split
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    words = text.split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
        'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can'
    }
    
    # Extract meaningful keywords
    keywords = []
    for word in words:
        if (len(word) > 2 and  # Minimum length
            word not in stop_words and  # Not a stop word
            not word.isdigit() and  # Not just numbers
            word.isalpha()):  # Only alphabetic characters
            keywords.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords[:10]  # Limit to top 10 keywords


def calculate_text_quality_score(text: str) -> float:
    """Calculate a quality score for text content.
    
    Args:
        text: Input text
        
    Returns:
        Quality score between 0 and 1
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    text = text.strip()
    if not text:
        return 0.0
    
    score = 0.0
    
    # Length score (reasonable length gets points)
    if 10 <= len(text) <= 500:
        score += 0.3
    elif len(text) > 500:
        score += 0.2  # Very long text gets fewer points
    elif len(text) > 5:
        score += 0.1  # Short text gets some points
    
    # Word count score
    words = text.split()
    if 3 <= len(words) <= 50:
        score += 0.3
    elif len(words) > 50:
        score += 0.2
    elif len(words) > 1:
        score += 0.1
    
    # Capitalization score (proper capitalization indicates quality)
    if text[0].isupper():
        score += 0.1
    
    # Punctuation score (proper punctuation indicates quality)
    if any(p in text for p in '.!?'):
        score += 0.1
    
    # Avoid excessive capitalization or punctuation
    upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
    if upper_ratio > 0.5:  # More than 50% uppercase
        score -= 0.2
    
    punct_ratio = sum(1 for c in text if c in string.punctuation) / len(text)
    if punct_ratio > 0.2:  # More than 20% punctuation
        score -= 0.1
    
    # Alphanumeric ratio (good mix of letters and numbers)
    alnum_ratio = sum(1 for c in text if c.isalnum()) / len(text)
    if alnum_ratio > 0.7:
        score += 0.1
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, score))


def normalize_brand_name(brand: str) -> str:
    """Normalize brand name for consistent filtering.
    
    Args:
        brand: Raw brand name
        
    Returns:
        Normalized brand name
    """
    if not brand:
        return ""
    
    brand = clean_text(brand.lower())
    
    # Remove common business suffixes
    suffixes = [
        ' inc', ' inc.', ' llc', ' llc.', ' ltd', ' ltd.', 
        ' corp', ' corp.', ' company', ' co.', ' co'
    ]
    
    for suffix in suffixes:
        if brand.endswith(suffix):
            brand = brand[:-len(suffix)].strip()
    
    return brand


def extract_price_from_text(text: str) -> tuple[float, float] | None:
    """Extract price range from text.
    
    Args:
        text: Text that may contain price information
        
    Returns:
        Tuple of (min_price, max_price) or None
    """
    if not text:
        return None
    
    # Common price patterns
    price_patterns = [
        r'\$(\d+(?:\.\d{2})?)',  # $19.99
        r'(\d+(?:\.\d{2})?)\s*dollars?',  # 19.99 dollars
        r'(\d+(?:\.\d{2})?)\s*usd',  # 19.99 USD
        r'price:\s*\$?(\d+(?:\.\d{2})?)',  # price: $19.99
    ]
    
    prices = []
    text_lower = text.lower()
    
    for pattern in price_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                price = float(match)
                if 0 < price < 10000:  # Reasonable price range
                    prices.append(price)
            except ValueError:
                continue
    
    if not prices:
        return None
    
    return (min(prices), max(prices))


def is_fashion_related(text: str) -> bool:
    """Check if text is fashion-related.
    
    Args:
        text: Input text
        
    Returns:
        True if text appears to be fashion-related
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    fashion_keywords = {
        # Clothing categories
        'dress', 'shirt', 'pants', 'jeans', 'skirt', 'jacket', 'coat', 'sweater',
        'hoodie', 'blouse', 'top', 'bottom', 'shorts', 'suit', 'blazer', 'cardigan',
        
        # Footwear
        'shoes', 'boots', 'sneakers', 'sandals', 'heels', 'flats', 'loafers',
        
        # Accessories
        'bag', 'purse', 'wallet', 'belt', 'hat', 'cap', 'scarf', 'jewelry',
        'necklace', 'bracelet', 'ring', 'earrings', 'watch', 'sunglasses',
        
        # Materials
        'cotton', 'silk', 'wool', 'leather', 'denim', 'polyester', 'linen',
        'cashmere', 'velvet', 'satin', 'chiffon',
        
        # Styles
        'casual', 'formal', 'business', 'party', 'evening', 'summer', 'winter',
        'vintage', 'modern', 'classic', 'trendy', 'bohemian', 'minimalist',
        
        # Fashion terms
        'fashion', 'style', 'outfit', 'wardrobe', 'clothing', 'apparel', 'wear'
    }
    
    # Count fashion-related words
    words = re.findall(r'\b\w+\b', text_lower)
    fashion_word_count = sum(1 for word in words if word in fashion_keywords)
    
    # If more than 10% of words are fashion-related, consider it fashion content
    if len(words) > 0:
        fashion_ratio = fashion_word_count / len(words)
        return fashion_ratio > 0.1
    
    return False 