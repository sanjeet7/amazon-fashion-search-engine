"""Text processing utilities for fashion products."""

import re
import tiktoken
from typing import List, Dict, Any, Optional
import json


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive punctuation but keep meaningful ones
    text = re.sub(r'[^\w\s\-\.,!?&%$]', '', text)
    
    return text


def extract_features(product_data: Dict[str, Any]) -> str:
    """Extract and combine product features for embedding."""
    
    features = []
    
    # Title (most important)
    if title := product_data.get('title'):
        features.append(clean_text(title))
    
    # Main category
    if main_category := product_data.get('main_category'):
        features.append(f"Category: {clean_text(main_category)}")
    
    # Product features
    if product_features := product_data.get('features'):
        if isinstance(product_features, list):
            cleaned_features = [clean_text(f) for f in product_features if f]
            if cleaned_features:
                features.append("Features: " + " | ".join(cleaned_features[:5]))  # Limit to 5 features
    
    # Description (first few items)
    if description := product_data.get('description'):
        if isinstance(description, list):
            cleaned_desc = [clean_text(d) for d in description if d]
            if cleaned_desc:
                features.append("Description: " + " ".join(cleaned_desc[:3]))  # Limit to 3 descriptions
    
    # Categories (additional context)
    if categories := product_data.get('categories'):
        if isinstance(categories, list):
            clean_categories = [clean_text(c) for c in categories if c]
            if clean_categories:
                features.append("Categories: " + ", ".join(clean_categories[:3]))  # Limit to 3 categories
    
    # Store information
    if store := product_data.get('store'):
        features.append(f"Brand: {clean_text(store)}")
    
    # Price information (for context)
    if price := product_data.get('price'):
        try:
            price_float = float(price)
            features.append(f"Price: ${price_float:.2f}")
        except (ValueError, TypeError):
            pass
    
    # Rating information
    if rating := product_data.get('average_rating'):
        try:
            rating_float = float(rating)
            features.append(f"Rating: {rating_float:.1f} stars")
        except (ValueError, TypeError):
            pass
    
    return " | ".join(features)


def calculate_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Calculate token count for given text and model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback estimation: roughly 4 characters per token
        return len(text) // 4


def extract_search_filters_with_llm(query: str) -> Dict[str, Any]:
    """Extract search filters from natural language query using LLM reasoning.
    
    This is a placeholder for LLM-based filter extraction.
    In a full implementation, this would use GPT-4 to intelligently 
    extract price ranges, categories, brands, etc. from natural language.
    """
    
    filters = {}
    query_lower = query.lower()
    
    # Price extraction (simple patterns for now, could be LLM-enhanced)
    price_patterns = [
        r'under\s*\$?(\d+)', r'below\s*\$?(\d+)', r'less than\s*\$?(\d+)',
        r'max\s*\$?(\d+)', r'maximum\s*\$?(\d+)'
    ]
    
    for pattern in price_patterns:
        if match := re.search(pattern, query_lower):
            filters['price_max'] = float(match.group(1))
            break
    
    # Category extraction (could be enhanced with LLM)
    category_keywords = {
        'dress': ['dress', 'dresses', 'gown', 'frock'],
        'shoes': ['shoes', 'sneakers', 'boots', 'heels', 'sandals'],
        'top': ['shirt', 'blouse', 'top', 't-shirt', 'tee'],
        'bottom': ['pants', 'jeans', 'trousers', 'shorts', 'skirt'],
        'jacket': ['jacket', 'coat', 'blazer', 'cardigan'],
        'accessory': ['bag', 'purse', 'belt', 'jewelry', 'watch', 'hat']
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            filters['category'] = category
            break
    
    return filters


def enhance_query_with_context(query: str) -> str:
    """Enhance search query with fashion domain context.
    
    This could be replaced with LLM-based query enhancement
    that understands fashion terminology and context.
    """
    
    # Simple enhancements (could be LLM-driven)
    enhanced = query
    
    # Add fashion context for ambiguous terms
    fashion_context = {
        'formal': 'formal wear business attire',
        'casual': 'casual everyday comfortable',
        'summer': 'summer lightweight breathable',
        'winter': 'winter warm cozy',
        'work': 'professional business office',
        'party': 'party evening festive',
        'wedding': 'wedding formal elegant'
    }
    
    query_lower = query.lower()
    for term, context in fashion_context.items():
        if term in query_lower:
            enhanced = f"{query} {context}"
            break
    
    return enhanced