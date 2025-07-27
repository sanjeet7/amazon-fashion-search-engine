"""Text processing utilities for fashion search."""

import re
import tiktoken
from typing import List, Dict, Any, Optional, Tuple

# Lazy import - only import openai when actually needed
# import openai  # Moved to be lazy


def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that don't add value
    text = re.sub(r'[^\w\s\-\.\,\:\;\!\?]', '', text)
    
    # Remove redundant phrases common in product descriptions
    noise_patterns = [
        r'\b(imported|made in|brand new|free shipping)\b',
        r'\b(asin|upc|model number)[\s:]+\w+\b',
        r'\b(visit our store|see more items)\b'
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


# STANDARDIZED FILTER VALUES - Used by both ingestion and query sides
STANDARD_CATEGORIES = {
    'dress': ['dress', 'dresses', 'gown', 'gowns', 'frock', 'frocks'],
    'shirt': ['shirt', 'shirts', 'blouse', 'blouses', 'top', 'tops', 't-shirt', 't-shirts', 'tee', 'tees'],
    'pants': ['pants', 'pant', 'trousers', 'trouser', 'jeans', 'jean'],
    'shoes': ['shoes', 'shoe', 'sneakers', 'sneaker', 'boots', 'boot', 'heels', 'heel', 'sandals', 'sandal'],
    'jacket': ['jacket', 'jackets', 'coat', 'coats', 'blazer', 'blazers'],
    'skirt': ['skirt', 'skirts'],
    'shorts': ['shorts', 'short'],
    'sweater': ['sweater', 'sweaters', 'pullover', 'pullovers', 'cardigan', 'cardigans'],
    'underwear': ['underwear', 'undergarment', 'bra', 'bras', 'panties', 'boxers'],
    'accessories': ['bag', 'bags', 'purse', 'purses', 'belt', 'belts', 'jewelry', 'watch', 'watches', 'hat', 'hats']
}

STANDARD_COLORS = {
    'black': ['black', 'noir'],
    'white': ['white', 'cream', 'ivory', 'off-white'],
    'blue': ['blue', 'navy', 'royal', 'cobalt', 'azure'],
    'red': ['red', 'crimson', 'scarlet', 'burgundy'],
    'green': ['green', 'olive', 'forest', 'emerald'],
    'yellow': ['yellow', 'gold', 'golden'],
    'pink': ['pink', 'rose', 'magenta'],
    'purple': ['purple', 'violet', 'lavender'],
    'orange': ['orange', 'coral', 'peach'],
    'brown': ['brown', 'tan', 'beige', 'khaki'],
    'gray': ['gray', 'grey', 'silver', 'charcoal'],
    'multicolor': ['multicolor', 'multi-color', 'print', 'printed', 'pattern', 'patterned']
}

STANDARD_MATERIALS = {
    'cotton': ['cotton', '100% cotton', 'organic cotton'],
    'polyester': ['polyester', 'poly'],
    'leather': ['leather', 'genuine leather', 'faux leather'],
    'denim': ['denim', 'jean'],
    'silk': ['silk', '100% silk'],
    'wool': ['wool', 'merino', 'cashmere'],
    'linen': ['linen', '100% linen'],
    'spandex': ['spandex', 'elastane', 'lycra'],
    'nylon': ['nylon'],
    'blend': ['blend', 'mixed', 'combination']
}

STANDARD_STYLES = {
    'casual': ['casual', 'everyday', 'relaxed'],
    'formal': ['formal', 'dress', 'business', 'professional'],
    'sporty': ['sporty', 'athletic', 'active', 'sport'],
    'vintage': ['vintage', 'retro', 'classic'],
    'bohemian': ['bohemian', 'boho', 'hippie'],
    'modern': ['modern', 'contemporary', 'trendy'],
    'elegant': ['elegant', 'sophisticated', 'chic']
}

STANDARD_OCCASIONS = {
    'work': ['work', 'office', 'business', 'professional'],
    'party': ['party', 'celebration', 'festive'],
    'wedding': ['wedding', 'bridal', 'formal event'],
    'casual': ['casual', 'everyday', 'daily'],
    'vacation': ['vacation', 'travel', 'holiday'],
    'date': ['date', 'romantic', 'dinner'],
    'sport': ['sport', 'gym', 'workout', 'athletic']
}


def normalize_brand_name(brand: str) -> str:
    """Normalize brand names for consistent matching."""
    if not brand:
        return ""
    
    brand = clean_text(brand).strip().lower()
    
    # Handle common brand variations
    brand_mappings = {
        'nike inc': 'nike',
        'adidas ag': 'adidas',
        'the gap': 'gap',
        'h&m hennes & mauritz': 'h&m',
        'zara sa': 'zara'
    }
    
    return brand_mappings.get(brand, brand).title()


def extract_standardized_category(text: str) -> Optional[str]:
    """Extract standardized category from text."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    for standard_cat, variations in STANDARD_CATEGORIES.items():
        for variation in variations:
            if variation in text_lower:
                return standard_cat
    
    return None


def extract_standardized_color(text: str) -> Optional[str]:
    """Extract standardized color from text."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    for standard_color, variations in STANDARD_COLORS.items():
        for variation in variations:
            if variation in text_lower:
                return standard_color
    
    return None


def extract_standardized_material(text: str) -> Optional[str]:
    """Extract standardized material from text."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    for standard_material, variations in STANDARD_MATERIALS.items():
        for variation in variations:
            if variation in text_lower:
                return standard_material
    
    return None


def extract_standardized_style(text: str) -> Optional[str]:
    """Extract standardized style from text."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    for standard_style, variations in STANDARD_STYLES.items():
        for variation in variations:
            if variation in text_lower:
                return standard_style
    
    return None


def extract_standardized_occasion(text: str) -> Optional[str]:
    """Extract standardized occasion from text."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    for standard_occasion, variations in STANDARD_OCCASIONS.items():
        for variation in variations:
            if variation in text_lower:
                return standard_occasion
    
    return None


def extract_product_filters(product_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standardized filters from product data during ingestion."""
    filters = {}
    
    # Combine all text fields for analysis
    all_text = " ".join(filter(None, [
        product_data.get('title', ''),
        product_data.get('main_category', ''),
        " ".join(product_data.get('categories', [])),
        " ".join(product_data.get('features', [])),
        " ".join(product_data.get('description', [])),
        str(product_data.get('details', ''))
    ])).lower()
    
    # Extract standardized filters
    filters['category'] = extract_standardized_category(all_text)
    filters['color'] = extract_standardized_color(all_text)
    filters['material'] = extract_standardized_material(all_text)
    filters['style'] = extract_standardized_style(all_text)
    filters['occasion'] = extract_standardized_occasion(all_text)
    
    # Brand (normalized)
    if product_data.get('brand') or product_data.get('store'):
        brand_raw = product_data.get('brand') or product_data.get('store')
        filters['brand'] = normalize_brand_name(brand_raw)
    
    # Price (standardized format)
    if product_data.get('price'):
        try:
            price = float(product_data['price'])
            filters['price'] = price
            filters['price_tier'] = (
                'budget' if price < 25 else
                'mid' if price < 100 else
                'premium'
            )
        except (ValueError, TypeError):
            pass
    
    # Quality indicators
    if product_data.get('average_rating'):
        try:
            rating = float(product_data['average_rating'])
            filters['rating'] = rating
            filters['rating_tier'] = (
                'high' if rating >= 4.0 else
                'medium' if rating >= 3.0 else
                'low'
            )
        except (ValueError, TypeError):
            pass
    
    # Remove None values
    return {k: v for k, v in filters.items() if v is not None}


def extract_features(product_data: Dict[str, Any]) -> str:
    """Extract and combine key features for embedding generation."""
    
    # Start with title (most important)
    components = []
    
    if product_data.get('title'):
        components.append(clean_text(product_data['title']))
    
    # Add features if available
    if product_data.get('features'):
        features = product_data['features']
        if isinstance(features, list):
            # Take first 3 most relevant features to control token count
            relevant_features = [clean_text(f) for f in features[:3] if f and len(f.strip()) > 10]
            if relevant_features:
                components.append(" ".join(relevant_features))
        elif isinstance(features, str):
            components.append(clean_text(features))
    
    # Add selective details (avoid operational noise)
    if product_data.get('details'):
        details = product_data['details']
        if isinstance(details, dict):
            # Focus on fashion-relevant details
            relevant_keys = ['material', 'fabric', 'style', 'fit', 'color', 'pattern', 'occasion']
            relevant_details = []
            for key, value in details.items():
                if any(rel_key in key.lower() for rel_key in relevant_keys):
                    if value and len(str(value).strip()) > 3:
                        relevant_details.append(f"{key}: {value}")
            
            if relevant_details:
                components.append(" ".join(relevant_details[:2]))  # Limit to 2 details
    
    # Add brand if available and not already in title
    if product_data.get('brand'):
        brand = clean_text(product_data['brand'])
        combined_text = " ".join(components).lower()
        if brand.lower() not in combined_text:
            components.append(f"Brand: {brand}")
    
    return " ".join(components)


def calculate_tokens(text: str, model: str = "gpt-4.1-mini") -> int:
    """Calculate token count for a given text and model."""
    try:
        # Use tiktoken for accurate token counting
        if "gpt-4" in model:
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation
        return len(text.split()) * 1.3


async def extract_search_filters_with_llm(query: str, client) -> Dict[str, Any]:
    """Extract search filters from user query using GPT-4.1-mini with standardized values."""
    
    # Lazy import openai only when actually using LLM functions
    try:
        import openai
    except ImportError:
        return extract_basic_filters(query)
    
    # Create examples using our standardized values
    category_examples = list(STANDARD_CATEGORIES.keys())[:5]
    color_examples = list(STANDARD_COLORS.keys())[:8]
    material_examples = list(STANDARD_MATERIALS.keys())[:5]
    style_examples = list(STANDARD_STYLES.keys())[:5]
    occasion_examples = list(STANDARD_OCCASIONS.keys())[:5]
    
    system_prompt = f"""You are a fashion search assistant. Extract structured filters from user queries using ONLY the standardized values below.

STANDARD VALUES (use exactly these, not variations):
- category: {', '.join(category_examples)} (and others from this set)
- color: {', '.join(color_examples)} (and others from this set) 
- material: {', '.join(material_examples)} (and others from this set)
- style: {', '.join(style_examples)} (and others from this set)
- occasion: {', '.join(occasion_examples)} (and others from this set)
- price_range: {{"min": number, "max": number}} (extract from "under $X", "between $X and $Y", etc.)
- brand: exact brand name mentioned (normalize: "nike" not "Nike Inc")

Return JSON with these fields (use null if not mentioned):
- brand, category, color, material, style, occasion, price_range

Examples:
"blue summer dress under $50" → {{"category": "dress", "color": "blue", "price_range": {{"max": 50}}, "occasion": "casual"}}
"Nike running shoes" → {{"brand": "nike", "category": "shoes", "style": "sporty"}}
"formal work shirts" → {{"category": "shirt", "style": "formal", "occasion": "work"}}
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        # Validate extracted values against our standards
        validated_result = validate_extracted_filters(result)
        return validated_result
        
    except Exception as e:
        # Fallback to basic extraction if LLM fails
        return extract_basic_filters(query)


def validate_extracted_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that extracted filters use standardized values."""
    validated = {}
    
    # Validate category
    if filters.get('category'):
        category = filters['category'].lower()
        if category in STANDARD_CATEGORIES:
            validated['category'] = category
    
    # Validate color
    if filters.get('color'):
        color = filters['color'].lower()
        if color in STANDARD_COLORS:
            validated['color'] = color
    
    # Validate material
    if filters.get('material'):
        material = filters['material'].lower()
        if material in STANDARD_MATERIALS:
            validated['material'] = material
    
    # Validate style
    if filters.get('style'):
        style = filters['style'].lower()
        if style in STANDARD_STYLES:
            validated['style'] = style
    
    # Validate occasion
    if filters.get('occasion'):
        occasion = filters['occasion'].lower()
        if occasion in STANDARD_OCCASIONS:
            validated['occasion'] = occasion
    
    # Pass through other fields as-is
    for key in ['brand', 'price_range', 'size', 'gender']:
        if filters.get(key):
            validated[key] = filters[key]
    
    return validated


def extract_basic_filters(query: str) -> Dict[str, Any]:
    """Fallback filter extraction using standardized patterns."""
    filters = {}
    query_lower = query.lower()
    
    # Extract category using standardized values
    filters['category'] = extract_standardized_category(query_lower)
    
    # Extract color using standardized values
    filters['color'] = extract_standardized_color(query_lower)
    
    # Extract material using standardized values
    filters['material'] = extract_standardized_material(query_lower)
    
    # Extract style using standardized values
    filters['style'] = extract_standardized_style(query_lower)
    
    # Extract occasion using standardized values
    filters['occasion'] = extract_standardized_occasion(query_lower)
    
    # Extract price
    price_pattern = r'under\s*\$?(\d+)|below\s*\$?(\d+)|less\s*than\s*\$?(\d+)'
    price_match = re.search(price_pattern, query_lower)
    if price_match:
        price = int(next(g for g in price_match.groups() if g))
        filters['price_range'] = {'max': price}
    
    # Remove None values
    return {k: v for k, v in filters.items() if v is not None}


async def enhance_query_with_context(query: str, client) -> str:
    """Enhance user query with fashion context using GPT-4.1-mini."""
    
    # Lazy import openai only when actually using LLM functions
    try:
        import openai
    except ImportError:
        return query
    
    system_prompt = """You are a fashion search expert. Enhance user queries by adding relevant fashion context and synonyms.

Rules:
1. Keep the original intent clear
2. Add fashion-relevant synonyms and context
3. Expand abbreviations and casual terms
4. Keep under 100 words
5. Focus on searchable terms

Examples:
"blue dress" → "blue dress navy azure sapphire casual formal summer party elegant"
"comfy shoes" → "comfortable shoes casual walking daily wear soft cushioned supportive"
"work outfit" → "professional work outfit business attire office formal workplace appropriate"
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhance this fashion query: {query}"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception:
        # Fallback to original query if enhancement fails
        return query


def extract_quality_filters(product_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract quality indicators for stratified sampling."""
    quality_indicators = {}
    
    # Rating quality
    if product_data.get('average_rating'):
        rating = float(product_data['average_rating'])
        quality_indicators['has_rating'] = True
        quality_indicators['rating_tier'] = 'high' if rating >= 4.0 else 'medium' if rating >= 3.0 else 'low'
    else:
        quality_indicators['has_rating'] = False
        quality_indicators['rating_tier'] = 'unknown'
    
    # Review count (business signal strength)
    review_count = product_data.get('rating_number', 0)
    if isinstance(review_count, str):
        try:
            review_count = int(review_count.replace(',', ''))
        except:
            review_count = 0
    
    quality_indicators['review_count'] = review_count
    quality_indicators['review_tier'] = (
        'high' if review_count >= 50 else
        'medium' if review_count >= 10 else
        'low'
    )
    
    # Content completeness
    title_complete = bool(product_data.get('title', '').strip())
    features_complete = bool(product_data.get('features'))
    brand_complete = bool(product_data.get('brand', '').strip())
    
    completeness_score = sum([title_complete, features_complete, brand_complete])
    quality_indicators['completeness_score'] = completeness_score
    quality_indicators['content_tier'] = (
        'complete' if completeness_score == 3 else
        'partial' if completeness_score >= 2 else
        'minimal'
    )
    
    # Overall quality score (for stratified sampling)
    base_score = 0
    if quality_indicators['has_rating']:
        base_score += 2
    if quality_indicators['rating_tier'] == 'high':
        base_score += 3
    elif quality_indicators['rating_tier'] == 'medium':
        base_score += 1
    
    if quality_indicators['review_tier'] == 'high':
        base_score += 3
    elif quality_indicators['review_tier'] == 'medium':
        base_score += 1
    
    base_score += completeness_score
    
    quality_indicators['quality_score'] = base_score
    quality_indicators['quality_tier'] = (
        'premium' if base_score >= 8 else
        'standard' if base_score >= 5 else
        'basic'
    )
    
    return quality_indicators