"""Query processing and enhancement using OpenAI GPT models."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
import logging

from openai import AsyncOpenAI

from shared.models import SearchAPIConfig, QueryAnalysis, QueryIntent, Season, Occasion
from shared.utils import setup_logger, extract_keywords


class QueryProcessor:
    """Processes and enhances search queries using LLM."""
    
    def __init__(self, config: SearchAPIConfig, logger: logging.Logger):
        """Initialize query processor.
        
        Args:
            config: Search API configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        
        # Query enhancement cache
        self.enhancement_cache: Dict[str, QueryAnalysis] = {}
        
        self.logger.info(f"Initialized QueryProcessor with model: {config.chat_model}")

    async def analyze_and_enhance_query(self, query: str) -> QueryAnalysis:
        """Analyze and enhance search query using LLM.
        
        Args:
            query: Original search query
            
        Returns:
            QueryAnalysis with enhanced query and extracted attributes
        """
        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self.enhancement_cache:
            self.logger.debug(f"Using cached query analysis for: {query}")
            return self.enhancement_cache[cache_key]
        
        try:
            start_time = time.time()
            
            # Generate query analysis
            analysis = await self._analyze_query_with_llm(query)
            
            # Cache the result
            self.enhancement_cache[cache_key] = analysis
            
            processing_time = time.time() - start_time
            self.logger.info(f"Query analysis completed in {processing_time:.2f}s: {query}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Query enhancement failed: {e}")
            # Return basic analysis as fallback
            return self._create_fallback_analysis(query)

    async def _analyze_query_with_llm(self, query: str) -> QueryAnalysis:
        """Analyze query using OpenAI LLM.
        
        Args:
            query: Search query
            
        Returns:
            QueryAnalysis object
        """
        system_prompt = """You are a fashion search expert. Analyze the user's search query and provide:

1. Enhanced query: Expand the query with fashion-relevant terms and synonyms
2. Intent classification: specific_item, style_search, occasion_based, or seasonal
3. Extracted attributes: brand, category, color, material, price range, season, occasion

Respond in JSON format:
{
    "enhanced_query": "expanded query with fashion terms",
    "intent": "intent_type",
    "extracted_attributes": {
        "brand": "brand_name or null",
        "category": "clothing_category or null", 
        "color": "color or null",
        "material": "material or null",
        "price_min": number_or_null,
        "price_max": number_or_null,
        "season": "spring/summer/fall/winter or null",
        "occasion": "casual/formal/work/party/sport or null",
        "size": "size or null",
        "style": "style_descriptor or null"
    },
    "confidence": 0.8
}"""

        user_prompt = f"Analyze this fashion search query: '{query}'"
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                analysis_data = json.loads(content)
                return self._create_query_analysis(query, analysis_data)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
                # Try to extract enhanced query from text response
                enhanced_query = self._extract_enhanced_query_from_text(content)
                return self._create_fallback_analysis(query, enhanced_query)
                
        except Exception as e:
            self.logger.error(f"LLM query analysis failed: {e}")
            raise

    def _create_query_analysis(self, original_query: str, analysis_data: Dict[str, Any]) -> QueryAnalysis:
        """Create QueryAnalysis object from LLM response.
        
        Args:
            original_query: Original search query
            analysis_data: Parsed LLM response
            
        Returns:
            QueryAnalysis object
        """
        # Extract main fields
        enhanced_query = analysis_data.get('enhanced_query', original_query)
        intent_str = analysis_data.get('intent', 'style_search')
        confidence = analysis_data.get('confidence', 0.5)
        
        # Map intent string to enum
        intent_mapping = {
            'specific_item': QueryIntent.SPECIFIC_ITEM,
            'style_search': QueryIntent.STYLE_SEARCH,
            'occasion_based': QueryIntent.OCCASION_BASED,
            'seasonal': QueryIntent.SEASONAL
        }
        intent = intent_mapping.get(intent_str, QueryIntent.STYLE_SEARCH)
        
        # Extract attributes
        attrs = analysis_data.get('extracted_attributes', {})
        
        # Map season string to enum
        season_str = attrs.get('season')
        season = None
        if season_str:
            season_mapping = {
                'spring': Season.SPRING,
                'summer': Season.SUMMER,
                'fall': Season.FALL,
                'winter': Season.WINTER
            }
            season = season_mapping.get(season_str.lower())
        
        # Map occasion string to enum
        occasion_str = attrs.get('occasion')
        occasion = None
        if occasion_str:
            occasion_mapping = {
                'casual': Occasion.CASUAL,
                'formal': Occasion.FORMAL,
                'work': Occasion.WORK,
                'party': Occasion.PARTY,
                'sport': Occasion.SPORT
            }
            occasion = occasion_mapping.get(occasion_str.lower())
        
        return QueryAnalysis(
            original_query=original_query,
            enhanced_query=enhanced_query,
            intent=intent,
            confidence=confidence,
            extracted_brand=attrs.get('brand'),
            extracted_category=attrs.get('category'),
            extracted_color=attrs.get('color'),
            extracted_material=attrs.get('material'),
            extracted_price_min=attrs.get('price_min'),
            extracted_price_max=attrs.get('price_max'),
            extracted_season=season,
            extracted_occasion=occasion,
            extracted_size=attrs.get('size'),
            extracted_style=attrs.get('style'),
            keywords=extract_keywords(enhanced_query)
        )

    def _extract_enhanced_query_from_text(self, text: str) -> str:
        """Extract enhanced query from non-JSON text response.
        
        Args:
            text: LLM text response
            
        Returns:
            Enhanced query string
        """
        # Look for common patterns
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Enhanced:') or line.startswith('enhanced_query:'):
                return line.split(':', 1)[1].strip().strip('"')
            elif 'enhanced' in line.lower() and ':' in line:
                return line.split(':', 1)[1].strip().strip('"')
        
        # Fallback: return first substantial line
        for line in lines:
            line = line.strip().strip('"')
            if len(line) > 10:  # Substantial content
                return line
        
        return text.strip()

    def _create_fallback_analysis(self, query: str, enhanced_query: Optional[str] = None) -> QueryAnalysis:
        """Create basic query analysis as fallback.
        
        Args:
            query: Original query
            enhanced_query: Enhanced query if available
            
        Returns:
            Basic QueryAnalysis object
        """
        enhanced = enhanced_query or query
        
        # Simple intent detection based on keywords
        intent = QueryIntent.STYLE_SEARCH
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['specific', 'exact', 'particular']):
            intent = QueryIntent.SPECIFIC_ITEM
        elif any(word in query_lower for word in ['wedding', 'party', 'work', 'office', 'meeting']):
            intent = QueryIntent.OCCASION_BASED
        elif any(word in query_lower for word in ['summer', 'winter', 'spring', 'fall', 'seasonal']):
            intent = QueryIntent.SEASONAL
        
        return QueryAnalysis(
            original_query=query,
            enhanced_query=enhanced,
            intent=intent,
            confidence=0.3,  # Low confidence for fallback
            keywords=extract_keywords(enhanced)
        )

    def extract_search_filters(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Extract search filters from query analysis.
        
        Args:
            analysis: Query analysis result
            
        Returns:
            Dictionary of filters for search
        """
        filters = {}
        
        if analysis.extracted_brand:
            filters['brand'] = analysis.extracted_brand
        
        if analysis.extracted_category:
            filters['category'] = analysis.extracted_category
        
        if analysis.extracted_price_min is not None:
            filters['price_min'] = analysis.extracted_price_min
        
        if analysis.extracted_price_max is not None:
            filters['price_max'] = analysis.extracted_price_max
        
        return filters

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get query enhancement statistics.
        
        Returns:
            Enhancement statistics
        """
        return {
            'cache_size': len(self.enhancement_cache),
            'total_enhancements': len(self.enhancement_cache),
            'cache_hit_rate': 1.0 if len(self.enhancement_cache) > 0 else 0.0
        } 