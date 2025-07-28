"""
Filter Manager Module

Handles product filtering logic including:
- Request-based filters (price, category, rating)
- LLM-extracted filters with graceful degradation
- Smart semantic matching for filter values
"""

import logging
from typing import List, Dict, Any
from shared.models import ProductResult, SearchRequest

logger = logging.getLogger(__name__)


class FilterManager:
    """Manages product filtering operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.filter_operations = 0
        self.strict_passes = 0
        self.lenient_passes = 0
        self.semantic_only_passes = 0
    
    def apply_filters(
        self, 
        products: List[ProductResult], 
        request: SearchRequest, 
        extracted_filters: Dict[str, Any]
    ) -> List[ProductResult]:
        """
        Apply filters with graceful degradation strategy.
        
        Strategy:
        1. Try strict filtering (all filters)
        2. If too few results, try lenient filtering
        3. If still too few, fall back to semantic-only
        """
        
        self.filter_operations += 1
        min_results_threshold = max(1, request.top_k // 2)
        
        if extracted_filters:
            # Try strict filtering first
            filtered_products = self._apply_strict_filters(products, request, extracted_filters)
            
            if len(filtered_products) >= min_results_threshold:
                self.strict_passes += 1
                self.logger.debug(f"Strict filtering yielded {len(filtered_products)} results")
                return filtered_products
            
            # Try lenient filtering
            filtered_products = self._apply_lenient_filters(products, request, extracted_filters)
            
            if len(filtered_products) >= min_results_threshold:
                self.lenient_passes += 1
                self.logger.debug(f"Lenient filtering yielded {len(filtered_products)} results")
                return filtered_products
            
            self.logger.debug(f"Both strict and lenient filtering yielded too few results, falling back to semantic-only")
        
        # Fall back to semantic-only (no extracted filters)
        self.semantic_only_passes += 1
        return self._apply_request_filters_only(products, request)
    
    def _apply_strict_filters(
        self, 
        products: List[ProductResult], 
        request: SearchRequest, 
        extracted_filters: Dict[str, Any]
    ) -> List[ProductResult]:
        """Apply all filters with strict matching."""
        
        filtered = []
        
        for product in products:
            # Similarity threshold
            if product.similarity_score < request.min_similarity:
                continue
            
            # Apply request-based filters
            if not self._check_request_filters(product, request):
                continue
            
            # Apply LLM-extracted filters with strict matching
            if not self._check_extracted_filters_strict(product, extracted_filters):
                continue
            
            filtered.append(product)
        
        return filtered
    
    def _apply_lenient_filters(
        self, 
        products: List[ProductResult], 
        request: SearchRequest, 
        extracted_filters: Dict[str, Any]
    ) -> List[ProductResult]:
        """Apply filters with lenient matching for better recall."""
        
        filtered = []
        
        for product in products:
            # Similarity threshold (keep strict)
            if product.similarity_score < request.min_similarity:
                continue
            
            # Apply request-based filters (keep strict)
            if not self._check_request_filters(product, request):
                continue
            
            # Apply LLM-extracted filters with lenient matching
            if not self._check_extracted_filters_lenient(product, extracted_filters):
                continue
            
            filtered.append(product)
        
        return filtered
    
    def _apply_request_filters_only(
        self, 
        products: List[ProductResult], 
        request: SearchRequest
    ) -> List[ProductResult]:
        """Apply only request-based filters (no extracted filters)."""
        
        filtered = []
        
        for product in products:
            # Similarity threshold
            if product.similarity_score < request.min_similarity:
                continue
            
            # Apply request-based filters only
            if not self._check_request_filters(product, request):
                continue
            
            filtered.append(product)
        
        return filtered
    
    def _check_request_filters(self, product: ProductResult, request: SearchRequest) -> bool:
        """Check request-based filters (price, category, rating)."""
        
        # Price filters
        if request.price_min is not None and (product.price is None or product.price < request.price_min):
            return False
        if request.price_max is not None and (product.price is None or product.price > request.price_max):
            return False
        
        # Category filter (case-insensitive matching)
        if request.category:
            category_match = False
            search_category = request.category.lower()
            
            # Check main category
            if product.main_category and search_category in product.main_category.lower():
                category_match = True
            
            # Check all categories
            for cat in product.categories:
                if search_category in cat.lower():
                    category_match = True
                    break
            
            if not category_match:
                return False
        
        # Rating filter
        if request.min_rating is not None and (product.average_rating is None or product.average_rating < request.min_rating):
            return False
        
        return True
    
    def _check_extracted_filters_strict(self, product: ProductResult, extracted_filters: Dict[str, Any]) -> bool:
        """Check LLM-extracted filters with strict matching."""
        
        if not extracted_filters or not product.matched_filters:
            return True  # No filters to apply
        
        # Check each extracted filter against product's standardized filter metadata
        for filter_name, filter_value in extracted_filters.items():
            if filter_name == 'price_range':
                # Handle price range filter
                if 'min' in filter_value and (product.price is None or product.price < filter_value['min']):
                    return False
                if 'max' in filter_value and (product.price is None or product.price > filter_value['max']):
                    return False
                continue
            
            # Check standardized filter values
            product_filter_value = product.matched_filters.get(filter_name)
            if product_filter_value is None:
                continue  # Product doesn't have this filter, but don't exclude
            
            # Exact match for standardized values
            if isinstance(filter_value, str) and isinstance(product_filter_value, str):
                if filter_value.lower() != product_filter_value.lower():
                    return False
            elif filter_value != product_filter_value:
                return False
        
        return True
    
    def _check_extracted_filters_lenient(self, product: ProductResult, extracted_filters: Dict[str, Any]) -> bool:
        """Check LLM-extracted filters with lenient matching."""
        
        if not extracted_filters or not product.matched_filters:
            return True  # No filters to apply
        
        # For lenient matching, require at least one filter to match
        any_filter_matched = False
        
        for filter_name, filter_value in extracted_filters.items():
            if filter_name == 'price_range':
                # Handle price range filter (keep strict for price)
                if 'min' in filter_value and (product.price is None or product.price < filter_value['min']):
                    continue
                if 'max' in filter_value and (product.price is None or product.price > filter_value['max']):
                    continue
                any_filter_matched = True
                continue
            
            # Check filter values with smart matching
            product_filter_value = product.matched_filters.get(filter_name)
            if product_filter_value is None:
                continue  # Product doesn't have this filter
            
            # Handle list-based filters
            if isinstance(filter_value, list):
                filter_values = filter_value
            else:
                filter_values = [filter_value]
            
            for fv in filter_values:
                if self._smart_filter_match(fv, product_filter_value, filter_name):
                    any_filter_matched = True
                    break
        
        return any_filter_matched if extracted_filters else True
    
    def _smart_filter_match(self, filter_value: str, product_value: str, filter_type: str) -> bool:
        """Smart matching for filter values with semantic understanding."""
        
        filter_lower = filter_value.lower().strip()
        product_lower = product_value.lower().strip()
        
        # Exact match
        if filter_lower == product_lower:
            return True
        
        # Partial match for broader categories
        if filter_lower in product_lower or product_lower in filter_lower:
            return True
        
        # Style-specific semantic matching
        if filter_type in ['style', 'occasion']:
            equivalences = {
                'casual': ['everyday', 'relaxed', 'informal', 'comfortable'],
                'formal': ['business', 'professional', 'dress', 'elegant'],
                'athletic': ['sport', 'workout', 'fitness', 'active', 'gym'],
                'elegant': ['sophisticated', 'classy', 'refined', 'chic'],
                'vintage': ['retro', 'classic', 'antique', 'old-fashioned'],
                'bohemian': ['boho', 'hippie', 'free-spirited'],
                'minimalist': ['simple', 'clean', 'basic'],
                'trendy': ['fashionable', 'stylish', 'contemporary', 'modern']
            }
            
            for key, values in equivalences.items():
                if (filter_lower == key and product_lower in values) or \
                   (product_lower == key and filter_lower in values):
                    return True
        
        # Color-specific semantic matching
        elif filter_type == 'color':
            color_equivalences = {
                'black': ['noir', 'ebony', 'jet'],
                'white': ['ivory', 'cream', 'off-white'],
                'blue': ['navy', 'royal', 'denim'],
                'red': ['crimson', 'scarlet', 'burgundy'],
                'green': ['olive', 'forest', 'emerald'],
                'brown': ['tan', 'beige', 'chocolate'],
                'gray': ['grey', 'charcoal', 'silver'],
                'pink': ['rose', 'blush', 'coral']
            }
            
            for key, values in color_equivalences.items():
                if (filter_lower == key and product_lower in values) or \
                   (product_lower == key and filter_lower in values):
                    return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter operation statistics."""
        
        total_ops = self.filter_operations
        if total_ops == 0:
            return {
                'total_filter_operations': 0,
                'strict_success_rate': 0.0,
                'lenient_success_rate': 0.0,
                'semantic_only_rate': 0.0
            }
        
        return {
            'total_filter_operations': total_ops,
            'strict_success_rate': self.strict_passes / total_ops,
            'lenient_success_rate': self.lenient_passes / total_ops,
            'semantic_only_rate': self.semantic_only_passes / total_ops
        }