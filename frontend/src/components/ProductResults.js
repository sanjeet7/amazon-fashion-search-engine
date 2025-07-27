import React from 'react';
import { Star, ShoppingBag, X } from 'lucide-react';
import './ProductResults.css';

const ProductResults = ({ results, loading, query, onClear }) => {
  if (loading) {
    return (
      <div className="product-results">
        <div className="results-header">
          <h2>Searching...</h2>
          <div className="loading-spinner" />
        </div>
        <div className="loading-grid">
          {Array(12).fill(0).map((_, index) => (
            <div key={index} className="product-card loading-card loading-pulse">
              <div className="loading-image"></div>
              <div className="loading-content">
                <div className="loading-line"></div>
                <div className="loading-line short"></div>
                <div className="loading-line"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!results) {
    return null;
  }

  const formatPrice = (price) => {
    if (!price || price === 0) return 'Price not available';
    return `$${price.toFixed(2)}`;
  };

  const formatRating = (rating, count) => {
    if (!rating) return null;
    return (
      <div className="rating">
        <div className="stars">
          {[...Array(5)].map((_, i) => (
            <Star
              key={i}
              size={14}
              className={i < Math.floor(rating) ? 'star-filled' : 'star-empty'}
              fill={i < Math.floor(rating) ? 'currentColor' : 'none'}
            />
          ))}
        </div>
        <span className="rating-text">
          {rating.toFixed(1)} {count && `(${count.toLocaleString()})`}
        </span>
      </div>
    );
  };

  const getImageUrl = (images) => {
    if (!images || images.length === 0) {
      return '/api/placeholder/300/300';
    }
    return images[0];
  };

  const truncateText = (text, maxLength = 100) => {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="product-results fade-in">
      <div className="results-header">
        <div className="results-info">
          <h2>Search Results</h2>
          <p className="results-meta">
            {results.total_results} products found for "{query}" 
            <span className="search-time">in {Math.round(results.search_time_ms)}ms</span>
          </p>
          {results.enhanced_query && results.enhanced_query !== query && (
            <p className="enhanced-query">
              Enhanced query: "{results.enhanced_query}"
            </p>
          )}
        </div>
        <button onClick={onClear} className="btn btn-secondary">
          <X size={16} />
          Clear Results
        </button>
      </div>

      {results.results.length === 0 ? (
        <div className="no-results">
          <ShoppingBag size={48} className="no-results-icon" />
          <h3>No products found</h3>
          <p>Try adjusting your search terms or filters</p>
        </div>
      ) : (
        <div className="products-grid">
          {results.results.map((result, index) => {
            const product = result.product;
            return (
              <div key={`${product.parent_asin}-${index}`} className="product-card">
                <div className="product-image-container">
                  <img
                    src={getImageUrl(product.images)}
                    alt={product.title}
                    className="product-image"
                    onError={(e) => {
                      e.target.src = '/api/placeholder/300/300';
                    }}
                  />
                  <div className="similarity-badge">
                    {Math.round(product.similarity_score * 100)}% match
                  </div>
                  <div className="rank-badge">#{result.rank}</div>
                </div>

                <div className="product-content">
                  <div className="product-header">
                    <h3 className="product-title" title={product.title}>
                      {truncateText(product.title, 80)}
                    </h3>
                    {product.main_category && (
                      <span className="product-category">{product.main_category}</span>
                    )}
                  </div>

                  <div className="product-pricing">
                    <span className="product-price">{formatPrice(product.price)}</span>
                    {product.average_rating && formatRating(product.average_rating, product.rating_number)}
                  </div>

                  {product.features && product.features.length > 0 && (
                    <div className="product-features">
                      <ul>
                        {product.features.slice(0, 3).map((feature, idx) => (
                          <li key={idx}>{truncateText(feature, 60)}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="product-footer">
                    {product.store && (
                      <span className="product-store">by {product.store}</span>
                    )}
                    <div className="product-categories">
                      {product.categories.slice(0, 2).map((category, idx) => (
                        <span key={idx} className="category-tag">
                          {category}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ProductResults;