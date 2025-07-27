import React, { useState } from 'react';
import { Search, Filter, Sparkles } from 'lucide-react';
import { SAMPLE_QUERIES } from '../services/api';
import './SearchBar.css';

const SearchBar = ({ onSearch, loading, disabled }) => {
  const [query, setQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    top_k: 12,
    min_similarity: 0.0,
    price_min: '',
    price_max: '',
    category: '',
    min_rating: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !disabled && !loading) {
      const searchFilters = {};
      
      // Only include non-empty filters
      Object.keys(filters).forEach(key => {
        const value = filters[key];
        if (value !== '' && value !== null && value !== undefined) {
          searchFilters[key] = typeof value === 'string' ? 
            (isNaN(value) ? value : Number(value)) : value;
        }
      });
      
      onSearch(query, searchFilters);
    }
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSampleQuery = (sampleQuery) => {
    setQuery(sampleQuery);
    onSearch(sampleQuery, {});
  };

  const clearFilters = () => {
    setFilters({
      top_k: 12,
      min_similarity: 0.0,
      price_min: '',
      price_max: '',
      category: '',
      min_rating: ''
    });
  };

  return (
    <div className="search-bar">
      <div className="search-main">
        <form onSubmit={handleSubmit} className="search-form">
          <div className="search-input-group">
            <Search className="search-icon" size={20} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe what you're looking for... e.g., 'comfortable summer dresses under $50'"
              className="search-input"
              disabled={disabled}
            />
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className={`filter-toggle ${showFilters ? 'active' : ''}`}
              disabled={disabled}
            >
              <Filter size={18} />
            </button>
            <button
              type="submit"
              className="search-button"
              disabled={disabled || loading || !query.trim()}
            >
              {loading ? (
                <div className="loading-spinner" />
              ) : (
                <Search size={18} />
              )}
            </button>
          </div>
        </form>

        {showFilters && (
          <div className="filters-panel fade-in">
            <div className="filters-grid">
              <div className="filter-group">
                <label className="filter-label">Results Count</label>
                <input
                  type="number"
                  value={filters.top_k}
                  onChange={(e) => handleFilterChange('top_k', e.target.value)}
                  min="1"
                  max="50"
                  className="filter-input"
                />
              </div>
              
              <div className="filter-group">
                <label className="filter-label">Min Price ($)</label>
                <input
                  type="number"
                  value={filters.price_min}
                  onChange={(e) => handleFilterChange('price_min', e.target.value)}
                  min="0"
                  step="0.01"
                  className="filter-input"
                  placeholder="0.00"
                />
              </div>
              
              <div className="filter-group">
                <label className="filter-label">Max Price ($)</label>
                <input
                  type="number"
                  value={filters.price_max}
                  onChange={(e) => handleFilterChange('price_max', e.target.value)}
                  min="0"
                  step="0.01"
                  className="filter-input"
                  placeholder="999.99"
                />
              </div>
              
              <div className="filter-group">
                <label className="filter-label">Min Rating</label>
                <select
                  value={filters.min_rating}
                  onChange={(e) => handleFilterChange('min_rating', e.target.value)}
                  className="filter-input"
                >
                  <option value="">Any Rating</option>
                  <option value="1">1+ Stars</option>
                  <option value="2">2+ Stars</option>
                  <option value="3">3+ Stars</option>
                  <option value="4">4+ Stars</option>
                  <option value="4.5">4.5+ Stars</option>
                </select>
              </div>
            </div>
            
            <div className="filters-actions">
              <button
                type="button"
                onClick={clearFilters}
                className="btn btn-secondary btn-sm"
              >
                Clear Filters
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="sample-queries">
        <div className="sample-queries-header">
          <Sparkles size={16} />
          <span>Try these examples:</span>
        </div>
        <div className="sample-queries-list">
          {SAMPLE_QUERIES.slice(0, 5).map((sampleQuery, index) => (
            <button
              key={index}
              onClick={() => handleSampleQuery(sampleQuery)}
              className="sample-query"
              disabled={disabled || loading}
            >
              {sampleQuery}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SearchBar;