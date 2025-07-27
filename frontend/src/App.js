import React, { useState, useEffect } from 'react';
import SearchBar from './components/SearchBar';
import ProductResults from './components/ProductResults';
import Header from './components/Header';
import StatusIndicator from './components/StatusIndicator';
import { searchProducts, checkHealth } from './services/api';
import './App.css';

function App() {
  const [searchResults, setSearchResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Check API health on component mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const health = await checkHealth();
        setApiHealth(health);
      } catch (err) {
        console.error('API health check failed:', err);
        setApiHealth({ status: 'error', message: 'API not available' });
      }
    };

    checkApiHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkApiHealth, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const handleSearch = async (query, filters = {}) => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setSearchQuery(query);

    try {
      const results = await searchProducts(query, filters);
      setSearchResults(results);
    } catch (err) {
      console.error('Search error:', err);
      setError(err.message || 'Search failed. Please try again.');
      setSearchResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClearResults = () => {
    setSearchResults(null);
    setSearchQuery('');
    setError(null);
  };

  return (
    <div className="app">
      <Header />
      
      <main className="app-main">
        <div className="container">
          <StatusIndicator health={apiHealth} />
          
          <div className="search-section">
            <SearchBar 
              onSearch={handleSearch}
              loading={loading}
              disabled={apiHealth?.status !== 'healthy'}
            />
            
            {error && (
              <div className="error-message fade-in">
                <p>{error}</p>
                <button 
                  onClick={handleClearResults}
                  className="btn btn-secondary btn-sm"
                >
                  Clear
                </button>
              </div>
            )}
          </div>

          <ProductResults 
            results={searchResults}
            loading={loading}
            query={searchQuery}
            onClear={handleClearResults}
          />
        </div>
      </main>
    </div>
  );
}

export default App;