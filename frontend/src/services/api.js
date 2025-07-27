import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || 
                     error.response.data?.message || 
                     `Server error: ${error.response.status}`;
      throw new Error(message);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Cannot connect to the server. Please check if the API is running.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

/**
 * Check API health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

/**
 * Search for products using semantic search
 */
export const searchProducts = async (query, filters = {}) => {
  try {
    const searchRequest = {
      query: query.trim(),
      top_k: filters.top_k || 12,
      min_similarity: filters.min_similarity || 0.0,
      ...filters
    };

    console.log('Search request:', searchRequest);
    
    const response = await api.post('/search', searchRequest);
    return response.data;
  } catch (error) {
    console.error('Search failed:', error);
    throw error;
  }
};

/**
 * Get search engine statistics
 */
export const getStats = async () => {
  try {
    const response = await api.get('/stats');
    return response.data;
  } catch (error) {
    console.error('Failed to get stats:', error);
    throw error;
  }
};

/**
 * Sample search queries for demo purposes
 */
export const SAMPLE_QUERIES = [
  "comfortable summer dresses under $50",
  "elegant wedding guest outfit",
  "professional work attire",
  "casual weekend wear",
  "running shoes for women",
  "warm winter jacket",
  "formal evening dress",
  "minimalist jewelry",
  "vintage style accessories",
  "sustainable fashion brands"
];

export default api;