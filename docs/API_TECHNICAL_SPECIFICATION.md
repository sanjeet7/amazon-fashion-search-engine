# Amazon Fashion Search Engine - Technical API Specification

**Version:** 1.0  
**Last Updated:** January 2025  
**Base URL:** `http://127.0.0.1:8000` (Development) | `https://your-domain.com` (Production)

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Example Responses](#example-responses)
8. [Frontend Integration Guide](#frontend-integration-guide)
9. [Deployment Instructions](#deployment-instructions)

---

## Overview

The Amazon Fashion Search Engine provides semantic search capabilities for fashion products using OpenAI embeddings and FAISS vector similarity search. The API supports natural language queries and returns relevant fashion products with intelligent ranking.

### Key Features
- **Semantic Search**: Natural language product search using AI embeddings
- **Intelligent Ranking**: Products ranked by similarity, ratings, and business signals
- **Rich Metadata**: Comprehensive product information including images, filters, and categories
- **Query Enhancement**: AI-powered query expansion for better search results
- **Advanced Filtering**: Price, rating, category, and attribute-based filters

### Technology Stack
- **Backend**: FastAPI (Python)
- **Search Engine**: FAISS with OpenAI embeddings
- **Data Processing**: Pandas, NumPy
- **AI Integration**: OpenAI GPT-4 and embeddings API

---

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Search API    │    │ Data Pipeline   │
│   (React/Next)  │◄──►│   (FastAPI)     │◄──►│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                        │
                               ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ FAISS Vector   │    │ Product Data    │
                       │ Search Index    │    │ (500 Products)  │
                       └─────────────────┘    └─────────────────┘
```

### Data Flow
1. User enters search query in frontend
2. Frontend sends request to Search API
3. API enhances query using OpenAI
4. Query converted to embeddings
5. FAISS performs vector similarity search
6. Results ranked and filtered
7. Structured response returned to frontend

---

## Authentication

**Current Implementation**: No authentication required (development)  
**Production Recommendation**: Implement API key authentication

```typescript
// Future authentication header
headers: {
  'Authorization': 'Bearer YOUR_API_KEY',
  'Content-Type': 'application/json'
}
```

---

## API Endpoints

### 1. Health Check
**Endpoint**: `GET /health`  
**Purpose**: Check API status and system health

**Response Schema**:
```typescript
interface HealthResponse {
  status: string;                    // "healthy" | "error" | "initializing"
  version: string;                   // "1.0.0"
  uptime_seconds: number;           // Server uptime
  total_products: number;           // Number of indexed products
  index_ready: boolean;             // Whether search index is ready
  embeddings_loaded: boolean;       // Whether embeddings are loaded
  avg_search_time_ms?: number;      // Average search response time
  total_searches: number;           // Total searches performed
}
```

**Example Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "total_products": 500,
  "index_ready": true,
  "embeddings_loaded": true,
  "avg_search_time_ms": 250.3,
  "total_searches": 42
}
```

### 2. Product Search
**Endpoint**: `POST /search`  
**Purpose**: Semantic search for fashion products

**Request Schema**:
```typescript
interface SearchRequest {
  query: string;                    // Search query (1-500 chars)
  top_k?: number;                   // Number of results (1-50, default: 10)
  min_similarity?: number;          // Minimum similarity (0-1, default: 0.0)
  price_min?: number;              // Minimum price filter
  price_max?: number;              // Maximum price filter
  category?: string;               // Category filter
  min_rating?: number;             // Minimum rating filter (0-5)
}
```

**Response Schema**:
```typescript
interface SearchResponse {
  query: string;                   // Original search query
  results: SearchResult[];         // Array of search results
  total_results: number;          // Total number of results
  search_time_ms: number;         // Search execution time
  enhanced_query?: string;        // AI-enhanced query
  detected_intent?: string;       // Detected search intent
  filters_applied?: FilterData;   // Applied filters summary
}

interface SearchResult {
  product: ProductResult;         // Product information
  rank: number;                  // Result ranking (1-based)
}
```

**Example Request**:
```json
{
  "query": "comfortable summer dresses under $50",
  "top_k": 5,
  "min_similarity": 0.1,
  "price_max": 50.0,
  "min_rating": 4.0
}
```

### 3. API Statistics
**Endpoint**: `GET /stats`  
**Purpose**: Get search engine performance statistics

**Response Schema**:
```typescript
interface StatsResponse {
  total_searches: number;
  avg_search_time_ms: number;
  index_size: number;
  total_products: number;
  embeddings_loaded: boolean;
  index_ready: boolean;
  service_uptime_seconds: number;
  initialization_error?: string;
}
```

### 4. Root Endpoint
**Endpoint**: `GET /`  
**Purpose**: Basic API information

**Response Schema**:
```typescript
interface RootResponse {
  service: string;
  version: string;
  status: string;
  docs: string;                   // "/docs" - Swagger documentation URL
}
```

---

## Data Models

### ProductResult
```typescript
interface ProductResult {
  parent_asin: string;            // Amazon product identifier
  title: string;                  // Product title
  main_category?: string;         // Primary category
  price?: number;                 // Product price (USD)
  average_rating?: number;        // Average rating (0-5)
  rating_number?: number;         // Number of ratings
  similarity_score: number;      // Search similarity score (0-1)
  features: string[];            // Product features
  description: string[];         // Product descriptions
  store?: string;                // Store/brand name
  categories: string[];          // Product categories
  images: string[];              // Product image URLs
  matched_filters?: FilterMetadata; // Matched filter metadata
}
```

### FilterMetadata
```typescript
interface FilterMetadata {
  metadata: {
    brand?: string;
    category?: string;
    color?: string;
    material?: string;
    occasion?: string;
    price?: number;
    price_tier?: string;          // "budget" | "mid" | "premium"
    rating?: number;
    rating_tier?: string;         // "low" | "medium" | "high"
    style?: string;
  };
  // Individual filter fields
  color?: string;
  style?: string;
  occasion?: string;
  brand?: string;
  rating?: number;
  rating_tier?: string;
  material?: string;
  category?: string;
  price?: number;
  price_tier?: string;
}
```

---

## Error Handling

### HTTP Status Codes
- **200**: Success
- **422**: Validation Error (invalid request parameters)
- **500**: Internal Server Error
- **503**: Service Unavailable (search engine not ready)

### Error Response Schema
```typescript
interface ErrorResponse {
  detail: string | ValidationError[];
}

interface ValidationError {
  type: string;
  loc: string[];
  msg: string;
  input: any;
}
```

### Common Error Examples

**Validation Error (422)**:
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "query"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

**Service Unavailable (503)**:
```json
{
  "detail": "Search service not ready. Please wait for initialization to complete."
}
```

**Internal Server Error (500)**:
```json
{
  "detail": "Search failed: similarity_score validation error"
}
```

---

## Example Responses

### Successful Search Response
```json
{
  "query": "women's comfortable shoes",
  "results": [
    {
      "product": {
        "parent_asin": "B07VWJM737",
        "title": "No Show Socks Women 4-6 pairs Low Cut Cotton Casual Ankle Socks",
        "main_category": "AMAZON FASHION",
        "price": null,
        "average_rating": 3.6,
        "rating_number": 18,
        "similarity_score": 0.95,
        "features": ["Machine Wash"],
        "description": [],
        "store": "BOTINDO",
        "categories": [],
        "images": [
          "https://m.media-amazon.com/images/I/71XZIGBxN6L._AC_UL1500_.jpg",
          "https://m.media-amazon.com/images/I/61GeqTgKHDL._AC_UL1500_.jpg"
        ],
        "matched_filters": {
          "metadata": {
            "brand": "Botindo",
            "category": null,
            "color": "gray",
            "material": "cotton",
            "occasion": "casual",
            "rating": 3.6,
            "rating_tier": "medium",
            "style": "casual"
          },
          "color": "gray",
          "style": "casual",
          "occasion": "casual",
          "brand": "Botindo",
          "rating": 3.6,
          "rating_tier": "medium",
          "material": "cotton"
        }
      },
      "rank": 1
    }
  ],
  "total_results": 1,
  "search_time_ms": 245.7,
  "enhanced_query": "women's comfortable shoes footwear casual athletic sneakers flats boots",
  "detected_intent": null,
  "filters_applied": {}
}
```

---

## Frontend Integration Guide

### React/Next.js Implementation Example

```typescript
// types/api.ts
export interface SearchParams {
  query: string;
  top_k?: number;
  min_similarity?: number;
  price_min?: number;
  price_max?: number;
  category?: string;
  min_rating?: number;
}

// hooks/useSearch.ts
import { useState } from 'react';

export const useSearch = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const search = async (params: SearchParams) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  return { search, loading, results, error };
};

// components/SearchForm.tsx
export const SearchForm: React.FC = () => {
  const { search, loading, results, error } = useSearch();
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      search({ query, top_k: 10 });
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search for fashion items..."
        className="w-full p-2 border rounded"
      />
      <button 
        type="submit" 
        disabled={loading}
        className="px-4 py-2 bg-blue-500 text-white rounded"
      >
        {loading ? 'Searching...' : 'Search'}
      </button>
      
      {error && <p className="text-red-500">{error}</p>}
      
      {results && (
        <div className="mt-4">
          <p>Found {results.total_results} results in {results.search_time_ms}ms</p>
          {results.results.map((result) => (
            <ProductCard key={result.product.parent_asin} product={result.product} />
          ))}
        </div>
      )}
    </form>
  );
};
```

### API Proxy Setup (Next.js)
```typescript
// pages/api/search.ts
import type { NextApiRequest, NextApiResponse } from 'next';

const API_BASE_URL = process.env.SEARCH_API_URL || 'http://127.0.0.1:8000';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    
    if (!response.ok) {
      return res.status(response.status).json(data);
    }

    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
}
```

---

## Deployment Instructions

### Backend Deployment (Railway/Heroku/DigitalOcean)

1. **Environment Variables**:
```bash
OPENAI_API_KEY=your_openai_api_key_here
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
DEVELOPMENT_MODE=false
```

2. **Docker Deployment**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
EXPOSE 8000

CMD ["uvicorn", "services.search-api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. **Start Commands**:
```bash
# Development
python services/search-api/main.py

# Production
uvicorn services.search-api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend Deployment (Vercel)

1. **Environment Variables**:
```bash
SEARCH_API_URL=https://your-backend-domain.com
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
```

2. **Build Configuration** (`vercel.json`):
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "functions": {
    "pages/api/**/*.ts": {
      "runtime": "nodejs18.x"
    }
  }
}
```



---

#

## Support & Documentation

- **Swagger UI**: `http://127.0.0.1:8000/docs` (Interactive API documentation)
- **ReDoc**: `http://127.0.0.1:8000/redoc` (Alternative API docs)
- **Health Check**: `http://127.0.0.1:8000/health`
- **GitHub Repository**: [amazon-fashion-search-engine](https://github.com/sanjeet7/amazon-fashion-search-engine)

---

**Note**: This API is currently configured for development with 500 test products. For production deployment, ensure proper data pipeline setup and consider scaling considerations for larger datasets. 