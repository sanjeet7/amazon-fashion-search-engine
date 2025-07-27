# Amazon Fashion Search Engine - Technical API Specification

**Project**: AI-Powered Fashion Product Search Engine  
**Backend**: FastAPI with OpenAI embeddings and FAISS vector search  
**Dataset**: 500 real Amazon fashion products with complete metadata  
**Search Capability**: Semantic search using natural language queries  

## API Base Information

**Development URL**: `http://127.0.0.1:8000`  
**Production URL**: `https://your-backend-domain.com` (to be deployed)  
**API Framework**: FastAPI with automatic OpenAPI documentation  
**Documentation**: Available at `/docs` (Swagger UI) and `/redoc`  

## Complete API Endpoints

### 1. Product Search
```
POST /search
Content-Type: application/json
```

**Request Schema**:
```typescript
interface SearchRequest {
  query: string;                    // Required: Search query (1-500 characters)
  top_k?: number;                   // Optional: Number of results (1-50, default: 10)
  min_similarity?: number;          // Optional: Minimum similarity threshold (0-1, default: 0.0)
  price_min?: number;              // Optional: Minimum price filter (USD)
  price_max?: number;              // Optional: Maximum price filter (USD)
  category?: string;               // Optional: Category filter
  min_rating?: number;             // Optional: Minimum rating filter (0-5)
}
```

**Response Schema**:
```typescript
interface SearchResponse {
  query: string;                   // Original search query
  results: SearchResult[];         // Array of search results
  total_results: number;          // Total number of results returned
  search_time_ms: number;         // Search execution time in milliseconds
  enhanced_query?: string;        // AI-enhanced query with synonyms
  detected_intent?: string;       // Detected search intent (currently null)
  filters_applied?: object;       // Summary of filters applied (currently empty object)
}

interface SearchResult {
  product: ProductResult;         // Complete product information
  rank: number;                  // Result ranking (1-based)
}
```

### 2. Health Check
```
GET /health
```

**Response Schema**:
```typescript
interface HealthResponse {
  status: string;                    // "healthy" | "error" | "initializing"
  version: string;                   // API version (e.g., "1.0.0")
  uptime_seconds: number;           // Server uptime in seconds
  total_products: number;           // Number of indexed products
  index_ready: boolean;             // Whether search index is ready
  embeddings_loaded: boolean;       // Whether embeddings are loaded
  avg_search_time_ms?: number;      // Average search response time
  total_searches: number;           // Total searches performed since startup
}
```

### 3. API Statistics
```
GET /stats
```

**Response Schema**:
```typescript
interface StatsResponse {
  total_searches: number;           // Total searches performed
  avg_search_time_ms: number;      // Average search time
  index_size: number;              // Number of vectors in search index
  total_products: number;          // Total products available
  embeddings_loaded: boolean;      // Embeddings loading status
  index_ready: boolean;            // Search index status
  service_uptime_seconds: number;  // Service uptime
  initialization_error?: string;   // Any initialization errors
}
```

### 4. Root Information
```
GET /
```

**Response Schema**:
```typescript
interface RootResponse {
  service: string;                 // "Amazon Fashion Search API"
  version: string;                 // API version
  status: string;                  // "running"
  docs: string;                    // "/docs" (Swagger documentation path)
}
```

## Core Data Models

### ProductResult
```typescript
interface ProductResult {
  parent_asin: string;            // Amazon Standard Identification Number
  title: string;                  // Product title/name
  main_category?: string;         // Primary category (e.g., "AMAZON FASHION")
  price?: number;                 // Product price in USD (null if not available)
  average_rating?: number;        // Average customer rating (0-5, null if not available)
  rating_number?: number;         // Number of customer ratings (null if not available)
  similarity_score: number;      // Search similarity score (0-1, clamped for validation)
  features: string[];            // Array of product features (e.g., ["Machine Wash"])
  description: string[];         // Array of product descriptions
  store?: string;                // Store/brand name (null if not available)
  categories: string[];          // Array of product categories
  images: string[];              // Array of product image URLs (Amazon CDN URLs)
  matched_filters?: FilterMetadata; // Rich metadata about matched search filters
}
```

### FilterMetadata
```typescript
interface FilterMetadata {
  metadata: {
    brand?: string;               // Normalized brand name
    category?: string;            // Standardized category
    color?: string;               // Detected color
    material?: string;            // Detected material (e.g., "cotton")
    occasion?: string;            // Usage occasion (e.g., "casual", "formal")
    price?: number;               // Product price
    price_tier?: string;          // "budget" | "mid" | "premium"
    rating?: number;              // Product rating
    rating_tier?: string;         // "low" | "medium" | "high"
    style?: string;               // Style classification (e.g., "casual", "athletic")
  };
  // Individual filter fields (flattened for easy access)
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

## API Response Examples

### Successful Search Response
```json
{
  "query": "comfortable women socks",
  "results": [
    {
      "product": {
        "parent_asin": "B07VWJM737",
        "title": "No Show Socks Women 4-6 pairs Low Cut Cotton Casual Ankle Socks with Non Slip Flat Boat Line 4 Pairs grey sizes 9-12",
        "main_category": "AMAZON FASHION",
        "price": null,
        "average_rating": 3.6,
        "rating_number": 18,
        "similarity_score": 1.0,
        "features": ["Machine Wash"],
        "description": [],
        "store": "BOTINDO",
        "categories": [],
        "images": [
          "https://m.media-amazon.com/images/I/71XZIGBxN6L._AC_UL1500_.jpg",
          "https://m.media-amazon.com/images/I/61GeqTgKHDL._AC_UL1500_.jpg",
          "https://m.media-amazon.com/images/I/61LaLe8aqJL._AC_UL1500_.jpg"
        ],
        "matched_filters": {
          "metadata": {
            "brand": "Botindo",
            "category": null,
            "color": "gray",
            "material": "cotton",
            "occasion": "casual",
            "price": null,
            "price_tier": null,
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
  "search_time_ms": 1938.002347946167,
  "enhanced_query": "socks hosiery foot coverings ankle crew knee-high cotton wool blend casual athletic dress formal comfortable breathable moisture-wicking",
  "detected_intent": null,
  "filters_applied": {}
}
```

### Health Check Response
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

## Error Handling

### HTTP Status Codes
- `200`: Success
- `422`: Validation Error (invalid request parameters)
- `500`: Internal Server Error
- `503`: Service Unavailable (search engine not ready)

### Error Response Schema
```typescript
interface ErrorResponse {
  detail: string | ValidationError[];
}

interface ValidationError {
  type: string;                    // Error type (e.g., "string_too_short")
  loc: string[];                   // Error location path (e.g., ["body", "query"])
  msg: string;                     // Human-readable error message
  input: any;                      // Invalid input value
}
```

### Example Error Responses

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

## Backend Technical Context

### Search Engine Capabilities
- **Semantic Search**: Uses OpenAI text-embedding-3-small model for query understanding
- **Query Enhancement**: AI automatically expands queries with related terms and synonyms
- **Vector Search**: FAISS (Facebook AI Similarity Search) for efficient similarity matching
- **Intelligent Ranking**: Combines similarity scores with business signals (ratings, completeness)
- **Real-time Processing**: Average search time ~250ms for 500 products

### Data Processing Pipeline
- **500 Real Products**: Actual Amazon fashion items with real ASINs, prices, ratings
- **Rich Metadata**: Extracted colors, materials, styles, occasions, categories
- **Image URLs**: Direct links to Amazon CDN product images
- **Quality Filtering**: Products filtered for completeness and relevance

### Search Query Examples
The API handles natural language queries such as:
- "comfortable summer dresses under $50"
- "casual work outfits for women"
- "athletic shoes for running"
- "vintage style accessories"
- "winter coats waterproof"
- "yoga pants comfortable"
- "formal dress shoes men"

### Filter Capabilities
Products can be filtered by:
- **Price Range**: Min/max price filtering
- **Ratings**: Minimum rating threshold (0-5 stars)
- **Categories**: Product categories and subcategories
- **Attributes**: Colors, materials, styles, occasions
- **Similarity**: Minimum similarity score threshold

## Authentication & Security

**Current**: No authentication required (development mode)  
**Production Recommendation**: Implement API key-based authentication

```typescript
// Future authentication header format
headers: {
  'Authorization': 'Bearer YOUR_API_KEY',
  'Content-Type': 'application/json'
}
```

## Performance Characteristics

- **Index Size**: 500 products with 1536-dimensional embeddings
- **Search Speed**: ~250ms average response time
- **Concurrency**: Supports multiple concurrent searches
- **Memory Usage**: ~3MB for embeddings + FAISS index
- **Startup Time**: ~15 seconds for full initialization

## Deployment Configuration

### Environment Variables Required
```bash
OPENAI_API_KEY=sk-...                    # Required: OpenAI API key
API_HOST=0.0.0.0                         # Host binding
API_PORT=8000                            # Port number
LOG_LEVEL=INFO                           # Logging level
DEVELOPMENT_MODE=false                   # Production mode
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "services.search-api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Data Files Required
- `data/processed/processed_products.parquet` (240KB)
- `data/embeddings/embeddings.npy` (3MB)  
- `data/embeddings/faiss_index.index` (3MB)
- `data/embeddings/metadata.json`
- `data/embeddings/product_ids.json`

## API Testing

### Test Endpoints
```bash
# Health check
curl http://127.0.0.1:8000/health

# Basic search
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "blue dress", "top_k": 5}'

# Advanced search with filters
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "comfortable shoes", 
    "top_k": 10,
    "price_max": 100,
    "min_rating": 4.0
  }'
```

### Interactive Documentation
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

## Production Considerations

### Scaling
- Current implementation handles 500 products efficiently
- Can scale to 10K+ products with minimal changes
- FAISS index supports millions of vectors for future growth

### Monitoring
- Health endpoint provides system status
- Stats endpoint tracks usage metrics
- Built-in logging for debugging and monitoring

### Data Pipeline
- Automated data processing pipeline available
- Can regenerate embeddings and index as needed
- Supports different data sources and sample sizes

---

**Repository**: https://github.com/sanjeet7/amazon-fashion-search-engine  
**Backend Status**: Fully functional and tested  
**Frontend Required**: React/Next.js application to consume this API 