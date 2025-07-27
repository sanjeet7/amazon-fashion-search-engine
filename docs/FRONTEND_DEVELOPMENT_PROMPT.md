# Frontend Development Prompt: Amazon Fashion Search Engine

## Project Overview

Build a modern, responsive fashion e-commerce search frontend for the Amazon Fashion Search Engine. This is a semantic search application that uses AI-powered search to find fashion products based on natural language queries.

## Core Requirements

### Primary Functionality
1. **Semantic Search Interface**: Users can search using natural language (e.g., "comfortable summer dresses under $50", "casual work outfit", "winter boots for hiking")
2. **Product Discovery**: Display search results with rich product information including images, prices, ratings, and descriptions
3. **Advanced Filtering**: Allow users to filter by price range, ratings, categories, colors, materials, and occasions
4. **Responsive Design**: Fully responsive for desktop, tablet, and mobile devices

### Tech Stack Requirements
- **Framework**: Next.js 14+ with TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Shadcn/ui or similar modern component library
- **State Management**: React hooks (useState, useEffect) and React Query for API state
- **Image Optimization**: Next.js Image component with lazy loading
- **Deployment**: Vercel

## API Integration Details

### Base API Endpoints
```typescript
// Backend API (to be deployed separately)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

// Main search endpoint
POST /search
GET /health
GET /stats
GET /
```

### Complete TypeScript Interface Definitions
```typescript
// types/api.ts
export interface SearchRequest {
  query: string;                    // Search query (1-500 chars)
  top_k?: number;                   // Number of results (1-50, default: 10)
  min_similarity?: number;          // Minimum similarity (0-1, default: 0.0)
  price_min?: number;              // Minimum price filter
  price_max?: number;              // Maximum price filter
  category?: string;               // Category filter
  min_rating?: number;             // Minimum rating filter (0-5)
}

export interface SearchResponse {
  query: string;                   // Original search query
  results: SearchResult[];         // Array of search results
  total_results: number;          // Total number of results
  search_time_ms: number;         // Search execution time
  enhanced_query?: string;        // AI-enhanced query
  detected_intent?: string;       // Detected search intent
  filters_applied?: FilterData;   // Applied filters summary
}

export interface SearchResult {
  product: ProductResult;         // Product information
  rank: number;                  // Result ranking (1-based)
}

export interface ProductResult {
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

export interface FilterMetadata {
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

export interface HealthResponse {
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

## Required Components & Features

### 1. Layout & Navigation
```typescript
// components/Layout.tsx
- Header with search bar
- Navigation menu
- Footer with links
- Responsive hamburger menu for mobile
```

### 2. Search Interface
```typescript
// components/SearchForm.tsx
- Large, prominent search input
- Search suggestions/autocomplete (bonus)
- Voice search capability (bonus)
- Recent searches dropdown (bonus)
- Search button with loading state
- Clear search functionality
```

### 3. Product Display
```typescript
// components/ProductGrid.tsx
- Grid layout (responsive: 1 col mobile, 2-3 tablet, 4+ desktop)
- Product cards with hover effects
- Lazy loading for images
- Masonry layout option (bonus)

// components/ProductCard.tsx
- Product image with fallback
- Product title (truncated if long)
- Price display (handle null prices)
- Star ratings display
- Brand/store name
- Quick view button (bonus)
- Add to favorites heart icon (bonus)
```

### 4. Filtering System
```typescript
// components/FilterSidebar.tsx
- Collapsible filter sections
- Price range slider
- Rating filter (star selection)
- Category checkboxes
- Color palette selector
- Material filter
- Occasion filter
- Clear all filters button
- Filter count badges
```

### 5. Search Results
```typescript
// components/SearchResults.tsx
- Results count and search time
- Sort options (relevance, price low/high, rating, newest)
- View toggle (grid/list)
- Load more / pagination
- Empty state when no results
- Loading skeletons
```

### 6. Product Detail (Bonus)
```typescript
// components/ProductModal.tsx or pages/product/[asin].tsx
- Large image gallery
- Product details
- Feature list
- Customer ratings
- Related products
```

## API Integration Implementation

### React Query Setup
```typescript
// hooks/useSearch.ts
import { useQuery, useMutation } from '@tanstack/react-query';

export const useSearch = () => {
  const [searchParams, setSearchParams] = useState<SearchRequest>({
    query: '',
    top_k: 12
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ['search', searchParams],
    queryFn: () => searchProducts(searchParams),
    enabled: !!searchParams.query.trim(),
    staleTime: 5 * 60 * 1000, // 5 minutes cache
  });

  return {
    searchResults: data,
    isLoading,
    error,
    searchParams,
    setSearchParams
  };
};

// api/search.ts
export const searchProducts = async (params: SearchRequest): Promise<SearchResponse> => {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }

  return response.json();
};
```

### API Route Proxy
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
    console.error('Search API Error:', error);
    res.status(500).json({ error: 'Search service unavailable' });
  }
}

// Similar proxy routes for /api/health and /api/stats
```

## Design Requirements

### Visual Design
- **Modern, clean aesthetic** inspired by top e-commerce sites (Amazon, Nordstrom, ASOS)
- **Fashion-forward color palette**: Use sophisticated colors (neutrals with accent colors)
- **High-quality typography**: Modern sans-serif fonts, clear hierarchy
- **Generous white space**: Avoid cluttered layouts
- **Professional photography treatment**: Showcase product images prominently

### UI/UX Guidelines
- **Mobile-first responsive design**
- **Accessibility compliance** (ARIA labels, keyboard navigation, color contrast)
- **Fast loading times** with skeleton states
- **Intuitive navigation** with breadcrumbs
- **Progressive enhancement** for advanced features
- **Error handling** with user-friendly messages

### Specific UI Components Needed
```typescript
// Use Shadcn/ui or build custom components:
- Button variants (primary, secondary, ghost)
- Input with search icon
- Card components for products
- Badge components for tags/filters
- Slider component for price range
- Star rating component
- Loading skeleton components
- Modal/dialog components
- Dropdown menus
- Checkbox and radio components
```

## Example Search Queries to Test
- "comfortable summer dresses under $50"
- "casual work outfits for women"
- "athletic shoes for running"
- "vintage style accessories"
- "winter coats waterproof"
- "yoga pants comfortable"
- "formal dress shoes men"
- "bohemian style jewelry"

## Performance & Technical Requirements

### Performance Optimizations
- **Image optimization**: Use Next.js Image component with lazy loading
- **Bundle optimization**: Code splitting and tree shaking
- **Caching**: React Query for API caching, CDN for images
- **Debounced search**: Prevent excessive API calls while typing
- **Virtual scrolling**: For large result sets (bonus)

### SEO & Analytics
- **Next.js SEO**: Meta tags, structured data for products
- **URL structure**: `/search?q=query&filters=...` for shareable searches
- **Analytics integration**: Track search queries, clicks, popular products
- **Sitemap generation**: For discovered products

### Error Handling
- **Network errors**: Graceful degradation when API is down
- **Empty states**: Helpful messages when no results found
- **Loading states**: Skeleton screens and spinners
- **Retry mechanisms**: For failed requests
- **Fallback images**: When product images fail to load

## Development Workflow

### Project Structure
```
src/
├── components/
│   ├── ui/              # Basic UI components (shadcn/ui)
│   ├── layout/          # Layout components
│   ├── search/          # Search-related components
│   ├── product/         # Product-related components
│   └── filters/         # Filter components
├── hooks/               # Custom React hooks
├── lib/                 # Utility functions
├── types/               # TypeScript type definitions
├── pages/               # Next.js pages
│   ├── api/            # API routes (proxies to backend)
│   ├── search/         # Search page
│   └── index.tsx       # Homepage
└── styles/             # Global styles and Tailwind config
```

### Environment Variables
```bash
# .env.local
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
SEARCH_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_APP_NAME="Amazon Fashion Search"
```

### Package.json Dependencies
```json
{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "@tanstack/react-query": "^5.0.0",
    "tailwindcss": "^3.3.0",
    "typescript": "^5.0.0",
    "@radix-ui/react-slider": "^1.1.0",
    "@radix-ui/react-dialog": "^1.0.0",
    "lucide-react": "^0.300.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  }
}
```

## Testing Requirements

### Testing Strategy
- **Unit tests**: For utility functions and hooks
- **Component tests**: Using React Testing Library
- **E2E tests**: Basic search flow with Playwright
- **API integration tests**: Mock API responses
- **Accessibility tests**: Automated a11y testing

### Key Test Scenarios
- Search with various queries
- Filter application and clearing
- Responsive behavior on different screen sizes
- Loading and error states
- Image lazy loading
- Keyboard navigation

## Deployment Configuration

### Vercel Configuration
```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "functions": {
    "pages/api/**/*.ts": {
      "runtime": "nodejs18.x"
    }
  },
  "env": {
    "SEARCH_API_URL": "@search-api-url"
  }
}
```

### Performance Targets
- **Lighthouse Score**: 90+ in all categories
- **First Contentful Paint**: < 1.5s
- **Largest Contentful Paint**: < 2.5s
- **Time to Interactive**: < 3.5s
- **Bundle Size**: < 300KB initial load

## Success Criteria

### Functional Requirements
✅ Users can search for fashion items using natural language  
✅ Search results display relevant products with images and details  
✅ Users can filter results by price, rating, category, and attributes  
✅ Application is fully responsive across devices  
✅ Search is fast and responsive (< 2 second response time)  
✅ Error states are handled gracefully  
✅ Images load efficiently with lazy loading  

### Quality Requirements
✅ Code is well-structured and maintainable  
✅ Components are reusable and properly typed  
✅ Accessibility standards are met  
✅ Performance metrics meet targets  
✅ SEO best practices are implemented  

## Additional Notes

- **Backend API**: The backend is already built and deployed. Frontend only needs to integrate with the provided API endpoints.
- **Product Data**: The system contains 500 real Amazon fashion products with actual images, prices, and ratings.
- **Search Intelligence**: The search uses AI to understand natural language queries and provides enhanced search results.
- **Scalability**: Design the frontend to easily handle larger datasets in the future.

## Example API Response
```json
{
  "query": "comfortable summer dresses",
  "results": [
    {
      "product": {
        "parent_asin": "B07VWJM737",
        "title": "Women's Casual Summer Floral Print Dress",
        "main_category": "AMAZON FASHION",
        "price": 29.99,
        "average_rating": 4.2,
        "rating_number": 156,
        "similarity_score": 0.89,
        "features": ["Machine Wash", "Breathable Fabric"],
        "description": ["Comfortable summer dress with floral print"],
        "store": "Summer Fashion Co",
        "categories": ["Dresses", "Summer Wear"],
        "images": [
          "https://m.media-amazon.com/images/I/61XZIGBxN6L._AC_UL1500_.jpg"
        ],
        "matched_filters": {
          "color": "floral",
          "style": "casual",
          "occasion": "summer",
          "material": "cotton"
        }
      },
      "rank": 1
    }
  ],
  "total_results": 15,
  "search_time_ms": 234.5,
  "enhanced_query": "comfortable summer dresses casual floral breathable cotton"
}
```

This comprehensive specification provides everything needed to build a production-ready fashion search frontend that integrates seamlessly with the existing Amazon Fashion Search Engine API. 