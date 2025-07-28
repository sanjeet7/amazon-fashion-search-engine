const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"

// --- Interfaces based on API Specification ---

export interface SearchRequest {
  query: string
  top_k?: number
  min_similarity?: number
  price_min?: number
  price_max?: number
  category?: string
  min_rating?: number
}

export interface ProductResult {
  parent_asin: string
  title: string
  main_category?: string
  price?: number
  average_rating?: number
  rating_number?: number
  similarity_score: number
  features: string[]
  description: string[]
  store?: string
  categories: string[]
  images: string[]
}

export interface SearchResult {
  product: ProductResult
  rank: number
}

export interface SearchResponse {
  query: string
  results: SearchResult[]
  total_results: number
  search_time_ms: number
  enhanced_query?: string
}

/**
 * Searches for products using the backend API.
 * @param request - The search request parameters.
 * @returns A promise that resolves to the search response.
 */
export async function searchProducts(request: SearchRequest): Promise<SearchResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || "An error occurred during the search.")
    }

    return await response.json()
  } catch (error) {
    console.error("API Search Error:", error)
    // In case of a network error or failed fetch, return a structured error response
    if (error instanceof Error) {
      throw new Error(`Failed to fetch search results: ${error.message}`)
    }
    throw new Error("An unknown error occurred while searching.")
  }
}

export async function getPopularProducts(): Promise<ProductResult[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/popular`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || "An error occurred while fetching popular products.")
    }

    return (await response.json()) as ProductResult[]
  } catch (error) {
    console.error("API Popular Products Error:", error)
    if (error instanceof Error) {
      throw new Error(`Failed to fetch popular products: ${error.message}`)
    }
    throw new Error("An unknown error occurred while fetching popular products.")
  }
}

export async function getProductById(id: string): Promise<ProductResult | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/product/${id}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      return null
    }

    return (await response.json()) as ProductResult
  } catch (error) {
    console.error("API Product by ID Error:", error)
    return null
  }
}
