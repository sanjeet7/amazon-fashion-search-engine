"use client"

import { useState, useEffect } from "react"
import ProductCard from "@/components/ProductCard"
import { searchProducts, type SearchResult } from "@/services/api"
import { RotateCcw, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface SearchResultsProps {
  query: string
}

export default function SearchResults({ query }: SearchResultsProps) {
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [totalResults, setTotalResults] = useState(0)

  useEffect(() => {
    if (query && query.trim()) {
      performSearch(query)
    } else {
      setResults([])
      setLoading(false)
      setError(null)
    }
  }, [query])

  const performSearch = async (searchQuery: string) => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await searchProducts({ query: searchQuery, top_k: 20 })
      setResults(response.results)
      setTotalResults(response.total_results)
    } catch (err) {
      console.error("Search error:", err)
      setError(err instanceof Error ? err.message : "An unknown error occurred.")
      setResults([])
      setTotalResults(0)
    } finally {
      setLoading(false)
    }
  }

  // Don't render anything if no query
  if (!query || !query.trim()) {
    return null
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="flex items-center space-x-3">
          <div className="w-6 h-6 border-2 border-[#10a37f] border-t-transparent rounded-full animate-spin"></div>
          <span className="text-[#acacbe]">Finding your perfect matches...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="w-12 h-12 bg-[#40414f] rounded-full flex items-center justify-center mx-auto mb-4">
            <AlertTriangle className="h-6 w-6 text-red-400" />
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">Oops! Something went wrong</h3>
          <p className="text-[#acacbe] mb-4 max-w-md">{error}</p>
          <Button 
            onClick={() => performSearch(query)} 
            className="bg-[#10a37f] hover:bg-[#0d8f68] text-white border-0"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Try Again
          </Button>
        </div>
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">🤷</div>
        <h3 className="text-lg font-semibold text-white mb-2">No matches found</h3>
        <p className="text-[#acacbe]">Try a different search term or check your spelling.</p>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <p className="text-[#acacbe]">
        Showing <span className="font-semibold text-white">{results.length}</span> of{" "}
        <span className="font-semibold text-white">{totalResults}</span> results.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {results.map((result) => (
          <ProductCard key={result.product.parent_asin} product={result.product} />
        ))}
      </div>
    </div>
  )
}
