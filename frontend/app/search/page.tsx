"use client"

import { Suspense } from "react"
import SearchBar from "@/components/SearchBar"
import SearchResults from "@/components/SearchResults"
import { ArrowLeft } from "lucide-react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"

export default function SearchPage() {
  const searchParams = useSearchParams()
  const query = searchParams.get('q') || ""

  return (
    <div className="min-h-screen bg-[#202123]">
      {/* Header */}
      <div className="bg-[#343541] border-b border-[#40414f] sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center space-x-4 mb-4">
            <Link href="/" className="flex items-center space-x-2 text-[#acacbe] hover:text-white transition-colors">
              <ArrowLeft className="h-5 w-5" />
              <span>Back to Home</span>
            </Link>
          </div>
          <SearchBar initialQuery={query} className="max-w-3xl" />
        </div>
      </div>

      {/* Search Results */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {query && (
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-white">Results for "{query}"</h1>
          </div>
        )}
        <Suspense fallback={<SearchResultsSkeleton />}>
          <SearchResults query={query} />
        </Suspense>
      </div>
    </div>
  )
}

function SearchResultsSkeleton() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {Array.from({ length: 12 }).map((_, i) => (
        <div key={i} className="bg-[#343541] rounded-lg border border-[#40414f] overflow-hidden">
          <div className="aspect-square bg-[#40414f] animate-pulse" />
          <div className="p-4 space-y-3">
            <div className="h-4 bg-[#40414f] rounded animate-pulse" />
            <div className="h-3 bg-[#40414f] rounded w-2/3 animate-pulse" />
            <div className="h-4 bg-[#40414f] rounded w-1/3 animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  )
}
