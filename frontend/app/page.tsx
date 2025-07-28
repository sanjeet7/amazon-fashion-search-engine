"use client"

import SearchBar from "@/components/SearchBar"
import { Search, Sparkles, TrendingUp } from "lucide-react"
import { useRouter } from "next/navigation"

export default function Home() {
  const router = useRouter()

  const handleExampleSearch = (query: string) => {
    const params = new URLSearchParams({ q: query })
    router.push(`/search?${params.toString()}`)
  }

  return (
    <div className="min-h-screen bg-[#202123]">
      {/* Hero Section */}
      <div className="flex flex-col items-center justify-center min-h-screen px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          {/* Logo/Title */}
          <div className="mb-8">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-[#10a37f] to-[#1a7f64] rounded-xl flex items-center justify-center">
                <Search className="h-6 w-6 text-white" />
              </div>
              <h1 className="text-4xl sm:text-5xl font-bold text-white">
                Fashion Search
              </h1>
            </div>
            <p className="text-xl text-[#acacbe] max-w-2xl mx-auto leading-relaxed">
              Discover your perfect style with AI-powered semantic search. 
              Find fashion items using natural language descriptions.
            </p>
          </div>

          {/* Search Bar */}
          <div className="mb-12">
            <SearchBar 
              placeholder="Try: 'comfortable summer dress under $50' or 'elegant wedding guest outfit'"
              className="max-w-2xl mx-auto"
            />
          </div>

          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto">
            <div className="bg-[#343541] border border-[#40414f] rounded-xl p-6">
              <div className="w-10 h-10 bg-[#10a37f]/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Sparkles className="h-5 w-5 text-[#10a37f]" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Smart Search</h3>
              <p className="text-[#acacbe] text-sm">
                Use natural language to find exactly what you're looking for
              </p>
            </div>

            <div className="bg-[#343541] border border-[#40414f] rounded-xl p-6">
              <div className="w-10 h-10 bg-[#10a37f]/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <TrendingUp className="h-5 w-5 text-[#10a37f]" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Relevance Scoring</h3>
              <p className="text-[#acacbe] text-sm">
                AI-powered matching shows how well each item fits your query
              </p>
            </div>

            <div className="bg-[#343541] border border-[#40414f] rounded-xl p-6">
              <div className="w-10 h-10 bg-[#10a37f]/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <Search className="h-5 w-5 text-[#10a37f]" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Advanced Filters</h3>
              <p className="text-[#acacbe] text-sm">
                Filter by price, rating, category, and more for precise results
              </p>
            </div>
          </div>

          {/* Sample Searches */}
          <div className="mt-12">
            <p className="text-[#acacbe] mb-4">Try these example searches:</p>
            <div className="flex flex-wrap justify-center gap-3">
              {[
                "summer dresses under $30",
                "professional work attire",
                "casual weekend outfit",
                "formal evening wear"
              ].map((example) => (
                <button
                  key={example}
                  onClick={() => handleExampleSearch(example)}
                  className="px-4 py-2 bg-[#40414f] hover:bg-[#565869] border border-[#6e6f7e] rounded-lg text-[#acacbe] hover:text-white transition-colors text-sm"
                >
                  "{example}"
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
