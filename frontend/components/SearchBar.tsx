"use client"

import type React from "react"
import { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { Search, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface SearchBarProps {
  placeholder?: string
  initialQuery?: string
  className?: string
}

export default function SearchBar({
  placeholder = "Search for fashion items...",
  initialQuery = "",
  className,
}: SearchBarProps) {
  const [query, setQuery] = useState(initialQuery)
  const inputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()

  const handleSearch = (searchQuery: string) => {
    if (!searchQuery.trim()) return
    const params = new URLSearchParams({ q: searchQuery })
    router.push(`/search?${params.toString()}`)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    handleSearch(query)
  }

  return (
    <div className={cn("relative w-full", className)}>
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative group">
          <Search className="absolute left-5 top-1/2 transform -translate-y-1/2 h-5 w-5 text-[#acacbe] group-focus-within:text-[#10a37f] transition-colors duration-200" />
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder}
            className="w-full pl-14 pr-14 py-4 text-base border border-[#6e6f7e] rounded-xl focus:ring-2 focus:ring-[#10a37f] focus:border-[#10a37f] outline-none transition-all duration-200 bg-[#565869] text-white placeholder-[#acacbe] hover:bg-[#6e6f7e] shadow-lg"
          />
          {query && (
            <button
              type="button"
              onClick={() => {
                setQuery("")
                inputRef.current?.focus()
              }}
              className="absolute right-5 top-1/2 transform -translate-y-1/2 p-1 hover:bg-[#6e6f7e] rounded-full transition-colors duration-200"
            >
              <X className="h-4 w-4 text-[#acacbe]" />
            </button>
          )}
        </div>
      </form>
    </div>
  )
}
