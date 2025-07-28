import Image from "next/image"
import { Star } from "lucide-react"
import { cn } from "@/lib/utils"
import type { ProductResult } from "@/services/api"

interface ProductCardProps {
  product: ProductResult
  className?: string
}

export default function ProductCard({ product, className }: ProductCardProps) {
  const imageUrl = product.images?.[0]

  return (
    <div
      className={cn(
        "group bg-[#343541] border border-[#40414f] rounded-xl overflow-hidden transition-all duration-200 hover:border-[#565869]",
        className,
      )}
    >
      {imageUrl ? (
        <div className="relative aspect-square bg-[#40414f]">
          <Image
            src={imageUrl}
            alt={product.title || "Product image"}
            fill
            className="object-cover"
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
            unoptimized
            onError={(e) => {
              const target = e.target as HTMLImageElement
              target.style.display = 'none'
            }}
          />
          <div className="absolute top-3 left-3">
            <div className="inline-flex items-center px-2 py-1 bg-gradient-to-r from-[#10a37f] to-[#1a7f64] text-white text-xs rounded-full font-medium">
              {Math.round(product.similarity_score * 100)}% Match
            </div>
          </div>
        </div>
      ) : (
        <div className="aspect-square bg-[#40414f] flex items-center justify-center">
          <span className="text-[#acacbe] text-sm">No Image</span>
        </div>
      )}

      <div className="p-4">
        {product.store && (
          <p className="text-xs text-[#acacbe] mb-1 uppercase tracking-wider truncate">
            {product.store}
          </p>
        )}
        
        <h3 
          className="font-medium text-white mb-2 line-clamp-2 leading-snug min-h-[2.5rem]" 
          title={product.title}
        >
          {product.title}
        </h3>

        <div className="flex items-center justify-between mt-auto">
          <div className="flex-1">
            {product.price ? (
              <span className="text-lg font-semibold text-white">
                ${product.price.toFixed(2)}
              </span>
            ) : (
              <span className="text-sm text-[#acacbe]">Price not available</span>
            )}
          </div>
          
          {product.average_rating && (
            <div className="flex items-center space-x-1 ml-2">
              <Star className="h-4 w-4 fill-yellow-400 text-yellow-400 flex-shrink-0" />
              <span className="text-sm text-[#acacbe] whitespace-nowrap">
                {product.average_rating.toFixed(1)}
                {product.rating_number && (
                  <span className="text-[#565869] ml-1">({product.rating_number})</span>
                )}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
