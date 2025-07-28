import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone build for Docker
  output: 'standalone',
  
  // Optimize for production
  compress: true,
  poweredByHeader: false,
  
  // Environment variable configuration
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  
  // Image optimization
  images: {
    domains: [],
    unoptimized: true, // For better Docker compatibility
  },
  
  // ESLint configuration
  eslint: {
    ignoreDuringBuilds: false,
  },
  
  // TypeScript configuration
  typescript: {
    ignoreBuildErrors: false,
  },
  
  // Experimental features for better performance
  experimental: {
    optimizeCss: true,
  },
};

export default nextConfig;
