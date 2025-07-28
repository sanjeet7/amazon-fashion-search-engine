import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Basic health check - ensure the frontend is responsive
    return NextResponse.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'fashion-search-frontend',
      version: '1.0.0'
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}