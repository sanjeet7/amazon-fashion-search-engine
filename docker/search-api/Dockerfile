FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
RUN pip install \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pandas==2.1.4 \
    numpy==1.25.2 \
    openai==1.12.0 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1 \
    faiss-cpu==1.8.0 \
    tiktoken==0.7.0 \
    pyarrow==16.1.0 \
    psutil==5.9.8 \
    packaging==24.0

# Copy backend code
COPY backend/ backend/

# Set Python path
ENV PYTHONPATH=/app

# Create data directories
RUN mkdir -p data/raw data/processed data/embeddings

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
WORKDIR /app/backend/search-api
CMD ["python", "src/main.py"] 