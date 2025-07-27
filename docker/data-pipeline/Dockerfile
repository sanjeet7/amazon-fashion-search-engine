FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with explicit httpx version
RUN pip install \
    pandas==2.1.4 \
    numpy==1.25.2 \
    httpx==0.24.1 \
    openai==1.40.0 \
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
RUN mkdir -p data/raw data/processed data/embeddings data/sample

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "backend.data_pipeline.src.main", "--mode=sample"] 