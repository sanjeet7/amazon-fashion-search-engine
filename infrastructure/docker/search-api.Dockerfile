FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock ./
COPY shared/ ./shared/
COPY services/search-api/ ./services/search-api/

# Install Python dependencies
RUN pip install uv && \
    uv venv && \
    . .venv/bin/activate && \
    uv sync

# Create data directories
RUN mkdir -p data/processed data/embeddings logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "services/search-api/main.py", "--host", "0.0.0.0", "--port", "8000"]