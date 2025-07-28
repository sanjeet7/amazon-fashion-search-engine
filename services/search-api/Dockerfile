# ===================================
# Search API Service Dockerfile  
# ===================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy project files
COPY pyproject.toml ./
COPY shared/ ./shared/
COPY services/search-api/ ./services/search-api/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create data directories with proper permissions (read-only)
RUN mkdir -p /app/data/{processed,embeddings} && \
    chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["python", "services/search-api/main.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]