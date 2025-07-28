# ===================================
# Data Pipeline Service Dockerfile
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
COPY services/data-pipeline/ ./services/data-pipeline/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create data directories with proper permissions
RUN mkdir -p /app/data/{raw,processed,embeddings,sample} && \
    chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
ENTRYPOINT ["python", "services/data-pipeline/main.py"]
CMD ["--validate-only"]