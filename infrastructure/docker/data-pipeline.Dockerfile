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
COPY services/data-pipeline/ ./services/data-pipeline/

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

# Expose port (not used for data pipeline but good practice)
EXPOSE 8080

# Default command
CMD ["python", "services/data-pipeline/main.py"]