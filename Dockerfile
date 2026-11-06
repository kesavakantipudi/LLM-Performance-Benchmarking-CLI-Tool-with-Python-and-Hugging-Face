FROM python:3.11-slim

# Install system dependencies for build, git, and monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and configs
COPY src/ /app/src/
COPY datasets/ /app/datasets/
COPY configs/ /app/configs/
COPY tests/ /app/tests/

# Set Python path and cache locations
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Default entry point
ENTRYPOINT ["python", "src/cli.py"]
CMD ["--help"]
