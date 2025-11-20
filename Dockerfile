FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for RAG
RUN mkdir -p /tmp/vector_store /tmp/data

# Collect static files
RUN python manage.py collectstatic --noinput

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "rag_backend.wsgi:application"]