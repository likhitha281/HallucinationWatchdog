# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# Install Node (needed for npx @arizeai/phoenix-mcp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pre-install phoenix-mcp globally so npx doesn't fetch on first request
RUN npm install -g @arizeai/phoenix-mcp@latest

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Cloud Run expects the app to listen on $PORT (default 8080)
ENV PORT=8080

EXPOSE 8080

# Use uvicorn with multiple workers for Cloud Run concurrency
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port $PORT --workers 2 --log-level info"]
