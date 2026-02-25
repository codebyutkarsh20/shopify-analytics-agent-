# ============================================
# Shopify Analytics Agent - Production Dockerfile
# Multi-stage build for minimal image size
# ============================================

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies for compiled packages (numpy, pandas, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies into a virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Production image
FROM python:3.12-slim AS production

# Labels
LABEL maintainer="Shopify Analytics Agent"
LABEL description="Telegram + WhatsApp bot for Shopify analytics with multi-LLM support"

# Security: run as non-root user
RUN groupadd -r botuser && useradd -r -g botuser -d /app -s /sbin/nologin botuser

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Disable Python buffering for real-time Docker logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY src/ ./src/
COPY main.py .
COPY requirements.txt .

# Create data and log directories with correct ownership
RUN mkdir -p /app/data /app/logs && \
    chown -R botuser:botuser /app

# Switch to non-root user
USER botuser

# Expose WhatsApp webhook port (only used if WHATSAPP_ENABLED=true)
EXPOSE 8080

# Health check: hit the actual WhatsApp webhook /health endpoint
# start-period=60s gives sentence-transformers model time to load
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=5)" || exit 1

# Default command
CMD ["python", "main.py"]
