# Stage 1: Build stage with dependencies
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy and install Python requirements
COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# Stage 2: Final production stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache --no-index --find-links=/wheels /wheels/*

# Copy application code
COPY . .

# No EXPOSE or CMD here, handled by docker-compose
