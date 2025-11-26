# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js (needed if filesystem server is used in other configs)
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for web app (Streamlit or Flask)
EXPOSE 8501
EXPOSE 3000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - can be overridden
CMD ["python", "Clients/mcp_slack.py"]
