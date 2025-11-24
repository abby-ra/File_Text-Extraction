# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_chithrakan.txt .
RUN pip install --no-cache-dir -r requirements_chithrakan.txt

# Try to install Chithrakan from various sources
RUN pip install chithrakan || \
    (git clone https://github.com/AI4Bharat/Chithrakan.git /tmp/chithrakan && \
     cd /tmp/chithrakan && \
     pip install -e . && \
     cd / && rm -rf /tmp/chithrakan) || \
    echo "Chithrakan installation attempted"

# Install alternative Malayalam OCR tools
RUN pip install indic-ocr || echo "indic-ocr not available"

# Copy application files
COPY enhanced_extract.py .
COPY chithrakan_docker.py .

# Create necessary directories
RUN mkdir -p /app/input_files /app/output

# Expose port for API (optional)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "chithrakan_docker.py"]
