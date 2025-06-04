FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies and optimization tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set optimization environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_HOME=/root/.cache/huggingface
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# CUDA optimization environment variables for 80GB VRAM
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Copy requirements and install Python dependencies first
COPY requirements.txt .

# Install PyTorch and dependencies with CUDA optimization
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /tmp && chmod 777 /tmp
RUN mkdir -p /root/.cache/huggingface && chmod -R 777 /root/.cache

# Pre-compile CUDA kernels for faster startup (optional warm-up)
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0

# Command to run the serverless handler
CMD ["python", "-u", "handler.py"]
