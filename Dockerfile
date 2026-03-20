# RunPod Serverless Dockerfile for Chess AI Service
# Uses CUDA-enabled Python base image for GPU inference

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 and system dependencies (required for numpy 2.3.4)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        build-essential \
        && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Create symlink for python command (remove existing if present)
RUN rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    python --version

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy handler.py (main entry point)
COPY handler.py .

# Copy engine directory (alphabeta.py, mcts.py, various_methods.py)
COPY engine/ ./engine/

# Copy policy_transformer directory (checkpoints excluded via .dockerignore)
COPY policy_transformer/ ./policy_transformer/

# Copy value_transformer directory (checkpoints excluded via .dockerignore)
COPY value_transformer/ ./value_transformer/

# Verify model file exists (the 4o1 model that handler.py loads)
RUN test -f value_transformer/mini_value_6o4.pt || echo "WARNING: Model file not found"

# Set Python path to include working directory
ENV PYTHONPATH=/app

# Expose port (RunPod may use this, though serverless typically doesn't need it)
EXPOSE 8000

# Run the handler (RunPod expects this entrypoint)
# Using -u for unbuffered output (important for RunPod logging)
CMD ["python", "-u", "handler.py"]
