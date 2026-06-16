# Use NVIDIA CUDA devel image for full CUDA toolkit
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    git \
    wget \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA_HOME for torch extensions
ENV CUDA_HOME=/usr/local/cuda

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Install packages directly into the system Python environment (no venv)
ENV UV_SYSTEM_PYTHON=1

# Copy project files first for dependency caching
COPY pyproject.toml ./

# Increase UV network timeout for large wheel downloads
ENV UV_HTTP_TIMEOUT=600

# Install PyTorch for CUDA 12.4
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1+cu124

# Copy the package code
COPY gemma4/ ./gemma4/

# Install the project and remaining dependencies
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e .

# Default command
CMD ["python", "-c", "from gemma4 import Gemma4Config; print('gemma4-e4b loaded successfully')"]
