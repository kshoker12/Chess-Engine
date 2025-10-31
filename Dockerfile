# --------------------------------------------------------------
# Builder Stage: Compile Stockfish 16 from source on ARM64
# --------------------------------------------------------------
FROM amazonlinux:2023 AS builder

# Install build tools with dnf (correct for AL2023)
RUN dnf update -y && \
    dnf install -y \
        git \
        make \
        gcc-c++ && \
    dnf clean all && \
    rm -rf /var/cache/yum

# Clone Stockfish, checkout SF16 tag, compile for ARM64 (armv8)
RUN git clone https://github.com/official-stockfish/Stockfish.git && \
    cd Stockfish && \
    git checkout sf_16 && \
    cd src && \
    make -j$(nproc) build ARCH=armv8

# --------------------------------------------------------------
# Final Stage: AWS Lambda Python 3.12 (aarch64)
# --------------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.12

# Install minimal runtime tools (for Stockfish if needed)
RUN microdnf update -y && \
    microdnf install -y \
        wget \
        unzip \
        tar \
        gzip \
        bzip2 \
        ca-certificates && \
    microdnf clean all && \
    rm -rf /var/cache/yum

# Copy compiled Stockfish binary from builder
COPY --from=builder /Stockfish/src/stockfish /usr/local/bin/stockfish
RUN chmod +x /usr/local/bin/stockfish

# Verify (fails build if compile broke)
RUN stockfish --version

# Lambda working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app.py .
COPY engine.py .
COPY features.py .
COPY eval_model.py .
COPY pytorch_model.py .

# Copy models
COPY sk_eval.joblib .

# Optional PyTorch model
# COPY chess_eval_pytorch.pt .

# Lambda handler
CMD [ "app.handler" ]