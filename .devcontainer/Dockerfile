FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgomp1 && \
    pip install --no-cache-dir \
    torch==2.0.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
    numpy \
    ipykernel \
    plotly && \
    pip install --upgrade pip && \
    apt-get purge -y --auto-remove gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*
