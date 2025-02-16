FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        torch==2.6.0 \
        torchaudio==2.6.0 \
        torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/test/cpu && \
    pip install --no-cache-dir \
        numpy \
        ipykernel \
        matplotlib \
        nbformat \
        plotly && \
        pip install --upgrade pip

RUN apt-get purge -y --auto-remove \
        gcc \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*