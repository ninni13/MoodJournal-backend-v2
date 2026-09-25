FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 && \
    pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
