FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system deps for reportlab/PDF rendering on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 \
    && rm -rf /var/lib/apt/lists/*

# Install project deps first (leverages build cache).
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

# Copy application.
COPY src ./src
COPY .env.example ./

# Output directory for generated PDFs.
ENV DIGEST_OUTPUT_DIR=/data/digests
RUN mkdir -p /data/digests

CMD ["python", "-m", "digest_agent.main"]