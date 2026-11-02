FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chewwatch/ chewwatch/

RUN useradd --system --no-create-home chewwatch \
    && mkdir -p /data \
    && chown chewwatch /data

USER chewwatch
VOLUME ["/data"]

# /config/config.yaml is expected to be mounted read-only.
CMD ["python", "-m", "chewwatch", "--config", "/config/config.yaml"]
