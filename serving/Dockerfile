# CPU image for local deployment of the FastAPI serving endpoint.
# For GPU serving, base off nvidia/cuda and install the cuda torch wheels instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    LOAD_4BIT=0

WORKDIR /app

# System deps for tokenizers / sentencepiece builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt ./serving/requirements.txt
RUN pip install -r serving/requirements.txt

COPY src ./src
COPY serving ./serving

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
