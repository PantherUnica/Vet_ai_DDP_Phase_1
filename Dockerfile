# VetAI Doctor UI — production-ish image
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    VETAI_DATA_DIR=/data \
    VETAI_RUNS_DIR=/runs

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --upgrade pip && pip install -r requirements-deploy.txt

COPY . .

RUN mkdir -p /data /runs && \
    useradd --create-home --shell /bin/bash vetai && \
    chown -R vetai:vetai /app /data /runs

USER vetai

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "doctor_ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
