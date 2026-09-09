# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace \
    HOME=/tmp
WORKDIR /workspace
COPY requirements.txt ./
RUN python -m pip install torch==2.10.0 --index-url ${TORCH_INDEX_URL} \
    && python -m pip install -r requirements.txt
COPY app ./app
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY docs ./docs
COPY Dockerfile requirements-dev.txt pyproject.toml README.md Makefile .gitignore .dockerignore docker-compose.yml docker-compose.gpu.yml ./
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]

FROM base AS test
COPY requirements-dev.txt pyproject.toml README.md ./
COPY tests ./tests
RUN python -m pip install -r requirements-dev.txt
CMD ["python", "-m", "pytest", "-q"]

FROM base AS runtime
USER 1000:1000
