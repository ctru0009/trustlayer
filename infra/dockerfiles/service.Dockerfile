# Model service: FastAPI + torch (CPU) over the pinned embed/classify extras.
#
# Build context is the REPO ROOT (see infra/docker-compose.yml): the
# Dockerfile copies python/ only. Model weights download on first use
# into a named volume (uv + HF caches persist across restarts, not
# across volume wipes). Needs the HF token + licence accept baked in
# only for /answer (Gemma is gated); /embed + /classify work unauth'd.
FROM python:3.12-slim-trixie

ENV UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=0

RUN pip install --no-cache-dir uv==0.10.9

WORKDIR /app
COPY python/pyproject.toml python/uv.lock python/.python-version ./
RUN uv sync --locked --extra service --extra embed --extra classify --no-dev

COPY python/src ./src
COPY python/tests ./tests

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=12 \
    CMD uv run --extra service python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uv", "run", "--extra", "service", "--extra", "embed", "--extra", "classify", \
     "uvicorn", "trustlayer.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
