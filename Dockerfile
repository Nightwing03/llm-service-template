FROM python:3.12-slim

# Bytecode caching is pointless in a layer that never changes; unbuffered
# output means container logs appear when they happen, not when a buffer fills.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so a source edit does not invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "uvicorn[standard]"

COPY llm_service/ ./llm_service/

# Non-root: nothing in here needs to write to the filesystem.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000

# Hits /health, which makes no upstream call -- so the healthcheck cannot
# burn rate-limit quota on a container that is merely idle.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "llm_service.app:app", "--host", "0.0.0.0", "--port", "8000"]