
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application and backend code.
COPY . .

# Create a non-root user and persistent-data directory.
RUN useradd --create-home app \
    && mkdir -p /data/faiss \
    && chown -R app:app /app /data

USER app

EXPOSE 8000

# Check that the API is responding.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run from the project root when main.py is at /app/main.py.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
