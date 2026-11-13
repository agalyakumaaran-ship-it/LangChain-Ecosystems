
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY Requirements.txt ./Requirements.txt
RUN pip install --no-cache-dir -r Requirements.txt
COPY backend/ ./backend/
COPY Hybrid_RAG.py ./Hybrid_RAG.py
RUN useradd --create-home app && chown -R app:app /app
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit","run","Hybrid_RAG.py","--server.port=8501","--server.address=0.0.0.0","--server.headless=true"]
