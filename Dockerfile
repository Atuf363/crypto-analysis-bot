# Updated build
FROM python:3.11-slim

# ... (other RUN commands for apt/gcc) ...

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep this copy command, it creates /app/backend
COPY backend/ ./backend/

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# *** CRITICAL FIX: Set PYTHONPATH to /app ***
CMD ["sh", "-c", "PYTHONPATH=/app uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
