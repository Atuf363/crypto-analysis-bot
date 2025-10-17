# Updated build
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- CHANGE 1: Copy contents of backend/ folder directly into the working directory ---
COPY backend/ .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# --- CHANGE 2: Run main.py directly as it is now at the root of the working directory ---
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
