# NextJS build stage
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Python backend server stage
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
COPY --chown=user:user . .
COPY --chown=user:user --from=frontend-builder /app/frontend/out ./frontend/out

# Create database folders and set permissions
RUN mkdir -p data/chroma_db

# Predownload and cache sentence transformer model weights
RUN python download_models.py

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "voice_agent.main:app", "--host", "0.0.0.0", "--port", "7860"]
