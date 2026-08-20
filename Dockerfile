# ---- Frontend build stage ----
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Backend runtime stage ----
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# We use a custom index for PyTorch CPU to save space
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Cached-outputs + user-upload deployment: no raw dataset is available at
# build or run time, so bake in a lightweight spaCy model instead of relying
# on the app's runtime download-on-first-use fallback.
ENV SPACY_MODEL=en_core_web_sm
RUN python -m spacy download en_core_web_sm \
    && python -c "import nltk; nltk.download('stopwords')"

# App code + pre-tagged outputs/ (this deployment has no raw dataset to mount,
# so the cached .json.gz metadata ships inside the image).
COPY . .
COPY --from=frontend /frontend/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces (Docker SDK) routes to this port by default.
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
