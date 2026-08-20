---
title: ScriptTagger
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ScriptTagger — AI-Powered Metadata Tagging from Movie Transcripts

Generative-AI / NLP pipeline that ingests a movie screenplay (a transcript of media content) and produces rich, structured metadata for indexing, archiving, recommendations and compliance:

- **Scene / time segmentation** — every scene becomes a timestamped segment
- **Speaker identification** — who said what, with gender enrichment
- **Topics & keywords** — RAKE + TF-IDF + KeyBERT keyphrases per scene and overall
- **Named entities** — people, organizations, locations, products (spaCy)
- **Sentiment & emotion** — VADER baseline + optional transformer emotion model (RoBERTa)
- **Content classification** — multi-label genre classifier trained on screenplay text
- **Ultra-Fast Storage** — `.json.gz` Gzip compression (~900MB reduced to ~100MB)
- **React 19 Web Dashboard** — Glassmorphism dark-mode React + Tailwind + Recharts frontend

---

## Architecture

```
screenplay (.txt)
      │
      ▼
 src/parser.py         structural parse (scenes, speakers, dialogue, action)
 src/segmentation.py   scene → pseudo-timestamps (mm:ss per segment)
 src/speakers.py       canonical speakers + gender enrichment
 src/topics.py         RAKE / TF-IDF / KeyBERT keyphrases
 src/ner.py            spaCy NER, per-scene + global, speaker linking
 src/sentiment.py      VADER + (optional) transformer emotion/sentiment
 src/classify.py       multi-label genre classifier (sklearn)
 src/pipeline.py       orchestrates the modules → compressed metadata (.json.gz)
 api/main.py           FastAPI service & Static Frontend host (http://localhost:8000)
 frontend/             React 19 + Vite + Tailwind + Recharts Web Dashboard
 ui/app.py             Streamlit dashboard (http://localhost:8501)
 evaluate/evaluate.py  offline evaluation vs rule_based/BERT annotations
 scripts/tag_corpus.py multi-threaded parallel batch processor
```

Metadata output (Compressed `.json.gz` or JSON API response) per script:

```json
{
  "title": "Ex Machina",
  "genres": [{"genre": "Drama", "score": 0.38}, ...],
  "known_genres": ["Drama", "Mystery", "Sci-Fi", "Thriller"],
  "overall": {
    "topics": [{"keyword": "...", "score": 0.12}, ...],
    "entities": [{"label": "PERSON", "text": "CALEB", "count": 326, "is_speaker": true}, ...],
    "sentiment": {"compound": 0.03, "label": "neutral"},
    "emotion": {"label": "neutral", "distribution": {...}},
    "num_scenes": 142, "num_dialogue_lines": 2227, "num_words": 11359
  },
  "segments": [
    {"segment_id": 1, "start": "00:00", "end": "00:06",
     "heading": "INT. OFFICE - DAY", "location": "OFFICE", "time_of_day": "DAY",
     "speakers": [...], "topics": [...], "entities": [...],
     "sentiment": {...}, "emotion": {...}, "dialogue": [...]}
  ],
  "speakers": [{"name": "NATHAN", "lines": 920, "words": 4760, "gender": null}, ...]
}
```

---

## Running with Docker

This project is fully Dockerized — no local Python/Node install needed. There
are two ways to run it, depending on whether you have the raw Kaggle dataset:

### A. Standalone container — cached outputs + user uploads (no dataset needed)

This is exactly what ships to production (Hugging Face Spaces / Render /
Cloud Run — see [Deploying](#deploying-on-hugging-face-spaces-cached-outputs--uploads-only)
below). Works with just this repo, nothing else required:

```bash
git clone https://github.com/Shrey7781/metadata_tagging.git
cd metadata_tagging/Metadata_Tagging
docker build -t scripttagger .
docker run -p 7860:7860 scripttagger
```

- Open **[http://localhost:7860](http://localhost:7860)** for the React dashboard.
- Interactive API docs: [http://localhost:7860/docs](http://localhost:7860/docs)
- Serves the ~200 pre-tagged movies baked into `outputs/`, and tags any
  user-uploaded script live (genre prediction included, via the pre-trained
  `data/models/genre_classifier.joblib`).

### B. docker-compose — local full-corpus dev (dataset optional)

```bash
git clone https://github.com/Shrey7781/metadata_tagging.git
cd metadata_tagging/Metadata_Tagging
docker compose up --build
```

- **React / FastAPI Web Application:** [http://localhost:8000](http://localhost:8000)
- **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)

`docker-compose.yml` mounts `./data` and `./outputs` as volumes. If you also
place the raw Kaggle dataset (see [Dataset](#dataset) below) where
`src/config.py` expects it — or point `DATASET_ROOT` at it — you get the
full 2,800+ movie catalog and can tag/browse anything, not just the ~200
pre-tagged movies. Without the dataset, this runs in the same cached +
uploads mode as the standalone container above.

---

## Local Setup (Virtual Environment)

### Step 1: Create a Virtual Environment

**On Windows:**
```powershell
python -m venv .venv
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
```

### Step 2: Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```cmd
.\.venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 (Optional): Full Setup & Model (Re)Training

Only needed if you have the raw Kaggle dataset (see [Dataset](#dataset))
and want the full 2,800+ movie catalog, NER on the larger `en_core_web_lg`
model, and/or to retrain the genre classifier yourself. Skip this if you're
fine with the ~200 pre-tagged movies already in `outputs/` plus live-tagging
your own script uploads — that works out of the box with no dataset at all
(genre prediction included, via the already-trained
`data/models/genre_classifier.joblib` committed in this repo).

```bash
# point at wherever you extracted the dataset (or hardcode it in src/config.py)
export DATASET_ROOT="/path/to/archive"
python setup.py
```

`setup.py` downloads `en_core_web_lg`, NLTK stopwords, builds the full
corpus index, and retrains the genre classifier against the dataset.

> ⚡ **Note**: No GPU required. All models are optimized to run on CPU.

---

## How to Run the Applications

Make sure your virtual environment is activated before running these commands.

### 1. Run React Frontend + FastAPI Backend (Main Web App)

```bash
python -m uvicorn api.main:app --port 8000
```
- Open **[http://localhost:8000](http://localhost:8000)** in your browser for the React Glassmorphism Web App.
- Interactive API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
- Endpoints: `POST /tag` (Tag script/text), `GET /scripts` (List corpus scripts), `GET /metadata/{imdb_id}` (Fetch metadata).

To run React Frontend in standalone Vite development mode:
```bash
cd frontend
npm install
npm run dev
```
- Opens at [http://localhost:5173](http://localhost:5173) with automatic proxying to FastAPI port 8000.

### 2. Run Streamlit UI Dashboard

```bash
streamlit run ui/app.py
```
- Opens automatically at [http://localhost:8501](http://localhost:8501)

### 3. Run Multi-Threaded Parallel Batch Processor

To generate `.json.gz` compressed metadata for all 2,800+ scripts using multi-threading:

```bash
python scripts/tag_corpus.py --workers 8
```

### 4. Run Model Evaluation Benchmark

To evaluate parser and tagging accuracy against annotated ground truth:

```bash
python evaluate/evaluate.py --sample 30
```

---

## Dataset

Kaggle — [Movie Scripts Corpus](https://www.kaggle.com/datasets/gufukuro/movie-scripts-corpus)
(`archive (3)` extracted to the path configured in `src/config.py`):
- `screenplay_data/raw_texts` — 2,857 screenplay transcripts
- `screenplay_data/rule_based_annotations` — ScreenPy scene/speaker ground truth (2,607)
- `screenplay_data/BERT_annotations`, `manual_annotations` — line-type labels
- `movie_characters/` — per-character dialogue + gender pickle
- `movie_metadata/movie_meta_data.csv` — genres, plot, keywords, cast, awards

---

## Deploying on Hugging Face Spaces (cached-outputs + uploads only)

The raw Kaggle corpus (~900MB) is too large to ship in the repo, so the Space
deployment intentionally runs a scaled-down mode: it ships only the
pre-tagged `outputs/*.json.gz` catalog (~17MB, baked into the Docker image)
plus support for tagging user-uploaded scripts on demand. It does **not**
require or expect the raw dataset at build or run time.

- `/scripts` browses the pre-tagged catalog built directly from `outputs/`
  (see `src/corpus._build_index_from_outputs`), not the full 2,800+ corpus.
- `/tag`, `/metadata/{imdb_id}` serve cached results for those pre-tagged
  IMDb ids; anything else returns 404 (raw script text isn't available to
  regenerate from).
- `POST /tag/upload` (and the "Upload file" mode in the UI) still runs the
  full pipeline on any script text a user pastes/uploads — this is the
  primary way to tag anything outside the pre-tagged catalog.
- Genre prediction works on uploaded/non-cached text too — `data/models/
  genre_classifier.joblib` is trained once locally (where the raw dataset
  is available) and committed/baked into the image, so the raw dataset
  itself never needs to ship. Retrain it with `DATASET_ROOT=/path/to/archive
  python -c "from src import classify; classify.train_model(force=True)"`
  if the dataset changes. Note the hold-out macro-F1 is low (~0.06) because
  several genres are heavily underrepresented in this corpus — predictions
  on common genres (Drama, Comedy, Action, Thriller, etc.) are reasonable,
  rare ones (Talk-Show, Short, Sport, War, Western) less so.
- The image bundles `en_core_web_sm` for NER (set via `SPACY_MODEL` env var)
  instead of the heavier `en_core_web_lg` used for local/full-dataset runs,
  to keep build time and image size reasonable on Spaces' free tier.

To deploy: create a new Space with SDK "Docker", push this repo to it (the
`Dockerfile` builds the React frontend and serves it + the API from a single
container on port 7860, matching the `app_port` in this README's frontmatter).
No dataset volume or extra secrets are required.

> **Note:** as of mid-2026, Hugging Face requires a paid PRO/Team/Enterprise
> plan to create a Docker SDK Space (CPU Basic hardware itself is still free
> — the SDK choice is what's gated). The same `Dockerfile` works unmodified
> on Render, Google Cloud Run, or any other platform that just runs a
> Dockerfile, if you'd rather avoid that.

---

## Changelog — Cached-Deployment Migration

The raw ~900MB Kaggle dataset can't be committed to this repo, so the app
was reworked to run fully without it at build/deploy time (still using it
locally, once, wherever available, to produce the artifacts below):

- **`Dockerfile`** — rewritten as a multi-stage build: compiles the React
  frontend (`frontend/dist`) in a Node stage, bakes in the `en_core_web_sm`
  spaCy model + NLTK stopwords at build time (previously undone — NER relied
  on an unreliable runtime download on first use), copies `outputs/` into
  the image, and serves everything from one container on port 7860 (the
  Hugging Face Docker SDK / Render / Cloud Run convention). Previously the
  frontend was never built at all in Docker, and the API only ever returned
  a bare JSON message at `/`.
- **`src/corpus.py`** — `build_index()` now builds a lightweight catalog
  directly from `outputs/*.json.gz` when the raw dataset isn't present,
  instead of crashing (this also happened to fix the Streamlit dashboard,
  which called the same code path unguarded). `script_path()` now raises a
  clean `KeyError` (→ HTTP 404) instead of an unhandled exception when raw
  script text isn't available for a given movie.
- **`src/ner.py`** — the spaCy model name is now configurable via a
  `SPACY_MODEL` env var (defaults to `en_core_web_lg` for local/full-dataset
  use; the Docker image sets it to the lighter `en_core_web_sm` to keep
  build time/image size reasonable).
- **`src/config.py`** — added a `DATASET_ROOT` env var override so a one-off
  local run (e.g. training the classifier) can point at wherever the
  dataset happens to be extracted, without hardcoding a personal machine
  path into the file.
- **`.dockerignore` / `.gitignore`** — previously blanket-excluded all of
  `outputs/` and `data/` (fine when relying on docker-compose volumes, but
  fatal for a standalone image with no volumes). Now only the regenerable
  `data/corpus_index.csv` is excluded, so the pre-tagged catalog and the
  trained genre classifier actually ship inside the image/repo.
- **`data/models/genre_classifier.joblib`** (new, committed, 13MB) — trained
  once locally against the full dataset (23 genres, 2,831 samples), so
  genre prediction works on live user uploads without needing the dataset
  at deploy time. Hold-out macro-F1 is low (~0.06) due to genre class
  imbalance in the corpus — reliable on common genres (Drama, Comedy,
  Action, Thriller...), weak on rare ones (Talk-Show, Short, Sport, War,
  Western).
- **`README.md`** — added the Hugging Face Spaces YAML frontmatter, the
  deployment-scope and free-hosting-platform notes above, and this section.

All of the above was validated end-to-end against a real `docker build` +
standalone `docker run` (no volumes, no dataset) before being committed:
catalog browsing, cached-movie metadata lookup, clean 404s for uncached
titles, and live tagging (NER, sentiment, genre prediction) on a freshly
uploaded script all confirmed working.

---

## Notes & Tradeoffs

- **Gzip Compression**: Outputs are stored as `.json.gz` files in `outputs/` directory. This reduces storage footprint from ~900MB to ~100MB while preserving full scene dialogue arrays.
- **Parallel Workers**: Batch generation uses `ThreadPoolExecutor` (default 8 workers) to process 2,800+ movies in ~15-20 minutes.
- **Transformer Emotion Model**: Transformer emotion models run on CPU and are optional; toggle them in the UI or pass `use_transformers=true` to the API.
- **Genre Classifier**: Trained on screenplay text + IMDb plot summaries; `known_genres` shows ground truth for corpus titles.