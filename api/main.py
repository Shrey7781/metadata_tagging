import logging
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import corpus, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="ScriptTagger API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def _current_index():
    """Load the corpus index fresh. Deliberately not cached at module scope:
    when running from outputs/ alone (no raw dataset), rebuilding is cheap
    and newly-tagged uploads/pastes should show up in /scripts immediately
    rather than only after a restart."""
    try:
        idx = corpus.load_index()
        return idx if not idx.empty else corpus.build_index()
    except Exception as exc:
        logger.error("Failed to load corpus index: %s", exc)
        return None


try:
    logger.info("Loaded corpus index with %s scripts", len(_current_index()))
except Exception:
    pass


class TagRequest(BaseModel):
    imdb_id: str = ""
    text: str = ""
    title: str = ""
    use_transformers: bool = False
    include_dialogue: bool = False


from fastapi.responses import FileResponse

@app.get("/")
def root():
    index_file = DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"app": "ScriptTagger", "endpoints": ["/health", "/scripts", "/tag", "/metadata/{id}", "/scripts/{id}"]}


@app.get("/api/info")
def info():
    return {"app": "ScriptTagger", "endpoints": ["/health", "/scripts", "/tag", "/metadata/{id}", "/scripts/{id}"]}


@app.get("/health")
@app.get("/api/health")
def health():
    idx = _current_index()
    return {"status": "ok", "scripts_indexed": int(len(idx)) if idx is not None else 0}


@app.get("/scripts")
@app.get("/api/scripts")
def scripts(query: str = "", limit: int = 0, offset: int = 0):
    df = _current_index()
    if df is None:
        raise HTTPException(503, "corpus index unavailable")
    if query:
        mask = df["title"].fillna("").str.contains(query, case=False, regex=False)
        df = df[mask]
    if limit > 0:
        df = df.iloc[offset : offset + limit]
    rows = []
    for _, r in df.iterrows():
        g_val = r.get("genres")
        genres_list = [g.strip() for g in str(g_val).split(",") if g.strip()] if pd.notna(g_val) and g_val else []
        rows.append(
            {
                "imdb_id": str(r["imdbid"]),
                "title": str(r.get("title")) if pd.notna(r.get("title")) else "",
                "year": int(r.get("year")) if pd.notna(r.get("year")) else None,
                "genres": genres_list[:5],
                "num_scenes": None,
            }
        )
    return {"total": len(df), "results": rows}


@app.get("/scripts/{imdb_id}")
@app.get("/api/scripts/{imdb_id}")
def raw_script(imdb_id: str):
    try:
        text = corpus.read_script(imdb_id)
    except KeyError:
        raise HTTPException(404, "script not found")
    return {"imdb_id": imdb_id, "title": corpus.metadata_for(imdb_id).get("title", ""), "text": text}


@app.post("/tag")
@app.post("/api/tag")
def tag(req: TagRequest):
    if req.imdb_id:
        cached = pipeline.load_cached_metadata(req.imdb_id)
        if cached is not None and not req.use_transformers:
            has_dialogue = any(bool(s.get("dialogue")) for s in cached.get("segments", []))
            if not req.include_dialogue or has_dialogue:
                return cached
        try:
            text = corpus.read_script(req.imdb_id)
        except KeyError:
            # No raw text to (re)tag from (e.g. this id is a saved upload,
            # which never has raw text available) -- best effort: serve
            # whatever's cached rather than 404ing when we do have *some*
            # result, even if it doesn't fully match include_dialogue /
            # use_transformers.
            if cached is not None:
                return cached
            raise HTTPException(404, "script not found")
        meta = pipeline.tag_script(
            text,
            imdb_id=req.imdb_id,
            title=corpus.metadata_for(req.imdb_id).get("title", ""),
            use_transformers=req.use_transformers,
            include_dialogue=req.include_dialogue,
        )
        pipeline.save_metadata(req.imdb_id, meta)
        return meta
    if req.text:
        cached = pipeline.load_cached_metadata_by_title(req.title) if req.title else None
        if cached is not None and not req.use_transformers:
            has_dialogue = any(bool(s.get("dialogue")) for s in cached.get("segments", []))
            if not req.include_dialogue or has_dialogue:
                return cached
        meta = pipeline.tag_script(
            req.text,
            imdb_id=req.imdb_id,
            title=req.title,
            use_transformers=req.use_transformers,
            include_dialogue=req.include_dialogue,
        )
        pipeline.save_metadata(req.imdb_id or req.title, meta)
        return meta
    raise HTTPException(400, "provide either imdb_id or text")


@app.post("/tag/upload")
@app.post("/api/tag/upload")
async def tag_upload(file: UploadFile = File(...), use_transformers: bool = False):
    raw = (await file.read()).decode("utf-8", errors="replace")
    title = Path(file.filename).stem if file.filename else ""
    cached = pipeline.load_cached_metadata_by_title(title) if title else None
    if cached is not None and not use_transformers:
        return cached
    meta = pipeline.tag_script(
        raw,
        title=title,
        use_transformers=use_transformers,
        include_dialogue=False,
    )
    pipeline.save_metadata(title, meta)
    return meta


@app.get("/metadata/{imdb_id}")
@app.get("/api/metadata/{imdb_id}")
def metadata(imdb_id: str):
    cached = pipeline.load_cached_metadata(imdb_id)
    if cached:
        return cached
    try:
        text = corpus.read_script(imdb_id)
    except KeyError:
        raise HTTPException(404, "script not found")
    meta = pipeline.tag_script(
        text,
        imdb_id=imdb_id,
        title=corpus.metadata_for(imdb_id).get("title", ""),
        use_transformers=False,
    )
    pipeline.save_metadata(imdb_id, meta)
    return meta


from fastapi.staticfiles import StaticFiles

DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")