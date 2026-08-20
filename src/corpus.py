import re
from pathlib import Path

import pandas as pd

from src.config import (
    BERT_ANNO_DIR,
    CHARACTER_TEXTS_DIR,
    INDEX_CSV,
    MANUAL_ANNO_DIR,
    META_CSV,
    OUTPUTS_DIR,
    RAW_TEXTS_DIR,
    RULE_BASED_DIR,
)

FILE_RE = re.compile(
    r"^(?P<title>.+?)_(?P<imdbid>\d{7,9})(?:_[A-Za-z0-9 _-]+)?\.(?:txt|json)$"
)


def _scan_dir(directory: Path, suffix: str):
    if not directory.exists():
        return {}
    out = {}
    for p in directory.iterdir():
        if p.name.endswith(suffix):
            m = FILE_RE.match(p.name)
            if m:
                out[m.group("imdbid")] = p
    return out


def _build_index_from_outputs() -> pd.DataFrame:
    """Lightweight catalog built from pre-tagged outputs/*.json.gz, used when the
    raw Kaggle screenplay corpus isn't available (e.g. cached-only deployments)."""
    import gzip
    import json

    rows = []
    for p in sorted(OUTPUTS_DIR.glob("*.json.gz")):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        imdbid = str(meta.get("imdb_id") or "").strip().zfill(7)
        if not imdbid or imdbid == "0000000":
            continue
        genres = meta.get("known_genres") or [
            g.get("genre") for g in meta.get("genres", []) if g.get("genre")
        ]
        rows.append(
            {
                "imdbid": imdbid,
                "title": meta.get("title", ""),
                "script_path": "",
                "genres": ", ".join(genres),
                "year": None,
                "has_rule_based": False,
                "has_bert_anno": False,
                "has_manual_anno": False,
                "has_characters": False,
                "rule_based_path": "",
                "bert_anno_path": "",
                "manual_anno_path": "",
                "characters_path": "",
            }
        )
    return pd.DataFrame(rows)


def build_index() -> pd.DataFrame:
    if not META_CSV.exists():
        idx = _build_index_from_outputs()
        idx.to_csv(INDEX_CSV, index=False)
        return idx

    raw = _scan_dir(RAW_TEXTS_DIR, ".txt")
    rule_based = _scan_dir(RULE_BASED_DIR, ".json")
    bert = _scan_dir(BERT_ANNO_DIR, ".txt")
    manual = _scan_dir(MANUAL_ANNO_DIR, ".txt")

    meta = pd.read_csv(META_CSV, dtype={"imdbid": str})
    meta["imdbid"] = meta["imdbid"].str.zfill(7)

    rows = []
    for imdbid, path in raw.items():
        char_dir = CHARACTER_TEXTS_DIR / path.stem
        rows.append(
            {
                "imdbid": imdbid,
                "title": path.stem.rsplit("_", 1)[0],
                "script_path": str(path),
                "has_rule_based": imdbid in rule_based,
                "has_bert_anno": imdbid in bert,
                "has_manual_anno": imdbid in manual,
                "has_characters": char_dir.exists(),
                "rule_based_path": str(rule_based[imdbid]) if imdbid in rule_based else "",
                "bert_anno_path": str(bert[imdbid]) if imdbid in bert else "",
                "manual_anno_path": str(manual[imdbid]) if imdbid in manual else "",
                "characters_path": str(char_dir) if char_dir.exists() else "",
            }
        )
    idx = pd.DataFrame(rows)
    idx = idx.merge(meta, on="imdbid", how="left")
    idx["title"] = idx["title_y"].fillna(idx["title_x"])
    idx = idx.drop(columns=["title_x", "title_y"])
    idx.to_csv(INDEX_CSV, index=False)
    return idx


def load_index() -> pd.DataFrame:
    if INDEX_CSV.exists():
        return pd.read_csv(INDEX_CSV, dtype={"imdbid": str})
    return build_index()


def script_path(imdbid: str) -> Path:
    imdbid = imdbid.zfill(7)
    row = load_index()
    row = row[row["imdbid"] == imdbid]
    raw_path = row.iloc[0].get("script_path") if not row.empty else None
    # An empty script_path written to the CSV index round-trips back as NaN
    # (float), not "" — and NaN is truthy in Python, so a plain falsy check
    # doesn't catch it. pd.isna() is required to detect both cases.
    if row.empty or pd.isna(raw_path) or not raw_path:
        raise KeyError(f"imdbid {imdbid} not in corpus")
    path = Path(raw_path)
    if not path.exists():
        # The index can outlive the filesystem layout it was built against
        # (e.g. a stale data/corpus_index.csv left over from a run with a
        # different DATASET_ROOT/mount) — treat that the same as "not in
        # corpus" rather than letting a raw FileNotFoundError surface.
        raise KeyError(f"imdbid {imdbid} script file missing: {path}")
    return path


def read_script(imdbid: str) -> str:
    return script_path(imdbid).read_text(encoding="utf-8", errors="replace")


def metadata_for(imdbid: str) -> dict:
    imdbid = imdbid.zfill(7)
    row = load_index()
    row = row[row["imdbid"] == imdbid]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def character_files(imdbid: str) -> list[tuple[str, str]]:
    """Return list of (character_name, dialogue_text) from the movie_characters split."""
    imdbid = imdbid.zfill(7)
    row = load_index()
    row = row[row["imdbid"] == imdbid]
    if row.empty or not row.iloc[0]["has_characters"]:
        return []
    d = Path(row.iloc[0]["characters_path"])
    out = []
    for p in sorted(d.glob("*.txt")):
        name = p.name[: -len("_text.txt")]
        out.append((name, p.read_text(encoding="utf-8", errors="replace")))
    return out


def rule_based_annotation(imdbid: str):
    """Return parsed rule-based annotation JSON for a movie (or None)."""
    import json

    imdbid = imdbid.zfill(7)
    row = load_index()
    row = row[row["imdbid"] == imdbid]
    if row.empty or not row.iloc[0]["has_rule_based"]:
        return None
    with open(row.iloc[0]["rule_based_path"], encoding="utf-8") as f:
        return json.load(f)


def line_label_annotations(imdbid: str):
    """Return (list of (label, data)) lines from BERT or manual annotations for training/eval."""
    imdbid = imdbid.zfill(7)
    row = load_index()
    row = row[row["imdbid"] == imdbid]
    if row.empty:
        return []
    r = row.iloc[0]
    path = r["bert_anno_path"] if isinstance(r["bert_anno_path"], str) and r["bert_anno_path"] else r["manual_anno_path"]
    if not isinstance(path, str) or not path:
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if ": " in line and line.split(": ", 1)[0] in {"dialog", "scene_heading", "speaker_heading", "text"}:
            label, data = line.split(": ", 1)
            out.append((label, data))
        else:
            out.append(("text", line))
    return out


if __name__ == "__main__":
    idx = build_index()
    print(f"Indexed {len(idx)} scripts")
    print(idx[["imdbid", "title", "has_rule_based", "has_characters"]].head(10))