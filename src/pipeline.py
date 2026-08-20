"""Orchestrates the full metadata-tagging pipeline for a script."""

import gzip
import json
import logging
from pathlib import Path

from src import classify, ner, parser, segmentation, sentiment, speakers, srt, topics
from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def tag_script(
    script_text: str,
    imdb_id: str = "",
    title: str = "",
    use_transformers: bool = False,
    include_dialogue: bool = False,
) -> dict:
    """Run the full pipeline on raw script text and return a metadata dict."""
    if srt.looks_like_srt(script_text):
        script_text = srt.srt_to_text(script_text)

    parsed = parser.parse_script(script_text, title=title, imdb_id=imdb_id)

    known_genres: list[str] = []
    if imdb_id:
        try:
            from src import corpus

            meta_row = corpus.metadata_for(imdb_id)
            raw_genres = meta_row.get("genres") or ""
            if isinstance(raw_genres, str) and raw_genres.strip():
                known_genres = [g.strip() for g in raw_genres.split(",") if g.strip()][:6]
        except Exception:
            known_genres = []

    if not parsed.scenes:
        # treat plain transcript (or converted subtitle text) as a single
        # scene, still picking up "SPEAKER: line" attribution where present
        parsed.scenes = [
            parser.Scene(index=0, heading="", interior=None, location=None, time_of_day=None)
        ]
        for line in script_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = parser.INLINE_SPEAKER_RE.match(line)
            if m:
                speaker = parser.normalize_speaker(m.group("speaker"))
                text = m.group("dialog").strip()
            else:
                speaker, text = None, line
            parsed.scenes[0].dialogue.append(parser.DialogueLine(speaker=speaker, text=text))

    segments = segmentation.assign_timestamps(parsed)
    scene_texts = [
        " ".join(d.text for d in s.dialogue) + " " + " ".join(s.action) for s in parsed.scenes
    ]

    # NER
    ner_extractor = ner.NERExtractor()
    ner_res = ner_extractor.extract(parsed)
    global_entities = ner.person_matches_speakers(ner_res["global_entities"], parsed)
    scene_entities = ner_res["scene_entities"]

    # Topics
    scene_topic_lists = topics.scene_topics(parsed, scene_texts, top_n=8)
    overall_topics = topics.overall_topics(parsed, top_n=25)

    # Sentiment (VADER) + optional transformer emotion per dialogue line
    dialogue_texts = [d.text for d in parsed.all_dialogue]
    line_sentiments = [sentiment.vader_sentiment(t) for t in dialogue_texts]
    if use_transformers:
        line_emotions = sentiment.transformer_emotion(dialogue_texts)
    else:
        line_emotions = [None] * len(dialogue_texts)

    # Build per-scene aggregates
    seg_meta = []
    di = 0
    for si, (seg, scene) in enumerate(zip(segments, parsed.scenes)):
        scene_sents = line_sentiments[di : di + len(scene.dialogue)]
        scene_emos = line_emotions[di : di + len(scene.dialogue)]
        di += len(scene.dialogue)
        speakers_in_scene = []
        for d in scene.dialogue:
            n = parser.normalize_speaker(d.speaker)
            if n and n not in speakers_in_scene:
                s = {"name": n, "lines": 1, "words": len(d.text.split()), "gender": None}
                if speakers._is_plausible_speaker(s, min_lines=0):
                    speakers_in_scene.append(n)
        dialogue_lines = []
        for j, d in enumerate(scene.dialogue):
            line = {
                "speaker": d.speaker,
                "text": d.text,
                "parenthetical": d.parenthetical,
                "sentiment": line_sentiments[di - len(scene.dialogue) + j]
                if scene_sents
                else None,
                "emotion": scene_emos[j] if scene_emos[j] else None,
            }
            dialogue_lines.append(line)
        seg_meta.append(
            {
                **seg,
                "heading": scene.heading,
                "interior": scene.interior,
                "location": scene.location,
                "time_of_day": scene.time_of_day,
                "speakers": speakers_in_scene,
                "topics": scene_topic_lists[si] if si < len(scene_topic_lists) else [],
                "entities": scene_entities.get(scene.index, []),
                "sentiment": sentiment.aggregate_sentiment(scene_sents),
                "emotion": sentiment.aggregate_emotion(scene_emos),
                "dialogue": dialogue_lines if include_dialogue else [],
            }
        )

    # Speaker stats
    speaker_stats = speakers.speaker_stats(parsed)

    # Content classification (genres)
    try:
        genre_text = " ".join(d.text for d in parsed.all_dialogue) + " " + " ".join(
            parsed.all_action
        )
        genres = classify.predict_genres(genre_text, top_n=5)
    except Exception as exc:
        logger.warning("Genre classification failed: %s", exc)
        genres = []

    meta = {
        "imdb_id": imdb_id,
        "title": parsed.title or title,
        "source": "script",
        "genres": genres,
        "known_genres": known_genres,
        "overall": {
            "topics": overall_topics,
            "entities": global_entities,
            "sentiment": sentiment.aggregate_sentiment(line_sentiments),
            "emotion": sentiment.aggregate_emotion(line_emotions),
            "num_scenes": len(parsed.scenes),
            "num_dialogue_lines": len(parsed.all_dialogue),
            "num_words": sum(len(t.split()) for t in dialogue_texts),
        },
        "segments": seg_meta,
        "speakers": speaker_stats,
    }
    return meta


def _clean_filename(title: str) -> str:
    if not title or not isinstance(title, str):
        return ""
    import re

    cleaned = re.sub(r"[^\w\-_ ]", "", title).strip().replace(" ", "_")
    return cleaned[:50]


def save_metadata(imdb_id: str, meta: dict, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    title = meta.get("title", "") if isinstance(meta, dict) else ""
    clean_t = _clean_filename(title)

    if clean_t and imdb_id:
        filename = f"{clean_t}_{imdb_id.zfill(7)}.json.gz"
    elif clean_t:
        filename = f"{clean_t}.json.gz"
    elif imdb_id:
        filename = f"{imdb_id.zfill(7)}.json.gz"
    else:
        filename = "script_metadata.json.gz"

    out = outputs_dir / filename
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))
    return out


def load_cached_metadata(imdb_id: str, outputs_dir: Path = OUTPUTS_DIR) -> dict | None:
    if not imdb_id:
        return None
    padded_id = imdb_id.zfill(7)
    matches = list(outputs_dir.glob(f"*{padded_id}*.json.gz")) + list(
        outputs_dir.glob(f"*{padded_id}*.json")
    )
    if matches:
        match_file = matches[0]
        if match_file.name.endswith(".gz"):
            with gzip.open(match_file, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            return json.loads(match_file.read_text(encoding="utf-8"))
    return None