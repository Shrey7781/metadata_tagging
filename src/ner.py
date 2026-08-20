"""Named Entity Recognition with spaCy, aggregated per scene and globally."""

import logging
import os
from functools import lru_cache

from src.parser import ParsedScript, normalize_speaker

logger = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_lg")

KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "WORK_OF_ART", "EVENT", "LAW"}
LABEL_ALIASES = {
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "ORG": "ORGANIZATION",
    "LAW": "ORGANIZATION",
    "WORK_OF_ART": "PRODUCT",
}


@lru_cache(maxsize=1)
def load_spacy(model: str = DEFAULT_SPACY_MODEL):
    try:
        import spacy

        try:
            nlp = spacy.load(model)
        except OSError:
            logger.info("Downloading spaCy model %s ...", model)
            spacy.cli.download(model)
            nlp = spacy.load(model)
        return nlp
    except Exception as exc:  # pragma: no cover
        logger.warning("spaCy unavailable: %s", exc)
        return None


def _canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


class NERExtractor:
    def __init__(self, model: str = DEFAULT_SPACY_MODEL):
        self.nlp = load_spacy(model)

    def _doc_for(self, texts: list[str]):
        chunks = [t for t in texts if t and len(t.strip()) > 1]
        if not chunks or self.nlp is None:
            return []
        # spaCy batch to reduce overhead
        return [doc for doc in self.nlp.pipe(chunks, batch_size=64)]

    def extract(self, script: ParsedScript) -> dict:
        """Return {global_entities: [...], scene_entities: {scene_index: [...]}}."""
        docs_by_scene = {}
        for scene in script.scenes:
            texts = [d.text for d in scene.dialogue] + scene.action
            docs_by_scene[scene.index] = self._doc_for(texts)

        global_agg: dict[str, dict] = {}
        scene_out: dict[int, list] = {}

        for scene in script.scenes:
            ents = {}
            for doc in docs_by_scene[scene.index]:
                for ent in doc.ents:
                    if ent.label_ not in KEEP_LABELS:
                        continue
                    key = (_canonical_label(ent.label_), _key(ent.text))
                    prev = ents.get(key)
                    if prev:
                        prev["count"] += 1
                    else:
                        ents[key] = {
                            "label": _canonical_label(ent.label_),
                            "text": _clean(ent.text),
                            "count": 1,
                        }
            scene_entities = sorted(ents.values(), key=lambda e: -e["count"])[:20]
            scene_out[scene.index] = scene_entities
            for e in scene_entities:
                gkey = (e["label"], _key(e["text"]))
                g = global_agg.get(gkey)
                if g:
                    g["count"] += e["count"]
                    g["scenes"].append(scene.index)
                else:
                    global_agg[gkey] = {**e, "scenes": [scene.index]}

        global_entities = []
        for g in global_agg.values():
            g["scenes"] = sorted(set(g["scenes"]))
            global_entities.append(g)
        global_entities.sort(key=lambda e: -e["count"])
        return {"global_entities": global_entities, "scene_entities": scene_out}


def _key(s: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]", "", s.lower())


def _clean(s: str) -> str:
    return " ".join(s.split())


def person_matches_speakers(global_entities: list[dict], script: ParsedScript) -> list[dict]:
    """Annotate PERSON entities that correspond to dialogue speakers."""
    speakers = set()
    for d in script.all_dialogue:
        n = normalize_speaker(d.speaker)
        if n:
            speakers.add(n.lower())
    for e in global_entities:
        e["is_speaker"] = e["text"].lower() in speakers or any(
            e["text"].lower() in s for s in speakers
        )
    return global_entities