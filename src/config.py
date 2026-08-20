import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = DATA_DIR / "models"
INDEX_CSV = DATA_DIR / "corpus_index.csv"

POSSIBLE_ROOTS = [
    Path("D:/Downloads/archive"),
    Path("C:/Downloads/archive"),
    Path("/home/shaury/Desktop/machine learning/dataset/archive (3)"),
]

# Lets a one-off local run (e.g. training the genre classifier) point at
# wherever the raw Kaggle dataset happens to be extracted, without needing
# to hardcode a machine-specific path into this file.
_env_root = os.environ.get("DATASET_ROOT")
if _env_root:
    POSSIBLE_ROOTS.insert(0, Path(_env_root))

DATASET_ROOT = POSSIBLE_ROOTS[0]
for p in POSSIBLE_ROOTS:
    if p.exists() and (p / "movie_metadata/movie_meta_data.csv").exists():
        DATASET_ROOT = p
        break

RAW_TEXTS_DIR = DATASET_ROOT / "screenplay_data/data/raw_texts/raw_texts"
RULE_BASED_DIR = DATASET_ROOT / "screenplay_data/data/rule_based_annotations/rule_based_annotations"
MANUAL_ANNO_DIR = DATASET_ROOT / "screenplay_data/data/manual_annotations/manual_annotations"
BERT_ANNO_DIR = DATASET_ROOT / "screenplay_data/data/BERT_annotations/BERT_annotations"
CHARACTER_TEXTS_DIR = DATASET_ROOT / "movie_characters/data/movie_character_texts/movie_character_texts"
GENDERS_PICKLE = DATASET_ROOT / "movie_characters/data/character_genders.pickle"
META_CSV = DATASET_ROOT / "movie_metadata/movie_meta_data.csv"

for d in (DATA_DIR, OUTPUTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)