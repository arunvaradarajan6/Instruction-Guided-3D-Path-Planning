"""Where everything lives -- the one module that knows the folder layout."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
FIGURES = ROOT / "figures"
DATASETS = ROOT / "datasets"
CACHE = DATASETS / "cache"
# Models and results trained on a non-paper corpus live in their own folders, so a v2 run
# can never overwrite the paper's v1 checkpoints. Caches are shared (file names differ).
_CORPUS = os.environ.get("IG3D_CORPUS", "v1").lower()
_SUFFIX = "" if _CORPUS == "v1" else "_" + _CORPUS
MODELS = DATASETS / ("models" + _SUFFIX)
RESULTS = DATASETS / ("results" + _SUFFIX)
SIM = DATASETS / "sim"
