"""
config.py – Central configuration: all paths, column names, and constants.
All paths use pathlib so they work on Windows, Mac, and Linux.
"""

from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()

# ── Raw data ─────────────────────────────────────────────────────────────────
AESLC_ROOT = ROOT / "AESLC-master" / "enron_subject_line"
RAW_CSV = AESLC_ROOT

# ── Derived data ─────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
AESLC_CSV = DATA_DIR / "aeslc.csv"

# ── Models ───────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"

# ── Outputs ──────────────────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT / "outputs"

# ── Training constants ────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TFIDF_MAX_FEAT   = 5_000
LSA_COMPONENTS   = 100
KMEANS_K_RANGE = range(5, 13)  # k = 5..12
W2V_SIZE         = 100
W2V_WINDOW       = 5
W2V_MIN_COUNT    = 3
