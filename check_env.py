"""
check_env.py – Verify the Python environment is ready.
Prints ENV OK or lists what is missing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENV_PREFIX = str(ROOT.parent / ".venv")

print(f"Python version : {sys.version}")
print(f"sys.prefix     : {sys.prefix}")
print(f"Expected venv  : {VENV_PREFIX}")

# Check we're inside the venv
if not sys.prefix.startswith(VENV_PREFIX):
    print("WARNING: Not running inside the project .venv!")
else:
    print("Venv check     : OK (running inside .venv)")

missing = []
packages = {
    "numpy":       "numpy",
    "pandas":      "pandas",
    "sklearn":     "scikit-learn",
    "matplotlib":  "matplotlib",
    "joblib":      "joblib",
    "nltk":        "nltk",
    "gradio":      "gradio",
    "scipy":       "scipy",
}

for mod, pkg in packages.items():
    try:
        __import__(mod)
        print(f"  {pkg:20s}: OK")
    except ImportError:
        print(f"  {pkg:20s}: MISSING")
        missing.append(pkg)

# gensim optional
try:
    from gensim.models import Word2Vec
    print(f"  {'gensim':20s}: OK")
except ImportError:
    print(f"  {'gensim':20s}: not available (PPMI+SVD fallback will be used)")

if missing:
    print(f"\nENV MISSING: {missing}")
    sys.exit(1)
else:
    print("\nENV OK")
