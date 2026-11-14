"""
tests/smoke_test.py – Automated checks for the email personalization model.
Exits with code 0 on full PASS, non-zero on any failure.
"""

import os
import sys
from pathlib import Path
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

PASS_COUNT = 0
FAIL_COUNT = 0
FAILURES   = []


def ok(msg: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {msg}")


def fail(msg: str, exc: Exception = None):
    global FAIL_COUNT
    FAIL_COUNT += 1
    info = f": {exc}" if exc else ""
    FAILURES.append(f"{msg}{info}")
    print(f"  [FAIL] {msg}{info}")


# ── 1. Models exist and load ──────────────────────────────────────────────────
print("\n── 1. Model files exist ──")
import joblib
import numpy as np
from scipy.sparse import load_npz

required_models = [
    "tfidf_vectorizer.joblib",
    "lsa_svd.joblib",
    "kmeans.joblib",
    "segment_profiles.json",
    "tfidf_matrix.npz",
    "email_vecs_w2v.npy",
    "embeddings.npz",
]
for fname in required_models:
    p = ROOT / "models" / fname
    if p.exists():
        ok(f"models/{fname} exists")
    else:
        fail(f"models/{fname} MISSING")

# ── 2. personalize() returns all keys for 5 different inputs ─────────────────
print("\n── 2. personalize() returns all required keys ──")
try:
    from personalize import personalize
    REQUIRED_KEYS = {"segment_id", "segment_label", "topics",
                     "tone", "retrieval_mode", "similar_emails",
                     "subject", "body"}

    test_inputs = [
        ("Alice",   "data science machine learning NLP", "follow_up",       "tfidf"),
        ("Bob",     "finance banking investment portfolio", "offer",          "tfidf"),
        ("Carol",   "marketing social media brand campaigns", "update",      "tfidf"),
        ("David",   "software engineering python backend API", "meeting_request", "tfidf"),
        ("Eve",     "healthcare medical clinical research", "introduction",   "tfidf"),
    ]
    for name, profile, purpose, retrieval in test_inputs:
        try:
            result = personalize(name, profile, purpose, retrieval)
            missing_keys = REQUIRED_KEYS - set(result.keys())
            if missing_keys:
                fail(f"personalize({name!r}) missing keys: {missing_keys}")
            elif not result["subject"]:
                fail(f"personalize({name!r}) empty subject")
            elif not result["body"]:
                fail(f"personalize({name!r}) empty body")
            else:
                ok(f"personalize({name!r}, {purpose}) → keys OK, subject={repr(result['subject'])[:40]}")
        except Exception as e:
            fail(f"personalize({name!r}) raised", e)
except Exception as e:
    fail("Could not import personalize", e)

# ── 3. Different profiles produce different emails ────────────────────────────
print("\n── 3. Different profiles produce different emails ──")
try:
    r1 = personalize("TechPerson", "machine learning data science artificial intelligence", "follow_up", "tfidf")
    r2 = personalize("FinancePerson", "banking investment portfolio management returns", "offer", "tfidf")
    if r1["body"] != r2["body"]:
        ok("Different profiles → different bodies")
    else:
        fail("Different profiles produced identical bodies")
    if r1["subject"] != r2["subject"]:
        ok("Different profiles → different subjects")
    else:
        fail("Different profiles produced identical subjects")
except Exception as e:
    fail("Diversity test raised", e)

# ── 4. Empty / very short input does not crash ────────────────────────────────
print("\n── 4. Empty input does not crash ──")
try:
    result = personalize("", "", "follow_up", "tfidf")
    ok(f"Empty input handled gracefully (body length={len(result['body'])})")
except Exception as e:
    fail("Empty input raised", e)

try:
    result = personalize("X", "hi", "offer", "tfidf")
    ok(f"Very short input handled gracefully")
except Exception as e:
    fail("Very short input raised", e)

# ── 5. Both retrieval modes work ──────────────────────────────────────────────
print("\n── 5. Both retrieval modes work ──")
for mode in ["tfidf", "word2vec", "random"]:
    try:
        result = personalize("TestUser", "software developer python machine learning", "follow_up", mode)
        ok(f"retrieval={mode!r} → subject={repr(result['subject'])[:40]}")
    except Exception as e:
        fail(f"retrieval={mode!r} raised", e)

# ── 6. app.py imports and builds interface without launching ──────────────────
print("\n── 6. app.py import and interface build ──")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", ROOT / "app.py")
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    assert hasattr(app_mod, "demo"), "app.py must define 'demo'"
    assert hasattr(app_mod, "build_interface"), "app.py must define 'build_interface'"
    ok("app.py imported and 'demo' object exists (server NOT launched)")
except Exception as e:
    fail("app.py import failed", e)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n══════════════════════════════════")
print(f"  PASS: {PASS_COUNT}   FAIL: {FAIL_COUNT}")
if FAILURES:
    print("\nFailure details:")
    for f in FAILURES:
        print(f"  ✗ {f}")
print(f"══════════════════════════════════\n")

sys.exit(0 if FAIL_COUNT == 0 else 1)
