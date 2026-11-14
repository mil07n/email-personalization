"""
embeddings.py – Word-vector embeddings with a backend switch.

backend = "gensim"   → Word2Vec (gensim)
backend = "ppmi_svd" → Co-occurrence matrix → PPMI → TruncatedSVD (scikit-learn)
                        ("count-based word embeddings (PPMI + SVD),
                          the matrix-factorization counterpart of word2vec")

The rest of the project only calls get_word_vectors() and email_vector().
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

import config

log = logging.getLogger(__name__)

# ── Try gensim ────────────────────────────────────────────────────────────────
try:
    from gensim.models import Word2Vec as _GensimW2V
    _GENSIM_OK = True
except Exception as _e:
    log.warning("gensim not available (%s). Will use PPMI+SVD fallback.", _e)
    _GENSIM_OK = False

# module-level state
_backend: str = "unset"
_word_vectors: Optional[dict] = None   # word -> np.ndarray
_gensim_model = None                   # gensim model object


# ── Public API ────────────────────────────────────────────────────────────────

def get_word_vectors() -> dict:
    """Return {word: vector_array} for the loaded model."""
    assert _word_vectors is not None, "Call train() or load() first."
    return _word_vectors


def email_vector(tokens: list) -> np.ndarray:
    """Return the average word vector for a list of tokens (email representation)."""
    if _word_vectors is None:
        return np.zeros(config.W2V_SIZE)
    vecs = [_word_vectors[t] for t in tokens if t in _word_vectors]
    if not vecs:
        return np.zeros(config.W2V_SIZE)
    return np.mean(vecs, axis=0)


def train(token_lists: list[list[str]]) -> str:
    """
    Train word embeddings on token_lists.
    Returns the backend name actually used ("gensim" or "ppmi_svd").
    """
    global _backend, _word_vectors, _gensim_model

    if _GENSIM_OK:
        log.info("Training Word2Vec (gensim) …")
        model = _GensimW2V(
            sentences=token_lists,
            vector_size=config.W2V_SIZE,
            window=config.W2V_WINDOW,
            min_count=config.W2V_MIN_COUNT,
            seed=config.RANDOM_SEED,
            workers=1,   # reproducible
        )
        _gensim_model = model
        _word_vectors = {w: model.wv[w] for w in model.wv.key_to_index}
        _backend = "gensim"
    else:
        log.info("Training PPMI+SVD embeddings (word2vec-style fallback) …")
        _word_vectors, _ = _train_ppmi_svd(token_lists, config.W2V_SIZE, config.W2V_WINDOW)
        _backend = "ppmi_svd"

    log.info("Embeddings backend: %s  |  vocab size: %d", _backend, len(_word_vectors))
    return _backend


def save(path: Path):
    """Save word vectors to disk as a .npz file (numpy archive)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    words = list(_word_vectors.keys())
    vecs  = np.array([_word_vectors[w] for w in words])
    np.savez(path, words=words, vecs=vecs, backend=np.array([_backend]))
    if _gensim_model is not None:
        _gensim_model.save(str(path.with_suffix(".gensim")))
    log.info("Saved embeddings to %s", path)


def load(path: Path):
    """Load word vectors from a .npz file saved by save()."""
    global _backend, _word_vectors, _gensim_model
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    words = data["words"].tolist()
    vecs  = data["vecs"]
    _backend = str(data["backend"][0])
    _word_vectors = dict(zip(words, vecs))
    # Try loading gensim model too (optional)
    gensim_path = path.with_suffix(".gensim")
    if _GENSIM_OK and gensim_path.exists():
        try:
            _gensim_model = _GensimW2V.load(str(gensim_path))
        except Exception:
            pass
    log.info("Loaded embeddings from %s  backend=%s  vocab=%d", path, _backend, len(_word_vectors))


# ── PPMI + SVD ────────────────────────────────────────────────────────────────

def _train_ppmi_svd(token_lists: list[list[str]],
                    n_components: int,
                    window: int) -> tuple[dict, object]:
    """
    Count-based word embeddings (PPMI + SVD):
      1. Build word-word co-occurrence matrix (window size = W2V_WINDOW)
      2. Convert counts to PPMI (Positive Pointwise Mutual Information)
      3. Reduce dimensionality with TruncatedSVD to n_components
    This is the matrix-factorization counterpart of word2vec.
    """
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import lil_matrix

    # --- vocabulary ---------------------------------------------------------
    freq: dict = defaultdict(int)
    for tokens in token_lists:
        for t in tokens:
            freq[t] += 1

    min_count = config.W2V_MIN_COUNT
    vocab = sorted(w for w, c in freq.items() if c >= min_count)
    w2i  = {w: i for i, w in enumerate(vocab)}
    V    = len(vocab)
    if V < 2:
        log.warning("Vocabulary too small for PPMI; returning zero vectors")
        return {w: np.zeros(n_components) for w in vocab}, None

    # Clamp components to vocab size
    n_components = min(n_components, V - 1)

    log.info("PPMI+SVD: vocab=%d  components=%d  building co-occurrence …", V, n_components)

    # --- co-occurrence matrix -----------------------------------------------
    cooc = lil_matrix((V, V), dtype=np.float32)
    for tokens in token_lists:
        idxs = [w2i[t] for t in tokens if t in w2i]
        for pos, i in enumerate(idxs):
            start = max(0, pos - window)
            end   = min(len(idxs), pos + window + 1)
            for j_pos in range(start, end):
                if j_pos != pos:
                    cooc[i, idxs[j_pos]] += 1.0

    cooc = cooc.tocsr()

    # --- PPMI ---------------------------------------------------------------
    total     = cooc.sum()
    row_sums  = np.array(cooc.sum(axis=1)).flatten()
    col_sums  = np.array(cooc.sum(axis=0)).flatten()

    # We compute PPMI only for non-zero entries (sparse-friendly)
    from scipy.sparse import csr_matrix
    cx    = cooc.tocoo()
    pmi_d = []
    for i, j, v in zip(cx.row, cx.col, cx.data):
        denom = (row_sums[i] * col_sums[j])
        if denom > 0:
            pmi = np.log2((v * total) / denom + 1e-10)
            pmi_d.append((i, j, max(pmi, 0.0)))  # PPMI: max(PMI, 0)

    if not pmi_d:
        log.warning("PPMI matrix is empty; returning zero vectors")
        return {w: np.zeros(n_components) for w in vocab}, None

    rows_a, cols_a, vals_a = zip(*pmi_d)
    ppmi = csr_matrix((vals_a, (rows_a, cols_a)), shape=(V, V), dtype=np.float32)

    # --- TruncatedSVD (= LSA on the PPMI matrix) ----------------------------
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    W   = svd.fit_transform(ppmi)   # shape (V, n_components)

    word_vecs = {vocab[i]: W[i] for i in range(V)}
    return word_vecs, svd
