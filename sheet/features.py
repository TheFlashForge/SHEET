"""Initial node features p_i (PLM lexical anchors).

The paper initialises every node with contextualised PLM features
(RoBERTa).  To keep this release runnable without GPU or model
downloads, two back-ends are provided:

  * ``"plm_cache"``  : load pre-computed PLM embeddings for the node
    strings from a word2vec-style text file (one line: ``<key> f1 f2 ...``).
    Generate it offline with any PLM (e.g. ``sentence-transformers`` /
    ``transformers`` mean-pooled RoBERTa) and pass it through unchanged.
  * ``"tfidf"``      : deterministic TF-IDF + SVD fallback (CPU, no
    downloads).  Useful for smoke tests and for reviewers to exercise
    the architecture end-to-end.

Numeric-value nodes additionally receive a min-max-normalised scalar
appended to their feature vector (paper Sec. CHTG).
"""

import os
from typing import Dict, List, Optional

import numpy as np


def try_parse_float(s: str) -> Optional[float]:
    try:
        v = float(s.replace(",", "").replace("$", "").strip())
        return v
    except (TypeError, ValueError):
        return None


def load_plm_cache(path: str) -> Dict[str, np.ndarray]:
    """Load word2vec-style cached features {node_string: vector}."""
    cache: Dict[str, np.ndarray] = {}
    if not os.path.exists(path):
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            cache[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return cache


class TfidfEncoder:
    """Lightweight deterministic fallback lexical encoder."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._fit = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            sublinear_tf=True, token_pattern=r"(?u)\b\w+\b", min_df=1
        )
        X = self.vec.fit_transform(texts)
        n_comp = max(2, min(self.dim, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
        Z = self.svd.fit_transform(X)
        self._fit = True
        return Z.astype(np.float32)

    def transform(self, texts: List[str]) -> np.ndarray:
        assert self._fit, "call fit_transform first"
        return self.svd.transform(self.vec.transform(texts)).astype(np.float32)


def build_text_features(
    strings: List[str],
    plm_cache: Optional[Dict[str, np.ndarray]] = None,
    fallback_dim: int = 256,
) -> np.ndarray:
    """Encode a list of node strings into [N, D].

    PLM cache entries are used when available; missing strings fall back
    to the TF-IDF encoder fitted on the corpus (zero rows if unseen).
    """
    if plm_cache:
        dims = {v.shape[0] for v in plm_cache.values()}
        assert len(dims) == 1, "PLM cache vectors must share one dimension"
        dim = dims.pop()
        out = np.zeros((len(strings), dim), dtype=np.float32)
        missing = [i for i, s in enumerate(strings) if s not in plm_cache]
        for i, s in enumerate(strings):
            if s in plm_cache:
                out[i] = plm_cache[s]
        if missing:  # fit fallback on the missing subset, scale to cache dim
            enc = TfidfEncoder(dim=dim)
            corpus = [strings[i] for i in missing]
            Z = enc.fit_transform(corpus) if len(set(corpus)) > 1 else np.zeros(
                (len(corpus), dim), dtype=np.float32
            )
            if Z.shape[1] < dim:
                Z = np.pad(Z, ((0, 0), (0, dim - Z.shape[1])))
            for j, i in enumerate(missing):
                out[i] = Z[j]
        return out

    enc = TfidfEncoder(dim=fallback_dim)
    if len(set(strings)) <= 1:
        return np.zeros((len(strings), fallback_dim), dtype=np.float32)
    return enc.fit_transform(strings)
