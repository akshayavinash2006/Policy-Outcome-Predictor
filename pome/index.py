"""Embedding and the vector store.

At 57 records an exact cosine scan over a NumPy matrix is both the fastest
and the most accurate search there is. A FAISS or Chroma index is a drop-in
replacement once the corpus grows to tens of thousands of records."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .data import file_hash, load_records


class Embedder:
    """Sentence-transformer embeddings, with TF-IDF as an offline fallback."""

    def __init__(self, backend: str = "auto", model_name: str = C.EMBED_MODEL):
        self.model_name = model_name
        self.backend = backend
        self._model = None
        self._tfidf = None
        if backend in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.backend = "sbert"
            except Exception as exc:  # broken torch, no network, ...
                if backend == "sbert":
                    raise
                print(f"[embed] sentence-transformers unavailable ({type(exc).__name__}); using TF-IDF")
                self.backend = "tfidf"

    def fit(self, texts: list[str]) -> "Embedder":
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(
                ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1
            ).fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sbert":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        vecs = self._tfidf.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    @property
    def label(self) -> str:
        return self.model_name if self.backend == "sbert" else "tfidf-1-2gram"


@dataclass
class VectorStore:
    df: pd.DataFrame
    vectors: np.ndarray          # (n, d) unit vectors of design_text
    embedder: Embedder
    threshold: float             # relevance gate on raw cosine

    def similarities(self, qvec: np.ndarray) -> np.ndarray:
        return self.vectors @ qvec

    def embed_query(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0]

    def pairwise_offdiag(self) -> np.ndarray:
        sims = self.vectors @ self.vectors.T
        return sims[~np.eye(len(sims), dtype=bool)]


def _threshold(vectors: np.ndarray, q: float) -> float:
    sims = vectors @ vectors.T
    return float(np.quantile(sims[~np.eye(len(sims), dtype=bool)], q))


def build_index(backend: str = "auto", save: bool = True) -> VectorStore:
    df = load_records()
    emb = Embedder(backend).fit(df["design_text"].tolist())
    vecs = emb.encode(df["design_text"].tolist())
    store = VectorStore(df, vecs, emb, _threshold(vecs, C.THRESHOLD_QUANTILE))
    if save:
        C.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(C.INDEX_DIR / "vectors.npy", vecs)
        meta = {
            "data_hash": file_hash(C.DATA_PATH),
            "backend": emb.backend,
            "model": emb.label,
            "threshold": store.threshold,
            "threshold_quantile": C.THRESHOLD_QUANTILE,
            "record_ids": df["record_id"].tolist(),
        }
        (C.INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
        if emb.backend == "tfidf":
            (C.INDEX_DIR / "tfidf.pkl").write_bytes(pickle.dumps(emb._tfidf))
    return store


def load_or_build_index(backend: str = "auto") -> VectorStore:
    """Reuse the saved index if it matches the current CSV; rebuild otherwise."""
    meta_path = C.INDEX_DIR / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        df = load_records()
        fresh = (
            meta.get("data_hash") == file_hash(C.DATA_PATH)
            and meta.get("record_ids") == df["record_id"].tolist()
            and (backend == "auto" or backend == meta.get("backend"))
        )
        if fresh:
            if meta["backend"] == "sbert":
                try:
                    emb = Embedder("sbert")
                except Exception:
                    return build_index("tfidf")
            else:
                emb = Embedder.__new__(Embedder)
                emb.model_name, emb.backend, emb._model = C.EMBED_MODEL, "tfidf", None
                emb._tfidf = pickle.loads((C.INDEX_DIR / "tfidf.pkl").read_bytes())
            vecs = np.load(C.INDEX_DIR / "vectors.npy")
            return VectorStore(df, vecs, emb, float(meta["threshold"]))
    return build_index(backend)
