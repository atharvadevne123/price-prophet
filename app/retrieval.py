import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.pkl")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss not installed — falling back to cosine similarity")


class ProductIndex:
    def __init__(self, n_features: int = 15):
        self.n_features = n_features
        self.index = None
        self.metadata: List[Dict] = []
        self._vectors: Optional[np.ndarray] = None

    def build(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        self.metadata = metadata
        normed = self._normalize(vectors)
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.n_features)
            self.index.add(normed.astype(np.float32))
        else:
            self._vectors = normed
        logger.info("Product index built with %d items", len(metadata))

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict]:
        if not self.metadata:
            return []
        normed_q = self._normalize(query.reshape(1, -1)).astype(np.float32)
        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(normed_q, min(k, len(self.metadata)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    item = dict(self.metadata[idx])
                    item["similarity"] = round(float(score), 4)
                    results.append(item)
            return results
        elif self._vectors is not None:
            sims = (self._vectors @ normed_q.T).flatten()
            top_k = np.argsort(sims)[::-1][:k]
            results = []
            for idx in top_k:
                item = dict(self.metadata[idx])
                item["similarity"] = round(float(sims[idx]), 4)
                results.append(item)
            return results
        return []

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return v / norms

    def save(self, path: str = FAISS_INDEX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"metadata": self.metadata, "vectors": self._vectors, "n_features": self.n_features}, f)

    def load(self, path: str = FAISS_INDEX_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.metadata = data["metadata"]
        self._vectors = data.get("vectors")
        self.n_features = data["n_features"]
        return True


_product_index: Optional[ProductIndex] = None


def get_index() -> ProductIndex:
    global _product_index
    if _product_index is None:
        _product_index = ProductIndex()
        if not _product_index.load():
            _seed_index()
    return _product_index


def _seed_index() -> None:
    from app.features import generate_synthetic_training_data, CATEGORY_MAP
    X, y = generate_synthetic_training_data(n_samples=500)
    categories = list(CATEGORY_MAP.keys())
    np.random.seed(99)
    meta = [
        {
            "product_id": f"PROD-{i:04d}",
            "category": categories[int(X[i, 7]) % len(categories)],
            "base_price": round(float(X[i, 0]), 2),
            "avg_demand": round(float(y[i]), 2),
        }
        for i in range(len(X))
    ]
    _product_index.build(X, meta)
    logger.info("Product index seeded with %d synthetic products", len(meta))
