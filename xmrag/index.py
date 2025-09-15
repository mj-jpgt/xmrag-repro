from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os, json
import numpy as np
import faiss

from .utils import l2_normalize_rows

def _build_flat_ip(d: int) -> faiss.Index:
    index = faiss.IndexFlatIP(d)
    return index

class ModalIndex:
    def __init__(self, dim: int, modality: str):
        self.dim = dim
        self.modality = modality
        self.index = _build_flat_ip(dim)
        self.items: List[Dict[str, Any]] = []

    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
        assert vecs.shape[1] == self.dim
        self.index.add(vecs.astype("float32"))
        self.items.extend(metas)

    def search(self, vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.index.search(vecs.astype("float32"), k)

    def save(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(outdir, f"{self.modality}.index"))
        with open(os.path.join(outdir, f"{self.modality}.meta.jsonl"), "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it) + "\n")

    @staticmethod
    def load(outdir: str, modality: str) -> "ModalIndex":
        idx_path = os.path.join(outdir, f"{modality}.index")
        meta_path = os.path.join(outdir, f"{modality}.meta.jsonl")
        index = faiss.read_index(idx_path)
        items = [json.loads(x) for x in open(meta_path, "r", encoding="utf-8")]
        mi = ModalIndex(index.d, modality)
        mi.index = index
        mi.items = items
        return mi
