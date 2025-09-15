from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

def fuse_scores(text_scores: np.ndarray, image_scores: np.ndarray, alpha: float=0.6) -> np.ndarray:
    """Weighted fusion: alpha * text + (1-alpha) * image, shape [Q, K] each"""
    if text_scores is None and image_scores is None:
        raise ValueError("At least one modality scores must be provided")
    if text_scores is None:
        return image_scores
    if image_scores is None:
        return text_scores
    return alpha * text_scores + (1.0 - alpha) * image_scores

def topk_from_index(index_items: List[Dict[str, Any]], I: np.ndarray, D: np.ndarray) -> List[List[Dict[str, Any]]]:
    """Turn FAISS results into per-query ranked lists with scores"""
    out: List[List[Dict[str, Any]]] = []
    for qi in range(I.shape[0]):
        hits = []
        for rank, (idx, score) in enumerate(zip(I[qi], D[qi])):
            if idx < 0: 
                continue
            meta = index_items[idx].copy()
            meta.update({"rank": int(rank), "score": float(score)})
            hits.append(meta)
        out.append(hits)
    return out

def rerank_union(text_hits: List[List[Dict[str,Any]]] | None,
                 image_hits: List[List[Dict[str,Any]]] | None,
                 alpha: float=0.6,
                 k: int=5) -> List[List[Dict[str,Any]]]:
    """Combine two ranked lists into a fused top-k using weighted scores per doc_id."""
    if text_hits is None: return [h[:k] for h in image_hits]
    if image_hits is None: return [h[:k] for h in text_hits]

    fused = []
    for th, ih in zip(text_hits, image_hits):
        scores: Dict[str,float] = {}
        keep: Dict[str,Dict[str,Any]] = {}
        for h in th:
            did = h.get("doc_id", f"text:{h.get('rank')}")
            scores[did] = scores.get(did, 0.0) + alpha * h["score"]
            keep.setdefault(did, h)
        for h in ih:
            did = h.get("doc_id", f"image:{h.get('rank')}")
            scores[did] = scores.get(did, 0.0) + (1.0 - alpha) * h["score"]
            keep.setdefault(did, h)
        # sort by fused score
        order = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        fused_hits = []
        for did, sc in order:
            meta = keep[did].copy()
            meta["score"] = float(sc)
            fused_hits.append(meta)
        fused.append(fused_hits)
    return fused
