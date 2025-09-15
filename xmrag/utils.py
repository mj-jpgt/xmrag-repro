from __future__ import annotations
import json, os, math, random, logging
from typing import Any, Dict, Iterable, List
import numpy as np

logger = logging.getLogger("xmrag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def set_seed(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom

def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T

def flatten_text_from_context(ctx: Dict[str, Any]) -> str:
    if ctx.get("type") == "text":
        return ctx.get("text", "")
    if ctx.get("type") == "image":
        # Prefer caption if present
        return ctx.get("caption") or ""
    return ""

def first_answer_lower(answers: List[str]) -> str:
    return (answers[0] if answers else "").strip().lower()
