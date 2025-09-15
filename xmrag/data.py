from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
import os
from PIL import Image
from .utils import load_jsonl

def iter_rows(path: str, split: str|None=None) -> Iterable[Dict[str, Any]]:
    for r in load_jsonl(path):
        if split is None or r.get("split") == split:
            yield r

def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def gather_corpus(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    '''Return (text_docs, image_docs) each item has keys: doc_id, text or path, meta'''
    text_docs, image_docs = [], []
    for r in rows:
        for c in r.get("contexts", []):
            if c.get("type") == "text" and c.get("text"):
                text_docs.append({"doc_id": c.get("doc_id"), "text": c.get("text"), "meta": c})
            elif c.get("type") == "image" and c.get("path"):
                image_docs.append({"doc_id": c.get("doc_id"), "path": c.get("path"), "caption": c.get("caption",""), "meta": c})
    return text_docs, image_docs
