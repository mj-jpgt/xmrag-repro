from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image

import torch
from transformers import (
    AutoTokenizer, AutoModel,
    CLIPModel, CLIPProcessor
)

from sentence_transformers import SentenceTransformer
from .utils import l2_normalize_rows

@torch.no_grad()
def encode_text(texts: List[str], model_name: str, device: str="cpu", normalize: bool=True) -> np.ndarray:
    # Fast path: sentence-transformers
    if "sentence-transformers" in model_name or model_name.startswith("gte-") or "e5" in model_name:
        st = SentenceTransformer(model_name, device=device)
        embs = st.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        return l2_normalize_rows(embs) if normalize else embs

    # Generic HF encoder (mean pooling)
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    all_out = []
    bs = 16
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        x = tok(chunk, padding=True, truncation=True, return_tensors="pt").to(device)
        out = enc(**x)
        # mean-pool last hidden
        last = out.last_hidden_state
        mask = x.attention_mask.unsqueeze(-1)
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
        all_out.append(pooled.cpu().numpy())
    embs = np.vstack(all_out)
    return l2_normalize_rows(embs) if normalize else embs

@torch.no_grad()
def encode_images(imgs: List[Image.Image], model_name: str, device: str="cpu", normalize: bool=True) -> np.ndarray:
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    all_out = []
    bs = 16
    for i in range(0, len(imgs), bs):
        batch = imgs[i:i+bs]
        inputs = proc(images=batch, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        all_out.append(out.cpu().numpy())
    embs = np.vstack(all_out)
    return l2_normalize_rows(embs) if normalize else embs
