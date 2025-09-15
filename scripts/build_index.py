#!/usr/bin/env python
from __future__ import annotations
import argparse, os, numpy as np
import torch
from PIL import Image

from xmrag.config import Config
from xmrag.utils import set_seed, load_jsonl
from xmrag.data import iter_rows, gather_corpus, load_image
from xmrag.embed import encode_text, encode_images
from xmrag.index import ModalIndex

def main(args):
    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = iter_rows(args.data, split=None)
    text_docs, image_docs = gather_corpus(rows)

    # TEXT
    text_strs = [d["text"] for d in text_docs]
    txt_embs = encode_text(text_strs, cfg.models.text_encoder, device=device, normalize=True)
    tindex = ModalIndex(dim=txt_embs.shape[1], modality="text")
    tindex.add(txt_embs.astype("float32"), text_docs)
    tindex.save(args.outdir)

    # IMAGE
    imgs = [load_image(d["path"]) for d in image_docs] if image_docs else []
    if imgs:
        img_embs = encode_images(imgs, cfg.models.image_encoder, device=device, normalize=True)
        iindex = ModalIndex(dim=img_embs.shape[1], modality="image")
        iindex.add(img_embs.astype("float32"), image_docs)
        iindex.save(args.outdir)

    print(f"[index] text={len(text_docs)} image={len(image_docs)} saved to {args.outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    main(ap.parse_args())
