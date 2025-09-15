#!/usr/bin/env python
from __future__ import annotations
import argparse, os, numpy as np, json
import torch
from xmrag.config import Config
from xmrag.utils import set_seed, load_jsonl, save_jsonl
from xmrag.embed import encode_text
from xmrag.index import ModalIndex
from xmrag.retriever import topk_from_index, rerank_union
from xmrag.refiner import RefinementEncoder, GeneratorWrapper
from xmrag.metrics import per_modality_metrics

def main(args):
    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_jsonl(args.data)
    q_texts = [r["question"] for r in rows]
    q_vecs = encode_text(q_texts, cfg.models.text_encoder, device=device, normalize=True)

    # Load indices
    tindex = ModalIndex.load(args.indexdir, "text")
    try:
        iindex = ModalIndex.load(args.indexdir, "image")
    except Exception:
        iindex = None

    # Search
    tk = cfg.retrieval.k
    Dt, It = tindex.search(q_vecs, tk)
    thits = topk_from_index(tindex.items, It, Dt)

    if iindex:
        Di, Ii = iindex.search(q_vecs, tk)   # NOTE: we use question text to query both, a common baseline
        ihits = topk_from_index(iindex.items, Ii, Di)
    else:
        ihits = None

    fused = rerank_union(thits, ihits, alpha=cfg.retrieval.alpha, k=tk)

    # Load refiner/generator
    refiner = RefinementEncoder(device=device)
    refdir = args.refiner if args.refiner else cfg.paths.refiner_dir
    if os.path.isdir(refdir):
        gen = GeneratorWrapper(model_name=refdir, device=device)
    else:
        gen = GeneratorWrapper(cfg.models.generator, device=device)

    preds = {}
    for r, hits in zip(rows, fused):
        ans = gen.generate(prompt=f"Q: {r['question']}\n" + "\n".join([h.get('text') or h.get('caption','') for h in hits]) + "\nA:")
        preds[r["id"]] = {
            "answer": ans,
            "retrieval_modality": "mixed" if iindex else "text",
            "ranked_doc_ids": [h.get("doc_id") for h in hits]
        }

    save_jsonl([{"id": k, **v} for k,v in preds.items()], args.out)
    print(f"[xmrag] saved predictions to {args.out}")

    # (Optional) quick metrics if gold doc ids annotated
    try:
        m = per_modality_metrics(rows, preds, k=tk)
        print("[metrics]", json.dumps(m, indent=2))
    except Exception as e:
        print(f"[metrics] skipped: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--indexdir", required=True)
    ap.add_argument("--refiner", default=None)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
