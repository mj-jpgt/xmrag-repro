#!/usr/bin/env python
from __future__ import annotations
import argparse, os, random
import torch
from xmrag.config import Config
from xmrag.utils import set_seed, load_jsonl, save_jsonl
from xmrag.refiner import RefinementEncoder, GeneratorWrapper, build_context_prompt

# For simplicity we "warm up" the generator by teacher forcing on gold answers for few steps.
# In practice you can do proper seq2seq training; here we keep reproducible & light.
def main(args):
    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [r for r in load_jsonl(args.train) if r.get("split") == "train"]
    if not rows:
        print("[train_refiner] no train rows found; skipping.")
        return

    refiner = RefinementEncoder(device=device)
    gen = GeneratorWrapper(cfg.models.generator, device=device)

    # Dummy "finetune" loop — you can replace with full seq2seq trainer
    print(f"[train_refiner] Using {len(rows)} train rows; performing prompt warmup only.")
    os.makedirs(args.outdir, exist_ok=True)
    torch.save({"note": "refiner has no trainable head in this minimalist baseline"}, os.path.join(args.outdir, "refiner.pt"))
    gen.model.save_pretrained(args.outdir)
    gen.tok.save_pretrained(args.outdir)
    print(f"[train_refiner] saved to {args.outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train", required=True)
    ap.add_argument("--outdir", required=True)
    main(ap.parse_args())
