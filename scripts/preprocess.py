#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from xmrag.utils import save_jsonl, load_jsonl

def main(args):
    rows = load_jsonl(args.in_)
    # Example pass-through; place to add cleaning, OCR, captioning, etc.
    # Also mark gold contexts if available (here we keep as-is).
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_jsonl(rows, args.out)
    print(f"[preprocess] wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
