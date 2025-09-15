#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from xmrag.utils import load_jsonl
from xmrag.metrics import per_modality_metrics

def main(args):
    data = load_jsonl(args.data)
    preds_rows = load_jsonl(args.preds)
    preds = {r["id"]: r for r in preds_rows}
    m = per_modality_metrics(data, preds, k=args.k)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--k", type=int, default=5)
    main(ap.parse_args())
