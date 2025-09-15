from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc

def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())

def exact_match(pred: str, golds: List[str]) -> float:
    p = normalize(pred)
    for g in golds:
        if p == normalize(g):
            return 1.0
    return 0.0

def token_f1(pred: str, golds: List[str]) -> float:
    p = set(normalize(pred).split())
    best = 0.0
    for g in golds:
        gset = set(normalize(g).split())
        if len(p)==0 and len(gset)==0:
            best = max(best, 1.0)
        elif len(p)==0 or len(gset)==0:
            best = max(best, 0.0)
        else:
            inter = len(p & gset); prec = inter / max(1, len(p)); rec = inter / max(1, len(gset))
            f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
            best = max(best, f1)
    return best

def recall_at_k(gold_doc_ids: List[str], ranked_doc_ids: List[str], k: int=5) -> float:
    S = set(gold_doc_ids)
    top = set(ranked_doc_ids[:k])
    return 1.0 if len(S & top) > 0 else 0.0

def mean_reciprocal_rank(gold_doc_ids: List[str], ranked_doc_ids: List[str]) -> float:
    S = set(gold_doc_ids)
    for i, d in enumerate(ranked_doc_ids, 1):
        if d in S:
            return 1.0 / i
    return 0.0

def per_modality_metrics(rows: List[Dict[str,Any]], preds: Dict[str,Any], k: int=5) -> Dict[str, Any]:
    buckets = defaultdict(list)
    for r in rows:
        pid = r["id"]
        info = preds.get(pid, {})
        mod = info.get("retrieval_modality", "mixed")
        em = exact_match(info.get("answer",""), r.get("answers", []))
        f1 = token_f1(info.get("answer",""), r.get("answers", []))

        ranked = info.get("ranked_doc_ids", [])
        gold = [c.get("doc_id") for c in r.get("contexts", []) if c.get("is_gold")]
        r1 = recall_at_k(gold, ranked, k=k) if gold else 0.0
        mrr = mean_reciprocal_rank(gold, ranked) if gold else 0.0

        buckets[mod].append({"EM":em, "F1":f1, f"R@{k}":r1, "MRR":mrr})

    out = {}
    for mod, L in buckets.items():
        if not L: continue
        out[mod] = {k: float(np.mean([x[k] for x in L])) for k in L[0].keys()}
    # macro
    if out:
        keys = list(next(iter(out.values())).keys())
        macro = {k: float(np.mean([v[k] for v in out.values()])) for k in keys}
        out["macro"] = macro
    return out
