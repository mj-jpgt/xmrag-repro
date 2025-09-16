
# XM-RAG: Cross‑Modal Retrieval‑Augmented Generation (Reproducible Base)

## Introduction
This repository contains a clean, modular base to **reproduce** and **extend** results for **cross‑modal, multi‑hop question answering** using **unified retrieval + generation**. The pipeline—**XM‑RAG**—builds **text** and **image** indices, retrieves top‑k evidence from both modalities, **fuses** them with a transparent weighting, optionally **refines** the evidence into a compact representation, and finally **generates** an answer with a seq2seq model.

This codebase is structured for **ICLR/GitHub reproducibility**: one‑command scripts, JSONL data schema, deterministic seeds, and per‑modality metrics.




---

## Datasets
XM‑RAG operates on a **unified JSONL schema** so you can plug in public datasets (e.g., **MultimodalQA**, **WebQA**) or your own (e.g., LR‑MMQA). Minimal requirements:

```json
{{
  "id": "unique-id",
  "question": "What ...?",
  "answers": ["gold answer", "aliases..."],
  "contexts": [
    {{"type": "text", "text": "passage text", "doc_id": "doc123"}},
    {{"type": "image", "path": "path/to/img.jpg", "caption": "optional caption", "doc_id": "doc123"}}
  ],
  "split": "train|val|test"
}}
```

- Only one of `text` or `path` is required per context.
- If you have gold supervision for retrieval, add `is_gold: true` in a context to enable retrieval metrics.

> **Tip:** For WebQA/MultimodalQA, convert their native formats to the schema above. Keep `doc_id` consistent across text and images from the same source.

---

## Model & Checkpoints
XM‑RAG uses pluggable encoders:
- **Text encoder** (default): `sentence-transformers/all-MiniLM-L6-v2`
- **Image encoder** (default): `openai/clip-vit-base-patch32`
- **Generator** (default): `t5-small`

Configure them in `config.yaml`:
```yaml
models:
  text_encoder: sentence-transformers/all-MiniLM-L6-v2
  image_encoder: openai/clip-vit-base-patch32
  generator: t5-small
retrieval:
  k: 5
  alpha: 0.6   # text vs image fusion weight
```

You can point `generator` to a local fine‑tuned checkpoint directory (e.g., `artifacts/refiner/`).

---

## Environment
- Python ≥ 3.10
- CPU‑only works out of the box (`faiss-cpu`); GPUs accelerate encoding/generation automatically.
- Install deps:
```bash
pip install -r requirements.txt
```

---

## Training procedure

### 0) Inspect sample & preprocess
```bash
python scripts/preprocess.py --in data/sample/sample.jsonl --out data/processed.jsonl
```
> Add your own cleaning, OCR, captioning, or translation logic in `scripts/preprocess.py` if needed.

### 1) Build retrieval indices (text + image)
```bash
python scripts/build_index.py --data data/processed.jsonl --outdir artifacts/index
```
- Builds FAISS (Inner‑Product) indices for each modality
- Stores `text.index`, `image.index` and their `*.meta.jsonl`

### 2) (Optional) Warm‑up / fine‑tune generator
```bash
python scripts/train_refiner.py --train data/processed.jsonl --outdir artifacts/refiner
```
- Baseline “warm‑up” scaffold that saves a generator checkpoint to `artifacts/refiner/`


### 3) Run XM‑RAG (retrieve → refine → generate)
```bash
python scripts/run_xmrag.py   --data data/processed.jsonl   --indexdir artifacts/index   --refiner artifacts/refiner   --out artifacts/preds.jsonl
```
- Uses the text encoder to embed questions and query both indices
- Fuses text/image results with `alpha` (configurable)
- Builds a compact prompt from top‑k contexts and generates answers

### 4) Evaluate
```bash
python scripts/evaluate.py --data data/processed.jsonl --preds artifacts/preds.jsonl --k 5
```
- Reports **EM**, **Token‑F1**, **R@k**, **MRR** per modality + **macro**

---

## Reproducing paper‑style experiments

1. **Datasets**: Prepare `train/val/test` splits in the JSONL schema. Ensure `doc_id` is stable per source and add `is_gold: true` where supervision exists.
2. **Encoders**: Set exact encoder names/checkpoints in `config.yaml` to match the paper.
3. **Retrieval knobs**: Use `retrieval.k` and `retrieval.alpha` for ablations (text‑only, image‑only, fused).
4. **Refinement**: Swap in your architecture in `xmrag/refiner.py` (e.g., cross‑attention over top‑k) while keeping the interfaces unchanged.
5. **Generation**: Point `models.generator` to your fine‑tuned checkpoint directory to reproduce final numbers.
6. **Logging**: Persist `artifacts/preds.jsonl` and `artifacts/index/` for auditability.

---

## Optional: Knowledge‑graph / Structured augmentation
If your setup uses entity/relation extraction or table linearization:
- Run extraction offline and attach additional contexts as `type: "text"` with `doc_id`s that map back to KG nodes/tables.
- XM‑RAG will index them like any other text source and co‑retrieve with images.
- Keep spans/alignments in `meta` if you want to analyze attribution later.

---

## Results & expected artifacts
- **Indices**: `artifacts/index/text.index`, `artifacts/index/image.index` + `*.meta.jsonl`
- **Checkpoints**: `artifacts/refiner/` (generator tokenizer + weights)
- **Predictions**: `artifacts/preds.jsonl`:
```json
{{ "id": "ex1",
   "answer": "a cat",
   "retrieval_modality": "mixed",
   "ranked_doc_ids": ["doc1","doc2","doc3","doc4","doc5"] }}
```
- **Metrics**: printed JSON with per‑modality + macro aggregates

---

## Common pitfalls
- **faiss‑cpu import**: If `faiss` fails to import, ensure you installed `faiss-cpu` from `requirements.txt` (or use a wheel compatible with your Python version).
- **Transformers cache**: If models fail to download (air‑gapped runs), pre‑download checkpoints and point `config.yaml` to local paths.
- **GPU vs CPU**: The code auto‑detects CUDA for encoders/generator. Mixed environments are fine but slower on CPU.

---

## Framework versions
- PyTorch ≥ 2.2
- Transformers ≥ 4.41
- sentence‑transformers ≥ 3.0
- FAISS (CPU) ≥ 1.8
- scikit‑learn ≥ 1.4

Exact versions are pinned in `requirements.txt` for reproducibility.


