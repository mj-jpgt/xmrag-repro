from __future__ import annotations
from typing import List, Dict, Any
import torch, math
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

def build_context_prompt(question: str, hits: List[Dict[str,Any]], max_passages: int=5) -> str:
    blocks = [f"Q: {question}"]
    for i, h in enumerate(hits[:max_passages], 1):
        # Use text or caption fallback
        ctx = h.get("text") or h.get("caption") or ""
        if ctx:
            blocks.append(f"[{i}] {ctx}")
    blocks.append("A:")
    return "\n".join(blocks)

class RefinementEncoder(torch.nn.Module):
    """Small text encoder to compress multiple passages into a short latent (mean pooled)."""
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str="cpu"):
        super().__init__()
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(device)

    @torch.no_grad()
    def encode_passages(self, passages: List[str], max_tokens: int=256) -> torch.Tensor:
        x = self.tok(passages, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(self.device)
        out = self.enc(**x).last_hidden_state
        mask = x.attention_mask.unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        # mean over passages
        return pooled.mean(0, keepdim=True)

class GeneratorWrapper:
    def __init__(self, model_name: str="t5-small", device: str="cpu"):
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tok = T5Tokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int=64) -> str:
        x = self.tok([prompt], return_tensors="pt").to(self.device)
        y = self.model.generate(**x, max_new_tokens=max_new_tokens)
        out = self.tok.batch_decode(y, skip_special_tokens=True)[0]
        return out.strip()

def refine_and_generate(question: str,
                        fused_hits: List[Dict[str,Any]],
                        refiner: RefinementEncoder,
                        generator: GeneratorWrapper,
                        max_passages: int=5) -> str:
    # Build prompt from top passages
    prompt = build_context_prompt(question, fused_hits, max_passages=max_passages)
    # (Optionally) Could inject latent vector into prompt; for simplicity, we pass text only.
    return generator.generate(prompt)
