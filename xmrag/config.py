from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class Paths:
    data: str = "data/processed.jsonl"
    index_dir: str = "artifacts/index"
    refiner_dir: str = "artifacts/refiner"
    preds: str = "artifacts/preds.jsonl"

@dataclass
class Models:
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_encoder: str = "openai/clip-vit-base-patch32"
    generator: str = "t5-small"

@dataclass
class Retrieval:
    k: int = 5
    alpha: float = 0.6
    normalize: bool = True
    use_pq: bool = False

@dataclass
class Train:
    batch_size: int = 8
    lr: float = 3e-4
    epochs: int = 2
    max_input_tokens: int = 512

@dataclass
class Config:
    seed: int = 42
    paths: Paths = Paths()
    models: Models = Models()
    retrieval: Retrieval = Retrieval()
    train: Train = Train()

    @staticmethod
    def from_yaml(path: Optional[str]) -> "Config":
        if not path:
            return Config()
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Manual load keeps clarity & defaults
        c = Config()
        if "seed" in data: c.seed = data["seed"]
        if "paths" in data:
            for k,v in data["paths"].items():
                setattr(c.paths, k, v)
        if "models" in data:
            for k,v in data["models"].items():
                setattr(c.models, k, v)
        if "retrieval" in data:
            for k,v in data["retrieval"].items():
                setattr(c.retrieval, k, v)
        if "train" in data:
            for k,v in data["train"].items():
                setattr(c.train, k, v)
        return c
