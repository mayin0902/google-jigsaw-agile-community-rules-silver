from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Paths:
    data_dir: Path = Path("/kaggle/input/jigsaw-agile-community-rules")
    output_dir: Path = Path("/kaggle/working")
    qwen_model: Path = Path("/kaggle/input/qwen-3/transformers/4b-base/1")
    gte_model: Path = Path("/kaggle/input/gte-base-en-v1-5")
    deberta_model: Path = Path("/kaggle/input/huggingfacedebertav3variants/deberta-v3-base")
    lora_dir: Path = Path("/kaggle/working/pseudo_lora")


@dataclass(slots=True)
class QwenConfig:
    seed: int = 22
    train_frac_from_test_examples: float = 0.05
    use_train_csv: bool = True
    max_model_len: int = 2048
    lora_rank: int = 32
    lora_alpha: int = 64
    learning_rate: float = 2e-4
    epochs: int = 1


@dataclass(slots=True)
class GteConfig:
    seed: int = 42
    max_seq_length: int = 256
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-5
    triplet_margin: float = 0.25
    augmentation_factor: int = 2


@dataclass(slots=True)
class DebertaConfig:
    seed: int = 42
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    label_smoothing: float = 0.10
    ema_decay: float = 0.99
    ema_start_ratio: float = 0.8
    fgm_alpha: float = 0.3
    fgm_epsilon: float = 1.0


@dataclass(slots=True)
class BlendConfig:
    deberta_weight: float = 0.5
    gte_weight: float = 0.3
    llm_weight: float = 0.7


DEFAULT_PATHS = Paths()
