from __future__ import annotations

import argparse
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from jigsaw_agile.config import DEFAULT_PATHS, DebertaConfig
from jigsaw_agile.data import flatten_examples_for_classifier, read_test
from jigsaw_agile.text import url_to_semantics


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, list[int]], labels: list[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


class ModelWithAttentionHead(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int = 2, label_smoothing: float = 0.10):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.output_hidden_states = True
        self.config.hidden_dropout_prob = 0.0
        self.config.return_dict = True
        self.config.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1),
            nn.Softmax(dim=1),
        )
        self.regressor = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]])

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if "deberta" in self.config.model_type:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < 2:
            raise ValueError("Backbone did not return enough hidden states.")

        last_layer = torch.mean(torch.stack([hidden_states[-1] * 2.2, hidden_states[-2]]), dim=0)
        weights = self.attention(last_layer)
        context_vector = torch.sum(weights * last_layer, dim=1)
        logits = torch.stack([self.regressor(drop(context_vector)) for drop in self.dropouts]).mean(dim=0)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits} if labels is not None else {"logits": logits}


class AdvancedTrainerMixin:
    """FGM-style adversarial training plus late-stage EMA weight swapping."""

    def __init__(
        self,
        *args,
        alpha: float = 0.3,
        epsilon: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_start_ratio: float = 0.8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.emb_name = "word_embeddings"
        self.backup = {}
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_start_ratio = ema_start_ratio
        self.ema_model = None
        self.backup_weights = None

    def _create_ema_model(self):
        self.ema_model = deepcopy(self.model)

    def _update_ema(self, model):
        if not self.use_ema or self.ema_model is None:
            return
        with torch.no_grad():
            for ema, param in zip(self.ema_model.parameters(), model.parameters()):
                ema.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    @contextmanager
    def _swap_ema_weights(self):
        if self.use_ema and self.ema_model is not None:
            self.backup_weights = {n: p.data.clone() for n, p in self.model.named_parameters()}
            ema_state = self.ema_model.state_dict()
            for n, p in self.model.named_parameters():
                p.data.copy_(ema_state[n])
            try:
                yield
            finally:
                for n, p in self.model.named_parameters():
                    p.data.copy_(self.backup_weights[n])
                self.backup_weights = None
        else:
            yield

    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        if self.use_ema:
            start_step = int(self.state.max_steps * self.ema_start_ratio)
            if self.state.global_step >= start_step:
                if self.ema_model is None:
                    self._create_ema_model()
                self._update_ema(model)
        return loss

    def evaluate(self, *args, **kwargs):
        with self._swap_ema_weights():
            return super().evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with self._swap_ema_weights():
            return super().predict(*args, **kwargs)

    def _save_restore_embeddings(self, restore: bool = False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if not restore:
                    self.backup[name] = param.data.clone()
                else:
                    param.data = self.backup[name]

    def _attack_embeddings(self, step_size: float):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                norm = torch.norm(param.grad)
                param.data.add_(step_size * param.grad / (norm + 1e-8))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        loss = loss.mean() if loss.ndim > 0 else loss
        if model.training:
            self.accelerator.backward(loss, retain_graph=True)
            self._save_restore_embeddings(restore=False)
            self._attack_embeddings(self.epsilon)
            for _ in range(2):
                with self.compute_loss_context_manager():
                    adv_out = model(**inputs)
                adv_loss = adv_out["loss"]
                adv_loss = adv_loss.mean() if adv_loss.ndim > 0 else adv_loss
                self.accelerator.backward(adv_loss)
                self._attack_embeddings(0.3)
            with self.compute_loss_context_manager():
                final_adv_out = model(**inputs)
            adv_loss = final_adv_out["loss"]
            adv_loss = adv_loss.mean() if adv_loss.ndim > 0 else adv_loss
            total_loss = loss + self.alpha * adv_loss
            self._save_restore_embeddings(restore=True)
            model.zero_grad()
        else:
            total_loss = loss
        return (total_loss, outputs) if return_outputs else total_loss


def build_trainer_class():
    from transformers import Trainer

    class AdvancedTrainer(AdvancedTrainerMixin, Trainer):
        pass

    return AdvancedTrainer


def make_input_text(df: pd.DataFrame) -> pd.Series:
    body_with_url = df["body"].apply(lambda x: str(x) + url_to_semantics(x))
    return df["rule"].astype(str) + "[SEP]" + body_with_url


def train_and_infer_deberta(
    data_dir: str | Path = DEFAULT_PATHS.data_dir,
    model_path: str | Path = DEFAULT_PATHS.deberta_model,
    output_path: str | Path = DEFAULT_PATHS.output_dir / "sub_deberta_v3.csv",
    output_dir: str | Path = DEFAULT_PATHS.output_dir / "save_deberta",
    config: DebertaConfig = DebertaConfig(),
) -> None:
    from transformers import AutoTokenizer, TrainingArguments

    seed_everything(config.seed)
    training_df = flatten_examples_for_classifier(data_dir)
    test_df = read_test(data_dir)
    if len(test_df) == 10:
        training_df = training_df.head(128)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    train_texts = make_input_text(training_df).tolist()
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=config.max_length,
    )
    train_dataset = JigsawDataset(train_encodings, training_df["rule_violation"].astype(int).tolist())
    model = ModelWithAttentionHead(
        str(model_path),
        num_labels=2,
        label_smoothing=config.label_smoothing,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="none",
        save_strategy="no",
        logging_steps=10,
    )
    trainer_cls = build_trainer_class()
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        alpha=config.fgm_alpha,
        epsilon=config.fgm_epsilon,
        ema_decay=config.ema_decay,
        ema_start_ratio=config.ema_start_ratio,
    )
    trainer.train()

    test_encodings = tokenizer(
        make_input_text(test_df).tolist(),
        truncation=True,
        padding=True,
        max_length=config.max_length,
    )
    predictions = trainer.predict(JigsawDataset(test_encodings))
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    submission = pd.DataFrame(
        {"row_id": test_df["row_id"].values, "rule_violation": probs}
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_PATHS.data_dir))
    parser.add_argument("--model", default=str(DEFAULT_PATHS.deberta_model))
    parser.add_argument("--output", default=str(DEFAULT_PATHS.output_dir / "sub_deberta_v3.csv"))
    parser.add_argument("--work-dir", default=str(DEFAULT_PATHS.output_dir / "save_deberta"))
    args = parser.parse_args()
    train_and_infer_deberta(args.data_dir, args.model, args.output, args.work_dir)
