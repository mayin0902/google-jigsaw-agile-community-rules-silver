from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from jigsaw_agile.config import DEFAULT_PATHS, GteConfig
from jigsaw_agile.data import read_test
from jigsaw_agile.text import clean_url_to_domain_path


def collect_all_texts(test_df: pd.DataFrame) -> list[str]:
    texts: set[str] = set()
    for body in test_df["body"]:
        if pd.notna(body):
            texts.add(clean_url_to_domain_path(body))
    for col in [
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
    ]:
        for example in test_df[col]:
            if pd.notna(example):
                texts.add(clean_url_to_domain_path(example))
    return list(texts)


def create_triplet_dataset(
    test_df: pd.DataFrame,
    augmentation_factor: int = 2,
    random_seed: int = 42,
    subsample_fraction: float = 1.0,
):
    from datasets import Dataset

    random.seed(random_seed)
    np.random.seed(random_seed)
    anchors: list[str] = []
    positives: list[str] = []
    negatives: list[str] = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="triplets"):
        rule = clean_url_to_domain_path(row["rule"])
        compliant = [
            clean_url_to_domain_path(row[col])
            for col in ["negative_example_1", "negative_example_2"]
            if pd.notna(row[col])
        ]
        violating = [
            clean_url_to_domain_path(row[col])
            for col in ["positive_example_1", "positive_example_2"]
            if pd.notna(row[col])
        ]
        for pos_ex in compliant:
            for neg_ex in violating:
                anchors.append(rule)
                positives.append(pos_ex)
                negatives.append(neg_ex)

    if augmentation_factor > 0:
        rule_positives: dict[str, list[str]] = {}
        rule_negatives: dict[str, list[str]] = {}
        for rule in test_df["rule"].unique():
            rule_df = test_df[test_df["rule"] == rule]
            pos_pool: list[str] = []
            neg_pool: list[str] = []
            for _, row in rule_df.iterrows():
                for col in ["negative_example_1", "negative_example_2"]:
                    if pd.notna(row[col]):
                        pos_pool.append(clean_url_to_domain_path(row[col]))
                for col in ["positive_example_1", "positive_example_2"]:
                    if pd.notna(row[col]):
                        neg_pool.append(clean_url_to_domain_path(row[col]))
            rule_positives[rule] = list(set(pos_pool))
            rule_negatives[rule] = list(set(neg_pool))

        for rule in test_df["rule"].unique():
            clean_rule = clean_url_to_domain_path(rule)
            pos_pool = rule_positives[rule]
            neg_pool = rule_negatives[rule]
            n_samples = min(
                augmentation_factor * len(pos_pool),
                len(pos_pool) * len(neg_pool),
            )
            for _ in range(n_samples):
                if pos_pool and neg_pool:
                    anchors.append(clean_rule)
                    positives.append(random.choice(pos_pool))
                    negatives.append(random.choice(neg_pool))

    combined = list(zip(anchors, positives, negatives))
    random.shuffle(combined)
    if subsample_fraction < 1.0:
        combined = combined[: int(len(combined) * subsample_fraction)]
    anchors, positives, negatives = zip(*combined) if combined else ([], [], [])
    return Dataset.from_dict(
        {
            "anchor": list(anchors),
            "positive": list(positives),
            "negative": list(negatives),
        }
    )


def fine_tune_model(
    model,
    train_dataset,
    output_dir: str | Path,
    config: GteConfig,
):
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import TripletLoss

    loss = TripletLoss(model=model, triplet_margin=config.triplet_margin)
    dataset_size = len(train_dataset)
    steps_per_epoch = max(1, dataset_size // config.batch_size)
    max_steps = steps_per_epoch * config.epochs
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        warmup_steps=0,
        learning_rate=config.learning_rate,
        logging_steps=max(1, max_steps // 4),
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        max_grad_norm=1.0,
        dataloader_drop_last=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    final_model_path = Path(output_dir) / "final"
    model.save_pretrained(str(final_model_path))
    return model


def create_rule_centroids(
    test_df: pd.DataFrame,
    text_to_embedding: dict[str, np.ndarray],
    rule_embeddings: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray | int]]:
    centroids: dict[str, dict[str, np.ndarray | int]] = {}
    for rule in test_df["rule"].unique():
        rule_data = test_df[test_df["rule"] == rule]
        pos_embeddings = []
        neg_embeddings = []
        for _, row in rule_data.iterrows():
            for col in ["positive_example_1", "positive_example_2"]:
                clean_text = clean_url_to_domain_path(row[col])
                if clean_text in text_to_embedding:
                    pos_embeddings.append(text_to_embedding[clean_text])
            for col in ["negative_example_1", "negative_example_2"]:
                clean_text = clean_url_to_domain_path(row[col])
                if clean_text in text_to_embedding:
                    neg_embeddings.append(text_to_embedding[clean_text])

        if pos_embeddings and neg_embeddings:
            pos_centroid = np.array(pos_embeddings).mean(axis=0)
            neg_centroid = np.array(neg_embeddings).mean(axis=0)
            pos_centroid = pos_centroid / np.linalg.norm(pos_centroid)
            neg_centroid = neg_centroid / np.linalg.norm(neg_centroid)
            centroids[rule] = {
                "positive": pos_centroid,
                "negative": neg_centroid,
                "pos_count": len(pos_embeddings),
                "neg_count": len(neg_embeddings),
                "rule_embedding": rule_embeddings[rule],
            }
    return centroids


def predict_by_centroid_distance(
    test_df: pd.DataFrame,
    text_to_embedding: dict[str, np.ndarray],
    rule_centroids: dict[str, dict[str, np.ndarray | int]],
) -> pd.DataFrame:
    row_ids: list[int] = []
    predictions: list[float] = []
    for rule in test_df["rule"].unique():
        if rule not in rule_centroids:
            continue
        rule_data = test_df[test_df["rule"] == rule]
        pos_centroid = rule_centroids[rule]["positive"]
        neg_centroid = rule_centroids[rule]["negative"]
        valid_embeddings = []
        valid_row_ids = []
        for _, row in rule_data.iterrows():
            body = clean_url_to_domain_path(row["body"])
            if body in text_to_embedding:
                valid_embeddings.append(text_to_embedding[body])
                valid_row_ids.append(row["row_id"])
        if not valid_embeddings:
            continue
        query_embeddings = np.array(valid_embeddings)
        pos_distances = np.linalg.norm(query_embeddings - pos_centroid, axis=1)
        neg_distances = np.linalg.norm(query_embeddings - neg_centroid, axis=1)
        rule_predictions = neg_distances - pos_distances
        row_ids.extend(valid_row_ids)
        predictions.extend(rule_predictions)
    return pd.DataFrame({"row_id": row_ids, "rule_violation": predictions})


def train_and_infer_gte(
    data_dir: str | Path = DEFAULT_PATHS.data_dir,
    model_path: str | Path = DEFAULT_PATHS.gte_model,
    output_path: str | Path = DEFAULT_PATHS.output_dir / "sub_gte.csv",
    work_dir: str | Path = DEFAULT_PATHS.output_dir / "models" / "gte_triplet",
    config: GteConfig = GteConfig(),
) -> None:
    from sentence_transformers import SentenceTransformer

    test_df = read_test(data_dir)
    model = SentenceTransformer(str(model_path), trust_remote_code=True)
    model.max_seq_length = config.max_seq_length
    triplets = create_triplet_dataset(
        test_df,
        augmentation_factor=config.augmentation_factor,
        random_seed=config.seed,
    )
    model = fine_tune_model(model, triplets, work_dir, config)
    model.half()

    all_texts = collect_all_texts(test_df)
    all_embeddings = model.encode(
        all_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )
    text_to_embedding = {text: emb for text, emb in zip(all_texts, all_embeddings)}
    rule_embeddings = {
        rule: model.encode(
            clean_url_to_domain_path(rule),
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        for rule in test_df["rule"].unique()
    }
    centroids = create_rule_centroids(test_df, text_to_embedding, rule_embeddings)
    submission = predict_by_centroid_distance(test_df, text_to_embedding, centroids)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_PATHS.data_dir))
    parser.add_argument("--model", default=str(DEFAULT_PATHS.gte_model))
    parser.add_argument("--output", default=str(DEFAULT_PATHS.output_dir / "sub_gte.csv"))
    parser.add_argument("--work-dir", default=str(DEFAULT_PATHS.output_dir / "models" / "gte_triplet"))
    args = parser.parse_args()
    train_and_infer_gte(args.data_dir, args.model, args.output, args.work_dir)
