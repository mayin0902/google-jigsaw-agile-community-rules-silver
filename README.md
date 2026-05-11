# Google Jigsaw Agile Community Rules Classification

Kaggle Silver Medal solution for **Google Jigsaw - Agile Community Rules Classification**.

The task is rule-conditioned moderation: given a community rule and a user post, predict whether the post violates that rule. The solution combines an instruction-tuned LLM judge, metric learning over rule examples, and a discriminative DeBERTa classifier.

## Result

| Field | Value |
| --- | --- |
| Competition | Google Jigsaw - Agile Community Rules Classification |
| Organizer | Google Jigsaw |
| Medal | Silver |
| Rank | 42 / 2445 |
| Awarded | October 24, 2025 |
| Metric | Column-averaged AUC |
| Final AUC | 0.92909 |

## Method Overview

- Reformulate the task as rule-conditioned violation detection instead of rule-id classification.
- Fine-tune Qwen3-4B with QLoRA under limited Kaggle GPU resources.
- Use DeepSpeed ZeRO-2 and bf16 mixed precision for memory-efficient training.
- Train a GTE metric-learning branch with triplet loss to improve generalization to unseen rules.
- Train a DeBERTa-v3 discriminative branch on `rule [SEP] comment`, with URL semantics, attention pooling, FGM adversarial training, and EMA.
- Blend branch outputs with rank-based ensembling because the evaluation metric is AUC.

## Repository Layout

```text
configs/
  accelerate_zero2.yaml
scripts/
  install_kaggle_deps.sh
  run_kaggle_pipeline.sh
  train_qwen_lora.py
  infer_qwen_lora.py
  train_infer_gte.py
  train_infer_deberta.py
  blend_submissions.py
src/jigsaw_agile/
  config.py
  data.py
  text.py
  qwen_lora.py
  gte_triplet.py
  deberta_cls.py
  blend.py
  metrics.py
```

## Kaggle Run

```bash
bash scripts/install_kaggle_deps.sh
bash scripts/run_kaggle_pipeline.sh
```

The final submission is written to `/kaggle/working/submission.csv`.

## Local Checks

```bash
python -m pip install -e .
python -m compileall src scripts
```

Heavy runtime dependencies such as `vllm`, `transformers`, `peft`, `trl`, and `sentence-transformers` are imported lazily inside the corresponding training and inference paths.

## Ensemble

The final blend is implemented in `src/jigsaw_agile/blend.py`:

```text
0.5 * rank(DeBERTa) + 0.3 * rank(GTE) + 0.7 * rank(Qwen)
```

Rank blending is stable for AUC because it avoids relying on the calibration of heterogeneous model probabilities.

## Notes

Raw competition data, trained LoRA weights, offline wheel caches, and generated submissions are excluded from the repository.
