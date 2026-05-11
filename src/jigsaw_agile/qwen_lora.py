from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_agile.config import DEFAULT_PATHS, QwenConfig
from jigsaw_agile.data import (
    NEGATIVE,
    POSITIVE,
    attach_random_examples,
    build_prompt,
    build_qwen_training_frame,
    read_test,
)


def seed_everything(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_sft_dataset(df: pd.DataFrame, dry_run_rows: int | None = None):
    from datasets import Dataset

    work = df.copy()
    work["prompt"] = work.apply(build_prompt, axis=1)
    work["completion"] = work["rule_violation"].map({1: POSITIVE, 0: NEGATIVE})
    if dry_run_rows:
        work = work.head(dry_run_rows)
    return Dataset.from_pandas(work[["prompt", "completion"]])


def train_qwen_lora(
    data_dir: str | Path = DEFAULT_PATHS.data_dir,
    base_model_path: str | Path = DEFAULT_PATHS.qwen_model,
    lora_path: str | Path = DEFAULT_PATHS.lora_dir,
    config: QwenConfig = QwenConfig(),
    dry_run_rows: int | None = None,
) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils import is_torch_bf16_gpu_available
    from trl import SFTConfig, SFTTrainer

    seed_everything(config.seed)
    base_model_path = str(base_model_path)
    lora_path = str(lora_path)
    use_gptq = "gptq" in base_model_path.lower()

    train_df = build_qwen_training_frame(
        data_dir=data_dir,
        use_train=config.use_train_csv,
        test_rule_frac=config.train_frac_from_test_examples,
        seed=config.seed,
    )
    train_dataset = to_sft_dataset(train_df, dry_run_rows=dry_run_rows)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        num_train_epochs=config.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=config.learning_rate,
        weight_decay=0.20,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
    )

    if use_gptq:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="balanced_low_0",
            trust_remote_code=True,
            use_cache=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="balanced_low_0",
            trust_remote_code=True,
            use_cache=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    pretrain_lora_path = os.getenv("JIGSAW_PRETRAIN_LORA")
    if pretrain_lora_path:
        model = PeftModel.from_pretrained(model, pretrain_lora_path)
        model = model.merge_and_unload()

    if len(train_dataset) > 0:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=lora_config,
        )
        trainer.train()
        trainer.save_model(lora_path)
    else:
        peft_model = get_peft_model(model, lora_config)
        peft_model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)


def _build_inference_dataset(df: pd.DataFrame):
    from datasets import Dataset

    work = df.copy()
    work["prompt"] = work.apply(build_prompt, axis=1)
    return Dataset.from_pandas(work[["prompt"]])


def _run_vllm_on_slice(
    df_slice: pd.DataFrame,
    base_model_path: str,
    lora_path: str,
    max_model_len: int,
) -> pd.DataFrame:
    import vllm
    from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
    from vllm.lora.request import LoRARequest

    use_gptq = "gptq" in base_model_path.lower()
    llm = vllm.LLM(
        base_model_path,
        quantization="gptq" if use_gptq else None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=max_model_len,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )
    tokenizer = llm.get_tokenizer()
    outputs = llm.generate(
        _build_inference_dataset(df_slice)["prompt"],
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[
                MultipleChoiceLogitsProcessor(tokenizer, choices=[POSITIVE, NEGATIVE])
            ],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("qwen_lora", 1, lora_path),
    )
    log_probs = [
        {lp.decoded_token: np.exp(lp.logprob) for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]
    predictions = pd.DataFrame(log_probs)[[POSITIVE, NEGATIVE]]
    predictions["row_id"] = df_slice["row_id"].values
    return predictions


def _worker(
    device_id: int,
    df_slice: pd.DataFrame,
    base_model_path: str,
    lora_path: str,
    max_model_len: int,
    return_dict,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return_dict[device_id] = _run_vllm_on_slice(
        df_slice=df_slice,
        base_model_path=base_model_path,
        lora_path=lora_path,
        max_model_len=max_model_len,
    )


def infer_qwen_lora(
    data_dir: str | Path = DEFAULT_PATHS.data_dir,
    base_model_path: str | Path = DEFAULT_PATHS.qwen_model,
    lora_path: str | Path = DEFAULT_PATHS.lora_dir,
    output_path: str | Path = DEFAULT_PATHS.output_dir / "sub_llm.csv",
    config: QwenConfig = QwenConfig(),
    num_gpus: int = 2,
) -> None:
    os.environ["VLLM_USE_V1"] = "0"
    test_df = attach_random_examples(read_test(data_dir), seed=config.seed)
    num_gpus = max(1, num_gpus)

    if num_gpus == 1:
        predictions = _run_vllm_on_slice(
            test_df,
            str(base_model_path),
            str(lora_path),
            config.max_model_len,
        )
    else:
        chunks = np.array_split(test_df.reset_index(drop=True), num_gpus)
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        for device_id, chunk in enumerate(chunks):
            proc = mp.Process(
                target=_worker,
                args=(
                    device_id,
                    chunk.reset_index(drop=True),
                    str(base_model_path),
                    str(lora_path),
                    config.max_model_len,
                    return_dict,
                ),
            )
            proc.start()
            processes.append(proc)
        for proc in processes:
            proc.join()
        predictions = pd.concat([return_dict[i] for i in range(num_gpus)], ignore_index=True)

    submission = predictions[["row_id", POSITIVE]].rename(columns={POSITIVE: "rule_violation"})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def train_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_PATHS.data_dir))
    parser.add_argument("--base-model", default=str(DEFAULT_PATHS.qwen_model))
    parser.add_argument("--lora-dir", default=str(DEFAULT_PATHS.lora_dir))
    parser.add_argument("--dry-run-rows", type=int, default=None)
    args = parser.parse_args()
    train_qwen_lora(args.data_dir, args.base_model, args.lora_dir, dry_run_rows=args.dry_run_rows)


def infer_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_PATHS.data_dir))
    parser.add_argument("--base-model", default=str(DEFAULT_PATHS.qwen_model))
    parser.add_argument("--lora-dir", default=str(DEFAULT_PATHS.lora_dir))
    parser.add_argument("--output", default=str(DEFAULT_PATHS.output_dir / "sub_llm.csv"))
    parser.add_argument("--num-gpus", type=int, default=2)
    args = parser.parse_args()
    infer_qwen_lora(args.data_dir, args.base_model, args.lora_dir, args.output, num_gpus=args.num_gpus)
