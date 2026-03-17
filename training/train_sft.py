"""QLoRA SFT training script.

Session 9:  Script written and reviewed (this session)
Session 10: Smoke test — 200 examples, 1 epoch, loss curves inspected
Session 11: Full training run with W&B logging, best checkpoint saved

Usage:
    # Smoke test (session 10)
    python training/train_sft.py --smoke-test

    # Full run (session 11)
    python training/train_sft.py

    # Custom model
    python training/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from extractor.data.splits import verify_no_leakage
from extractor.utils.logging import configure_logging, get_logger
from training.config import TrainingConfig
from training.utils import TrainingHealthCallback, log_gpu_memory, log_trainable_params

configure_logging("info")
logger = get_logger(__name__)


def load_model_and_tokenizer(cfg: TrainingConfig):
    """Load base model in 4-bit NF4 and prepare for LoRA training."""

    logger.info("loading tokenizer", extra={"model": cfg.model_name})
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 4-bit quantization config ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.qlora.load_in_4bit,
        bnb_4bit_quant_type=cfg.qlora.bnb_4bit_quant_type,  # "nf4"
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg.qlora.bnb_4bit_use_double_quant,
    )

    logger.info("loading base model in 4-bit", extra={"model": cfg.model_name})
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        # torch_dtype is ignored when quantization_config is set, but
        # the compute dtype in BitsAndBytesConfig controls actual precision.
        trust_remote_code=True,
    )

    # prepare_model_for_kbit_training does three things:
    # 1. Freezes all base model parameters
    # 2. Upcasts normalization layers (RMSNorm) to float32 for stability
    # 3. Enables gradient checkpointing (trades VRAM for recomputation)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, tokenizer


def add_lora_adapters(model, cfg: TrainingConfig):
    """Inject LoRA adapter matrices into the model."""

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA adapters injected",
        extra={
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 3),
            "r": cfg.lora.r,
            "target_modules": cfg.lora.target_modules,
        },
    )
    return model


def build_data_collator(tokenizer, response_template: str):
    """Build the collator that masks loss on prompt tokens.

    DataCollatorForCompletionOnlyLM finds all occurrences of response_template
    in each tokenized sequence and sets labels to -100 for all preceding tokens.
    Only the assistant's response tokens contribute to the cross-entropy loss.

    For Qwen2.5 (ChatML format), the assistant turn starts with:
        <|im_start|>assistant\n
    For Llama 3.2, it would be:
        <|start_header_id|>assistant<|end_header_id|>\n\n
    """
    response_ids = tokenizer.encode(response_template, add_special_tokens=False)
    logger.info(
        "response template tokenized",
        extra={
            "template": response_template,
            "token_ids": response_ids,
            "tokens": [tokenizer.decode([t]) for t in response_ids],
        },
    )
    return DataCollatorForCompletionOnlyLM(
        response_template=response_ids,
        tokenizer=tokenizer,
    )


def load_datasets(hf_dir: str, smoke_test: bool, smoke_n: int = 200):
    """Load HF Datasets from disk. For smoke test, take first N examples."""
    base = Path(hf_dir)
    train_ds = load_from_disk(str(base / "train"))
    val_ds = load_from_disk(str(base / "val"))

    if smoke_test:
        train_ds = train_ds.select(range(min(smoke_n, len(train_ds))))
        val_ds = val_ds.select(range(min(50, len(val_ds))))
        logger.info(
            "smoke test: using reduced dataset",
            extra={"train": len(train_ds), "val": len(val_ds)},
        )
    else:
        logger.info(
            "full dataset loaded",
            extra={"train": len(train_ds), "val": len(val_ds)},
        )

    return train_ds, val_ds


def build_trainer(
    model,
    tokenizer,
    train_ds,
    val_ds,
    cfg: TrainingConfig,
) -> SFTTrainer:
    """Assemble the TRL SFTTrainer."""

    sft_args = SFTConfig(
        # Output
        output_dir=cfg.output_dir,
        # Epochs and batch
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        # Learning rate
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        # Precision — bf16 required on Ampere+ for stable training
        bf16=True,
        fp16=False,
        # Sequence
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        # Logging and checkpointing
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # W&B
        report_to="wandb" if cfg.use_wandb else "none",
        run_name=cfg.run_name,
    )

    collator = build_data_collator(tokenizer, cfg.response_template)

    return SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on 200 examples for 1 epoch (session 10 smoke test).")
    parser.add_argument("--model", default=None,
                        help="Override model_name from config.")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file to override defaults.")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = TrainingConfig()
    if args.config:
        overrides = json.loads(Path(args.config).read_text())
        cfg = cfg.model_copy(update=overrides)
    if args.model:
        cfg = cfg.model_copy(update={"model_name": args.model})
    if args.smoke_test:
        cfg = cfg.smoke_test_overrides()

    logger.info("training config", extra=cfg.model_dump())

    # ── Leakage check ─────────────────────────────────────────────────────────
    verify_no_leakage()

    # ── W&B setup ─────────────────────────────────────────────────────────────
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.model_dump())

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = add_lora_adapters(model, cfg)
    log_trainable_params(model)
    log_gpu_memory("after model load")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_ds, val_ds = load_datasets(
        cfg.hf_dataset_dir,
        smoke_test=args.smoke_test,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = build_trainer(model, tokenizer, train_ds, val_ds, cfg)

    trainer.add_callback(TrainingHealthCallback(fail_on_nan=True))
    logger.info("starting training")
    log_gpu_memory("before training")
    train_result = trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    # Saves only the LoRA adapter weights — not the full model.
    # To merge adapters into base weights: model.merge_and_unload()
    trainer.save_model()
    logger.info("model saved", extra={"output_dir": cfg.output_dir})

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("training complete", extra=metrics)

    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
