"""DPO alignment training script.

Session 15: Script written and reviewed (theory session)
Session 17: Full DPO training run with preference pairs

DPO directly optimizes the preference objective without a separate reward model.
The SFT checkpoint serves as both the starting policy and the frozen reference.

Loss (simplified):
    L = -E[log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x))
                   - β · (log π_θ(y_l|x) - log π_ref(y_l|x)))]

where:
    π_θ     = trained policy (updated by gradient descent)
    π_ref   = frozen SFT reference policy
    y_w     = chosen (preferred) completion
    y_l     = rejected (dispreferred) completion
    β       = KL temperature (default 0.1)

Usage:
    # Full DPO run (session 17)
    python training/train_dpo.py

    # Smoke test
    python training/train_dpo.py --smoke-test

    # Custom beta
    python training/train_dpo.py --beta 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer

from extractor.data.splits import verify_no_leakage
from extractor.utils.logging import configure_logging, get_logger
from training.config import DPOConfig, LoRAConfig
from training.utils import TrainingHealthCallback, log_gpu_memory, log_trainable_params

configure_logging("info")
logger = get_logger(__name__)


def load_preference_dataset(path: str, smoke_test: bool, smoke_n: int = 100) -> tuple[Dataset, Dataset]:
    """Load JSONL preference pairs and split into train/val."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Preference dataset not found: {path}\n"
            "Generate it first: python scripts/generate_preferences.py"
        )

    records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if smoke_test:
        records = records[:smoke_n]
        logger.info("smoke test: reduced preference dataset", extra={"n": len(records)})

    # 90/10 split — val used to monitor reward margin during training
    split = int(len(records) * 0.9)
    train_records = records[:split]
    val_records = records[split:]

    # DPOTrainer expects columns: prompt, chosen, rejected
    train_ds = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in train_records
    ])
    val_ds = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in val_records
    ])

    logger.info("preference dataset loaded", extra={"train": len(train_ds), "val": len(val_ds)})
    return train_ds, val_ds


def load_model_and_tokenizer(cfg: DPOConfig):
    """Load the SFT model in 4-bit as the starting point for DPO."""
    logger.info("loading tokenizer", extra={"model": cfg.model_name})
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("loading base model in 4-bit", extra={"model": cfg.model_name})
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load SFT LoRA weights on top of the base model
    logger.info("loading SFT adapter", extra={"adapter_dir": cfg.sft_adapter_dir})
    model = PeftModel.from_pretrained(model, cfg.sft_adapter_dir, is_trainable=True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, tokenizer


def build_trainer(model, tokenizer, train_ds, val_ds, cfg: DPOConfig) -> DPOTrainer:
    """Assemble the TRL DPOTrainer.

    DPOTrainer handles the reference model internally: it freezes a copy of
    the model at construction time and uses it to compute log π_ref(y|x) for
    each batch. This is why we do NOT need to load two separate model objects.
    """
    dpo_args = TRLDPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=True,
        fp16=False,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_length,
        precompute_ref_log_probs=True,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        # reward_margins logged automatically by DPOTrainer:
        # "rewards/chosen", "rewards/rejected", "rewards/margins"
        report_to="wandb" if cfg.use_wandb else "none",
        run_name=cfg.run_name,
    )

    return DPOTrainer(
        model=model,
        ref_model=None,   # DPOTrainer creates the frozen reference internally
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on 100 preference pairs for 1 epoch.")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override DPO beta (KL temperature).")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file to override defaults.")
    args = parser.parse_args()

    cfg = DPOConfig()
    if args.config:
        overrides = json.loads(Path(args.config).read_text())
        cfg = cfg.model_copy(update=overrides)
    if args.beta is not None:
        cfg = cfg.model_copy(update={"beta": args.beta})
    if args.smoke_test:
        cfg = cfg.model_copy(update={
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "eval_steps": 20,
            "save_steps": 20,
            "use_wandb": False,
            "run_name": "extractor-dpo-smoke",
            "output_dir": "checkpoints/dpo-smoke",
        })

    logger.info("DPO config", extra=cfg.model_dump())

    # Leakage check — same guard as SFT
    verify_no_leakage()

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.model_dump())

    model, tokenizer = load_model_and_tokenizer(cfg)
    log_trainable_params(model)
    log_gpu_memory("after model load")

    train_ds, val_ds = load_preference_dataset(
        cfg.preference_data_path,
        smoke_test=args.smoke_test,
    )

    trainer = build_trainer(model, tokenizer, train_ds, val_ds, cfg)
    trainer.add_callback(TrainingHealthCallback(fail_on_nan=True))

    logger.info("starting DPO training")
    log_gpu_memory("before training")
    train_result = trainer.train()

    trainer.save_model()
    logger.info("DPO model saved", extra={"output_dir": cfg.output_dir})

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("DPO training complete", extra=metrics)

    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
