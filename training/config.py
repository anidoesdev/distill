"""Training hyperparameter configuration.

All hyperparameters are in one place with documented rationale.
Pass --smoke-test to override key settings for a quick sanity run.

Design: Pydantic model rather than argparse flags or YAML. This gives type
validation, default documentation, and JSON serialization for W&B logging.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    r: int = Field(default=16, description="Rank of the LoRA matrices. Higher = more capacity, more VRAM.")
    lora_alpha: int = Field(
        default=32,
        description="Scaling factor. Effective LR multiplier for LoRA updates = alpha/r. "
                    "alpha=2r is a safe default.",
    )
    target_modules: list[str] = Field(
        default=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP
        ],
        description="Which linear layers to add LoRA adapters to. "
                    "Including all attention + MLP projections gives best results. "
                    "q_proj+v_proj only is faster but slightly worse.",
    )
    lora_dropout: float = Field(default=0.05, description="Dropout on LoRA outputs. Small but helps generalization.")
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Whether to train bias terms. 'none' is standard for QLoRA.",
    )


class QLoRAConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = Field(
        default="nf4",
        description="NF4 outperforms FP4 for normally distributed weights.",
    )
    bnb_4bit_compute_dtype: str = Field(
        default="bfloat16",
        description="Precision for dequantized matmuls. BF16 preferred on Ampere+.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Quantize scale factors from FP32 to FP8. Saves ~0.37 bits/param.",
    )


class TrainingConfig(BaseModel):
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_length: int = Field(
        default=1024,
        description="Set from sequence length analysis output in session 8.",
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora: LoRAConfig = LoRAConfig()
    qlora: QLoRAConfig = QLoRAConfig()

    # ── Data ──────────────────────────────────────────────────────────────────
    hf_dataset_dir: str = "data/processed/hf_dataset"
    packing: bool = Field(
        default=True,
        description="Sequence packing enabled (confirmed working in session 10 smoke test). "
                    "Gives ~35-40% throughput improvement when avg seq length << max_seq_length. "
                    "Requires TRL >= 0.12 for correct interaction with DataCollatorForCompletionOnlyLM.",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    num_train_epochs: int = Field(
        default=3,
        description="Adjusted after smoke-test loss curves (session 10). "
                    "3 epochs is typical for instruction fine-tuning at this scale.",
    )
    per_device_train_batch_size: int = Field(
        default=4,
        description="Reduce to 2 on T4-16GB if OOM during activation checkpointing.",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        description="Effective batch size = batch_size × grad_accum × n_gpus = 4×4×1 = 16.",
    )
    learning_rate: float = Field(
        default=2e-4,
        description="Higher than full fine-tuning because only LoRA params are updated. "
                    "2e-4 is the QLoRA paper default for instruction fine-tuning.",
    )
    lr_scheduler_type: Literal["cosine", "linear", "constant"] = Field(
        default="cosine",
        description="Cosine annealing decays LR smoothly to 0. "
                    "Better final loss than linear for short runs.",
    )
    warmup_ratio: float = Field(
        default=0.05,
        description="5% warmup prevents loss spikes at training start.",
    )
    weight_decay: float = Field(default=0.01)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    output_dir: str = "checkpoints/sft"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = Field(
        default=3,
        description="Keep only the 3 best checkpoints to save disk space.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "extractor"
    run_name: str = "extractor-sft-qlora"

    # ── Response template ─────────────────────────────────────────────────────
    response_template: str = Field(
        default="<|im_start|>assistant\n",
        description="Token sequence marking the start of the assistant turn. "
                    "DataCollatorForCompletionOnlyLM masks all tokens before this. "
                    "Qwen2.5 uses ChatML format. Llama 3.2 uses a different template.",
    )

    def smoke_test_overrides(self) -> TrainingConfig:
        """Return a copy with settings reduced for a quick smoke test."""
        return self.model_copy(
            update={
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "eval_steps": 20,
                "save_steps": 20,
                "use_wandb": False,
                "run_name": "extractor-sft-smoke",
                "output_dir": "checkpoints/sft-smoke",
            }
        )
