"""Training hyperparameter configuration.

All hyperparameters are in one place with documented rationale.
Pass --smoke-test to override key settings for a quick sanity run.

Design: Pydantic model rather than argparse flags or YAML. This gives type
validation, default documentation, and JSON serialization for W&B logging.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DPOConfig(BaseModel):
    """Hyperparameters for the DPO alignment phase (session 17).

    DPO trains on (prompt, chosen, rejected) triples using the SFT checkpoint
    as both the starting point and the frozen reference policy.
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    sft_adapter_dir: str = Field(
        default="checkpoints/sft",
        description="LoRA adapter from SFT training. Loaded as both the trainable "
                    "policy and (frozen copy) reference policy.",
    )

    # ── DPO loss ──────────────────────────────────────────────────────────────
    beta: float = Field(
        default=0.1,
        description="KL temperature. Controls how far the trained policy is allowed "
                    "to deviate from the SFT reference. "
                    "Low β (0.01–0.05): aggressive, risk of forgetting. "
                    "High β (0.5+): conservative, small preference margin. "
                    "0.1 is the DPO paper default.",
    )
    loss_type: Literal["sigmoid", "hinge", "ipo"] = Field(
        default="sigmoid",
        description="DPO loss variant. "
                    "'sigmoid': standard DPO (Rafailov et al. 2023). "
                    "'ipo': Identity PO, removes the log-sigmoid, more stable. "
                    "'hinge': max-margin variant, ignores easy pairs.",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    preference_data_path: str = "data/processed/preference_pairs.jsonl"
    max_prompt_length: int = Field(
        default=512,
        description="Max tokens for the prompt portion. Chosen+rejected each get "
                    "max_seq_length - max_prompt_length tokens.",
    )
    max_length: int = Field(
        default=1024,
        description="Max total sequence length (prompt + completion).",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    num_train_epochs: int = 1
    per_device_train_batch_size: int = Field(
        default=2,
        description="DPO processes chosen+rejected in the same forward pass, "
                    "so effective memory use is ~2× SFT. Reduce to 1 on T4-16GB.",
    )
    gradient_accumulation_steps: int = Field(
        default=8,
        description="Effective batch = 2×8 = 16, matching SFT effective batch.",
    )
    learning_rate: float = Field(
        default=5e-5,
        description="Lower than SFT (2e-4). DPO objective is more sensitive; "
                    "the SFT reference acts as an implicit regularizer.",
    )
    lr_scheduler_type: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora: "LoRAConfig | None" = None

    # ── Checkpointing ─────────────────────────────────────────────────────────
    output_dir: str = "checkpoints/dpo"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2

    # ── Logging ───────────────────────────────────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "extractor"
    run_name: str = "extractor-dpo"


class QuantizationConfig(BaseModel):
    """Configuration for post-training quantization (session 19/20).

    Two targets:
      AWQ  — fast inference on Ampere+ GPUs via activation-aware weight scaling
      GGUF — portable CPU+GPU inference via llama.cpp

    The merged model (BF16, ~6 GB) is the input for both.
    """

    model_dir: str = Field(
        default="checkpoints/dpo-merged",
        description="Path to the merged BF16 model to quantize.",
    )

    # ── AWQ ───────────────────────────────────────────────────────────────────
    awq_output_dir: str = "checkpoints/awq"
    awq_bits: int = Field(
        default=4,
        description="Bit-width for AWQ. 4 is standard; 8 available but less common.",
    )
    awq_group_size: int = Field(
        default=128,
        description="Number of weights per quantization group. Smaller = more accurate, "
                    "slightly larger model. 128 is the AWQ paper default.",
    )
    awq_zero_point: bool = Field(
        default=True,
        description="Use zero-point quantization (asymmetric). Slightly more accurate "
                    "than symmetric (zero_point=False) at no size cost.",
    )
    awq_calib_data: str = Field(
        default="data/processed/hf_dataset/train",
        description="HF dataset directory used for calibration (128 samples, no labels needed).",
    )
    awq_calib_samples: int = Field(
        default=128,
        description="Number of calibration examples. 128 is sufficient to find salient channels.",
    )
    awq_calib_seq_len: int = Field(
        default=512,
        description="Sequence length for calibration. Longer = more accurate channel detection, "
                    "but requires more VRAM during calibration.",
    )

    # ── GGUF ──────────────────────────────────────────────────────────────────
    gguf_output_dir: str = "checkpoints/gguf"
    gguf_type: Literal["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0"] = Field(
        default="Q4_K_M",
        description="GGUF quantization type. "
                    "Q4_K_M: 4-bit k-quant (attention Q6, MLP Q4). Recommended. "
                    "Q5_K_M: 5-bit, better quality, larger. "
                    "Q8_0: near-lossless, 2× compression only. "
                    "Q4_0: simple 4-bit, legacy.",
    )
    llama_cpp_dir: str = Field(
        default="./llama.cpp",
        description="Path to llama.cpp repository (cloned separately). "
                    "Needed for convert-hf-to-gguf.py and llama-quantize.",
    )


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
        default=False,
        description="Packing is incompatible with DataCollatorForCompletionOnlyLM in TRL 0.13. "
                    "Keep False to preserve loss masking on prompt tokens.",
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
