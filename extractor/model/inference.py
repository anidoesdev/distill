"""Local HuggingFace inference wrapper.

Used for:
  - Zero-shot baseline evaluation (sessions 2–3)
  - Smoke-testing a newly trained checkpoint (sessions 10–11)
  - Generating DPO candidate pairs (session 16)

In production (session 22+) the API calls vLLM directly via VLLMClient.
This class intentionally has no vLLM dependency so it runs anywhere torch runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from extractor.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0       # 0.0 = greedy decoding; >0 = sampling
    top_p: float = 0.95
    repetition_penalty: float = 1.1


class HFInference:
    """Wraps a HuggingFace causal LM for chat-template inference.

    Args:
        model_name: HuggingFace model ID or local checkpoint path.
        load_in_4bit: Use 4-bit NF4 quantization (bitsandbytes). Reduces VRAM
            by ~75% with ~1-2% accuracy drop. Required on <8GB VRAM GPUs.
        device_map: Passed to from_pretrained. "auto" shards across all GPUs.
        torch_dtype: Weight precision when NOT quantizing. bfloat16 is preferred
            over float16 on Ampere+ GPUs — same VRAM, more numerical stability.
    """

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_name = model_name

        logger.info("loading tokenizer", extra={"model": model_name})
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Ensure a pad token exists — some models omit it (pad with eos instead)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        bnb_config: Optional[BitsAndBytesConfig] = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",         # NormalFloat4: better than int4 for LLM weights
                bnb_4bit_compute_dtype=torch.bfloat16,  # dequantize to bf16 for matmuls
                bnb_4bit_use_double_quant=True,    # quantize the quantization constants too (~0.4 bit/param extra savings)
            )
            logger.info("4-bit NF4 quantization enabled")

        logger.info("loading model weights", extra={"model": model_name})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype if not load_in_4bit else None,
            trust_remote_code=True,
        )
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        logger.info(
            "model ready",
            extra={"params_b": round(n_params, 2), "vram_gb": round(vram_gb, 2)},
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> tuple[str, dict]:
        """Apply the model's chat template and generate a response.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
                      Roles: "system", "user", "assistant".
            config: Generation hyperparameters. Defaults to greedy decoding.

        Returns:
            (response_text, metadata) where metadata contains token counts
            and latency.
        """
        cfg = config or GenerationConfig()

        # apply_chat_template reads the Jinja2 template from tokenizer_config.json
        # and formats messages in the model's expected format.
        # add_generation_prompt=True appends the assistant turn opener so the
        # model knows to start generating (not re-summarize the conversation).
        input_ids: torch.Tensor = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_len = input_ids.shape[-1]
        t0 = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature if cfg.temperature > 0 else None,
                do_sample=cfg.temperature > 0,
                top_p=cfg.top_p if cfg.temperature > 0 else None,
                repetition_penalty=cfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        elapsed = time.perf_counter() - t0

        # output_ids contains prompt + completion. Slice off the prompt.
        new_token_ids = output_ids[0][prompt_len:]
        response = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        meta = {
            "model": self.model_name,
            "prompt_tokens": prompt_len,
            "completion_tokens": len(new_token_ids),
            "latency_s": round(elapsed, 3),
            "tokens_per_s": round(len(new_token_ids) / max(elapsed, 1e-6), 1),
        }
        return response.strip(), meta

    def inspect_chat_template(self) -> None:
        """Print the model's raw Jinja2 chat template. Useful for debugging."""
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            print(f"[{self.model_name}] chat_template:\n{template}\n")
        else:
            print(f"[{self.model_name}] no chat_template found in tokenizer config")

    def tokenize_example(self, text: str) -> None:
        """Show token IDs and decoded tokens for a string. Useful for understanding
        how the model sees special tokens and scientific text."""
        ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([i]) for i in ids]
        print(f"Text: {text!r}")
        print(f"Token count: {len(ids)}")
        for tid, tok in zip(ids, tokens):
            print(f"  {tid:6d}  {tok!r}")
