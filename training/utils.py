"""Training utilities: memory reporting, loss analysis, health callbacks.

These are used in all training sessions (10, 11, 17) to monitor training health.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from extractor.utils.logging import get_logger

logger = get_logger(__name__)


# ── GPU memory ────────────────────────────────────────────────────────────────

def log_gpu_memory(label: str = "") -> dict[str, float]:
    """Log current GPU VRAM allocation. Returns dict for further logging."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    info = {
        "gpu_allocated_gb": round(allocated, 2),
        "gpu_reserved_gb": round(reserved, 2),
    }
    logger.info("gpu memory", extra={"label": label, **info})
    return info


# ── Parameter counting ────────────────────────────────────────────────────────

def log_trainable_params(model: Any) -> dict[str, Any]:
    """Count and log trainable vs. frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
    }
    logger.info("parameter counts", extra=info)
    print(
        f"  Trainable params: {trainable:,}  ({info['trainable_pct']}% of {total:,})"
    )
    return info


# ── Loss curve analysis ───────────────────────────────────────────────────────

def analyze_trainer_state(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read trainer_state.json from a checkpoint and analyze the loss curve.

    Returns a dict with loss statistics and a pass/fail assessment.
    """
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.exists():
        # Try the parent directory (trainer saves state at the best checkpoint)
        for candidate in Path(checkpoint_dir).rglob("trainer_state.json"):
            state_path = candidate
            break
        else:
            raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_dir}")

    state = json.loads(state_path.read_text())
    log_history: list[dict] = state.get("log_history", [])

    train_losses = [
        (e["step"], e["loss"])
        for e in log_history
        if "loss" in e and "eval_loss" not in e
    ]
    eval_losses = [
        (e["step"], e["eval_loss"])
        for e in log_history
        if "eval_loss" in e
    ]
    grad_norms = [
        (e["step"], e["grad_norm"])
        for e in log_history
        if "grad_norm" in e
    ]

    result: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "total_steps": state.get("global_step", 0),
        "train_loss_first": train_losses[0][1] if train_losses else None,
        "train_loss_last": train_losses[-1][1] if train_losses else None,
        "eval_loss_best": min((l for _, l in eval_losses), default=None),
        "eval_loss_last": eval_losses[-1][1] if eval_losses else None,
        "n_nan": sum(1 for _, l in train_losses if math.isnan(l)),
        "grad_norm_max": max((g for _, g in grad_norms), default=None),
        "grad_norm_last": grad_norms[-1][1] if grad_norms else None,
    }

    # ── Checks ────────────────────────────────────────────────────────────────
    checks: dict[str, bool] = {}

    if result["train_loss_first"] and result["train_loss_last"]:
        checks["loss_decreased"] = result["train_loss_last"] < result["train_loss_first"]
        checks["initial_loss_sane"] = 1.0 < result["train_loss_first"] < 10.0
        checks["final_loss_nonzero"] = result["train_loss_last"] > 0.01
    checks["no_nan"] = result["n_nan"] == 0

    if result["grad_norm_max"] is not None:
        checks["grad_norm_reasonable"] = result["grad_norm_max"] < 50.0

    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result


def print_smoke_test_report(result: dict[str, Any]) -> None:
    """Print a human-readable smoke test report."""
    print("\n" + "=" * 58)
    print("SMOKE TEST REPORT")
    print("=" * 58)
    print(f"  Checkpoint:        {result['checkpoint_dir']}")
    print(f"  Steps completed:   {result['total_steps']}")
    print()

    if result["train_loss_first"] and result["train_loss_last"]:
        delta = result["train_loss_first"] - result["train_loss_last"]
        print(f"  Train loss:  {result['train_loss_first']:.3f} → {result['train_loss_last']:.3f}  (↓ {delta:.3f})")
    if result["eval_loss_last"]:
        print(f"  Val loss:    {result['eval_loss_last']:.3f}  (best: {result['eval_loss_best']:.3f})")
    if result["grad_norm_max"] is not None:
        print(f"  Grad norm:   max={result['grad_norm_max']:.2f}  last={result['grad_norm_last']:.2f}")
    if result["n_nan"] > 0:
        print(f"  NaN steps:   {result['n_nan']}  ← CRITICAL FAILURE")

    print()
    print("  Checks:")
    for check, passed in result.get("checks", {}).items():
        icon = "✓" if passed else "✗"
        print(f"    {icon} {check}")

    print()
    if result.get("passed"):
        print("  RESULT: PASS — proceed to full training (session 11)")
    else:
        failed = [k for k, v in result.get("checks", {}).items() if not v]
        print(f"  RESULT: FAIL — fix before proceeding: {failed}")

    print("=" * 58)


def print_ascii_loss_curve(checkpoint_dir: str | Path) -> None:
    """Print an ASCII loss curve from trainer_state.json."""
    state_path = next(Path(checkpoint_dir).rglob("trainer_state.json"), None)
    if not state_path:
        print("No trainer_state.json found.")
        return

    log_history = json.loads(state_path.read_text()).get("log_history", [])
    train_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    if not train_losses:
        print("No training loss entries found.")
        return

    # Normalize to 0-1 for display
    lo, hi = min(train_losses), max(train_losses)
    rows = 10
    cols = min(len(train_losses), 60)
    step = max(1, len(train_losses) // cols)
    sampled = train_losses[::step][:cols]

    print("\nTraining loss curve:")
    for row in range(rows, 0, -1):
        threshold = lo + (hi - lo) * (row / rows)
        line = "".join("█" if l <= threshold else " " for l in sampled)
        loss_label = f"{threshold:5.2f} |"
        print(f"  {loss_label} {line}")
    print(f"  {'':>6}  " + "─" * len(sampled))
    print(f"  {'':>6}  step 1{' ' * (len(sampled) - 12)}step {len(train_losses)}")


# ── Health callback ───────────────────────────────────────────────────────────

class TrainingHealthCallback(TrainerCallback):
    """Logs gradient norms and checks for NaN loss at each logging step.

    Attach to SFTTrainer via trainer.add_callback(TrainingHealthCallback()).
    """

    def __init__(self, fail_on_nan: bool = True) -> None:
        self.fail_on_nan = fail_on_nan
        self._nan_steps = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return

        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")

        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            self._nan_steps += 1
            logger.error(
                "NaN/Inf loss detected",
                extra={"step": state.global_step, "loss": loss},
            )
            if self.fail_on_nan:
                control.should_training_stop = True

        if grad_norm is not None and grad_norm > 50.0:
            logger.warning(
                "large gradient norm",
                extra={"step": state.global_step, "grad_norm": grad_norm},
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._nan_steps > 0:
            logger.error("training ended with NaN losses", extra={"nan_steps": self._nan_steps})
        else:
            logger.info("training completed without NaN losses")
