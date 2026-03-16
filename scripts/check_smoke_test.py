"""Inspect smoke test output and give a go/no-go for full training.

Run AFTER `python training/train_sft.py --smoke-test` completes.

Reads trainer_state.json from the smoke test checkpoint directory,
checks all health indicators, and prints a pass/fail report with
specific guidance on what to fix if anything failed.

Usage:
    python scripts/check_smoke_test.py
    python scripts/check_smoke_test.py --checkpoint checkpoints/sft-smoke
"""

import argparse
from pathlib import Path

from training.utils import analyze_trainer_state, print_ascii_loss_curve, print_smoke_test_report

DEFAULT_SMOKE_DIR = "checkpoints/sft-smoke"


def guidance(result: dict) -> None:
    """Print actionable guidance based on failed checks."""
    checks = result.get("checks", {})
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        return

    print("\nGuidance for failed checks:")
    for f in failed:
        if f == "loss_decreased":
            print(
                "  loss_decreased FAILED:\n"
                "    → Gradient may not be reaching LoRA parameters.\n"
                "    → Verify prepare_model_for_kbit_training() was called.\n"
                "    → Check that target_modules match actual layer names:\n"
                "       python -c \"from transformers import AutoModelForCausalLM; "
                "m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); "
                "print([n for n,_ in m.named_modules()][:30])\""
            )
        elif f == "final_loss_nonzero":
            print(
                "  final_loss_nonzero FAILED (loss = 0):\n"
                "    → All labels are -100. Response template not found.\n"
                "    → The DataCollatorForCompletionOnlyLM found no assistant turns.\n"
                "    → Check that 'response_template' in config matches the model's\n"
                "      chat template assistant opener exactly (including whitespace).\n"
                "    → Debug: tokenizer.encode('<|im_start|>assistant\\n') and compare\n"
                "      against actual token IDs in a sample batch."
            )
        elif f == "no_nan":
            print(
                "  no_nan FAILED:\n"
                "    → NaN loss detected. Check:\n"
                "    → Is bf16=True set? (required for Ampere+ GPUs)\n"
                "    → Is the learning rate too high? Try lr=5e-5.\n"
                "    → Are there any degenerate examples (empty section_text)?\n"
                "    → Add clip_grad_norm=1.0 to SFTConfig if not already set."
            )
        elif f == "initial_loss_sane":
            print(
                "  initial_loss_sane FAILED:\n"
                f"    → Initial loss {result['train_loss_first']:.2f} is outside [1.0, 10.0].\n"
                "    → If loss < 1: model may already be overfitted (wrong checkpoint loaded).\n"
                "    → If loss > 10: loss masking may be misconfigured; check labels."
            )
        elif f == "grad_norm_reasonable":
            print(
                f"  grad_norm_reasonable FAILED (max={result['grad_norm_max']:.1f}):\n"
                "    → Very large gradient norms suggest learning rate is too high.\n"
                "    → Try lr=5e-5 or lr=1e-4. Gradient clipping (max_grad_norm=1.0)\n"
                "      is set by default in SFTConfig but verify it's not overridden."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_SMOKE_DIR)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(
            f"Checkpoint directory not found: {ckpt}\n"
            "Run the smoke test first:\n"
            "  python training/train_sft.py --smoke-test"
        )
        return

    try:
        result = analyze_trainer_state(ckpt)
    except FileNotFoundError as e:
        print(f"Could not read training state: {e}")
        return

    print_ascii_loss_curve(ckpt)
    print_smoke_test_report(result)
    guidance(result)

    if result.get("passed"):
        print("\nNext steps:")
        print("  1. Review the loss curve above — expect a clear downward trend.")
        print("  2. Check W&B (if enabled) for gradient norm history.")
        print(f"  3. Note your final train loss: {result['train_loss_last']:.3f}")
        print("  4. Run full training: python training/train_sft.py")
        print("     Consider enabling packing=True in training/config.py for ~30% speedup.")


if __name__ == "__main__":
    main()
