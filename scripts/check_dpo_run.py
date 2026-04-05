"""Inspect DPO training output and give a go/no-go for session 18 eval.

Run AFTER `python training/train_dpo.py` completes.

Reads trainer_state.json from the DPO checkpoint directory, checks reward
margin health indicators, and prints a pass/fail report with specific guidance
on what to fix if anything failed.

Usage:
    python scripts/check_dpo_run.py
    python scripts/check_dpo_run.py --checkpoint checkpoints/dpo
"""

import argparse
from pathlib import Path

from training.utils import (
    analyze_dpo_trainer_state,
    print_ascii_reward_curve,
    print_dpo_report,
)

DEFAULT_DPO_DIR = "checkpoints/dpo"


def guidance(result: dict) -> None:
    checks = result.get("checks", {})
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        return

    print("\nGuidance for failed checks:")
    for f in failed:
        if f == "margin_increased":
            print(
                "  margin_increased FAILED:\n"
                "    → Reward margin did not grow during training.\n"
                "    → Check that preference pairs have real quality differences.\n"
                "      Run: python scripts/inspect_preferences.py\n"
                "    → If most pairs have F1 gap < 0.1, degradation was too subtle.\n"
                "    → Try lowering beta (e.g. --beta 0.05) for stronger optimization."
            )
        elif f == "accuracy_above_chance":
            print(
                "  accuracy_above_chance FAILED:\n"
                f"    → Reward accuracy {result.get('accuracy_mean', 0):.3f} ≤ 0.60.\n"
                "    → The model cannot reliably distinguish chosen from rejected.\n"
                "    → Most likely cause: preference pairs too similar (small margin).\n"
                "    → Regenerate dataset with stronger degradation:\n"
                "       python scripts/generate_preferences.py --stronger"
            )
        elif f == "margin_positive":
            print(
                "  margin_positive FAILED:\n"
                "    → chosen_reward ≤ rejected_reward at final step.\n"
                "    → The model has inverted its preferences — something is wrong\n"
                "      with the (chosen, rejected) assignment.\n"
                "    → Verify the dataset: python scripts/inspect_preferences.py\n"
                "    → Check that 'chosen' really is the better output in each pair."
            )
        elif f == "no_nan":
            print(
                "  no_nan FAILED:\n"
                "    → NaN loss during DPO. Most common causes:\n"
                "    → Learning rate too high — try lr=1e-5.\n"
                "    → Beta too low (< 0.01) causes log-ratio overflow — try beta=0.1.\n"
                "    → Add max_grad_norm=1.0 if not already set."
            )
        elif f == "grad_norm_reasonable":
            print(
                f"  grad_norm_reasonable FAILED (max={result.get('grad_norm_max', 0):.1f}):\n"
                "    → Large gradients. Try lr=1e-5 or beta=0.2."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_DPO_DIR)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(
            f"Checkpoint directory not found: {ckpt}\n"
            "Run DPO training first:\n"
            "  python training/train_dpo.py"
        )
        return

    try:
        result = analyze_dpo_trainer_state(ckpt)
    except FileNotFoundError as e:
        print(f"Could not read training state: {e}")
        return

    print_ascii_reward_curve(ckpt)
    print_dpo_report(result)
    guidance(result)

    if result.get("passed"):
        print("\nNext steps:")
        print("  1. Review the reward margin curve — expect a clear upward trend.")
        print(f"  2. Final margin: {result['margin_last']:.4f}  accuracy: {result['accuracy_last']:.3f}")
        print("  3. Export DPO checkpoint: python scripts/export_checkpoint.py \\")
        print(f"       --adapter {args.checkpoint} --output checkpoints/dpo-merged")
        print("  4. Run DPO eval: python scripts/eval_sft.py --model-dir checkpoints/dpo-merged")
        print("  5. Compare: python scripts/compare_evals.py  (session 18)")


if __name__ == "__main__":
    main()
