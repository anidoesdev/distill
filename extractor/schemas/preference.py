"""Preference pair schema for DPO training.

A preference pair is a (prompt, chosen, rejected) triple where:
  - prompt:   the full formatted conversation up to the assistant turn opener
  - chosen:   the preferred completion (higher-quality extraction)
  - rejected: the dispreferred completion (lower-quality extraction)

The prompt string uses the model's chat template format so it can be passed
directly to DPOTrainer without additional formatting.

Generation strategies (session 16):
  - Teacher vs. degraded:  GPT-4o-mini output as chosen, programmatically
                           degraded version (dropped fields, noise) as rejected
  - Teacher vs. base:      GPT-4o-mini as chosen, base model output as rejected
                           (only for examples where base model was clearly worse)
  - Human correction:      human-audited correction as chosen, original
                           teacher output as rejected (for audited examples)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PreferencePair(BaseModel):
    """One (prompt, chosen, rejected) triple for DPO training."""

    id: str = Field(description="Unique identifier, e.g. arxiv_id + field hash.")
    prompt: str = Field(
        description="Full formatted prompt including system and user turns, "
                    "ending with the assistant turn opener token sequence. "
                    "Produced by applying the model's chat template with "
                    "add_generation_prompt=True."
    )
    chosen: str = Field(description="Preferred completion — the better extraction.")
    rejected: str = Field(description="Dispreferred completion — the worse extraction.")
    source: str = Field(
        default="degraded",
        description="How this pair was generated: "
                    "'degraded' (teacher vs. programmatic noise), "
                    "'base_model' (teacher vs. base model output), "
                    "'human' (human correction vs. teacher output).",
    )
    chosen_score: float | None = Field(
        default=None,
        description="Macro F1 of chosen against human reference, if available.",
    )
    rejected_score: float | None = Field(
        default=None,
        description="Macro F1 of rejected against human reference, if available.",
    )

    @property
    def margin(self) -> float | None:
        """Score gap between chosen and rejected. None if scores unavailable."""
        if self.chosen_score is not None and self.rejected_score is not None:
            return self.chosen_score - self.rejected_score
        return None
