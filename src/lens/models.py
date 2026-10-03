from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, model_validator

# A model role resolved through config.MODEL_REGISTRY — never a raw model ID.
Role = Literal["root", "sub", "dev"]

# JEV question types.
QuestionType = Literal["noul", "choice", "score"]

@dataclass
class CallResult:
    """Structured result of one chat() transport call.

    Returned to the orchestrator, which owns message history. 
    The transport is
    stateless
    """

    text: str
    message: dict[str, Any]
    model: str
    provider: str
    model_snapshot: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    finish_reason: str | None = None


@dataclass
class JevResult:
    """Structured result of one jev() decision call"""

    answers: dict[str, dict[str, Any]]
    model: str
    provider: str
    tokens_in: int
    cost_usd: float
    latency_ms: int

class Question(BaseModel):
    """One typed question for the JEV decisions endpoint.

    `criteria` shape depends on `type`:
      - "noul"   -> dict[str, str]   exactly {"true": ..., "false": ...}
      - "choice" -> dict[str, str]   label -> description, >= 2 options
      - "score"  -> list[str]        ordered legend, low -> high, >= 2
    """

    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str]

    @model_validator(mode="after")
    def _enforce_criteria_shape(self) -> "Question":

        # Score type must be a list[str] with >= 2 entries
        if self.type == "score":
            if isinstance(self.criteria, dict):
                raise ValueError("Criteria for Score type questions should be a list.")

            if len(self.criteria) < 2:
                raise ValueError("Criteria list for Score type questions should atleast contain 2 labels.")

        # Choice type must be a dict[str, str] with >= 2 entries
        elif self.type == "choice":
            if isinstance(self.criteria, list):
                raise ValueError("Criteria for Choice type questions should be a dictionary.")
            
            if len(self.criteria) < 2:
                raise ValueError("Criteria dictionary for Choice type questions must be 2 key-value pairs or more.")

        # Noul type must be a dict[str, str] with only 2 entries where keys must be 'true' and 'false'
        else: 
            if isinstance(self.criteria, list):
                raise ValueError("Criteria for Noul type questions should be a dictionary.")

            if set(self.criteria) != {"true", "false"}:
                raise ValueError("Criteria dictionary for Noul type questions must have 2 key-value pairs which are 'true' and 'false'")

        return self