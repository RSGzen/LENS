"""Shared data contracts for the LENS harness (transport + decision layer).

Spec: ``Thesis Drafts/Tech Stack (Draft).md`` §2a (model routing + JEV question
types) and §4a (transport responsibilities).

Types
-----
- ``Role``        role selector, resolved through ``config.MODEL_REGISTRY``.
- ``Question``    validated input contract for the JEV decisions endpoint.
- ``CallResult``  one chat-completion result (returned to the orchestrator).
- ``JevResult``   one JEV decisions result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, model_validator

# A model role resolved through config.MODEL_REGISTRY — never a raw model ID.
Role = Literal["root", "sub", "dev"]

# JEV question kinds (Tech Stack §2a).
QuestionType = Literal["noul", "choice", "score"]


@dataclass
class CallResult:
    """Result of one ``chat()`` transport call.

    Returned to the orchestrator, which owns message history; the transport
    itself is stateless (Tech Stack §4a).

    ``text`` is the assistant's visible reply; ``message`` is the raw assistant
    message dict (``{"role": "assistant", "content": ...}``) that the
    orchestrator appends to history. ``model`` is the requested model id, and
    ``model_snapshot`` is the dated model that actually served the request (E-04).
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
    """Result of one ``jev()`` decision call (Tech Stack §2a).

    ``answers`` maps each question name to its typed answer object; ``model`` is
    the dated snapshot returned by the (single-provider) decision layer.
    """

    answers: dict[str, dict[str, Any]]
    model: str
    provider: str
    tokens_in: int
    cost_usd: float
    latency_ms: int


class Question(BaseModel):
    """One typed question for the JEV decisions endpoint.

    ``criteria`` shape depends on ``type``:
      - ``"noul"``   -> ``dict[str, str]``   exactly ``{"true": ..., "false": ...}``
      - ``"choice"`` -> ``dict[str, str]``   label -> description, >= 2 options
      - ``"score"``  -> ``list[str]``        ordered legend, low -> high, >= 2

    Validation runs at *construction* time (pydantic), so callers must build
    ``Question`` objects before passing them to ``clients.jev()``.
    """

    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str]

    @model_validator(mode="after")
    def _enforce_criteria_shape(self) -> "Question":
        """Enforce the per-``type`` criteria shape; raises ``ValidationError``."""

        # Score must be a list[str] with >= 2 ordered labels.
        if self.type == "score":
            if isinstance(self.criteria, dict):
                raise ValueError("Criteria for Score type questions should be a list.")
            if len(self.criteria) < 2:
                raise ValueError("Criteria list for Score type questions should at least contain 2 labels.")

        # Choice must be a dict[str, str] with >= 2 options.
        elif self.type == "choice":
            if isinstance(self.criteria, list):
                raise ValueError("Criteria for Choice type questions should be a dictionary.")
            if len(self.criteria) < 2:
                raise ValueError("Criteria dictionary for Choice type questions must have 2 key-value pairs or more.")

        # Noul must be a dict whose keys are exactly {"true", "false"}.
        else:
            if isinstance(self.criteria, list):
                raise ValueError("Criteria for Noul type questions should be a dictionary.")
            if set(self.criteria) != {"true", "false"}:
                raise ValueError("Criteria dictionary for Noul type questions must have keys 'true' and 'false'.")

        return self
