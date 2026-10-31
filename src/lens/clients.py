"""Host transport for OpenRouter: ``chat()`` and ``jev()``.

Call contract
-------------
- Outgoing HTTP goes through ``self.session.post(url, headers=...,
  data=json.dumps(payload), timeout=self.timeout)``; the response exposes
  ``.status_code`` (int) and ``.json()`` (dict).
- ``chat()`` logs one event with ``call_type="chat"``; ``jev()`` one with
  ``call_type="jev"``; each failed attempt logs an event carrying an ``error`` dict.
- Full request/response payloads are content-addressed via
  ``trajectory.store_blob()``; events carry ``prompt_ref`` / ``completion_ref``
  (never the raw text or the API key).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import requests

from lens import config
from lens.config import ModelSpec
from lens.models import CallResult, JevResult, Question, Role
from lens.observability.trajectory import CallType, TraceCtx, Trajectory

# Transient HTTP statuses worth retrying.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LensTransportError(RuntimeError):
    """A transport call failed after exhausting retries.

    ``status`` is the last HTTP status (or ``None`` for a network error);
    ``attempts`` is the number of POSTs made.
    """

    def __init__(self, message: str, *, status: int | None = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.status = status
        self.attempts = attempts


class LensClient:
    """Stateless OpenRouter transport shared by ``chat()`` and ``jev()``.

    Parameters
    ----------
    trajectory:
        The injected observability sink. Every call stores blobs + logs one event.
    api_key:
        OpenRouter key. If ``None``, resolved from the environment via
        ``config.require_api_key()`` (fail-loud). Never stored in a file or logged.
    registry:
        Role -> :class:`ModelSpec` map. Defaults to ``config.MODEL_REGISTRY``;
        injectable so tests can use a spec without provider pins.
    session:
        A ``requests.Session``-like object with ``post(...)``. Injectable for tests.
    sleeper:
        Backoff function (defaults to ``time.sleep``). Injectable so tests never wait.
    """

    def __init__(
        self,
        trajectory: Trajectory,
        *,
        api_key: str | None = None,
        registry: Mapping[Role, ModelSpec] | None = None,
        base_url: str = config.OPENROUTER_BASE_URL,
        jev_endpoint: str = config.JEV_ENDPOINT,
        timeout: float = config.DEFAULT_TIMEOUT_S,
        max_retries: int = config.MAX_RETRIES,
        session: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.trajectory = trajectory
        self.api_key = api_key if api_key is not None else config.require_api_key()
        self.registry = registry if registry is not None else config.MODEL_REGISTRY
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = f"{self.base_url}/chat/completions"
        self.jev_endpoint = jev_endpoint
        self.timeout = timeout
        # Total POST attempts = max_retries + 1 (one initial attempt + retries).
        self.max_retries = max_retries
        self.session = session if session is not None else requests.Session()
        self.sleeper = sleeper

    # ------------------------------------------------------------------ public
    def chat(
        self,
        role: Role,
        messages: Sequence[Mapping[str, Any]],
        *,
        trace_ctx: TraceCtx,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> CallResult:
        """One chat-completions call for ``role``; return a :class:`CallResult`.

        1. Resolve ``spec = self.registry[role]``; build the model slug (append
           ``":exacto"`` when ``spec.exacto``).
        2. Build the request body: ``model``, ``messages`` (as a list), plus
           ``tools`` / ``response_format`` / ``max_tokens`` only when provided
           **and** the spec supports them (gate on ``spec.supports_tools`` /
           ``spec.supports_response_format``). Add ``temperature`` **only when
           ``spec.temperature is not None``** and ``seed`` **only when
           ``spec.seed is not None``** (omit each key otherwise — for GPT-5
           reasoning roles the provider default applies). Add ``"provider"`` from
           :meth:`_provider_payload` **only when it returns a non-empty dict**.
        3. ``store_blob`` the serialized request body -> ``prompt_ref``.
        4. POST via ``self._post(self.chat_endpoint, body, trace_ctx=trace_ctx,
           call_type="chat")`` (handles retries + error logs).
        5. Extract text (``choices[0].message.content``), token counts and cost;
           ``store_blob`` the raw completion text -> ``completion_ref``; measure latency.
        6. ``log_event(call_type="chat", action="chat.completion", ...)`` once.
        7. Return ``CallResult`` (model = requested ``spec.id``; model_snapshot =
           the served ``response["model"]``; provider = ``response.get("provider")``
           or the first pinned provider, else ``"unknown"``).

        ``extra`` rides on the event untouched. It is how a caller that is not a
        loop turn — query segmentation is the one — says so in the log, instead of
        looking like a turn at ``step_idx = 0``.

        ``result_ref`` is **not** used here (it is for host-tool / ``emit()`` outputs).
        """

        model_spec = self.registry[role]
        model_slug = self._model_slug(model_spec)

        # Construct payload for request — omit any absent/unsupported key.
        payload: dict[str, Any] = {
            "model": model_slug,
            "messages": list(messages),
        }

        provider = self._provider_payload(model_spec)
        if provider:
            payload["provider"] = provider

        if tools is not None and model_spec.supports_tools:
            payload["tools"] = list(tools)

        if response_format is not None and model_spec.supports_response_format:
            payload["response_format"] = response_format

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if model_spec.reasoning_effort is not None:
            payload["reasoning_effort"] = model_spec.reasoning_effort

        if model_spec.temperature is not None:
            payload["temperature"] = model_spec.temperature

        if model_spec.seed is not None:
            payload["seed"] = model_spec.seed

        # Store prompt text as blob
        prompt_ref = self.trajectory.store_blob(json.dumps(payload))

        start_timer = time.perf_counter()

        response_json = self._post(
            endpoint=self.chat_endpoint, payload=payload, trace_ctx=trace_ctx, call_type="chat"
        )

        answer_text = response_json["choices"][0]["message"]["content"]

        usage = self._usage(response_json)
        tokens_in = usage["tokens_in"]
        tokens_out = usage["tokens_out"]
        cost = usage["cost_usd"]
        self._note_missing_cost(usage, model_spec.id, trace_ctx)

        latency_ms = int((time.perf_counter() - start_timer) * 1000)

        completion_ref = self.trajectory.store_blob(answer_text)

        self.trajectory.log_event(
            trace_ctx=trace_ctx,
            call_type="chat",
            action="chat.completion",
            model=model_spec.id,
            provider=response_json.get("provider", "unknown"),
            model_snapshot=response_json.get("model"),
            prompt_ref=prompt_ref,
            completion_ref=completion_ref,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens=usage["cached_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            latency_ms=latency_ms,
            cost_usd=cost,
            extra=dict(extra) if extra else None,
        )

        call_result = CallResult(
            text=answer_text,
            message=response_json["choices"][0]["message"],
            model=model_spec.id,
            provider=response_json.get("provider", "unknown"),
            model_snapshot=response_json.get("model"),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
            cached_tokens=usage["cached_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
        )

        return call_result

    def jev(
        self,
        state: str,
        questions: Mapping[str, Question],
        *,
        trace_ctx: TraceCtx,
    ) -> JevResult:
        """One JEV decisions call; return a :class:`JevResult`.

        The body is ``{"model": config.JEV_MODEL_ID, "state": state, "questions":
        {name: q.model_dump() for name, q in questions.items()}}`` — JEV is
        single-provider and not in ``MODEL_REGISTRY``. The serialized
        body is blobbed, POSTed via ``_post`` (retries + error logs), and the
        **top-level** ``usage`` / ``answers`` / dated ``model`` are extracted; the
        raw response is blobbed as ``completion_ref``, one ``log_event`` is
        written, and a :class:`JevResult` is returned.

        Note: JEV ``usage`` is at the top level of the response, not inside
        ``answers`` (an easy mistake).
        """

        if not questions:
            raise ValueError("jev() requires at least one question.")

        # Decision layer is single-provider and not in MODEL_REGISTRY.
        model_slug = config.JEV_MODEL_ID

        payload: dict[str, Any] = {
            "model": model_slug,
            "state": state,
            "questions": {name: q.model_dump() for name, q in questions.items()},
        }

        # Store prompt text as blob
        prompt_ref = self.trajectory.store_blob(json.dumps(payload))

        start_timer = time.perf_counter()

        response_json = self._post(
            endpoint=self.jev_endpoint, payload=payload, trace_ctx=trace_ctx, call_type="jev"
        )

        latency_ms = int((time.perf_counter() - start_timer) * 1000)

        answers = response_json.get("answers", {})
        usage = self._usage(response_json)
        tokens_in = usage["tokens_in"]
        tokens_out = usage["tokens_out"]
        cost = usage["cost_usd"]
        self._note_missing_cost(usage, config.JEV_MODEL_ID, trace_ctx)

        completion_ref = self.trajectory.store_blob(json.dumps(response_json))

        self.trajectory.log_event(
            trace_ctx=trace_ctx,
            call_type="jev",
            action="jev.decisions",
            model=model_slug,
            provider=response_json.get("provider"),
            model_snapshot=response_json.get("model"),
            prompt_ref=prompt_ref,
            completion_ref=completion_ref,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

        return JevResult(
            answers=answers,
            model=response_json.get("model", model_slug),
            provider=response_json.get("provider", "TypeSafe"),
            tokens_in=tokens_in,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    # ----------------------------------------------------------------- private
    def _headers(self) -> dict[str, str]:
        """Auth + content headers for every request.

        Contract: ``{"Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"}``. The key is never written to a log
        or blob.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        return headers

    def _provider_payload(self, spec: ModelSpec) -> dict[str, Any]:
        """Build the OpenRouter ``provider`` routing object for ``spec``.

        Returns ``{}`` when there are no provider pins (so no bare
        ``allow_fallbacks`` is sent); otherwise a dict with ``allow_fallbacks``
        and, when set, ``only`` / ``order``.
        """
        if not spec.provider_only and not spec.provider_order:
            return {}

        provider: dict[str, Any] = {"allow_fallbacks": spec.allow_fallbacks}
        if spec.provider_only:
            provider["only"] = list(spec.provider_only)
        if spec.provider_order:
            provider["order"] = list(spec.provider_order)
        return provider

    def _model_slug(self, spec: ModelSpec) -> str:
        """Return the request model id, appending ``":exacto"`` when ``spec.exacto``."""

        return f"{spec.id}:exacto" if spec.exacto else spec.id

    def _request_error_handling(
        self,
        error_type: str,
        status_code: int | None,
        attempt: int,
        wait_time: float | int,
        trace_ctx: TraceCtx,
        call_type: CallType,
    ) -> None:
        """Log one failed attempt as a ``{call_type}.retry`` event, then back off."""

        error_dict = {"type": error_type, "status": status_code, "attempt": attempt}

        self.trajectory.log_event(
            trace_ctx=trace_ctx, call_type=call_type, action=f"{call_type}.retry", error=error_dict
        )

        self.sleeper(wait_time)

    def _post(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        *,
        trace_ctx: TraceCtx,
        call_type: CallType,
    ) -> dict[str, Any]:
        """POST ``payload`` with bounded retries; return the parsed response dict.

        - Attempt at most ``self.max_retries + 1`` times.
        - Retry when ``status in RETRYABLE_STATUS`` or on ``requests.RequestException``.
        - Between attempts call ``self.sleeper(delay)`` with exponential backoff
          ``base * 2 ** attempt`` (e.g. base 1.0 s); no jitter needed.
        - On every failed attempt, log an error event
          (``action=f"{call_type}.retry"``) carrying ``type``, ``status`` (when
          known), and ``attempt``.
        - After the final failure, raise :class:`LensTransportError` with the last
          status + attempt count.
        - On success, return ``response.json()`` (do not log here; the caller logs
          the single success event).
        """

        attempts_made = 0
        last_status = None
        last_exc = None

        for attempt in range(1, self.max_retries + 2):
            attempts_made = attempt
            try:
                response = self.session.post(
                    url=endpoint,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )

            except requests.exceptions.RequestException as e:
                last_exc, last_status = e, None
                self._request_error_handling(
                    "RequestException", None, attempt, 1.0 * 2**attempt, trace_ctx, call_type
                )
                continue

            if response.status_code == 200:
                return response.json()

            last_status = response.status_code

            if response.status_code in RETRYABLE_STATUS:
                retry_after = getattr(response, "headers", {}).get("Retry-After")
                wait_time = int(retry_after) if retry_after else 2**attempt
                self._request_error_handling(
                    "Retryable_Status",
                    response.status_code,
                    attempt,
                    wait_time,
                    trace_ctx,
                    call_type,
                )
                continue

            self._request_error_handling(
                "NonRetryable_Status", response.status_code, attempt, 0, trace_ctx, call_type
            )
            break
        raise LensTransportError(
            f"{call_type} failed after {attempts_made} attempt(s); last={last_status}",
            status=last_status,
            attempts=attempts_made,
        ) from last_exc

    def _note_missing_cost(
        self, usage: Mapping[str, int | float], model: str, trace_ctx: TraceCtx
    ) -> None:
        """Record a call that reported tokens but no cost.

        A provider that renames or drops ``usage.cost`` makes every call free as far
        as the harness can see: no cap binds, ``totals()`` reports ``$0.00``, and
        nothing fails. The note is the only signal that a number is missing rather
        than zero.
        """
        tokens_in = int(usage["tokens_in"])
        tokens_out = int(usage["tokens_out"])

        if (tokens_in or tokens_out) == 0 or float(usage["cost_usd"]) != 0.0:
            return

        self.trajectory.log_event(
            trace_ctx=trace_ctx,
            call_type="note",
            action="cost.anomaly",
            extra={
                "text": (
                    f"{model} reported {tokens_in} in / {tokens_out} out tokens with no "
                    "usage.cost: this call is billed as $0.00, so the cap and every "
                    "reported $/query miss it"
                ),
                "model": model,
            },
        )

    def _usage(self, response: Mapping[str, Any]) -> dict[str, int | float]:
        """Extract the billing fields from an OpenRouter response.

        Chat uses ``usage.prompt_tokens`` / ``usage.completion_tokens``; the decision
        layer uses ``usage.input_tokens`` / ``usage.output_tokens``; ``cost_usd`` is
        ``usage.cost`` in both. Missing values default to ``0`` / ``0.0``.

        Two details are read when the provider reports them, because they are the
        only way to check the cost model afterwards:

        - ``prompt_tokens_details.cached_tokens`` is the part of the prompt that hit
          the provider's prefix cache and is billed at a lower rate. The design
          assumes a stable system prompt makes this most of the input, so it has to
          be measured rather than believed.
        - ``completion_tokens_details.reasoning_tokens`` is the reasoning half of the
          output (gpt-5 bills it as output). On a recorded run it is the largest
          single cost item.
        """

        usage = response.get("usage") or {}
        tokens_in = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        tokens_out = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        cost = usage.get("cost", 0.0) or 0.0

        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached = prompt_details.get("cached_tokens", 0) or 0
        reasoning = completion_details.get("reasoning_tokens", 0) or 0

        return {
            "tokens_in": int(tokens_in),
            "tokens_out": int(tokens_out),
            "cost_usd": float(cost),
            "cached_tokens": int(cached),
            "reasoning_tokens": int(reasoning),
        }
