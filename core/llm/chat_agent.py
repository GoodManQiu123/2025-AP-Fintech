"""OpenAI chat wrapper with hyperparameters, JSON mode, and full conversation export.

This module centralizes:
  - A lightweight, stateful chat client (adjustable context window).
  - Global/per-call hyperparameters (temperature, top_p, penalties, max_tokens, seed, stop).
  - Strict JSON mode using typed `ResponseFormatJSONObject`.
  - Complete conversation capture and rich, per-turn metadata export.

Requires: openai >= 1.0.0
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError
from openai._types import NOT_GIVEN, NotGiven
from openai.types import ResponseFormatJSONObject


# --------------------------------------------------------------------------- #
#                                  Data types                                  #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ChatMessage:
    """Simple container mirroring the OpenAI chat schema."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# --------------------------------------------------------------------------- #
#                                  ChatAgent                                   #
# --------------------------------------------------------------------------- #
class ChatAgent:
    """Stateful chat wrapper with hyperparameters, JSON mode, and full logging.

    Example:
      agent = ChatAgent(
          system_prompt="You are a JSON-only assistant.",
          model="gpt-4o-mini",
          json_mode=True,
          temperature=0.2,
          max_tokens=200,
          verbose=True,
      )
      reply = agent.send('{"task":"decide"}')

    Notes:
      - `json_mode=True` enforces JSON output via typed response_format.
      - Per-call overrides are supported: `agent.send(msg, temperature=0.0, max_tokens=64)`.
      - Use `export_dialog_json()` for a minimal message dump,
        and `export_full_json()` for complete experiment metadata.
    """

    # ------------------------------- lifecycle ------------------------------ #
    def __init__(
        self,
        system_prompt: str,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        # context window
        max_history: int = 20,
        # console echo
        verbose: bool = True,
        # JSON mode
        json_mode: bool = True,
        # default hyperparameters
        temperature: float = 0.2,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        # optional HTTP headers
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            default_headers=extra_headers or None,
        )

        # histories
        self._messages: List[ChatMessage] = [ChatMessage("system", system_prompt)]
        self._full_history: List[ChatMessage] = [ChatMessage("system", system_prompt)]
        self._turn_records: List[Dict[str, Any]] = []

        # config
        self._model = model
        self._max_history = max(3, int(max_history))
        self._verbose = verbose
        self._json_mode = bool(json_mode)

        # defaults (overridable per-call)
        self._defaults: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "frequency_penalty": float(frequency_penalty),
            "presence_penalty": float(presence_penalty),
            "max_tokens": int(max_tokens),
            "seed": int(seed) if seed is not None else None,
            "stop": list(stop) if stop is not None else None,
        }

    # ------------------------------- core API -------------------------------- #
    def send(self, user_msg: str, **overrides: Any) -> str:
        """Send a user message and return the assistant text reply."""
        # Update histories
        user = ChatMessage("user", user_msg)
        self._messages.append(user)
        self._full_history.append(user)
        self._trim_history()

        # Prepare payload & params
        payload_messages = [m.as_dict() for m in self._messages]
        req = self._merge_params(overrides)

        # Typed response_format per JSON mode (or NotGiven)
        rf: ResponseFormatJSONObject | NotGiven = NOT_GIVEN
        json_mode_override = overrides.get("json_mode", None)
        json_mode = bool(json_mode_override) if json_mode_override is not None else self._json_mode
        if json_mode:
            rf = ResponseFormatJSONObject(type="json_object")

        # Call OpenAI
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=payload_messages,
                response_format=rf,
                temperature=req["temperature"],
                top_p=req["top_p"],
                frequency_penalty=req["frequency_penalty"],
                presence_penalty=req["presence_penalty"],
                max_tokens=req["max_tokens"],
                seed=req["seed"] if req["seed"] is not None else NOT_GIVEN,
                stop=req["stop"] if req["stop"] is not None else NOT_GIVEN,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        # Extract content
        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = ""

        # Update histories
        assistant = ChatMessage("assistant", content)
        self._messages.append(assistant)
        self._full_history.append(assistant)
        self._trim_history()

        # Capture usage/metadata
        usage: Dict[str, Optional[int]] = {}
        try:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
        except Exception:
            usage = {}

        self._turn_records.append(
            {
                "request": {
                    "model": self._model,
                    "messages": payload_messages,
                    "params": {
                        "json_mode": json_mode,
                        "temperature": req["temperature"],
                        "top_p": req["top_p"],
                        "frequency_penalty": req["frequency_penalty"],
                        "presence_penalty": req["presence_penalty"],
                        "max_tokens": req["max_tokens"],
                        "seed": req["seed"],
                        "stop": req["stop"],
                        "response_format": "json_object" if json_mode else "text",
                    },
                },
                "response": {
                    "id": getattr(response, "id", None),
                    "created": getattr(response, "created", None),
                    "model": getattr(response, "model", self._model),
                    "usage": usage,
                    "message": assistant.as_dict(),
                },
            }
        )

        if self._verbose:
            print("\n[USER]\n" + user_msg)
            print("\n[ASSISTANT]\n" + content)
            print("-" * 60)

        return content

    # ----------------------------- helpers ---------------------------------- #
    def _merge_params(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge defaults with per-call overrides (with light validation)."""
        merged: Dict[str, Any] = dict(self._defaults)
        for key in (
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
            "seed",
            "stop",
        ):
            if key in overrides and overrides[key] is not None:
                merged[key] = overrides[key]

        merged["temperature"] = float(merged["temperature"])
        merged["top_p"] = float(merged["top_p"])
        merged["frequency_penalty"] = float(merged["frequency_penalty"])
        merged["presence_penalty"] = float(merged["presence_penalty"])
        merged["max_tokens"] = int(merged["max_tokens"])
        merged["seed"] = int(merged["seed"]) if merged["seed"] is not None else None
        if merged["stop"] is not None:
            merged["stop"] = list(merged["stop"])
        return merged

    def _trim_history(self) -> None:
        """Keep inference history ≤ max_history (preserve first system message)."""
        excess = len(self._messages) - self._max_history
        if excess > 0:
            del self._messages[1 : 1 + excess]

    # ------------------------------ mutation -------------------------------- #
    def set_params(self, **kwargs: Any) -> None:
        """Update default hyperparameters for future `send()` calls."""
        self._defaults.update(self._merge_params(kwargs))

    def set_json_mode(self, enabled: bool) -> None:
        """Toggle JSON mode globally."""
        self._json_mode = bool(enabled)

    def set_verbose(self, enabled: bool) -> None:
        """Toggle console echo globally."""
        self._verbose = bool(enabled)

    # ------------------------------- exports -------------------------------- #
    def export_dialog_json(self, path: str | os.PathLike) -> None:
        """Export ONLY the full conversation as a JSON array of messages."""
        data = [m.as_dict() for m in self._full_history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def export_full_json(self, path: str | os.PathLike) -> None:
        """Export complete metadata: config, per-turn snapshots, and full history."""
        blob: Dict[str, Any] = {
            "model": self._model,
            "config": {
                "max_history": self._max_history,
                "verbose": self._verbose,
                "json_mode": self._json_mode,
                "defaults": self._defaults,
            },
            "turns": self._turn_records,
            "conversation": [m.as_dict() for m in self._full_history],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)

    # ------------------------------- accessors ------------------------------ #
    @property
    def history(self) -> List[ChatMessage]:
        """Return a copy of the trimmed history used for inference."""
        return list(self._messages)

    @property
    def full_history(self) -> List[ChatMessage]:
        """Return the full, untrimmed conversation history."""
        return list(self._full_history)
