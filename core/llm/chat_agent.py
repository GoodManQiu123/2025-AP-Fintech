"""Lightweight OpenAI Chat wrapper with conversation capture & export."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from openai import OpenAI, OpenAIError


@dataclass(slots=True)
class ChatMessage:
    """Internal container mirroring OpenAI chat schema."""
    role: str    # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        """Convert to the dict format expected by the OpenAI client."""
        return {"role": self.role, "content": self.content}


class ChatAgent:
    """Stateful chat wrapper with optional verbose output and full logging."""

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_history: int = 20,
        verbose: bool = True,
    ) -> None:
        """
        Args:
          system_prompt: initial instruction for the assistant.
          model: OpenAI model name.
          api_key: override env var OPENAI_API_KEY if provided.
          max_history: maximum messages to keep in context (for inference).
          verbose: if True, print each user message and assistant reply.
        """
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._messages: List[ChatMessage] = [ChatMessage("system", system_prompt)]
        # keep FULL history separately, never trimmed (for analysis export)
        self._full_history: List[ChatMessage] = [ChatMessage("system", system_prompt)]
        # per-turn request/response records for rich metadata export
        self._turn_records: List[Dict[str, Any]] = []

        self._model = model
        self._max_history = max_history
        self._verbose = verbose

    # ----------------------------- Core API --------------------------------
    def send(self, user_msg: str) -> str:
        """
        Send a user message and receive assistant reply.
        - Maintains a trimmed context window for inference.
        - Keeps complete conversation & per-turn metadata for export.
        """
        # Append user message
        user = ChatMessage("user", user_msg)
        self._messages.append(user)
        self._full_history.append(user)
        self._trim_history()

        # Prepare payload for API
        payload = [m.as_dict() for m in self._messages]

        # Call OpenAI API
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=payload,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        assistant_content: str = response.choices[0].message.content
        assistant = ChatMessage("assistant", assistant_content)
        self._messages.append(assistant)
        self._full_history.append(assistant)
        self._trim_history()

        # Store per-turn metadata (robust to SDK differences)
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
                    # snapshot trimmed messages used for this turn
                    "messages": payload,
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

        # Optional verbose printing
        if self._verbose:
            print("\n[USER] ")
            print(user_msg)
            print("\n[ASSISTANT] ")
            print(assistant_content)
            print("-" * 60)

        return assistant_content

    # ----------------------------- Utilities -------------------------------
    def _trim_history(self) -> None:
        """Ensure trimmed message list length ≤ max_history (keep system prompt)."""
        excess = len(self._messages) - self._max_history
        if excess > 0:
            del self._messages[1 : 1 + excess]  # preserve index 0 (system)

    # ----------------------------- Exports ---------------------------------
    def export_dialog_json(self, path: str | os.PathLike) -> None:
        """
        Export ONLY the full conversation as a simple JSON array of messages,
        without timestamps or extra metadata (system/user/assistant roles kept).
        """
        data = [m.as_dict() for m in self._full_history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def export_full_json(self, path: str | os.PathLike) -> None:
        """
        Export complete experiment metadata:
        - model & config
        - per-turn request/response snapshots (messages, usage, ids, etc.)
        - full conversation for redundancy
        """
        blob: Dict[str, Any] = {
            "model": self._model,
            "config": {"max_history": self._max_history, "verbose": self._verbose},
            "turns": self._turn_records,
            "conversation": [m.as_dict() for m in self._full_history],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)

    # ----------------------------- Accessors -------------------------------
    @property
    def history(self) -> List[ChatMessage]:
        """Return a copy of the trimmed message history used for inference."""
        return list(self._messages)

    @property
    def full_history(self) -> List[ChatMessage]:
        """Return the complete (untrimmed) conversation history."""
        return list(self._full_history)
