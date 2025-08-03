"""Lightweight OpenAI Chat wrapper with conversation context."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from openai import OpenAI, OpenAIError


@dataclass(slots=True)
class ChatMessage:
    """Internal container mirroring OpenAI chat schema."""
    role: str    # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        """Convert to the dict format expected by the OpenAI client."""
        return {"role": self.role, "content": self.content}

DEFAULT_KEY = "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"

class ChatAgent:
    """Stateful chat wrapper with optional verbose output."""

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        api_key: str | None = DEFAULT_KEY,
        max_history: int = 20,
        verbose: bool = True,
    ) -> None:
        """
        Args:
          system_prompt: initial instruction for the assistant.
          model: OpenAI model name.
          api_key: override env var OPENAI_API_KEY if provided.
          max_history: maximum messages to keep in context.
          verbose: if True, print each user message and assistant reply.
        """
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._messages: List[ChatMessage] = [ChatMessage("system", system_prompt)]
        self._model = model
        self._max_history = max_history
        self._verbose = verbose

    def send(self, user_msg: str) -> str:
        """
        Send a user message, receive assistant reply.

        If verbose, prints user_msg and assistant reply to console.
        """
        print(f"\n[USER] {user_msg}" if self._verbose else "")
        # Append user message
        self._messages.append(ChatMessage("user", user_msg))
        self._trim_history()

        # Prepare payload
        payload = [msg.as_dict() for msg in self._messages]

        # Call OpenAI API
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=payload,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        assistant_content = response.choices[0].message.content

        # Record assistant reply
        self._messages.append(ChatMessage("assistant", assistant_content))
        self._trim_history()

        # Verbose output
        if self._verbose:
            print("\n[USER] ")
            print(user_msg)
            print("\n[ASSISTANT] ")
            print(assistant_content)
            print("-" * 60)

        return assistant_content

    def _trim_history(self) -> None:
        """Ensure message history length ≤ max_history (keep system prompt)."""
        excess = len(self._messages) - self._max_history
        if excess > 0:
            # preserve index 0 (system), drop oldest user/assistant messages
            del self._messages[1 : 1 + excess]

    @property
    def history(self) -> List[ChatMessage]:
        """Get a copy of the current message history."""
        return list(self._messages)