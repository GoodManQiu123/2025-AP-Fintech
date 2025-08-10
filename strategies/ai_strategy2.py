# strategies/ai_strategy3.py
"""
AIStrategy2: LLM-driven trading strategy with JSON-only I/O, hard guardrails,
and optional 'notes' to steer each decision turn.

Design goals
------------
1) Keep stable style/risk/metric definitions in the *system* message.
2) Each user turn sends only current_time, metrics, portfolio, optional ohlcv,
   and optional notes (hard constraints for the current turn).
3) The model replies with EXACTLY ONE-LINE JSON:
   {"signal":"BUY|SELL|HOLD","units":int,"reason":"...","feedback":"..."}.
4) If JSON parsing fails, retry ONCE with a stricter hint; still failing raises.
5) Engine/portfolio integration:
   - Supports multi-unit scaling in/out.
   - Enforces cash/position constraints (no sell when flat; no shorting; no
     overbuy beyond cash/capacity).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

from core.llm.chat_agent import ChatAgent
from core.metrics import RollingWindow
from core.strategy_base import Strategy
from core.types import MarketData, Signal


# ------------------------------ Style guides ------------------------------ #
_STYLE_GUIDES: Dict[str, str] = {
    "scalp": (
        "STYLE=SCALP\n"
        "- Objective: capture very short swings; prefer quick exits.\n"
        "- Entry bias: buy when price << short_sma with compression (RSI<35 & low vol);\n"
        "  exit on rebounds (RSI>60 or zscore>1) or when signals conflict.\n"
        "- Risk: small size; avoid chasing large candles.\n"
    ),
    "swing": (
        "STYLE=SWING\n"
        "- Objective: hold days–weeks; combine trend and mean reversion.\n"
        "- Entry bias: in uptrend (short_sma>long_sma) buy pullbacks (zscore<-1);\n"
        "  in downtrend (short_sma<long_sma) sell/trim on rebounds (zscore>1).\n"
        "- Risk: scale in/out; reduce trades when signals conflict.\n"
    ),
    "invest": (
        "STYLE=INVEST\n"
        "- Objective: hold weeks–months; focus on regime & momentum.\n"
        "- Entry bias: accumulate after bullish cross (short_sma>long_sma, RSI 45–60);\n"
        "  trim on overbought (RSI>70) or extreme deviation (zscore>2) when momentum fades.\n"
        "- Risk: fewer, higher-conviction trades; preserve capital in drawdowns.\n"
    ),
}


def _build_system_prompt(style: str) -> str:
    """Compose the system message with style & hard rules."""
    guide = _STYLE_GUIDES.get(style, _STYLE_GUIDES["swing"])
    return (
        "You are an elite trading agent. Output MUST be ONE LINE of STRICT JSON:\n"
        '{"signal":"BUY|SELL|HOLD","units":<int>,"reason":"<short>","feedback":"<1 short sentence>"}\n'
        "NO extra text. NO markdown. NO multi-line.\n"
        "\n"
        "Metrics provided each turn:\n"
        "- price: latest close; short_sma: short moving average of closes;\n"
        "- long_sma: long moving average; volatility: stddev of short window;\n"
        "- rsi: Relative Strength Index (0–100) over recent changes;\n"
        "- zscore: (price - short_sma) / volatility (stabilized for near-zero vol).\n"
        "If available, you may also see OHLCV fields (open/high/low/close/volume).\n"
        "\n"
        f"{guide}\n"
        "HARD CONSTRAINTS you must obey in the JSON decision:\n"
        "- If portfolio.position_units==0 then SELL is invalid; answer HOLD with units=0.\n"
        "- If SELL then units ≤ portfolio.position_units (no shorting).\n"
        "- If BUY then units ≤ portfolio.affordable_units AND ≤ portfolio.available_capacity.\n"
        "- For BUY or SELL, units must be > 0.\n"
        "- Treat any 'notes' provided by user as HARD, turn-specific constraints.\n"
        "The 'reason' explains this turn's decision; 'feedback' gives a brief\n"
        "reflection on the previous action in light of current metrics (use 'initial' if none).\n"
    )


# ------------------------------ Utilities ------------------------------ #
def _safe_int(v: int | float | str) -> int:
    """Robust int conversion (JSON may deliver strings)."""
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        try:
            return int(float(v))  # type: ignore[arg-type]
        except Exception:
            return 0


# ------------------------------ Strategy ------------------------------ #
class AIStrategy2(Strategy):
    """LLM trading strategy with strict JSON, retry-once, and notes support."""

    def __init__(
        self,
        *,
        style: str = "swing",
        start_cash: float = 10_000.0,
        short_win: int = 10,
        long_win: int = 30,
        rsi_win: int = 14,
        max_units: int = 500,
        history_days: Optional[int] = 180,
        verbose_llm: bool = True,
        # inference history window length (trimmed context); keep generous
        max_history: int = 64,
        # retry policy when JSON parse fails
        retry_once_on_parse_error: bool = True,
    ) -> None:
        self._style = style if style in _STYLE_GUIDES else "swing"
        self._retry_once = retry_once_on_parse_error

        # System message holds stable guidance & constraints.
        system_prompt = _build_system_prompt(self._style)

        # Chat agent: if your ChatAgent supports JSON mode & hyperparams,
        # it will enforce JSON outputs; otherwise, we still do a single retry.
        try:
            # Newer ChatAgent (with JSON mode & hyperparameters).
            self._chat = ChatAgent(
                system_prompt=system_prompt,
                verbose=verbose_llm,
                max_history=max_history,
                # Prefer JSON mode if supported by your ChatAgent implementation.
                json_mode=True,              # type: ignore[call-arg]
                temperature=0.1,             # type: ignore[call-arg]
                top_p=1.0,                   # type: ignore[call-arg]
                frequency_penalty=0.0,       # type: ignore[call-arg]
                presence_penalty=0.0,        # type: ignore[call-arg]
                max_tokens=120,              # type: ignore[call-arg]
            )
        except TypeError:
            # Backward-compatible fall-back (older ChatAgent signature).
            self._chat = ChatAgent(
                system_prompt=system_prompt,
                verbose=verbose_llm,
                max_history=max_history,
            )

        # Indicators
        self._short = RollingWindow(short_win)
        self._long = RollingWindow(long_win)
        self._rsi = RollingWindow(rsi_win)

        # Prompt-side portfolio mirror
        self._cash = float(start_cash)
        self._units = 0
        self._avg_cost = 0.0
        self._max_units = int(max_units)

        # Historical warm-up
        self._history_days = history_days
        self._hist: Deque[float] = deque()
        self._history_sent = False

        # Engine reads to size execution
        self.last_units: int = 0

    # ------------------------------ Helpers ------------------------------ #
    @staticmethod
    def _zscore(price: float, short_sma: float, vol: float) -> float:
        vol = max(vol, 1e-12)
        return (price - short_sma) / vol

    def _affordable(self, price: float) -> int:
        return int(self._cash // price)

    def _capacity(self) -> int:
        return max(self._max_units - self._units, 0)

    def _compute_metrics(self, price: float) -> Dict[str, float]:
        short_sma = self._short.sma()
        long_sma = self._long.sma()
        vol = max(self._short.std(), 1e-12)
        rsi = self._rsi.rsi()
        z = self._zscore(price, short_sma, vol)
        return {
            "price": price,
            "short_sma": short_sma,
            "long_sma": long_sma,
            "volatility": vol,
            "rsi": rsi,
            "zscore": z,
        }

    def _build_user_turn(self, bar: MarketData) -> str:
        """Build compact user payload; include notes when applicable."""
        price = float(bar.price)  # type: ignore[arg-type]
        metrics = self._compute_metrics(price)

        payload: Dict[str, object] = {
            "current_time": bar.time,
            "metrics": metrics,
            "portfolio": {
                "cash": self._cash,
                "position_units": self._units,
                "position_value": self._units * price,
                "avg_cost": self._avg_cost if self._units > 0 else 0.0,
                "affordable_units": self._affordable(price),
                "available_capacity": self._capacity(),
                "max_units": self._max_units,
            },
        }

        # Optional OHLCV if available
        ohlcv: Dict[str, float] = {}
        if bar.open is not None:
            ohlcv["open"] = float(bar.open)  # type: ignore[arg-type]
        if bar.high is not None:
            ohlcv["high"] = float(bar.high)  # type: ignore[arg-type]
        if bar.low is not None:
            ohlcv["low"] = float(bar.low)  # type: ignore[arg-type]
        if bar.close is not None:
            ohlcv["close"] = float(bar.close)  # type: ignore[arg-type]
        if bar.volume is not None:
            try:
                ohlcv["volume"] = float(bar.volume)  # type: ignore[arg-type]
            except Exception:
                pass
        if ohlcv:
            payload["ohlcv"] = ohlcv

        # Optional notes (hard, turn-specific constraints)
        notes: list[str] = []
        if self._units == 0:
            notes.append("position_units=0: SELL is disallowed; only BUY or HOLD permitted this turn.")
        # Add a few lightweight programmatic hints
        if metrics["rsi"] >= 80:
            notes.append("RSI very high: consider taking profit or waiting; avoid fresh BUY at extremes.")
        if metrics["rsi"] <= 20:
            notes.append("RSI very low: avoid panic SELL; consider mean-reversion entries.")
        if abs(metrics["zscore"]) >= 2.0:
            notes.append("Absolute zscore >= 2 indicates extreme deviation; act cautiously.")
        if notes:
            payload["notes"] = notes

        return json.dumps(payload, separators=(",", ":"))

    def _ask_model_once(self, user_payload: str) -> Dict[str, object]:
        """Send one request to the LLM and parse JSON into a dict."""
        reply = self._chat.send(user_payload)
        obj = json.loads(reply)  # Let this raise if not JSON
        # Normalize and keep only required keys (robustness)
        return {
            "signal": str(obj.get("signal", "HOLD")).upper(),
            "units": _safe_int(obj.get("units", 0)),
            "reason": str(obj.get("reason", "")).strip(),
            "feedback": str(obj.get("feedback", "")).strip() or "initial",
        }

    def _decide_with_retry(self, user_payload: str) -> Dict[str, object]:
        """
        Ask the model for a decision; retry once with a stricter prefix if JSON parse fails.
        This keeps runtime simple while providing robustness.
        """
        try:
            return self._ask_model_once(user_payload)
        except Exception:
            if not self._retry_once:
                raise
            strict = "JSON_ONLY Strictly output the JSON object only. " + user_payload
            return self._ask_model_once(strict)

    # ------------------------------ API hooks ------------------------------ #
    def observe(self, bar: MarketData) -> None:
        """Warm up metrics and send one-time pre-entry history summary."""
        if bar.price is None:
            return
        price = float(bar.price)
        self._short.push(price)
        self._long.push(price)
        self._rsi.push(price)

        # accumulate history for a single summary message
        self._hist.append(price)
        if (
            not self._history_sent
            and self._history_days is not None
            and len(self._hist) >= self._history_days
        ):
            first = self._hist[0]
            last = self._hist[-1]
            stats = {
                "period_days": len(self._hist),
                "min": min(self._hist),
                "max": max(self._hist),
                "mean": sum(self._hist) / len(self._hist),
                "first_price": first,
                "last_price": last,
                "change_pct": ((last - first) / first * 100.0) if first else 0.0,
            }
            # Keep historical context light; the system prompt already defines behavior.
            self._chat.send("HISTORY_SUMMARY " + json.dumps(stats, separators=(",", ":")))
            self._history_sent = True

    def generate_signal(self, bar: MarketData) -> Signal:
        """Main decision loop: build user payload, query LLM, enforce constraints, mirror state."""
        price_opt = bar.price
        if price_opt is None:
            self.last_units = 0
            return Signal.HOLD
        price = float(price_opt)

        # Continue warming if history_days=None
        if not self._history_sent and self._history_days is None:
            self.observe(bar)

        # Update indicators
        self._short.push(price)
        self._long.push(price)
        self._rsi.push(price)

        # Require full windows before acting
        if not (self._short.full and self._long.full and self._rsi.full):
            self.last_units = 0
            return Signal.HOLD

        # Build compact user payload and query model
        payload = self._build_user_turn(bar)
        decision = self._decide_with_retry(payload)

        sig_text = str(decision["signal"]).upper()
        req_units = _safe_int(decision["units"])
        # reason = decision["reason"]  # reserved for logging/analysis if needed
        # feedback = decision["feedback"]

        # ---------------- Hard constraints (double safety) ----------------
        if self._units == 0 and sig_text == "SELL":
            sig_text, req_units = "HOLD", 0

        if sig_text == "BUY":
            cap = min(self._affordable(price), self._capacity())
            if cap <= 0:
                sig_text, req_units = "HOLD", 0
            else:
                req_units = min(max(req_units, 1), cap)

        elif sig_text == "SELL":
            if self._units <= 0:
                sig_text, req_units = "HOLD", 0
            else:
                req_units = min(max(req_units, 1), self._units)

        elif sig_text != "HOLD":
            # Unknown signal → hold
            sig_text, req_units = "HOLD", 0

        # ---------------- Mirror cash/position for prompts ----------------
        if sig_text == "BUY" and req_units > 0:
            cost = req_units * price
            self._cash -= cost
            if self._units == 0:
                self._avg_cost = price
            else:
                total_before = self._avg_cost * self._units
                self._avg_cost = (total_before + cost) / (self._units + req_units)
            self._units += req_units
            self.last_units = req_units
            return Signal.BUY

        if sig_text == "SELL" and req_units > 0:
            self._cash += req_units * price
            self._units -= req_units
            if self._units == 0:
                self._avg_cost = 0.0
            self.last_units = req_units
            return Signal.SELL

        # HOLD
        self.last_units = 0
        return Signal.HOLD

    # ------------------------------ Exports ------------------------------ #
    def export_chat_logs(self, dst_dir: Path) -> None:
        """Export dialog (messages only) and full metadata JSON via ChatAgent."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        self._chat.export_dialog_json(dst_dir / "dialog.json")
        self._chat.export_full_json(dst_dir / "conversation_full.json")


# Factory for engine dynamic import
def build(**kwargs) -> Strategy:
    return AIStrategy2(**kwargs)
