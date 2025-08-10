"""
AIStrategy2: LLM-driven trading strategy with guardrails and FIFO-friendly behavior.

Highlights
----------
- Strict JSON output: {"signal":"BUY|SELL|HOLD","units":int,"reason":"..."}.
- Uses full pre-entry history to warm up indicators.
- Supports multi-unit scaling in/out; never sells when flat; never oversells/buys beyond cash.
- Robust JSON parsing with regex fallback.
- Periodic self-critique memory (reasons + reflections) fed back to prompts.
- Switchable trading styles (scalp / swing / invest) via constructor.
- Exports dialog to JSON via ChatAgent hooks.

Note:
Portfolio now uses FIFO lot accounting, so partial sells generate multiple Trade
records. This strategy mirrors cash/units for prompts and hard-constraints but
does not attempt to perfectly mirror FIFO lots internally (YAGNI).
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

from core.llm.chat_agent import ChatAgent
from core.metrics import RollingWindow
from core.strategy_base import Strategy
from core.types import MarketData, Signal

# --------------------------------------------------------------------------- #
#                             Prompt style guides                              #
# --------------------------------------------------------------------------- #
STYLE_GUIDES: Dict[str, str] = {
    "scalp": (
        "STYLE: SCALP\n"
        "- Objective: capture very short swings; prefer quick exits.\n"
        "- Bias: buy when price ≪ short_sma with compression (RSI<35 & low vol);\n"
        "        sell when price ≫ short_sma with expansion (RSI>65 or zscore>1).\n"
        "- Risk: small size and tight control; avoid chasing.\n"
    ),
    "swing": (
        "STYLE: SWING\n"
        "- Objective: hold days–weeks; combine trend and mean-reversion.\n"
        "- Bias: buy pullbacks in uptrend (short_sma>long_sma & zscore<-1);\n"
        "        sell rebounds in downtrend (short_sma<long_sma & zscore>1).\n"
        "- Risk: scale in/out; avoid overtrading on conflicting signals.\n"
    ),
    "invest": (
        "STYLE: INVEST\n"
        "- Objective: hold weeks–months; prioritise regime & trend.\n"
        "- Bias: accumulate after bullish cross (short_sma>long_sma, RSI 45–60);\n"
        "        trim when momentum fades (RSI>70 then rolls) or zscore>2.\n"
        "- Risk: fewer but higher-conviction trades; preserve capital.\n"
    ),
}

SYSTEM_PROMPT = (
    "You are an elite trading agent. You MUST reply with EXACTLY ONE LINE of STRICT JSON:\n"
    '{"signal":"BUY|SELL|HOLD","units": <integer>=0,"reason":"<short rationale>"}\n'
    "HARD RULES:\n"
    "- Do not add any extra text, no markdown, no multi-line, only the JSON object.\n"
    "- If position_units==0 then SELL is invalid → output HOLD, units=0, reason='no_position'.\n"
    "- If SELL then units ≤ position_units (no shorting). If BUY then units ≤ affordable_units AND ≤ available_slots.\n"
    "- Use confluence of trend (short_sma vs long_sma), momentum (RSI), and deviation (zscore=(price-short_sma)/volatility).\n"
    "- Prefer HOLD when signals conflict; reasons must be concise.\n"
)

# Accepts: {"signal":"BUY|SELL|HOLD","units":<int>,"reason":"..."}
_JSON_RE = re.compile(
    r'"?\s*signal"?\s*:\s*"?([A-Za-z]+)"?\s*,\s*"?units"?\s*:\s*("?[-+]?\d+"?)\s*,\s*"?reason"?\s*:\s*"([^"]*)"\s*}',
    re.S,
)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class AIStrategy2(Strategy):
    """LLM strategy with strict guardrails, memory and multi-unit capability."""

    # ----------------------------- init ----------------------------------- #
    def __init__(
        self,
        *,
        # style: "scalp" | "swing" | "invest"
        style: str = "swing",
        start_cash: float = 10_000.0,
        metrics_window_short: int = 10,
        metrics_window_long: int = 30,
        rsi_window: int = 14,
        verbose_llm: bool = True,
        max_units: int = 10,
        # memory & review
        memory_size: int = 16,
        review_interval: int = 10,
        # pre-entry history warmup length (None → stream until filled)
        history_days: Optional[int] = 180,
        # soft risk limits (inform prompts; portfolio controls realised risk)
        risk_max_loss_pct: float = 5.0,
        risk_max_loss_abs: float = 0.0,
    ) -> None:
        # portfolio mirror for prompts / constraints
        self._cash = start_cash
        self._units = 0
        self._avg_cost = 0.0  # simple weighted-average for prompts only
        self._max_units = max_units

        # indicators
        self._short = RollingWindow(metrics_window_short)
        self._long = RollingWindow(metrics_window_long)
        self._rsi = RollingWindow(rsi_window)

        # warmup history
        self._history_days = history_days
        self._history: Deque[float] = deque()
        self._history_sent = False

        # memory & self-review
        self._recent_reasons: Deque[str] = deque(maxlen=memory_size)
        self._recent_reflections: Deque[str] = deque(maxlen=memory_size)
        self._review_every = max(1, int(review_interval))
        self._step = 0
        self._last_sig: Optional[str] = None
        self._last_units: int = 0
        self._last_price: Optional[float] = None
        self._last_time: Optional[str] = None

        # risk hints
        self._risk_max_loss_pct = float(risk_max_loss_pct)
        self._risk_max_loss_abs = float(risk_max_loss_abs)

        # style
        key = style.lower().strip()
        if key == "scalping":  # alias
            key = "scalp"
        if key == "position":  # alias
            key = "invest"
        self._style = key if key in STYLE_GUIDES else "swing"
        self._style_guide = STYLE_GUIDES[self._style]

        # chat
        self._chat = ChatAgent(system_prompt=SYSTEM_PROMPT, verbose=verbose_llm)

        # engine reads this to send to Portfolio.execute(...)
        self.last_units: int = 0

    # --------------------------- observe / warmup -------------------------- #
    def observe(self, bar: MarketData) -> None:
        """Warm up indicators using pre-entry bars."""
        if bar.price is None:
            return
        self._short.push(bar.price)
        self._long.push(bar.price)
        self._rsi.push(bar.price)

        # historical summary once
        self._history.append(bar.price)
        if (
            not self._history_sent
            and self._history_days is not None
            and len(self._history) >= self._history_days
        ):
            first = self._history[0]
            last = self._history[-1]
            stats = {
                "period_days": len(self._history),
                "min": min(self._history),
                "max": max(self._history),
                "mean": sum(self._history) / len(self._history),
                "first_price": first,
                "last_price": last,
                "change_pct": (last - first) / first * 100.0 if first else 0.0,
            }
            self._chat.send("HISTORY_SUMMARY " + json.dumps(stats, separators=(",", ":")))
            self._history_sent = True

    # ------------------------------- prompts -------------------------------- #
    def _zscore(self, price: float, short_sma: float, vol: float) -> float:
        vol = max(vol, 1e-12)
        return (price - short_sma) / vol

    def _build_decide_prompt(self, price: float) -> str:
        short_sma = self._short.sma()
        long_sma = self._long.sma()
        vol = max(self._short.std(), 1e-12)
        rsi_val = self._rsi.rsi()
        zscore = self._zscore(price, short_sma, vol)

        affordable = int(self._cash // price)
        available_slots = max(self._max_units - self._units, 0)

        payload = {
            "style": self._style,
            "style_guide": self._style_guide,
            "metrics": {
                "price": price,
                "short_sma": short_sma,
                "long_sma": long_sma,
                "volatility": vol,
                "rsi": rsi_val,
                "zscore": zscore,
            },
            "portfolio": {
                "cash": self._cash,
                "position_units": self._units,
                "position_value": self._units * price,
                "avg_cost": self._avg_cost if self._units > 0 else 0.0,
                "affordable_units": affordable,
                "available_slots": available_slots,
                "max_units": self._max_units,
            },
            "risk_limits": {
                "max_loss_pct": self._risk_max_loss_pct,
                "max_loss_abs": self._risk_max_loss_abs,
            },
            "memory": {
                "recent_reasons": list(self._recent_reasons),
                "recent_reflections": list(self._recent_reflections),
            },
        }
        return "DECIDE_JSON_ONLY " + json.dumps(payload, separators=(",", ":"))

    # --------------------------- parsing utilities ------------------------- #
    @staticmethod
    def _parse_json_reply(text: str) -> Tuple[str, int, str]:
        """Robustly parse {"signal","units","reason"}; fallback to regex."""
        try:
            obj = json.loads(text)
            sig = str(obj.get("signal", "HOLD")).upper()
            units = int(obj.get("units", 0))
            reason = str(obj.get("reason", "")).strip()
            return sig, units, reason
        except Exception:
            m = _JSON_RE.search(text)
            if m:
                sig = m.group(1).upper()
                units = int(m.group(2).strip('"'))
                reason = m.group(3).strip()
                return sig, units, reason
        return "HOLD", 0, "parse_error"

    # ------------------------------ fallback -------------------------------- #
    def _fallback(self, price: float) -> Tuple[str, int, str]:
        """Very conservative backup when model output is invalid."""
        if not (self._short.full and self._long.full and self._rsi.full):
            return "HOLD", 0, "warming_up"
        short_sma = self._short.sma()
        long_sma = self._long.sma()
        vol = max(self._short.std(), 1e-12)
        rsi_val = self._rsi.rsi()
        z = self._zscore(price, short_sma, vol)

        up, down = short_sma > long_sma, short_sma < long_sma

        units = min(1 + int(min(2, abs(z) // 1.5)), self._max_units, int(self._cash // price))

        if up and z < -1.0 and rsi_val < 55 and units > 0:
            return "BUY", units, "fallback_pullback_in_uptrend"
        if down and z > 1.0 and rsi_val > 45 and self._units > 0:
            return "SELL", min(self._units, max(1, units)), "fallback_rebound_in_downtrend"
        if self._units > 0 and rsi_val > 70 and z > 1.5:
            return "SELL", min(self._units, 2), "fallback_take_profit"
        return "HOLD", 0, "fallback_hold"

    # --------------------------- self review -------------------------------- #
    def _maybe_self_review(self, cur_price: float, cur_time: str) -> None:
        if self._step % self._review_every != 0:
            return
        payload = {
            "last_signal": self._last_sig or "NONE",
            "last_units": self._last_units,
            "last_price": self._last_price if self._last_price is not None else 0.0,
            "current_price": cur_price,
            "elapsed_bars": self._review_every,
            "prior_reasons": list(self._recent_reasons),
        }
        text = self._chat.send(
            "REVIEW_DECISIONS Return ≤2 compact sentences: (1) what was good, (2) what to avoid next.\n"
            + json.dumps(payload, separators=(",", ":"))
        )
        if text:
            self._recent_reflections.append(text.strip())

    # ------------------------------- main API -------------------------------- #
    def generate_signal(self, bar: MarketData) -> Signal:
        price = bar.price
        if price is None:
            self.last_units = 0
            return Signal.HOLD

        # If history_days=None keep warming while streaming
        if not self._history_sent and self._history_days is None:
            self.observe(bar)

        # Update indicators
        self._short.push(price)
        self._long.push(price)
        self._rsi.push(price)

        # Periodic self-review before decision
        self._step += 1
        self._maybe_self_review(price, bar.time)

        if not (self._short.full and self._long.full and self._rsi.full):
            self.last_units = 0
            return Signal.HOLD

        prompt = self._build_decide_prompt(price)
        reply = self._chat.send(prompt)
        sig_text, req_units, reason = self._parse_json_reply(reply)

        # Enforce hard guardrails
        sig_text = sig_text.upper()
        req_units = max(0, int(req_units))

        if sig_text == "SELL" and self._units <= 0:
            sig_text, req_units, reason = "HOLD", 0, "no_position"

        if sig_text == "BUY":
            affordable = int(self._cash // price)
            available_slots = max(self._max_units - self._units, 0)
            cap = max(min(affordable, available_slots), 0)
            req_units = min(req_units if req_units > 0 else 1, cap)
            if req_units <= 0:
                sig_text, reason = "HOLD", "insufficient_cash_or_capacity"

        elif sig_text == "SELL":
            req_units = min(req_units if req_units > 0 else 1, self._units)
            if req_units <= 0:
                sig_text, reason = "HOLD", "nothing_to_sell"

        # If model suggested something invalid → conservative fallback
        if sig_text not in {"BUY", "SELL", "HOLD"}:
            sig_text, req_units, reason = self._fallback(price)

        # Mirror updates & emit signal
        if sig_text == "BUY":
            cost = req_units * price
            self._cash -= cost
            # update avg_cost (weighted average for remaining position)
            if self._units == 0:
                self._avg_cost = price
            else:
                total_before = self._avg_cost * self._units
                self._avg_cost = (total_before + cost) / (self._units + req_units)
            self._units += req_units
            self._recent_reasons.append(reason or "buy")
            self.last_units = req_units

            # record last action
            self._last_sig, self._last_units, self._last_price, self._last_time = (
                "BUY",
                req_units,
                price,
                bar.time,
            )
            return Signal.BUY

        if sig_text == "SELL":
            revenue = req_units * price
            self._cash += revenue
            self._units -= req_units
            if self._units == 0:
                self._avg_cost = 0.0  # reset when flat
            self._recent_reasons.append(reason or "sell")
            self.last_units = req_units

            self._last_sig, self._last_units, self._last_price, self._last_time = (
                "SELL",
                req_units,
                price,
                bar.time,
            )
            return Signal.SELL

        # HOLD
        self._recent_reasons.append(reason or "hold")
        self.last_units = 0
        self._last_sig, self._last_units, self._last_price, self._last_time = (
            "HOLD",
            0,
            price,
            bar.time,
        )
        return Signal.HOLD

    # --------------------------- conversation dumps ------------------------- #
    def export_chat_logs(self, dst_dir: Path) -> None:
        """Write dialog.json (messages only) and conversation_full.json (rich metadata)."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        self._chat.export_dialog_json(dst_dir / "dialog.json")
        self._chat.export_full_json(dst_dir / "conversation_full.json")


# Factory for engine dynamic import
def build(**kwargs) -> Strategy:
    return AIStrategy2(**kwargs)
