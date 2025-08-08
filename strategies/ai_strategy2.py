"""
AIStrategy2: LLM-driven trading strategy with:
- strict JSON {signal, units, reason}
- pre-entry indicator warm-up (uses full history before entry date)
- adaptive position sizing and safety constraints
- rule-based fallback when LLM output is invalid
- self-critique loop: model summarizes prior decision quality and we feed that back
- pluggable trading style guidance (scalping / swing / position)
"""
from __future__ import annotations

import json
import re
from collections import deque
from typing import Deque, Optional

from core.llm.chat_agent import ChatAgent
from core.metrics import RollingWindow
from core.strategy_base import Strategy
from core.types import MarketData, Signal

# Accepts: {"signal":"BUY|SELL|HOLD","units":<int>,"reason":"..."}
_JSON_RE = re.compile(
    r'"?signal"?\s*:\s*"?(\w+)"?\s*,\s*"?units"?\s*:\s*"?([0-9]+)"?(?:\s*,\s*"?reason"?\s*:\s*"(.*?)")?',
    re.IGNORECASE | re.DOTALL,
)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class AIStrategy2(Strategy):
    """LLM strategy with dynamic reasoning memory and periodic self-review."""

    # ----------------------------- init -----------------------------------
    def __init__(
        self,
        *,
        start_cash: float = 10_000.0,
        history_days: Optional[int] = 120,
        short_window: int = 20,
        long_window: int = 50,
        rsi_window: int = 14,
        max_units: int = 10,
        trading_style: str = "swing",  # "scalping" | "swing" | "position"
        verbose_llm: bool = True,
        # risk controls (can be tuned externally)
        risk_max_loss_pct: float = 5.0,     # stop if unrealized loss < -5%
        risk_max_loss_abs: float = 0.0,     # or absolute stop in currency; 0 disables
        review_interval: int = 10,          # ask model to self-critique every N bars
        memory_limit: int = 24,             # how many reasons/reflections to remember
    ) -> None:
        # portfolio mirror (cash-aware prompts)
        self._cash = start_cash
        self._units = 0
        self._avg_cost = 0.0
        self._max_units = max_units

        # history warm-up
        self._history_days = history_days
        self._hist_prices: Deque[float] = deque()
        self._history_sent = False

        # rolling metrics
        self._short = RollingWindow(short_window)
        self._long = RollingWindow(long_window)
        self._rsiw = RollingWindow(rsi_window)

        # style guide text
        self._style = trading_style.lower().strip()
        self._style_text = self._build_style_text(self._style)

        # risk
        self._risk_max_loss_pct = float(risk_max_loss_pct)
        self._risk_max_loss_abs = float(risk_max_loss_abs)

        # self-review cadence
        self._review_every = max(1, int(review_interval))
        self._step = 0

        # memory of model's reasons + periodic reflections
        self._reasons: Deque[str] = deque(maxlen=memory_limit)
        self._reflections: Deque[str] = deque(maxlen=max(6, memory_limit // 2))

        # last decision snapshot for review
        self._last_sig: Optional[str] = None
        self._last_units: int = 0
        self._last_price: Optional[float] = None
        self._last_time: Optional[str] = None

        # strict JSON system prompt + guardrails
        system_prompt = (
            "You are an elite trading agent. You ONLY reply with ONE LINE of STRICT JSON:\n"
            '{"signal":"BUY|SELL|HOLD","units": <integer>=0,"reason":"<short rationale>"}\n'
            "Rules:\n"
            "- Never add text before/after JSON. No markdown. No commentary.\n"
            "- Respect cash and position constraints. If BUY, units <= affordable and <= max limit. "
            "If SELL, units <= current position_units.\n"
            "- Use confluence: trend (short_sma vs long_sma), momentum (RSI), and deviation (zscore= (price-short_sma)/volatility).\n"
            "- Consider prior reasons and reflections to avoid repeating mistakes.\n"
            "- If signal is weak/ambiguous, HOLD with units=0 and give a concise reason.\n"
        )
        self._chat = ChatAgent(system_prompt=system_prompt, verbose=verbose_llm)

        # expose last executed units for engine/portfolio
        self.last_units: int = 0

    # ------------------------ style templates ------------------------------
    @staticmethod
    def _build_style_text(style: str) -> str:
        if style == "scalping":
            return (
                "STYLE: SCALPING\n"
                "- Objective: capture small intraday swings; prefer quick in/out.\n"
                "- Bias: prioritize mean-reversion near short_sma; react to extreme zscores.\n"
                "- Risk: tight sizing; avoid holding through adverse momentum; HOLD when unclear.\n"
            )
        if style == "position":
            return (
                "STYLE: POSITION (long-term)\n"
                "- Objective: ride multi-week trends; avoid noise.\n"
                "- Bias: trade with long_sma direction; only fade extremes with strong confirmation.\n"
                "- Risk: fewer but larger conviction trades; avoid frequent flips.\n"
            )
        # default swing
        return (
            "STYLE: SWING\n"
            "- Objective: hold for days-weeks; combine trend and reversion.\n"
            "- Bias: buy pullbacks in uptrend (short_sma>long_sma & zscore<-1), "
            "sell rebounds in downtrend (short_sma<long_sma & zscore>1).\n"
            "- Risk: scale in/out; avoid overtrading when signals conflict.\n"
        )

    # --------------------------- parsing -----------------------------------
    @staticmethod
    def _parse_llm_json(reply: str) -> tuple[str, int, str]:
        """Parse LLM JSON; fallback to regex; on failure return ('HOLD',0,'invalid')."""
        try:
            obj = json.loads(reply.strip())
            sig = str(obj.get("signal", "")).upper()
            units = int(obj.get("units", 0))
            reason = str(obj.get("reason", "")).strip()
            return sig, units, reason
        except Exception:
            m = _JSON_RE.search(reply)
            if m:
                sig = m.group(1).upper()
                units = int(m.group(2))
                reason = (m.group(3) or "").strip()
                return sig, units, reason
        return "HOLD", 0, "invalid_or_non_json"

    # --------------------------- utilities ---------------------------------
    def _affordable_units(self, price: float) -> int:
        return max(0, int(self._cash // price))

    def _bounded_buy(self, requested: int, price: float) -> int:
        return _clamp_int(requested if requested > 0 else 1, 0, min(self._max_units, self._affordable_units(price)))

    def _bounded_sell(self, requested: int) -> int:
        return _clamp_int(requested if requested > 0 else 1, 0, self._units)

    def _zscore(self, price: float, short_sma: float, vol: float) -> float:
        vol = max(vol, 1e-12)
        return (price - short_sma) / vol

    # --------------------------- warm-up -----------------------------------
    def observe(self, bar: MarketData) -> None:
        """Warm-up all indicators with history before entry date."""
        if bar.price is None:
            return
        self._short.push(bar.price)
        self._long.push(bar.price)
        self._rsiw.push(bar.price)

        self._hist_prices.append(bar.price)
        if (
            not self._history_sent
            and self._history_days is not None
            and len(self._hist_prices) >= self._history_days
        ):
            prices = list(self._hist_prices)
            stats = {
                "period_days": len(prices),
                "min": min(prices),
                "max": max(prices),
                "mean": sum(prices) / len(prices),
                "first_price": prices[0],
                "last_price": prices[-1],
                "change_pct": ((prices[-1] / prices[0] - 1) * 100.0) if prices[0] else 0.0,
            }
            # One-shot history summary
            self._chat.send("HISTORY_SUMMARY " + json.dumps(stats, separators=(",", ":")))
            self._history_sent = True

    # --------------------------- fallback rule -----------------------------
    def _fallback(self, price: float, short_sma: float, long_sma: float, vol: float, rsi: float) -> tuple[str, int, str]:
        """Conservative backup when LLM output is invalid."""
        z = self._zscore(price, short_sma, vol)
        up = short_sma > long_sma
        down = short_sma < long_sma

        # basic signal strength → units
        units = 1 + int(min(2, abs(z) // 1.5))
        units = min(units, self._max_units, self._affordable_units(price))

        # BUY in uptrend pullback + moderate RSI
        if up and z < -1.0 and rsi < 55 and units > 0:
            return "BUY", units, "fallback_buy_pullback"
        # SELL in downtrend rebound + moderate RSI
        if down and z > 1.0 and rsi > 45 and self._units > 0:
            sell_units = min(self._units, max(1, units))
            return "SELL", sell_units, "fallback_sell_rebound"
        # profit take if strongly overbought while holding
        if self._units > 0 and rsi > 70 and z > 1.5:
            sell_units = max(1, min(self._units, 2))
            return "SELL", sell_units, "fallback_take_profit"
        return "HOLD", 0, "fallback_hold"

    # --------------------------- self review -------------------------------
    def _maybe_self_review(self, cur_price: float, cur_time: str) -> None:
        """Ask the model to reflect on the previous decision and summarize lessons."""
        if self._step % self._review_every != 0:
            return
        # Build a thin review payload (does not require JSON)
        last = {
            "last_signal": self._last_sig or "NONE",
            "last_units": self._last_units,
            "last_price": self._last_price if self._last_price is not None else 0.0,
            "current_price": cur_price,
            "elapsed_bars": self._review_every,
            "prior_reasons": list(self._reasons),
        }
        msg = (
            "REVIEW_DECISIONS\n"
            "Summarize in <=2 bullet-like sentences: (1) what was done well, (2) what to avoid next time.\n"
            "Return plain text (no JSON). Keep it concise.\n"
            + json.dumps(last, separators=(",", ":"))
        )
        text = self._chat.send(msg)
        if text:
            self._reflections.append(text.strip())

    # --------------------------- main decision -----------------------------
    def generate_signal(self, bar: MarketData) -> Signal:
        if bar.price is None:
            self.last_units = 0
            return Signal.HOLD

        # Continuous warm-up if history_days=None
        if not self._history_sent and self._history_days is None:
            self.observe(bar)

        # Update windows with live bar
        self._short.push(bar.price)
        self._long.push(bar.price)
        self._rsiw.push(bar.price)

        # Review (periodic) before making the next decision
        self._step += 1
        self._maybe_self_review(bar.price, bar.time)

        # Need indicators ready
        if not (self._short.full and self._long.full and self._rsiw.full):
            self.last_units = 0
            return Signal.HOLD

        short_sma = self._short.sma()
        long_sma = self._long.sma()
        vol = self._short.std()
        rsi = self._rsiw.rsi()
        z = self._zscore(bar.price, short_sma, vol)

        pos_val = self._units * bar.price
        unreal = 0.0
        unreal_pct = 0.0
        if self._units > 0 and self._avg_cost > 0.0:
            unreal = pos_val - (self._avg_cost * self._units)
            unreal_pct = (bar.price / self._avg_cost - 1.0) * 100.0

        # hard stop check (strategy-side override)
        if self._units > 0:
            stop_hit_pct = unreal_pct <= -abs(self._risk_max_loss_pct)
            stop_hit_abs = self._risk_max_loss_abs > 0 and unreal <= -abs(self._risk_max_loss_abs)
            if stop_hit_pct or stop_hit_abs:
                units = self._units
                # execute forced SELL
                self._cash += units * bar.price
                self._units = 0
                self._avg_cost = 0.0
                self.last_units = units
                # record last action for next review
                self._last_sig, self._last_units, self._last_price, self._last_time = "SELL", units, bar.price, bar.time
                return Signal.SELL

        # Build compact memory (recent reasons + reflections)
        memory_blob = {
            "recent_reasons": list(self._reasons),
            "recent_reflections": list(self._reflections),
        }

        # Decision context
        context = {
            "style": self._style,
            "style_guide": self._style_text,
            "metrics": {
                "price": bar.price,
                "short_sma": short_sma,
                "long_sma": long_sma,
                "volatility": vol,
                "rsi": rsi,
                "zscore": z,
            },
            "portfolio": {
                "cash": self._cash,
                "position_units": self._units,
                "position_value": pos_val,
                "avg_cost": self._avg_cost if self._units > 0 else 0.0,
                "unrealized_gain": unreal,
                "unrealized_gain_pct": unreal_pct,
                "max_units": self._max_units,
            },
            "risk_limits": {
                "max_loss_pct": self._risk_max_loss_pct,
                "max_loss_abs": self._risk_max_loss_abs,
            },
            "memory": memory_blob,
        }

        # Strongly remind: JSON ONLY
        user_msg = "DECIDE_JSON_ONLY " + json.dumps(context, separators=(",", ":"))
        reply = self._chat.send(user_msg)
        sig_text, req_units, reason = self._parse_llm_json(reply)

        # If invalid -> fallback
        if sig_text not in {"BUY", "SELL", "HOLD"}:
            sig_text, req_units, reason = self._fallback(bar.price, short_sma, long_sma, vol, rsi)

        # Enforce constraints + update local portfolio mirror
        if sig_text == "BUY":
            units = self._bounded_buy(req_units, bar.price)
            if units <= 0:
                self.last_units = 0
                self._reasons.append(reason or "buy_blocked_no_cash")
                # record last action (hold) for future review
                self._last_sig, self._last_units, self._last_price, self._last_time = "HOLD", 0, bar.price, bar.time
                return Signal.HOLD
            cost = units * bar.price
            self._cash -= cost
            if self._units == 0:
                self._avg_cost = bar.price
            else:
                total_before = self._avg_cost * self._units
                self._avg_cost = (total_before + cost) / (self._units + units)
            self._units += units
            self.last_units = units
            self._reasons.append(reason or "buy_exec")
            self._last_sig, self._last_units, self._last_price, self._last_time = "BUY", units, bar.price, bar.time
            return Signal.BUY

        if sig_text == "SELL":
            units = self._bounded_sell(req_units)
            if units <= 0:
                self.last_units = 0
                self._reasons.append(reason or "sell_blocked_no_position")
                self._last_sig, self._last_units, self._last_price, self._last_time = "HOLD", 0, bar.price, bar.time
                return Signal.HOLD
            revenue = units * bar.price
            self._cash += revenue
            self._units -= units
            if self._units == 0:
                self._avg_cost = 0.0
            self.last_units = units
            self._reasons.append(reason or "sell_exec")
            self._last_sig, self._last_units, self._last_price, self._last_time = "SELL", units, bar.price, bar.time
            return Signal.SELL

        # HOLD
        self.last_units = 0
        self._reasons.append(reason or "hold")
        self._last_sig, self._last_units, self._last_price, self._last_time = "HOLD", 0, bar.price, bar.time
        return Signal.HOLD


def build(**kwargs) -> Strategy:
    """Factory for dynamic import by the engine."""
    return AIStrategy2(**kwargs)
