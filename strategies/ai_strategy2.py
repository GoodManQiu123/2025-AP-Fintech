"""AIStrategy3 — JSON-native, LLM-driven trading with guardrails and scaling.

This module defines a strategy that delegates decision-making to a Large
Language Model (LLM) via `ChatAgent`. The model receives a compact JSON
payload that includes price-derived metrics and a mirrored portfolio state,
and it must respond with a single-line JSON object specifying the trading
signal and the number of units to act on. The local strategy enforces hard
risk constraints (no shorting, no selling when flat, no overbuy beyond
cash/capacity) and mirrors cash/position state after every trade.

Key behavior:
- System prompt: Embeds stable trading principles, risk appetite, JSON
  contract, and (optionally) scaling guidance.
- Per-turn payload: Includes current_time, metrics, portfolio (cash, units,
  capacity, affordability, unrealized PnL), and optional notes (hard,
  turn-specific constraints).
- Strict JSON I/O: The assistant must reply with one-line JSON:
  {"signal":"BUY|SELL|HOLD","units":int,"reason":"...","feedback":"...","insight":"..."}
- Robustness: If parsing fails, the strategy retries once with a stricter
  "JSON_ONLY ..." prefix; otherwise it raises.
- Guardrails: Local enforcement of hard constraints, optional cooldown, and
  RSI/z-score nudges in notes for safer behavior.
- Scaling: Optional multi-step scaling in/out across consecutive bars.

This file is self-contained and integrates with the provided engine/portfolio.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

from core.llm.chat_agent import ChatAgent
from core.metrics import RollingWindow
from core.strategy_base import Strategy
from core.types import MarketData, Signal

# -----------------------------------------------------------------------------
# Style guides embedded in the system prompt.
# -----------------------------------------------------------------------------
_STYLE_GUIDES: Dict[str, str] = {
    "scalp": (
        "STYLE=SCALP\n"
        "- Objective: capture very short swings; prefer quick exits.\n"
        "- Entry bias: buy when price ≪ short_sma with compression (RSI<35 & low vol);\n"
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
    # Free-form, high-risk/high-reward (intentionally minimal guidance to allow creativity).
    "free_high": (
        "STYLE=FREE_HIGH\n"
        "- Objective: pursue high reward with tolerance for drawdowns.\n"
        "- Creative freedom: you may exploit momentum bursts or sharp reversals as you see fit.\n"
        "- Risk: accept variance; still obey hard constraints and JSON contract.\n"
    ),
    # Conservative, low-risk style (also minimal, avoids prescriptive rules).
    "conservative": (
        "STYLE=CONSERVATIVE\n"
        "- Objective: capital preservation with modest returns.\n"
        "- Creative freedom: prefer selective entries and partial profit-taking; reduce churn.\n"
        "- Risk: minimize drawdowns; always honor hard constraints and JSON contract.\n"
    ),
}


def _build_system_prompt(style: str, *, enable_scaling: bool) -> str:
    """Compose the system message containing style, metrics, and hard rules.

    Args:
        style: Strategy style key. Falls back to "swing" if unknown.
        enable_scaling: Whether to allow multi-step scaling guidance.

    Returns:
        A system prompt string to initialize the ChatAgent.
    """
    guide = _STYLE_GUIDES.get(style, _STYLE_GUIDES["swing"])
    scaling = (
        "- Scaling (optional): you may scale in/out across turns (partial BUY/SELL over time). "
        "When SELLing with a remaining position, you may trim partially. "
        "Do not exceed hard constraints.\n"
        if enable_scaling
        else "- Scaling: treat each entry/exit as a single block; avoid frequent partials.\n"
    )
    return (
        "You are an elite trading agent. Output MUST be ONE LINE of STRICT JSON:\n"
        '{"signal":"BUY|SELL|HOLD","units":<int>,"reason":"<short>","feedback":"<1 short sentence>",'
        '"insight":"<short>"}\n'
        "\n"
        "Metrics each turn:\n"
        "- price: latest close; short_sma: short moving average of closes;\n"
        "- long_sma: long moving average; volatility: stddev over short window;\n"
        "- rsi: Relative Strength Index (0–100) over recent price changes;\n"
        "- zscore: (price - short_sma) / volatility (stabilized for near-zero vol).\n"
        "OHLCV may also be provided (open/high/low/close/volume).\n"
        "\n"
        f"{guide}"
        f"{scaling}"
        "HARD CONSTRAINTS:\n"
        "- If portfolio.position_units==0 then SELL is invalid; reply HOLD units=0.\n"
        "- SELL units ≤ portfolio.position_units (no shorting).\n"
        "- BUY units ≤ portfolio.affordable_units AND ≤ portfolio.available_capacity.\n"
        "- Units must be strictly positive for BUY and SELL.\n"
        "The 'reason' explains THIS turn's decision; 'feedback' gives a brief\n"
        "reflection about recent behavior/decision given current metrics.\n"
        "'insight' captures the market's key observation/signal/trend this turn.\n"
    )


def _to_int(value: object) -> int:
    """Robust integer conversion from JSON values.

    Args:
        value: Any JSON-serializable value.

    Returns:
        Integer value parsed from `value`. Returns 0 on failure.
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        try:
            return int(float(value))  # type: ignore[arg-type]
        except Exception:
            return 0


class AIStrategy3(Strategy):
    """LLM strategy with JSON-only I/O, retry-once, scaling, and hard guardrails.

    This class mirrors cash/position locally to inform prompts and enforce
    limits. It can optionally provide a one-time history summary before
    trading and supports a cooldown period after trades.

    Args:
        style: Strategy style key; choices include "scalp", "swing", "invest",
            "free_high", and "conservative". Unknown values fall back to "swing".
        enable_scaling: If True, allows multi-step scaling in/out behavior.
        short_win: Window length for short moving average and volatility.
        long_win: Window length for long moving average.
        rsi_win: Window length for RSI-like metric.
        start_cash: Initial cash balance.
        max_units: Maximum total units allowed (position cap).
        history_days: If set, number of prices to accumulate before sending a
            one-time history summary to the LLM. If None, history accumulates
            continuously but no summary is sent.
        cooldown_bars_after_trade: Number of bars to block new BUYs after any
            trade (SELL/HOLD still allowed).
        model: Model name forwarded to ChatAgent.
        temperature: LLM sampling temperature.
        top_p: Nucleus sampling parameter.
        frequency_penalty: LLM frequency penalty.
        presence_penalty: LLM presence penalty.
        max_tokens: Maximum reply tokens expected from the LLM.
        json_mode: Whether to request JSON-mode behavior from the ChatAgent.
        max_history: Maximum conversation messages to retain.
        verbose_llm: Whether ChatAgent logs additional details.
        retry_once_on_parse_error: Whether to retry once with "JSON_ONLY ..." on
            JSON parsing failures.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *,
        # Behavior & style
        style: str = "free_high",
        enable_scaling: bool = True,
        # Metrics windows
        short_win: int = 10,
        long_win: int = 30,
        rsi_win: int = 14,
        # Cash/position caps
        start_cash: float = 10_000.0,
        max_units: int = 500,
        # Warm-up history
        history_days: Optional[int] = 180,
        # Cooldown after any trade (bars)
        cooldown_bars_after_trade: int = 0,
        # LLM wiring / hyper-parameters (forwarded to ChatAgent)
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 120,
        json_mode: bool = True,
        max_history: int = 64,
        verbose_llm: bool = True,
        retry_once_on_parse_error: bool = True,
    ) -> None:
        # Strategy style & scaling mode.
        self._style = style if style in _STYLE_GUIDES else "swing"
        self._enable_scaling = bool(enable_scaling)

        # Build system prompt.
        system_prompt = _build_system_prompt(
            self._style, enable_scaling=self._enable_scaling
        )

        # Chat agent (supports JSON mode & hyperparameters).
        self._chat = ChatAgent(
            system_prompt=system_prompt,
            model=model,
            max_history=max_history,
            verbose=verbose_llm,
            json_mode=json_mode,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
        )

        # Indicators.
        self._short = RollingWindow(short_win)
        self._long = RollingWindow(long_win)
        self._rsi = RollingWindow(rsi_win)

        # Mirrored portfolio state for prompts and local enforcement.
        self._cash = float(start_cash)
        self._units = 0
        self._avg_cost = 0.0
        self._max_units = int(max_units)

        # Warm-up state.
        self._history_days = history_days
        self._hist: Deque[float] = deque()
        self._history_sent = False

        # Trade cooldown state.
        self._cooldown_cfg = max(0, int(cooldown_bars_after_trade))
        self._cooldown_left = 0

        # Engine consults this for execution sizing.
        self.last_units: int = 0

        # Retry policy on JSON parse errors.
        self._retry_once = bool(retry_once_on_parse_error)

    # -------------------------------------------------------------- utilities
    @staticmethod
    def _zscore(price: float, short_sma: float, vol: float) -> float:
        """Compute the z-score of `price` relative to `short_sma` and `vol`.

        Uses a stabilized denominator to avoid division by zero.

        Args:
            price: Latest price.
            short_sma: Short-term simple moving average.
            vol: Standard deviation over the short window.

        Returns:
            Z-score as a float.
        """
        vol = max(vol, 1e-12)
        return (price - short_sma) / vol

    def _affordable(self, price: float) -> int:
        """Return the maximum whole units purchasable given current cash."""
        return int(self._cash // price)

    def _capacity(self) -> int:
        """Return the remaining capacity before reaching `self._max_units`."""
        return max(self._max_units - self._units, 0)

    def _compute_metrics(self, price: float) -> Dict[str, float]:
        """Compute per-turn metrics used by the LLM.

        Args:
            price: Latest price.

        Returns:
            A dictionary with: price, short_sma, long_sma, volatility, rsi, zscore.
        """
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

    def _build_user_payload(self, bar: MarketData) -> str:
        """Build the compact user JSON string for one decision turn.

        The payload includes metrics, a mirrored portfolio snapshot, optional
        OHLCV fields, and turn-specific notes to reinforce hard constraints.

        Args:
            bar: Latest market data.

        Returns:
            A JSON string (compact separators) to send to the LLM.
        """
        price = float(bar.price)  # type: ignore[arg-type]
        metrics = self._compute_metrics(price)

        # Derived portfolio stats.
        pos_val = self._units * price
        unreal_pnl = (
            (price - self._avg_cost) * self._units if self._units > 0 else 0.0
        )
        unreal_pct = (
            (unreal_pnl / (self._avg_cost * self._units) * 100.0)
            if self._units > 0
            else 0.0
        )

        payload: Dict[str, object] = {
            "current_time": bar.time,
            "metrics": metrics,
            "portfolio": {
                "cash": self._cash,
                "position_units": self._units,
                "position_value": pos_val,
                "avg_cost": self._avg_cost if self._units > 0 else 0.0,
                "affordable_units": self._affordable(price),
                "available_capacity": self._capacity(),
                "max_units": self._max_units,
                "unrealized_pnl": unreal_pnl,
                "unrealized_pnl_pct": unreal_pct,
            },
        }

        # Optional OHLCV fields if present.
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
                # Ignore non-numeric volume gracefully.
                pass
        if ohlcv:
            payload["ohlcv"] = ohlcv

        # Turn-specific, hard notes for the model.
        notes: list[str] = []

        # Disallow SELL when flat (redundant reinforcement).
        if self._units == 0:
            notes.append(
                "position_units=0: SELL disallowed; only BUY or HOLD permitted this turn."
            )

        # Cooldown hint.
        if self._cooldown_left > 0:
            notes.append(
                f"cooldown_bars_remaining={self._cooldown_left}: "
                "avoid new BUY; allow SELL/trim only if signals align."
            )

        # Capacity & cash hints.
        affordable = self._affordable(price)
        capacity = self._capacity()
        if affordable <= 0:
            notes.append("affordable_units=0: only HOLD or SELL.")
        if capacity <= 0:
            notes.append(
                "available_capacity=0: position at max_units; do not BUY more."
            )

        # Extreme RSI / z-score nudges (risk-aware guidance).
        if metrics["rsi"] >= 80:
            notes.append(
                "RSI very high: be cautious with new BUY; consider partial profit-taking."
            )
        if metrics["rsi"] <= 20:
            notes.append(
                "RSI very low: avoid panic SELL; consider mean-reversion entries."
            )
        if abs(metrics["zscore"]) >= 2.0:
            notes.append("abs(zscore)≥2: extreme deviation; act carefully.")

        if notes:
            payload["notes"] = notes

        return json.dumps(payload, separators=(",", ":"))

    def _ask_once(self, user_payload: str) -> Dict[str, object]:
        """Send one request to the LLM and parse the reply as JSON.

        Args:
            user_payload: The compact JSON string to send to the LLM.

        Returns:
            A dict with keys: signal, units, reason, feedback.

        Raises:
            Exception: If the LLM reply is not valid JSON.
        """
        reply = self._chat.send(user_payload)
        obj = json.loads(reply)  # Let this raise if malformed (handled by caller).
        return {
            "signal": str(obj.get("signal", "HOLD")).upper(),
            "units": _to_int(obj.get("units", 0)),
            "reason": str(obj.get("reason", "")).strip(),
            "feedback": str(obj.get("feedback", "")).strip() or "initial",
        }

    def _decide_with_retry(self, user_payload: str) -> Dict[str, object]:
        """Retry once with a strict prefix when JSON parsing fails.

        Args:
            user_payload: The compact JSON string for the LLM.

        Returns:
            Parsed decision dict with keys: signal, units, reason, feedback.

        Raises:
            Exception: If both attempts fail parsing as valid JSON.
        """
        try:
            return self._ask_once(user_payload)
        except Exception:
            if not self._retry_once:
                raise
            strict = "JSON_ONLY Strictly output the JSON object only. " + user_payload
            return self._ask_once(strict)

    # ---------------------------------------------------------------- lifecycle
    def observe(self, bar: MarketData) -> None:
        """Warm up metrics and emit a one-time history summary before trading.

        Accumulates prices for history and pushes values to rolling indicators.
        When `history_days` observations have accumulated (and only once),
        sends a compact summary to the LLM for additional context.

        Args:
            bar: Latest market data (only `price` is required here).
        """
        if bar.price is None:
            return

        price = float(bar.price)
        self._short.push(price)
        self._long.push(price)
        self._rsi.push(price)

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
            self._chat.send(
                "HISTORY_SUMMARY " + json.dumps(stats, separators=(",", ":"))
            )
            self._history_sent = True

    def generate_signal(self, bar: MarketData) -> Signal:
        """Main loop: build user JSON, call LLM, enforce constraints, mirror state.

        Args:
            bar: Latest market data tick/candle.

        Returns:
            A `Signal` value: BUY, SELL, or HOLD.
        """
        if bar.price is None:
            self.last_units = 0
            return Signal.HOLD

        price = float(bar.price)

        # If history_days is None, keep warming indicators via observe().
        if not self._history_sent and self._history_days is None:
            self.observe(bar)

        # Update indicators.
        self._short.push(price)
        self._long.push(price)
        self._rsi.push(price)

        # Require fully initialized windows.
        if not (self._short.full and self._long.full and self._rsi.full):
            self.last_units = 0
            # Still tick down cooldown while warming up.
            if self._cooldown_left > 0:
                self._cooldown_left -= 1
            return Signal.HOLD

        # Build turn payload and ask the model.
        payload = self._build_user_payload(bar)
        decision = self._decide_with_retry(payload)

        sig_text = str(decision["signal"]).upper()
        req_units = _to_int(decision["units"])

        # ------------------------ Hard constraints -------------------------
        # SELL is invalid when flat.
        if self._units == 0 and sig_text == "SELL":
            sig_text, req_units = "HOLD", 0

        if sig_text == "BUY":
            cap = min(self._affordable(price), self._capacity())
            if self._cooldown_left > 0:
                # Respect cooldown: block new BUYs; SELL/HOLD still allowed.
                cap = 0
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
            # Any unknown signal defaults to HOLD.
            sig_text, req_units = "HOLD", 0

        # ---------------- Mirror cash/position & cooldown updates ----------
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

            # Reset cooldown after a trade.
            self._cooldown_left = self._cooldown_cfg
            return Signal.BUY

        if sig_text == "SELL" and req_units > 0:
            self._cash += req_units * price
            self._units -= req_units
            if self._units == 0:
                self._avg_cost = 0.0
            self.last_units = req_units

            # Reset cooldown after a trade.
            self._cooldown_left = self._cooldown_cfg
            return Signal.SELL

        # HOLD path: decrement cooldown (if any) and do nothing.
        self.last_units = 0
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
        return Signal.HOLD

    # ------------------------------------------------------------ export logs
    def export_chat_logs(self, dst_dir: Path) -> None:
        """Export dialog.json (messages only) and conversation_full.json (metadata).

        Args:
            dst_dir: Destination directory. It will be created if it doesn't exist.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        self._chat.export_dialog_json(dst_dir / "dialog.json")
        self._chat.export_full_json(dst_dir / "conversation_full.json")


# -----------------------------------------------------------------------------
# Factory for engine dynamic import.
# -----------------------------------------------------------------------------
def build(**kwargs) -> Strategy:
    """Factory function returning a configured strategy instance.

    Args:
        **kwargs: Keyword arguments forwarded to `AIStrategy3`.

    Returns:
        A `Strategy` instance wrapping `AIStrategy3`.
    """
    return AIStrategy3(**kwargs)
