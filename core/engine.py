# core/engine.py
"""Run backtests, pass strategy kwargs directly, and record command line.

This module:
  1) Loads market data from CSV.
  2) Dynamically imports a strategy module and instantiates it via ``build()``.
     - Strategy-specific CLI params are passed straight through as **kwargs.
     - If the selected strategy doesn't accept a given param, its constructor
       will raise TypeError (intended validation path).
  3) Streams bars to the strategy and executes trades with a local portfolio.
  4) Exports summary/logs and (if exposed) strategy conversation logs.
  5) Prepends the exact command line to the beginning of ``trade.log``.

CLI notes:
  * This script exposes the **main** knobs of the three strategies in this repo:
      - strategies.threshold_strategy.AdaptiveThresholdStrategy
          --lookback, --buy-pct, --sell-pct
      - strategies.ai_strategy.AIStrategy
          --history-days, --metrics-window, --rsi-window, --verbose-llm, --max-units
      - strategies.ai_strategy2.AIStrategy2
          --style, --enable-scaling, --short-win, --long-win, --rsi-win,
          --max-units, --history-days, --cooldown-bars-after-trade,
          --model, --temperature, --top-p, --frequency-penalty,
          --presence-penalty, --max-tokens, --json-mode,
          --max-history, --verbose-llm, --retry-on-parse-error
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

from core.data_feed import CSVFeed
from core.portfolio import Portfolio
from core.types import Signal


# ───────────────────────────────────────── helpers ─────────────────────────────────────────
def _cmdline_str() -> str:
    """Return the exact command line used to invoke this process."""
    return " ".join(shlex.quote(x) for x in sys.argv)


def _append_command_to_trade_log(log_dir: Path, cmdline: str) -> None:
    """Prepend the exact command line to 'trade.log' (after export)."""
    trade_log_path = log_dir / "trade.log"
    if not trade_log_path.exists():
        return
    original = trade_log_path.read_text(encoding="utf-8", errors="replace")
    header = [
        "========== Command Line ==========",
        cmdline,
        "==================================",
        "\n",
    ]
    trade_log_path.write_text("\n".join(header) + original, encoding="utf-8")


def _load_strategy_module(dotted_path: str):
    """Import a strategy module by dotted path and return the module object."""
    try:
        return importlib.import_module(dotted_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Failed to import strategy module: {dotted_path}") from exc


# ────────────────────────────────────────── CLI ───────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with engine + three-strategy core options."""
    p = argparse.ArgumentParser(description="Run trading back-test.")

    # Engine (data/run) options
    p.add_argument("--asset", default="AAPL", help="Asset symbol (e.g., AAPL).")
    p.add_argument("--data-dir", default="data", help="Directory containing CSV data files.")
    p.add_argument(
        "--strategy",
        default="strategies.threshold_strategy",
        help="Dotted path to the strategy module exposing build().",
    )
    p.add_argument("--cash", type=float, default=10_000.0, help="Starting cash for the portfolio.")
    p.add_argument(
        "--entry-date",
        help=(
            "ISO date (YYYY-MM-DD). Bars strictly before this date are used for "
            "warm-up (observe) only; trading starts on/after this date."
        ),
    )

    # Threshold strategy
    p.add_argument("--lookback", type=int, default=None, help="[threshold] lookback bars")
    p.add_argument("--buy-pct", type=float, default=None, help="[threshold] buy threshold fraction (e.g., 0.02)")
    p.add_argument("--sell-pct", type=float, default=None, help="[threshold] sell threshold fraction (e.g., 0.02)")

    # AIStrategy (v1)
    p.add_argument("--history-days", type=int, default=None, help="[ai_strategy] warm-up history size")
    p.add_argument("--metrics-window", type=int, default=None, help="[ai_strategy] SMA/vol window")
    p.add_argument("--rsi-window", type=int, default=None, help="[ai_strategy] RSI window")
    p.add_argument("--verbose-llm", type=str, default=None, help="[ai_strategy/ai_strategy2] 'true' or 'false'")
    p.add_argument("--max-units", type=int, default=None, help="[ai_strategy/ai_strategy2] max units (cap/per action)")

    # AIStrategy2 (v2)
    p.add_argument("--style", type=str, default=None, help="[ai_strategy2] style (scalp/swing/invest/free_high/conservative)")
    p.add_argument("--enable-scaling", type=str, default=None, help="[ai_strategy2] 'true' or 'false'")
    p.add_argument("--short-win", type=int, default=None, help="[ai_strategy2] short window")
    p.add_argument("--long-win", type=int, default=None, help="[ai_strategy2] long window")
    p.add_argument("--rsi-win", type=int, default=None, help="[ai_strategy2] RSI window")
    p.add_argument("--cooldown-bars-after-trade", type=int, default=None, help="[ai_strategy2] cooldown bars after trade")
    p.add_argument("--model", type=str, default=None, help="[ai_strategy2] LLM model name")
    p.add_argument("--temperature", type=float, default=None, help="[ai_strategy2] temperature")
    p.add_argument("--top-p", type=float, default=None, help="[ai_strategy2] top-p")
    p.add_argument("--frequency-penalty", type=float, default=None, help="[ai_strategy2] frequency penalty")
    p.add_argument("--presence-penalty", type=float, default=None, help="[ai_strategy2] presence penalty")
    p.add_argument("--max-tokens", type=int, default=None, help="[ai_strategy2] max tokens")
    p.add_argument("--json-mode", type=str, default=None, help="[ai_strategy2] 'true' or 'false'")
    p.add_argument("--max-history", type=int, default=None, help="[ai_strategy2] trimmed chat history size")
    p.add_argument("--retry-on-parse-error", type=str, default=None, help="[ai_strategy2] 'true' or 'false'")

    return p


def _parse_cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    return _build_parser().parse_args()


# ───────────────────────────────────────── runner ─────────────────────────────────────────
def run() -> None:
    """Run the backtest loop and export summary/logs."""
    args = _parse_cli()
    csv_path = Path(args.data_dir) / f"{args.asset.upper()}.csv"
    if not csv_path.exists():
        sys.exit(f"Data file not found: {csv_path}")

    # Build strategy kwargs by *directly* collecting all explicitly provided
    # strategy-related args (value is not None). No external validation table.
    raw_kwargs: Dict[str, Any] = {
        # threshold
        "lookback": args.lookback,
        "buy_pct": args.buy_pct,
        "sell_pct": args.sell_pct,
        # ai_strategy
        "history_days": args.history_days,
        "metrics_window": args.metrics_window,
        "rsi_window": args.rsi_window,
        "verbose_llm": args.verbose_llm,
        "max_units": args.max_units,
        # ai_strategy2
        "style": args.style,
        "enable_scaling": args.enable_scaling,
        "short_win": args.short_win,
        "long_win": args.long_win,
        "rsi_win": args.rsi_win,
        "cooldown_bars_after_trade": args.cooldown_bars_after_trade,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "max_tokens": args.max_tokens,
        "json_mode": args.json_mode,
        "max_history": args.max_history,
        "retry_once_on_parse_error": args.retry_on_parse_error,
    }
    # Keep only keys that the user *explicitly* provided (not None). This keeps engine simple.
    strategy_kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}

    # Import strategy module and instantiate via build(**kwargs).
    mod = _load_strategy_module(args.strategy)
    if not hasattr(mod, "build"):
        sys.exit(f"[error] strategy module '{args.strategy}' must expose build()")
    # Let the strategy constructor do the validation for unsupported kwargs.
    strategy = mod.build(**strategy_kwargs)

    # Feed and portfolio.
    entry_dt = dt.datetime.fromisoformat(args.entry_date) if args.entry_date else None
    feed = CSVFeed(csv_path, asset=args.asset)
    portfolio = Portfolio(starting_cash=float(args.cash))

    # Main loop.
    for bar in feed.stream():
        bar_dt = dt.datetime.fromisoformat(bar.time)

        # Warm-up before entry date.
        if entry_dt and bar_dt < entry_dt:
            if hasattr(strategy, "observe"):
                strategy.observe(bar)
            continue

        # Generate & execute, then mark-to-market on this bar.
        sig = strategy.generate_signal(bar)
        if sig is not Signal.HOLD:
            units = getattr(strategy, "last_units", 1)
            portfolio.execute(sig, bar, units=units)

        portfolio.mark_from_bar(bar)

    # Console outputs.
    print(portfolio.summary())
    print(portfolio.trade_logs())

    # Export logs.
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"{args.asset.upper()}_{ts}"
    portfolio.export_logs(log_dir, csv_path, entry_dt, asset=args.asset.upper())

    # Prepend *exact* command line to trade.log for traceability.
    _append_command_to_trade_log(log_dir, _cmdline_str())

    # Conversation dumps (if provided by the strategy).
    if hasattr(strategy, "export_chat_logs"):
        try:
            strategy.export_chat_logs(log_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to export conversation logs: {exc}")

    print(f"\nLogs & chart saved to: {log_dir.resolve()}")


if __name__ == "__main__":
    run()
