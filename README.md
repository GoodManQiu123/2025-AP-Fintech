# Real-Time Machine Learning Trading System for High-Frequency Trading Simulation

**Author UID:** 06000219  
**Programme:** MSc Financial Technology (2024–2025), Imperial College Business School  
**Project Type:** Final report — financial software

---

## 1. Overview

A compact backtesting engine for experimenting with machine‑learning trading ideas.

- **Data Feed** (CSV → `MarketData`)
- **Strategy** (pluggable module, returns `BUY` / `SELL` / `HOLD`)
- **Portfolio** (FIFO lots; realised PnL; mark‑to‑market; exports)

```
CSV Feed  →  Strategy  →  Portfolio (FIFO lots & analytics)
```

Everything is plain Python; components are small and independent.

---

## 2. Repository Layout

```
.
├── core/
│   ├── engine.py              # CLI runner: feed → strategy → portfolio
│   ├── data_feed.py           # CSV → MarketData adapter (alias handling)
│   ├── portfolio.py           # FIFO lots, PnL, equity/plots, trade.log
│   ├── metrics.py             # RollingWindow, SMA, stddev, RSI-like
│   ├── types.py               # Signal enum, MarketData dataclass
│   ├── strategy_base.py       # Strategy interface
│   └── llm/
│       └── chat_agent.py      # OpenAI wrapper (JSON mode, logging/export)
├── strategies/
│   ├── threshold_strategy.py  # AdaptiveThresholdStrategy
│   ├── ai_strategy.py         # AIStrategy (LLM v1)
│   └── ai_strategy2.py        # AIStrategy2 (LLM v2, JSON-only, guardrails)
├── data/                      # Example datasets
│   ├── AAPL.csv
│   └── sample_prices.csv
├── logs/                      # Created per run (ASSET_YYYYMMDD_HHMMSS)
├── tests/                     # Pytest smoke/integration tests
├── requirements.txt
└── README.md
```

---

## 3. Environment Set‑up (Python 3.10)

### 3.1 Windows (PowerShell)

```powershell
# Check Python
python --version

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 macOS / Linux (bash/zsh)

```bash
# Check Python
python3 --version

# Create and activate a virtual environment
python3 -m venv .venv
source ./.venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 OpenAI key for LLM strategies (optional)

> Sign up at [platform.openai.com](https://platform.openai.com/) and create an API key.

```powershell
# Windows PowerShell
$Env:OPENAI_API_KEY = "YOUR_API_KEY_HERE"
```

```bash
# macOS / Linux
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

---

## 4. Data

### 4.1 CSV format

Minimum column: `time` (ISO date or datetime).  
Optional: `open`, `high`, `low`, `close`, `adj_close`, `volume`.  
Aliases accepted, e.g. **`Adj Close` → `adj_close`**, **`Vol` → `volume`**.

### 4.2 Download market data (Yahoo Finance)

`core/data_downloader.py` writes a canonical CSV.

```bash
# Last 365 days (interval 1d)
python -m core.data_downloader AAPL --days 365 --interval 1d --out data/AAPL.csv

# Date range
python -m core.data_downloader AAPL --start 2024-01-01 --end 2024-12-31 --interval 1d --out data/AAPL.csv

# Adjusted prices
python -m core.data_downloader AAPL --days 365 --interval 1d --auto-adjust --out data/AAPL.csv
```

Output columns: `asset,time,open,high,low,close,adj_close,volume`.

---

## 5. Run a Backtest

### 5.1 Engine flags

```bash
python -m core.engine --asset AAPL --data-dir data --strategy strategies.threshold_strategy   --start-cash 10000 --entry-date 2024-01-01
```

- `--asset`: looks for `data/<ASSET>.csv`
- `--data-dir`: CSV directory
- `--strategy`: dotted import path exposing `build(**kwargs)`
- `--start-cash`: initial cash for the portfolio
- `--entry-date`: bars **before** this date are observed only (warm‑up)

### 5.2 Strategy parameters

Any **extra** flags are forwarded as `**kwargs` to the strategy’s `build(**kwargs)` (dashes → underscores). If a kwarg is not accepted, Python raises `TypeError` on start.

#### Threshold

```bash
python -m core.engine --strategy strategies.threshold_strategy   --asset AAPL --data-dir data   --lookback 30 --buy-pct 0.02 --sell-pct 0.02
```

#### AI v1

```bash
python -m core.engine --strategy strategies.ai_strategy   --asset AAPL --data-dir data --start-cash 20000   --history-days 60 --metrics-window 20 --rsi-window 14   --verbose-llm true --max-units 10
```

#### AI v2

```bash
python -m core.engine --strategy strategies.ai_strategy2   --asset AAPL --data-dir data --start-cash 20000   --style swing --enable-scaling true   --short-win 10 --long-win 30 --rsi-win 14   --max-units 500 --history-days 180 --cooldown-bars-after-trade 0   --model gpt-4o-mini --temperature 0.1 --top-p 1.0   --frequency-penalty 0.0 --presence-penalty 0.0 --max-tokens 120   --json-mode true --max-history 64 --verbose-llm true --retry-on-parse-error true
```

### 5.3 Outputs

Each run writes to `logs/<ASSET_YYYYMMDD_HHMMSS>/`:

- `trade.log` — portfolio summary + trade list; **first lines show the exact command line**
- `trades.png` — price + BUY/SELL markers
- (console also prints the summary and trade log)

---

## 6. Strategies (what the engine calls)

```python
class Strategy(ABC):
    def generate_signal(self, bar: MarketData) -> Signal: ...
    def observe(self, bar: MarketData) -> None:  # optional
        pass
```
- `MarketData` (`core/types.py`) exposes `.price` (prefers `close`, then `adj_close`).
- `Signal` is an enum: `BUY`, `SELL`, `HOLD`.

Bundled implementations:
- `strategies/threshold_strategy.py` — AdaptiveThresholdStrategy
- `strategies/ai_strategy.py` — AIStrategy (LLM v1)
- `strategies/ai_strategy2.py` — AIStrategy2 (LLM v2; JSON‑only, scaling, guardrails; uses `core/llm/chat_agent.py`)

---

## 7. Tests

```bash
pytest -q
```
Covers metrics, portfolio, CSV feed aliasing, end‑to‑end engine run with a dummy strategy, and a chat‑agent export smoke test (skips if `openai` is missing).  
Windows tip (for subprocess encoding): `setx PYTHONIOENCODING utf-8`.

---

## 8. Marker Notes

- The engine forwards unknown CLI flags **directly** to `build(**kwargs)`; unsupported flags raise `TypeError` at init (single source of truth in the strategy).
- `trade.log` begins with the **full command line** for reproducibility.

---

## 9. License

MIT
