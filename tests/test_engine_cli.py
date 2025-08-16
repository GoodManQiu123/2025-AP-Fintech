import os
import subprocess
import sys
from pathlib import Path


def test_engine_cli_with_temp_csv_and_dummy_strategy(tmp_path, make_csv):
    """Run `python -m core.engine` end-to-end with temporary CSV and strategy.

    Forces UTF-8 in the child process to avoid Windows stdout encoding issues.
    """
    # Prepare CSV under a temp "data" dir.
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    make_csv(
        [
            {"time": "2024-01-01", "close": 10.0},
            {"time": "2024-01-02", "close": 11.0},
            {"time": "2024-01-03", "close": 11.0},
        ],
        filename="TEST.csv",
    ).rename(data_dir / "TEST.csv")

    # Ensure tests package (with dummy_strategy) is on PYTHONPATH for subprocess.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd()), str(Path(__file__).parent.parent)]
    )
    # Force UTF-8 stdout/stderr in the child process to handle unicode symbols.
    env["PYTHONIOENCODING"] = "utf-8"

    # Launch engine module using our dummy strategy.
    cmd = [
        sys.executable,
        "-m",
        "core.engine",
        "--asset",
        "TEST",
        "--data-dir",
        str(data_dir),
        "--strategy",
        "tests.dummy_strategy",
        "--cash",
        "1000",
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, env=env, check=True, encoding="utf-8", errors="replace"
    )

    # The console summary should contain these anchor strings.
    for needle in [
        "Portfolio Summary",
        "Realised PnL",
        "Max drawdown",
        "Exposure (time in market)",
    ]:
        assert needle in res.stdout
