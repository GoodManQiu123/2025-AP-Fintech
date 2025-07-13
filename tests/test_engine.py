import subprocess
from pathlib import Path


def test_engine_runs():
    """Ensure the demo loop completes without error."""
    root = Path(__file__).parent.parent
    result = subprocess.run(
        ["python", "-m", "core.engine"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Realised PnL" in result.stdout
