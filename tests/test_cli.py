import subprocess
import sys


def test_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pretrained_llm_adaptation.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "0.1.0"
