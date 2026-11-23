from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry", [
    [sys.executable, "-m", "dg_wave"],
    [str(Path(sys.executable).with_name("dg-wave"))],
])
def test_installed_help(entry, tmp_path):
    result = subprocess.run(
        [*entry, "--help"], cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "Stage 0" in result.stdout
    assert "--diagnostics" in result.stdout


def test_config_validation_cli():
    config = Path(__file__).parents[1] / "stage0.toml"
    result = subprocess.run(
        [sys.executable, "-m", "dg_wave", "--config", str(config)],
        capture_output=True, text=True, check=True,
    )
    assert "Configuration valid" in result.stdout


def test_bad_config_cli(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[broken")
    result = subprocess.run(
        [sys.executable, "-m", "dg_wave", "--config", str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "invalid TOML" in result.stderr
