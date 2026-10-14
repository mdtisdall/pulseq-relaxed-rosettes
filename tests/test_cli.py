from pathlib import Path

import pytest

from relaxed_rosettes.cli import main
from relaxed_rosettes.config import Config


def test_config_init_then_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "design.toml"
    assert main(["config", "init", str(path)]) == 0
    assert Config.load(path) == Config()

    capsys.readouterr()
    assert main(["config", "show", str(path), "--set", "seed=5"]) == 0
    out = capsys.readouterr().out
    expected = Config().with_overrides(["seed=5"])
    assert f"# config digest: {expected.digest()}" in out
    assert Config.from_toml(out) == expected


def test_config_init_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "design.toml"
    path.write_text("")
    assert main(["config", "init", str(path)]) == 1
    assert main(["config", "init", str(path), "--force"]) == 0


def test_invalid_override_reports_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "show", "--set", "readout.adc_dwell_s=4.5e-6"]) == 1
    assert "undersamples" in capsys.readouterr().err
