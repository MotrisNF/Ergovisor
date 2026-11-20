"""Tests for ``src.env_file``: the settings of ``.env``."""
import os
from pathlib import Path

import pytest

from src.env_file import load_env


def test_loads_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                      ) -> None:
    """Plain, quoted and exported values; comments and blanks skipped."""
    for name in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nA_KEY=one\nB_KEY=\"two\"\n"
                   "export C_KEY='three'\nnot a setting\n")
    load_env(env)
    assert (os.environ["A_KEY"], os.environ["B_KEY"],
            os.environ["C_KEY"]) == ("one", "two", "three")


def test_shell_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                    ) -> None:
    """A value already in the environment is kept."""
    monkeypatch.setenv("A_KEY", "shell")
    env = tmp_path / ".env"
    env.write_text("A_KEY=file\n")
    load_env(env)
    assert os.environ["A_KEY"] == "shell"


def test_empty_value_and_no_file(tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty value (as in .env-example) sets nothing; no file, no
    error."""
    monkeypatch.delenv("A_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("A_KEY=\n")
    load_env(env)
    load_env(tmp_path / "missing")
    assert "A_KEY" not in os.environ
