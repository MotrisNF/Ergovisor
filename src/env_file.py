"""The settings kept in a ``.env`` file (the Gemini API key), so they
need not be exported in the shell; see ``.env-example``."""
import os
from pathlib import Path

# The project's root: next to ``src``.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Put each ``NAME=value`` line of the file in the environment,
    unless it is already set there (the shell wins). Blank lines and
    lines starting with ``#`` are skipped; quotes around a value are
    removed. Nothing happens without the file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name and value:
            os.environ.setdefault(name, value)
