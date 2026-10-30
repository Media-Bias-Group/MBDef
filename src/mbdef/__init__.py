"""The pipeline behind the MBDef datasets as one package with one command line, `mbdef`.

Each pipeline step is a subcommand; docs/README.md has one page per step.
"""

# The Python version the code and its recorded runs were tested with
TESTED_PYTHON = (3, 14)


def untested_python(version: tuple[int, ...]) -> str | None:
    """The warning for a Python newer than the tested one, or None for the tested one (or an older one, which `requires-python` in pyproject.toml refuses)."""
    if tuple(version[:2]) <= TESTED_PYTHON:
        return None
    tested = ".".join(map(str, TESTED_PYTHON))
    running = ".".join(map(str, version[:3]))
    return f"Warning: this code was only tested with Python {tested}; this is Python {running}."
