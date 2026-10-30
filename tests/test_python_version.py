"""The code was tested with Python 3.14: a newer Python runs, with a warning, and the steps that need vLLM in-process say what to do where it cannot be installed."""

import sys

import pytest
from cli_helpers import run_cli

import mbdef
from mbdef import embed


def test_the_tested_python_runs_without_a_warning():
    assert mbdef.untested_python((3, 14, 8)) is None


def test_a_newer_python_gets_a_warning_that_names_the_tested_one():
    assert mbdef.untested_python((3, 15, 0)) == (
        "Warning: this code was only tested with Python 3.14; this is Python 3.15.0."
    )


def test_the_command_line_runs_on_the_tested_python_without_a_warning(tmp_path):
    done = run_cli(tmp_path, "--help")

    assert done.returncode == 0
    if sys.version_info[:2] == mbdef.TESTED_PYTHON:
        assert done.stderr == ""


def test_embedding_with_vllm_where_it_is_not_installed_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)  # an import of vllm now fails

    with pytest.raises(SystemExit) as stopped:
        embed.encode_vllm(["a text"], {"max_length": 8})

    assert str(stopped.value) == (
        "This step runs its model with vLLM, which is not installed (vLLM supports Python 3.14 but not 3.15 or newer). "
        "Run it in a Python 3.14 environment: `uv sync --python 3.14`."
    )
