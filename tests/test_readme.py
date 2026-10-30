"""The README's commands are ones the command line has: every `mbdef` command in its shell blocks names an existing group and action, and every flag it passes is one that action takes.

The README's outputs come from live runs with a model (they are not replayed here); this test keeps its commands from going stale when a flag changes.
"""

import re
import shlex
from pathlib import Path

import pytest
from cli_helpers import run_cli

from mbdef import dataset
from mbdef.dataset.card import BADGES

REPO = Path(__file__).resolve().parents[1]


def readme_commands() -> list[list[str]]:
    """The words after `mbdef` of each command in the README's bash blocks."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            line = line.split("#", 1)[0]
            command = re.search(r"\bmbdef (\S.*)", line)  # not `cd mbdef`
            if command:
                commands.append(shlex.split(command.group(1)))
    return commands


COMMANDS = readme_commands()


def test_the_readme_has_the_four_tasks_and_no_developer_commands():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    for heading in (
        "### 1. Extract definitions from your papers",
        "### 2. Score an extractor with DefScore",
        "### 3. Train your own extractor",
        "### 4. Swap the model",
        "## Reproduce the paper",
        "## Appendix",
    ):
        assert heading in text
    assert "pytest" not in text and "check-links" not in text
    assert len(COMMANDS) >= 10


@pytest.mark.parametrize(
    "words", COMMANDS, ids=[" ".join(words[:2]) for words in COMMANDS]
)
def test_each_readme_command_and_its_flags_exist(tmp_path, words):
    group, action = words[0], words[1]
    done = run_cli(tmp_path, group, action, "--help")

    assert done.returncode == 0, done.stderr
    flags = [word.split("=", 1)[0] for word in words if word.startswith("--")]
    unknown = [flag for flag in flags if flag not in done.stdout]
    assert not unknown, f"mbdef {group} {action} has no {unknown}"


def test_the_readme_opens_the_explorer_from_under_the_poster():
    """The buttons under the poster are the cards' own row, so the Explorer's address is the one constant `dataset.EXPLORER_URL`."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    poster = "(docs/images/mbdef-explorer-poster.png)\n\n"
    assert poster + BADGES in text
    assert (
        f'<a href="{dataset.EXPLORER_URL}"><img alt="Open the MBDef Explorer"' in BADGES
    )
    assert f'<a href="{dataset.PROJECT_URL}"><img alt="Project page"' in BADGES
    explorer_links = re.findall(r"https://media-bias-group\.github\.io/MBDef/", text)
    assert explorer_links == [dataset.EXPLORER_URL], explorer_links
