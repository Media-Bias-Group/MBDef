"""The installed package runs on its own: it imports no module by a path of the repository.

An import such as `src.mbdef.files` or `scripts.helper` works only when Python runs from the repository root, not from the installed package, so the package must import itself as `mbdef`.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def imports_of(folder: Path, *packages: str) -> list[str]:
    """'file: module' for every import of one of `packages` (or a module in it) below `folder`."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            found += [
                f"{path.relative_to(folder)}: {name}"
                for name in names
                if name in packages or name.startswith(tuple(p + "." for p in packages))
            ]
    return found


def test_the_package_imports_nothing_by_a_repository_path(tmp_path):
    """The finder sees every form of such an import, and the package has none."""
    planted = tmp_path / "pkg"
    planted.mkdir()
    (planted / "a.py").write_text("from scripts.helper import read_rows\n")
    (planted / "b.py").write_text("import src.mbdef.files\n")
    (planted / "c.py").write_text("from src import mbdef\n")
    (planted / "d.py").write_text("from mbdef.extract import relevance\n")

    assert imports_of(planted, "scripts", "src") == [
        "a.py: scripts.helper",
        "a.py: scripts.helper.read_rows",
        "b.py: src.mbdef.files",
        "c.py: src",
        "c.py: src.mbdef",
    ]
    assert imports_of(SRC / "mbdef", "scripts", "src") == []
