"""The closure step depends on no file outside the package and its inputs.

No closure module loads code by its file path or reads git history, and the loss audit that `closure bundle` copies into the release is the frozen file in git, with the hash of the released copy.
"""

import ast
import hashlib
from pathlib import Path

from mbdef.closure import bundle

REPO = Path(__file__).resolve().parents[1]
CLOSURE = REPO / "src" / "mbdef" / "closure"
# The loss audit as the released closure holds it (`provenance/` of the release).
LOSS_AUDIT_SHA256 = "888e2142f459945a2046dbc21705ba499a034490b625f07db3480c418fa48689"


def code_of(path: Path) -> str:
    """The module's statements without its docstrings and comments (prose may name paths)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                body[0].value.value = ""
    return ast.unparse(tree)


def test_no_closure_module_loads_a_file_by_path_or_reads_git_history():
    modules = sorted(CLOSURE.glob("*.py"))
    assert len(modules) >= 12
    for path in modules:
        code = code_of(path)
        for forbidden in (
            "spec_from_file_location",
            "importlib.util",
            "sys.path",
            "git show",
            "'git'",
        ):
            assert forbidden not in code, f"{path.name} uses {forbidden}"


def test_the_loss_audit_the_bundle_copies_is_a_frozen_input_with_the_released_hash():
    frozen = REPO / bundle.LOSS_AUDIT
    assert bundle.LOSS_AUDIT == Path("data/frozen/docs/closure-loss-audit.md")
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == LOSS_AUDIT_SHA256
