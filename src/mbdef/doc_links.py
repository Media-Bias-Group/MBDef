"""`mbdef repo check-links`: every internal link of the docs points at something in the tree.

It reads the markdown and LaTeX files of a git tree without a checkout and needs no network. A link is internal if it is a relative path or a path from the repository root. A `#section` anchor must be a heading (or an HTML `id`) of the markdown page it points at. Web addresses are not checked.
"""

import posixpath
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from mbdef import repo

# Files whose text a stored hash pins, so a broken link in them is not edited away: the check does not read them (it still accepts links *to* them). Each entry says why.
HASH_GATED = (
    "data/frozen/**",  # the frozen inputs of the released runs, kept byte for byte as the runs read them
)
PAGES = (".md", ".tex")  # the files whose links are read
OTHER_ADDRESS = re.compile(r"[a-z][a-z0-9+.-]*:", re.IGNORECASE)  # https:, mailto:, …

# Markdown inline links and images, with an optional "title" and one level of parentheses in the target; `<...>` lets a target hold spaces.
INLINE_LINK = re.compile(r"\]\((<[^>]*>|(?:[^()\s]|\([^()\s]*\))*)(?:\s+\"[^\"]*\")?\)")
REFERENCE_LINK = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")  # [name]: target
HTML_LINK = re.compile(
    r"""\b(?:href|src)=["']([^"']*)["']"""
)  # <a href="…">, <img src="…">
TEX_LINK = re.compile(r"\\href\{([^}]*)\}")
CODE_SPAN = re.compile(r"`[^`]*`")
FENCE = re.compile(r"^ {0,3}(```|~~~)")
HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$")
HTML_ANCHOR = re.compile(r"""\b(?:id|name)=["']([^"']+)["']""")  # <a id="x"></a>
LINE_ANCHOR = re.compile(
    r"L\d+(?:-L\d+)?"
)  # #L12 or #L12-L20, as GitHub reads them in a file


def read_texts(root: Path, rev: str | None, paths: list[str]) -> dict[str, str]:
    """The text of each file in `rev`'s tree, or in the index if `rev` is None."""
    blobs = repo.read_blobs(root, rev, paths)
    return {path: blob.decode("utf-8", "replace") for path, blob in blobs.items()}


def links_in(path: str, text: str) -> list[tuple[int, str]]:
    """Each (line number, target) of the links in a file's text, leaving out markdown code."""
    found, fence = [], None
    for number, line in enumerate(text.splitlines(), 1):
        if path.endswith(".tex"):
            if not line.lstrip().startswith("%"):
                found += [(number, m.group(1)) for m in TEX_LINK.finditer(line)]
            continue
        opening = FENCE.match(line)
        if fence:
            fence = None if opening and opening.group(1) == fence else fence
            continue
        if opening:
            fence = opening.group(1)
            continue
        line = CODE_SPAN.sub("", line)
        targets = [m.group(1) for m in INLINE_LINK.finditer(line)]
        targets += [m.group(1) for m in HTML_LINK.finditer(line)]
        if reference := REFERENCE_LINK.match(line):
            targets.append(reference.group(1))
        found += [(number, target.strip("<>")) for target in targets]
    return found


def slug(heading: str) -> str:
    """The anchor GitHub makes of a heading: link text kept, lower case, punctuation dropped, spaces made hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors_in(text: str) -> set[str]:
    """The anchors of a markdown page: its headings, a repeated heading with -1, -2, … after it, and the `id` or `name` of its HTML tags. Headings inside code are not headings."""
    found, seen, fence = set(), {}, None
    for line in text.splitlines():
        opening = FENCE.match(line)
        if fence:
            fence = None if opening and opening.group(1) == fence else fence
            continue
        if opening:
            fence = opening.group(1)
            continue
        found.update(HTML_ANCHOR.findall(line))
        if heading := HEADING.match(line):
            base = slug(heading.group(1))
            found.add(base if base not in seen else f"{base}-{seen[base]}")
            seen[base] = seen.get(base, 0) + 1
    return found


def problem(
    target: str,
    page: str,
    files: set[str],
    folders: set[str],
    anchors: dict[str, set[str]],
) -> str | None:
    """Why an internal link does not resolve: "" if it does, None if it is not an internal link."""
    base, _, fragment = target.partition("#")
    base = base.split("?", 1)[0]
    if OTHER_ADDRESS.match(target) or "{{" in target:
        return None  # an address, or a template's placeholder that is filled in when the page is built
    path = unquote(base)
    if not path and not fragment:
        return None  # an empty link
    if not path:
        joined = page  # a #section of the same page
    elif path.startswith("/"):  # from the repository root, as GitHub reads it
        joined = posixpath.normpath(path.lstrip("/"))
    else:
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
    if joined not in files and joined not in folders:
        return "no such file or folder in the tree"
    fragment = unquote(fragment)
    if (
        fragment
        and joined in anchors
        and not LINE_ANCHOR.fullmatch(fragment)
        and fragment not in anchors[joined]
    ):
        return f"no heading with that anchor in {joined}"
    return ""


def plural(count: int, noun: str) -> str:
    return f"{count:,} {noun}" + ("" if count == 1 else "s")


def check_links(rev: str | None = None) -> int:
    """Print each broken internal link with its file and line. Return 1 if any is broken, else 0."""
    root = Path(repo.git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    files = [path for path, _ in repo.tree_files(root, rev)]
    where = f"commit {rev}" if rev else "the index"
    known = set(files)
    folders = {str(parent) for path in files for parent in PurePosixPath(path).parents}
    pages = [path for path in files if path.endswith(PAGES)]
    skipped = [path for path in pages if repo.matches_any(path, list(HASH_GATED))]
    pages = [path for path in pages if path not in skipped]
    markdown = [path for path in files if path.endswith(".md")]
    texts = read_texts(root, rev, sorted(set(markdown) | set(pages)))
    anchors = {path: anchors_in(texts[path]) for path in markdown}
    checked = broken = 0
    for path in pages:
        for number, target in links_in(path, texts[path]):
            why = problem(target, path, known, folders, anchors)
            if why is None:
                continue
            checked += 1
            if why:
                broken += 1
                print(f"{path}:{number}: {target}: {why}")
    if skipped:
        print(
            f"Skipped {plural(len(skipped), 'hash-gated file')} ({', '.join(HASH_GATED)})."
        )
    links = plural(checked, "internal link")
    if broken:
        print(
            f"Broken: {broken:,} of {links} in {plural(len(pages), 'file')} of {where}."
        )
        return 1
    print(
        f"OK: all {links} in {plural(len(pages), 'file')} of {where} "
        "point to files, folders and headings in the tree."
    )
    return 0
