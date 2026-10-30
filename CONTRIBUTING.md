# Contributing

Install as in the [README](README.md#install), then run the checks before a change goes in:

```bash
uv run ruff check && uv run ruff format --check   # lint and format
uv run ty check src                               # types
uv run pytest -q -m "not slow"                    # the fast tests: no network, GPU, cluster or paid call
uv run mbdef repo check-links                     # every internal link of the docs points at a file, folder or heading
```

`uv run prek install` runs the link check before every commit.
The slow tests (`-m slow`) need the data root, and some of them a GPU; each says what it needs in its docstring.
On a shared machine, run GPU work under a lock (`flock`).

[docs/README.md](docs/README.md) says how the step pages are written, and [docs/setup/toolchain.md](docs/setup/toolchain.md) how the locked Python, packages and tools are refreshed.

## The link check: `mbdef repo check-links`

```bash
uv run mbdef repo check-links              # the index
uv run mbdef repo check-links --rev HEAD   # a commit's tree
```

It reads the markdown and LaTeX files of the tree with `git cat-file`, offline, and checks every internal link: a **relative path** (`../data/x.json`, `step.md#section`), read from the folder of the file that holds it, or a path from the repository root (`/docs/README.md`), as GitHub reads both, and a bare `#section` of the same page.
It makes no number of the paper; it keeps the docs' links in order, runs in the pre-commit hook ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)), and is not a general git tool.

The target must be a file or a folder of the tree, and a `#section` must be a heading of that markdown page (a repeated heading is `#run-it`, then `#run-it-1`) or an HTML `id` or `name`; `#L12` line anchors are accepted. Not checked: web addresses, and anything in code (a fenced block or backticks). Files that a stored hash pins cannot be edited to fix a link, so the check does not read them (`HASH_GATED` in `doc_links.py`): `data/frozen/**`. It still accepts links *to* them.

The result for a failing link:

```text
README.md:3: docs/gone.md: no such file or folder in the tree
Broken: 1 of 2 internal links in 2 files of the index.
```

and for a clean tree `OK: all <n> internal links in <m> files of the index point to files, folders and headings in the tree.` (`n` is the number of links checked, `m` the number of pages read). Before the last line it says how many hash-gated files it skipped (`Skipped 3 hash-gated files (data/frozen/**).` in this repository).

The code is [`src/mbdef/doc_links.py`](src/mbdef/doc_links.py) (the check) and [`repo.py`](src/mbdef/repo.py) (the git helpers it reads the tree with). [`tests/test_repo_check_links.py`](tests/test_repo_check_links.py) runs the command on throwaway git repositories with a fixed author and date: relative links, anchors, code that is not a link, other ways to write a link (`<…>` targets with spaces, titles, references, `href`), web addresses, LaTeX, hash-gated files, `--rev` against the index, and paths from the root.

To adapt it:

- **A file whose links cannot be edited:** add its glob to `HASH_GATED` in `doc_links.py` with the reason.
- **Other kinds of page:** add their extension to `PAGES` in `doc_links.py` and, if their links look different, a pattern next to `TEX_LINK`.
