"""Machine settings, file hashes, JSON and atomic file writing, and receipts: the one copy of each of these helpers that the steps share (git calls are in repo.py).

- `.env` holds only machine-specific values (`.env.example` lists them). `load_env` reads it once, at the start of every command, and `env(KEY)` fails with a pointer to `.env.example` when a command needs a key that is not set. Commands ask only for the keys they use.
- Hashes are SHA-256. A folder's hash is its folder digest (`digest`, below).
- JSON is written with its byte format as parameters, because each released output keeps the format it was released with. A file that a crash must never leave half-written (a ledger, a TEI file) is written with `write_atomic`.
- A receipt records what produced an output: the command, its settings, the hashes of its inputs and outputs, the code commit and the time.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from mbdef import repo

ENV_FILE = Path(
    ".env"
)  # in the folder the command runs from, normally the repository root
ENV_EXAMPLE = ".env.example"
# Ids written into recorded files (here and in closure/, evaluate/ and extract/) keep the name the files were recorded under: a reader checks them, so recorded files still read and recorded bytes still reproduce.
RECEIPT_SCHEMA = "taxonomy_pipeline/receipt-1"
RECEIPT = "receipt.json"  # every step writes its receipt here, inside its output folder
# What a publishable file must not hold (the dataset build and the pseudonymised exports check it): an e-mail address, and a path into one machine's own folders
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
LOCAL_FOLDERS = (
    "/home/",
    "/Users/",
    "/root/",
    "/tmp/",
    "/scratch/",
    "/mnt/",
    "/media/",
    "~/",
)


def load_env(path: Path = ENV_FILE) -> None:
    """Put the KEY=value lines of `path` into the environment. Values already set win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def env(key: str) -> str:
    """The value of a machine-specific key; stop with a pointer to `.env.example` if it is unset."""
    value = os.environ.get(key, "")
    if not value:
        raise SystemExit(
            f"{key} is not set. Copy {ENV_EXAMPLE} to .env and set {key} there; "
            f"{ENV_EXAMPLE} says what it is."
        )
    return value


def data_root() -> Path:
    """The folder that holds the data kept outside git (`DATA_ROOT`; docs/setup/data-root.md)."""
    return Path(env("DATA_ROOT")).expanduser()


def located(given: str | None, under_root: str) -> tuple[Path, str]:
    """An input's path, and how the outputs name it: as given, or `$DATA_ROOT/<path>` (UNDER_ROOT below DATA_ROOT)."""
    if given:
        return Path(given), given
    return data_root() / under_root, f"$DATA_ROOT/{under_root}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def folder_files(folder: Path) -> list[tuple[str, Path]]:
    """('./relative/path', path) for every regular file below `folder`, sorted bytewise.

    Symbolic links are skipped, as `find -type f` skips them.
    """
    found = [
        ("./" + p.relative_to(folder).as_posix(), p)
        for p in folder.rglob("*")
        if p.is_file() and not p.is_symlink()
    ]
    return sorted(found, key=lambda pair: pair[0].encode())


def file_sha256(path: Path) -> str:
    """The sha256 of one file, read in pieces of 1 MiB (PDF and TEI files can be large)."""
    hashed = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            hashed.update(piece)
    return hashed.hexdigest()


def digest(path: Path) -> str:
    """The sha256 of a file, or the folder digest of a folder.

    The folder digest is the sha256 of the output of `(cd DIR && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)`.
    """
    if path.is_dir():
        return listing_digest(
            {name[2:]: file_sha256(p) for name, p in folder_files(path)}
        )
    return file_sha256(path)


def listing_digest(hashes: dict[str, str]) -> str:
    """The folder digest from the sha256 of each file, keyed by its path inside the folder."""
    names = sorted(hashes, key=lambda name: ("./" + name).encode())
    return sha256("".join(f"{hashes[n]}  ./{n}\n" for n in names).encode())


def describe(path: Path) -> dict:
    """Path, sha256 and size of a file or folder, as a receipt lists it."""
    if path.is_dir():
        found = folder_files(path)
        size = sum(p.stat().st_size for _, p in found)
        return {
            "path": str(path),
            "sha256": digest(path),
            "bytes": size,
            "files": len(found),
        }
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def json_text(
    data,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
    escape_slashes: bool = False,
    end: str = "\n",
) -> str:
    """JSON text in the given byte format: `json.dumps`'s own settings, every "/" written as "\\/" (as some exporters write it), and what follows the text (a final newline, or nothing)."""
    text = json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        separators=separators,
    )
    return (text.replace("/", "\\/") if escape_slashes else text) + end


def write_json(path: Path, data, **byte_format) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(data, **byte_format), encoding="utf-8")


def write_atomic(contents: dict[Path, bytes | str]) -> None:
    """Write each {path: content} of CONTENTS (text is written as UTF-8) so that a reader sees the old file or the new one, never a part: every file is first written in full to a temporary file beside it and flushed to disk, and only then are they all renamed over their paths. So a run that fails while writing (a full disk) changes none of them."""
    staged: list[tuple[Path, Path]] = []
    try:
        for path, content in contents.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                staged.append((Path(handle.name), path))
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def git_state(folder: Path) -> dict:
    """The commit checked out in the repository that holds `folder`, and whether it has changes."""
    commit = repo.git_or_none(folder, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = repo.git_or_none(folder, "status", "--porcelain") or b""
    return {"commit": commit.decode().strip(), "dirty": bool(status.strip())}


DATA_ROOT_MANIFEST = Path("data/data-root.tsv")  # path, sha256, bytes, source
DEPOSIT_LINE = "#deposit"  # data-root.tsv's line with the public deposit's address
PLACEHOLDER = ".invalid"  # a deposit address on this domain: nothing is uploaded yet
# The sources of data-root.tsv.
HF = "hf:"  # "hf:<dataset>/<path>": a file of one of the two datasets on Hugging Face (HF_DATASETS)
DEPOSIT = "https:"  # "https:<path>": a file of the public deposit, below its address
FLAT = "__"  # the deposit's file names are flat (a Zenodo record allows no "/"): "/" becomes this
DERIVE = "derive:"  # "derive:<command>": the command that makes the file
MEMBER = (
    "#"  # "<archive>.tar.zst#<member>": one file or folder inside a zstd tar archive
)

# The two datasets on Hugging Face: the one place that names their repositories (private for now).
# MBDef is the definitions extracted from the corpus (`dataset build`); MBDef-Gold is the ground truth (`groundtruth build`).
MBDEF_REPO = "mediabiasgroup/MBDef"
MBDEF_GOLD_REPO = "mediabiasgroup/MBDef-Gold"
# Each dataset's repository and the revision `data fetch` downloads: the commit of the upload of version 1.0.0 (tagged v1.0.0).
HF_DATASETS = {
    "MBDef": (MBDEF_REPO, "fa97dbb3f049f2664eac260c9b363ef8cff61e26"),
    "MBDef-Gold": (MBDEF_GOLD_REPO, "e2efedfb56c1c2253f5db4c72aa85fa38945f6b6"),
}
# Their folders in the data root: `data fetch` puts each file at its path in the repository below them.
MBDEF = "MBDef"
MBDEF_GOLD = "MBDef-Gold"
# HF_ENDPOINT in the environment replaces it, as it does for the Hugging Face tools.
HF_ENDPOINT = "https://huggingface.co"


def mbdef_gold(name: str) -> Path:
    """The file NAME of MBDef-Gold, the ground truth, in the data root.
    Stops with a pointer to `data fetch` if it is not there."""
    path = data_root() / MBDEF_GOLD / name
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. It is a file of MBDef-Gold, the ground truth, on Hugging Face ({MBDEF_GOLD_REPO}): "
            f"`mbdef data fetch {MBDEF_GOLD}/` downloads it into the data root (docs/setup/data-fetch.md)."
        )
    return path


def read_list(path: Path) -> tuple[str | None, list[list[str]]]:
    """The deposit address (its `#deposit` line) and the rows of a data-root list, without its header and without lines that start with `#`."""
    deposit, rows = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields[0] == DEPOSIT_LINE:
            deposit = fields[1]
        elif line.strip() and not line.startswith("#"):
            rows.append(fields)
    return deposit, rows[1:]


def deposit_name(path: str) -> str:
    """The file name that PATH has in the public deposit: flat, each "/" replaced by FLAT."""
    return path.replace("/", FLAT)


def deposit_url(deposit: str | None, path: str) -> str | None:
    """PATH's address in the public deposit (its flat name below the `#deposit` address, for example https://zenodo.org/records/<id>/files/<name>); None while that is a placeholder."""
    if deposit is None or (urlsplit(deposit).hostname or PLACEHOLDER).endswith(
        PLACEHOLDER
    ):
        return None
    return deposit.rstrip("/") + "/" + deposit_name(path)


def hf_url(dataset: str, path: str) -> str:
    """The address of PATH in DATASET's repository on Hugging Face, at its pinned revision."""
    repo_id, revision = HF_DATASETS[dataset]
    endpoint = os.environ.get("HF_ENDPOINT") or HF_ENDPOINT
    return f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/{path}"


def download(url: str, target: Path, token: str | None = None) -> str | None:
    """Copy URL to TARGET, sending TOKEN as a bearer key if given; the error, or None."""
    request = Request(
        url, headers={"Authorization": f"Bearer {token}"} if token else {}
    )
    try:
        with urlopen(request, timeout=60) as answer, target.open("wb") as out:
            shutil.copyfileobj(answer, out, 1 << 20)
    except (URLError, OSError) as error:
        target.unlink(missing_ok=True)
        return f"{url}: {error}"
    return None


def unpack(archive: Path, member: str, folder: Path) -> str | None:
    """Unpack MEMBER of the zstd tar ARCHIVE (packed with a long window) into FOLDER; the error, or None."""
    with subprocess.Popen(
        ["zstd", "-d", "-q", "--long=31", "-c", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as zstd:
        done = subprocess.run(
            ["tar", "-x", "-f", "-", "-C", str(folder), member],
            stdin=zstd.stdout,
            capture_output=True,
            text=True,
        )
        zstd.wait()
    if zstd.returncode != 0 or done.returncode != 0:
        return f"cannot unpack {member}: {done.stderr.strip()}"
    return None


def download_dataset_file(source: str, part: Path) -> str | None:
    """Download the file of an `hf:<dataset>/<path>` source to PART (with the key HF_TOKEN, if it is set).
    Returns the error, or None."""
    dataset, _, path = source.removeprefix(HF).partition("/")
    if dataset not in HF_DATASETS:
        return f"{dataset} is not a dataset of files.HF_DATASETS ({', '.join(HF_DATASETS)})"
    error = download(hf_url(dataset, path), part, os.environ.get("HF_TOKEN") or None)
    if error is None:
        return None
    repo_id, revision = HF_DATASETS[dataset]
    return (
        f"the dataset {dataset} ({repo_id} on Hugging Face, revision {revision}) has no file {path}, "
        f"or is not uploaded or not public yet: {error}"
    )


def fetch(prefixes: list[str], manifest: Path = DATA_ROOT_MANIFEST) -> int:
    """Put every data-root file listed under PREFIXES (default: all) into DATA_ROOT.

    Files already there are only checked.
    A missing file is downloaded from Hugging Face (an `hf:` source: a file of one of the two datasets, HF_DATASETS) or from the public deposit (an `https:` source, below the address of the list's `#deposit` line), and kept only if its sha256 matches the list.
    A `derive:` source prints the command that makes the file.
    A file that is there but differs is reported and never overwritten.
    A source `<archive>#<member>` is a file or folder inside a zstd tar archive: the archive is downloaded once per run, the member unpacked and checked the same way (a folder by its folder digest).
    Returns 1 if any listed file is missing or differs.
    """
    root = data_root()
    deposit, rows = read_list(manifest)
    chosen = [row for row in rows if not prefixes or row[0].startswith(tuple(prefixes))]
    bad = 0
    archives: dict[str, Path | str] = {}  # archive -> its local copy, or why it failed
    scratch: tempfile.TemporaryDirectory | None = None
    for path, expected, _size, source in chosen:
        target = root / path
        if target.exists():
            found = digest(target)
            if found == expected:
                print(f"{path}: already there, sha256 matches")
            else:
                print(
                    f"{path}: CHANGED, its sha256 is {found}, not {expected}; not overwritten"
                )
                bad += 1
            continue
        where, _, member = source.removeprefix(DEPOSIT).partition(MEMBER)
        url = deposit_url(deposit, where) if source.startswith(DEPOSIT) else None
        if source.startswith(DERIVE):
            print(f"{path}: MISSING; make it with: {source.removeprefix(DERIVE)}")
            bad += 1
            continue
        if source.startswith(DEPOSIT) and url is None:
            print(
                f"{path}: MISSING; the public deposit is not uploaded yet ({manifest} "
                f"gives the placeholder address {deposit})"
            )
            bad += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if url and member:
            if scratch is None:  # on the data root's file system, so a folder moves in
                scratch = tempfile.TemporaryDirectory(dir=root, prefix=".fetch-")
            work = Path(scratch.name)
            if url not in archives:
                local = work / f"archive-{len(archives)}"
                archives[url] = download(url, local) or local
            local = archives[url]
            part = work / f"part-{path.replace('/', '_')}"
            part.mkdir()
            error = local if isinstance(local, str) else unpack(local, member, part)
            part = part / member
        else:
            part = target.with_name(target.name + ".part")
            if url:
                error = download(url, part)
            elif source.startswith(HF):
                error = download_dataset_file(source, part)
            else:
                error = f"{source} is not a source data fetch knows ({HF}, {DEPOSIT} or {DERIVE})"
        found = digest(part) if error is None and part.exists() else None
        if found == expected:
            part.replace(target)
            print(f"{path}: fetched, sha256 matches")
            continue
        if part.is_dir():
            shutil.rmtree(part)
        else:
            part.unlink(missing_ok=True)
        reason = (
            f"the download's sha256 is {found}, not {expected}"
            if found
            else error or f"{member or where} is not in the download"
        )
        print(f"{path}: FAILED, {reason}")
        bad += 1
    if scratch is not None:
        scratch.cleanup()
    where = " under " + ", ".join(prefixes) if prefixes else ""
    if bad:
        print(f"{bad} of {len(chosen)} files are missing or differ from {manifest}.")
        return 1
    print(
        f"All {len(chosen)} files{where} are in the data root with the listed sha256."
    )
    return 0


def receipt(
    command: str,
    settings: dict,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
) -> dict:
    """What produced some outputs: the command, settings, hashes, code commit and time."""
    return {
        "schema": RECEIPT_SCHEMA,
        "command": command,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": git_state(Path(__file__).resolve().parent),
        "python": platform.python_version(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "settings": settings,
        "inputs": {name: describe(path) for name, path in inputs.items()},
        "outputs": {name: describe(path) for name, path in outputs.items()},
    }
