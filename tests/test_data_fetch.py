"""`mbdef data fetch`: fill DATA_ROOT and check every file's sha256.

data/data-root.tsv gives each file a source: a file of one of the two datasets on Hugging Face (`hf:`), a path in the public deposit (`https:`) or the command that makes it (`derive:`). Two stand-in HTTP servers on the test machine play Hugging Face and the deposit, so no test touches the network.
"""

import functools
import hashlib
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from cli_helpers import run_cli, tool_dir, write_env

from mbdef import files

REPO = Path(__file__).resolve().parents[1]
NOT_UPLOADED = "https://deposit.invalid/mbdef-data/"  # a placeholder address

ONE = b"first input\n"
TWO = b"second input\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_list(work: Path, rows, deposit: str = NOT_UPLOADED) -> None:
    """data/data-root.tsv with the deposit address DEPOSIT and ROWS (path, sha256, bytes, source)."""
    (work / "data").mkdir(parents=True, exist_ok=True)
    lines = [
        f"#deposit\t{deposit}",
        "path\tsha256\tbytes\tsource",
        *("\t".join(map(str, row)) for row in rows),
    ]
    (work / "data/data-root.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup(tmp_path, rows, deposit: str = NOT_UPLOADED) -> tuple[Path, Path]:
    """A work folder with the list and a .env naming an empty data root: (work folder, data root)."""
    work, root = tmp_path / "work", tmp_path / "data-root"
    write_list(work, rows, deposit)
    write_env(work, DATA_ROOT=str(root))
    return work, root


class Server:
    """A stand-in web server on this machine: serves the files of a folder and logs every path and Authorization header it is asked with."""

    def __init__(self, folder: Path):
        self.folder, self.asked, self.authorization = folder, [], []
        folder.mkdir(parents=True, exist_ok=True)
        server = self

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                server.asked.append(self.path)
                server.authorization.append(self.headers.get("Authorization"))

        handler = functools.partial(Handler, directory=str(folder))
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.address = f"http://127.0.0.1:{self.http.server_port}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def put(self, name: str, content: bytes) -> None:
        (self.folder / name).parent.mkdir(parents=True, exist_ok=True)
        (self.folder / name).write_bytes(content)


@pytest.fixture
def deposit(tmp_path):
    """The public deposit's stand-in: its files are below /records/7/files/, under flat names (no "/"), the way a Zenodo record gives them."""
    server = Server(tmp_path / "deposit")
    yield server
    server.http.shutdown()


@pytest.fixture
def hub(tmp_path):
    """Hugging Face's stand-in: a dataset's file is at /datasets/<repo id>/resolve/<revision>/<path>."""
    server = Server(tmp_path / "hub")
    yield server
    server.http.shutdown()


DEPOSIT = "/records/7/files/"
GOLD, GOLD_REVISION = files.HF_DATASETS["MBDef-Gold"]
GOLD_ROWS = [
    (
        "MBDef-Gold/validated_gt.json",
        sha(ONE),
        len(ONE),
        "hf:MBDef-Gold/validated_gt.json",
    ),
    ("MBDef-Gold/figures/a.png", sha(TWO), len(TWO), "hf:MBDef-Gold/figures/a.png"),
]


def on_hub(hub: Server, dataset: str, path: str, content: bytes) -> None:
    repo, revision = files.HF_DATASETS[dataset]
    hub.put(f"datasets/{repo}/resolve/{revision}/{path}", content)


def test_fetch_downloads_the_files_of_a_dataset_from_hugging_face_and_checks_them(
    tmp_path, hub
):
    work, root = setup(tmp_path, GOLD_ROWS)
    on_hub(hub, "MBDef-Gold", "validated_gt.json", ONE)
    on_hub(hub, "MBDef-Gold", "figures/a.png", TWO)

    done = run_cli(
        work, "data", "fetch", "MBDef-Gold/", env={"HF_ENDPOINT": hub.address}
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [
        "MBDef-Gold/validated_gt.json: fetched, sha256 matches",
        "MBDef-Gold/figures/a.png: fetched, sha256 matches",
        "All 2 files under MBDef-Gold/ are in the data root with the listed sha256.",
    ]
    assert (root / "MBDef-Gold/validated_gt.json").read_bytes() == ONE
    assert (root / "MBDef-Gold/figures/a.png").read_bytes() == TWO
    # at the pinned revision of the dataset's repository
    assert hub.asked == [
        f"/datasets/{GOLD}/resolve/{GOLD_REVISION}/validated_gt.json",
        f"/datasets/{GOLD}/resolve/{GOLD_REVISION}/figures/a.png",
    ]
    assert hub.authorization == [None, None]  # no key unless HF_TOKEN is set


def test_a_dataset_that_is_not_on_hugging_face_yet_fails_naming_its_repository(
    tmp_path, hub
):
    work, root = setup(tmp_path, GOLD_ROWS[:1])

    done = run_cli(work, "data", "fetch", env={"HF_ENDPOINT": hub.address})

    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        f"MBDef-Gold/validated_gt.json: FAILED, the dataset MBDef-Gold ({GOLD} on Hugging Face, revision "
        f"{GOLD_REVISION}) has no file validated_gt.json, or is not uploaded or not public yet: "
        f"{hub.address}/datasets/{GOLD}/resolve/{GOLD_REVISION}/validated_gt.json: HTTP Error 404: File not found",
        "1 of 1 files are missing or differ from data/data-root.tsv.",
    ]
    assert [p for p in root.rglob("*") if p.is_file()] == []


def test_a_dataset_file_with_another_hash_is_not_kept(tmp_path, hub):
    work, root = setup(tmp_path, GOLD_ROWS[:1])
    on_hub(hub, "MBDef-Gold", "validated_gt.json", TWO)

    done = run_cli(work, "data", "fetch", env={"HF_ENDPOINT": hub.address})

    assert done.returncode == 1
    assert done.stdout.splitlines()[0] == (
        f"MBDef-Gold/validated_gt.json: FAILED, the download's sha256 is {sha(TWO)}, not {sha(ONE)}"
    )
    assert [p for p in root.rglob("*") if p.is_file()] == []


def test_with_hf_token_set_the_download_sends_it(tmp_path, hub):
    """A private dataset repository (before it is made public) needs a Hugging Face key."""
    work, root = setup(tmp_path, GOLD_ROWS[:1])
    on_hub(hub, "MBDef-Gold", "validated_gt.json", ONE)

    done = run_cli(
        work, "data", "fetch", env={"HF_ENDPOINT": hub.address, "HF_TOKEN": "hf_test"}
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert hub.authorization == ["Bearer hf_test"]


def test_a_dataset_row_must_name_one_of_the_two_datasets(tmp_path, hub):
    work, _ = setup(tmp_path, [("x/y.json", sha(ONE), len(ONE), "hf:Other/y.json")])

    done = run_cli(work, "data", "fetch", env={"HF_ENDPOINT": hub.address})

    assert done.returncode == 1
    assert done.stdout.splitlines()[0] == (
        "x/y.json: FAILED, Other is not a dataset of files.HF_DATASETS (MBDef, MBDef-Gold)"
    )
    assert hub.asked == []


# The public deposit and the commands that make a file.


def test_fetch_downloads_from_the_deposit_and_checks_each_file(tmp_path, deposit):
    rows_files = {"R1/status.json": b'{"row": "R1"}\n', "R1/papers/p.json": b"{}\n"}
    rows = [
        ("groundtruth/one.txt", sha(ONE), len(ONE), "https:groundtruth/one.txt"),
        (
            "frozen/run/rows",
            folder_digest(rows_files),
            17,
            "https:frozen/run.tar.zst#test-eval/rows",
        ),
        (
            "frozen/run/results.json",
            sha(TWO),
            len(TWO),
            "https:frozen/run.tar.zst#test-eval/final/results.json",
        ),
    ]
    work, root = setup(tmp_path, rows, deposit.address + DEPOSIT)
    deposit.put("records/7/files/groundtruth__one.txt", ONE)
    pack(
        deposit,
        "records/7/files/frozen__run.tar.zst",
        {**{f"rows/{n}": c for n, c in rows_files.items()}, "final/results.json": TWO},
    )

    done = run_cli(
        work, "data", "fetch", path=tool_dir(tmp_path / "bin", "zstd", "tar")
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [
        "groundtruth/one.txt: fetched, sha256 matches",
        "frozen/run/rows: fetched, sha256 matches",
        "frozen/run/results.json: fetched, sha256 matches",
        "All 3 files are in the data root with the listed sha256.",
    ]
    assert (root / "groundtruth/one.txt").read_bytes() == ONE
    assert (root / "frozen/run/rows/R1/papers/p.json").read_bytes() == b"{}\n"
    assert (root / "frozen/run/results.json").read_bytes() == TWO
    assert deposit.asked == [
        f"{DEPOSIT}groundtruth__one.txt",
        f"{DEPOSIT}frozen__run.tar.zst",
    ]  # once each, by the flat name ("/" becomes "__")
    assert sorted(p.name for p in root.iterdir()) == ["frozen", "groundtruth"]


def test_fetch_takes_only_the_paths_under_a_prefix(tmp_path, deposit):
    rows = [
        ("groundtruth/one.txt", sha(ONE), len(ONE), "https:groundtruth/one.txt"),
        ("frozen/two.txt", sha(TWO), len(TWO), "https:frozen/two.txt"),
    ]
    work, root = setup(tmp_path, rows, deposit.address + DEPOSIT)
    deposit.put("records/7/files/groundtruth__one.txt", ONE)

    done = run_cli(work, "data", "fetch", "groundtruth/")

    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [
        "groundtruth/one.txt: fetched, sha256 matches",
        "All 1 files under groundtruth/ are in the data root with the listed sha256.",
    ]
    assert not (root / "frozen").exists()


def test_fetch_leaves_files_that_are_already_there(tmp_path, deposit):
    rows = [("groundtruth/one.txt", sha(ONE), len(ONE), "https:groundtruth/one.txt")]
    work, root = setup(tmp_path, rows, deposit.address + DEPOSIT)
    (root / "groundtruth").mkdir(parents=True)
    (root / "groundtruth/one.txt").write_bytes(ONE)

    done = run_cli(work, "data", "fetch")

    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        done.stdout.splitlines()[0]
        == "groundtruth/one.txt: already there, sha256 matches"
    )
    assert deposit.asked == []


def test_fetch_never_overwrites_a_file_that_differs(tmp_path, deposit):
    rows = [("groundtruth/one.txt", sha(ONE), len(ONE), "https:groundtruth/one.txt")]
    work, root = setup(tmp_path, rows, deposit.address + DEPOSIT)
    (root / "groundtruth").mkdir(parents=True)
    (root / "groundtruth/one.txt").write_bytes(b"edited by hand\n")

    done = run_cli(work, "data", "fetch")

    assert done.returncode == 1
    assert done.stdout.splitlines()[0] == (
        "groundtruth/one.txt: CHANGED, its sha256 is "
        f"{sha(b'edited by hand' + bytes([10]))}, not {sha(ONE)}; not overwritten"
    )
    assert (root / "groundtruth/one.txt").read_bytes() == b"edited by hand\n"
    assert deposit.asked == []


def test_a_derived_file_is_listed_with_the_command_that_makes_it(tmp_path):
    rows = [
        (
            "frozen/big.json",
            sha(b"x"),
            1,
            "derive:mbdef evaluate grid --out $DATA_ROOT/frozen",
        )
    ]
    work, _ = setup(tmp_path, rows)

    done = run_cli(work, "data", "fetch", "frozen/")

    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        "frozen/big.json: MISSING; make it with: "
        "mbdef evaluate grid --out $DATA_ROOT/frozen",
        "1 of 1 files are missing or differ from data/data-root.tsv.",
    ]


def folder_digest(content: dict[str, bytes]) -> str:
    """The folder digest of CONTENT ({path inside the folder: bytes}), as the record defines it."""
    listing = "".join(
        f"{sha(data)}  ./{name}\n"
        for name, data in sorted(content.items(), key=lambda item: item[0].encode())
    )
    return sha(listing.encode())


def pack(server: Server, archive: str, content: dict[str, bytes]) -> None:
    """Put CONTENT ({path: bytes}) under `test-eval/` into the zstd tar ARCHIVE that SERVER serves."""
    run = server.folder.parent / "run/test-eval"
    for name, data in content.items():
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_bytes(data)
    tar = subprocess.run(
        ["tar", "-cf", "-", "-C", str(run.parent), "test-eval"],
        capture_output=True,
        check=True,
    )
    (server.folder / archive).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["zstd", "-q", "--long=31", "-o", str(server.folder / archive)],
        input=tar.stdout,
        check=True,
    )


def test_an_archive_member_with_another_hash_is_not_kept(tmp_path, deposit):
    rows = [
        (
            "frozen/results.json",
            sha(ONE),
            len(ONE),
            "https:frozen/run.tar.zst#test-eval/results.json",
        )
    ]
    work, root = setup(tmp_path, rows, deposit.address + DEPOSIT)
    pack(deposit, "records/7/files/frozen__run.tar.zst", {"results.json": TWO})

    done = run_cli(
        work, "data", "fetch", path=tool_dir(tmp_path / "bin", "zstd", "tar")
    )

    assert done.returncode == 1
    assert done.stdout.splitlines()[0] == (
        f"frozen/results.json: FAILED, the download's sha256 is {sha(TWO)}, not {sha(ONE)}"
    )
    assert [p for p in root.rglob("*") if p.is_file()] == []  # nothing is kept


def test_a_deposit_file_with_another_hash_is_not_kept(tmp_path, deposit):
    rows = [("groundtruth/one.txt", sha(ONE), len(ONE), "https:groundtruth/one.txt")]
    work, root = setup(tmp_path, rows, deposit.address + DEPOSIT)
    deposit.put("records/7/files/groundtruth__one.txt", TWO)

    done = run_cli(work, "data", "fetch")

    assert done.returncode == 1
    assert done.stdout.splitlines()[0] == (
        f"groundtruth/one.txt: FAILED, the download's sha256 is {sha(TWO)}, not {sha(ONE)}"
    )
    assert [p for p in root.rglob("*") if p.is_file()] == []


def test_a_file_missing_from_the_deposit_fails_with_the_address(tmp_path, deposit):
    address = deposit.address + DEPOSIT
    rows = [("groundtruth/one.txt", sha(ONE), len(ONE), "https:groundtruth/one.txt")]
    work, _ = setup(tmp_path, rows, address)

    done = run_cli(work, "data", "fetch")

    assert done.returncode == 1
    assert done.stdout.splitlines()[0] == (
        f"groundtruth/one.txt: FAILED, {address}groundtruth__one.txt: HTTP Error 404: File not found"
    )


def test_before_the_upload_a_deposit_file_is_listed_as_not_uploaded_yet(tmp_path):
    rows = [("groundtruth/one.txt", sha(ONE), len(ONE), "https:groundtruth/one.txt")]
    work, _ = setup(tmp_path, rows)

    done = run_cli(work, "data", "fetch")

    assert done.returncode == 1
    assert done.stdout.splitlines()[0] == (
        "groundtruth/one.txt: MISSING; the public deposit is not uploaded yet "
        f"(data/data-root.tsv gives the placeholder address {NOT_UPLOADED})"
    )


# The repository's own list.


def read_list() -> list[list[str]]:
    lines = (REPO / "data/data-root.tsv").read_text(encoding="utf-8").splitlines()
    return [line.split("\t") for line in lines if line and not line.startswith("#")][1:]


def test_every_listed_file_has_a_public_or_derived_source():
    """The list names no private store, no commit and no machine: each file is in one of the two datasets, in the deposit, or made by a command."""
    text = (REPO / "data/data-root.tsv").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("#deposit\thttps://")
    rows = read_list()
    kinds = {row[3].split(":", 1)[0] for row in rows}
    assert kinds <= {"hf", "https", "derive"}, kinds
    # The deposit's file names are flat ("/" becomes "__"): no path may already hold "__", and a Zenodo record holds at most 100 files by default.
    deposit = {
        r[3].removeprefix("https:").partition("#")[0]
        for r in rows
        if r[3][:6] == "https:"
    }
    assert [name for name in deposit if "__" in name] == []
    assert len(deposit) <= 100
    for word in ("rclone", "git:", "/home/", "/scratch/", "/projects/", "~/"):
        assert word not in text.lower(), word


def test_the_two_datasets_come_from_hugging_face_into_their_own_folders():
    """Each `hf:` row names one of the two datasets and lies in that dataset's folder of the data root, at the path the file has in the dataset's repository."""
    rows = [row for row in read_list() if row[3].startswith("hf:")]
    folders = {"MBDef": files.MBDEF, "MBDef-Gold": files.MBDEF_GOLD}
    for path, _, _, source in rows:
        dataset, _, inside = source.removeprefix("hf:").partition("/")
        assert path == f"{folders[dataset]}/{inside}", path
    assert {row[3].removeprefix("hf:").partition("/")[0] for row in rows} == set(
        folders
    )
    assert f"{files.MBDEF_GOLD}/validated_gt.json" in {row[0] for row in rows}


def test_the_two_datasets_are_not_in_git():
    """MBDef-Gold's records, report, statistics and fixed texts come from Hugging Face; data/ keeps only the inputs of the steps."""
    tracked = subprocess.run(
        ["git", "ls-files", "data"], cwd=REPO, capture_output=True, text=True
    ).stdout.split()
    if not tracked:
        pytest.skip("not a git checkout")
    assert [p for p in tracked if Path(p).name.startswith("validated_gt.")] == []
    assert (
        [p for p in tracked if Path(p).name.startswith("validated_gt_")]
        == [
            "data/validated_gt_citations.json",  # the closure's resolved citations (Step 4's input)
            "data/validated_gt_citations_corpus.json",
        ]
    )
