"""Publishable copies of the ground truth's annotation exports (paper Step 3, its ground truth).

`groundtruth build` and `evaluate agreement` read two zips that may name the annotators: the raw Label Studio exports (`Annotations.zip`) and the curator's export (`validated-v3.zip`, which holds the raw exports again as `annotations-original.zip`). `pseudonymise` writes copies of both that name no one and hold no local path and no paper text, and keeps everything the two commands read:

1. an unsubmitted draft's user name and e-mail become the pseudonym of its Label Studio account (`account-<number>`); the account numbers stay as they are, as the released statistics use them;
2. the local path of each paper's PDF becomes `papers/<file name>`;
3. the paper passages behind the assistant's proposals are removed (their paragraph numbers stay);
4. the hashes that name what changed are recomputed: the fingerprint of the raw exports in three members, and the final audit's seals; `README.txt` says what was done.

A changed JSON file is written back in its own format, so a copy differs from its original only in the replaced values; the zips keep their member order, times and compression. Before anything is written, every member of both copies is scanned for an e-mail address, a local path and a person's name field: one that is left stops the command, because the copies would not be publishable. docs/step-3-extraction/groundtruth.md has the worked example and the pinned hashes of the copies.
"""

import json
import re
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from mbdef import files, groundtruth

# Outputs, in --out: the copies, at the paths the originals have below groundtruth/ in DATA_ROOT.
EXPORT_COPY = "validated-v3.zip"
ANNOTATIONS_COPY = "newAnnotations/Annotations.zip"

# The rules.
PSEUDONYM = "account-{}"  # a Label Studio account's pseudonym, from its account number
PDF_FOLDER = "papers"  # what the local folder of the papers' PDFs becomes
# The note is written into the published copy, whose sha256 is pinned (`groundtruth.PSEUDONYMISED_EXPORT_SHA256`), so it keeps the command's name as released.
README_NOTE = (
    "\nThis is a pseudonymised copy for publication, made by `taxonomy_pipeline groundtruth "
    "pseudonymise`. An unsubmitted draft's user name and e-mail are replaced by the pseudonym of "
    "its Label Studio account (account-<number>), the local paths of the PDFs by "
    f"{PDF_FOLDER}/<file name>, and the paper passages in curation-suggestions.json are removed "
    "(their block numbers stay; its sha256 still names the file the curator worked with). "
    "annotations-original.zip is the pseudonymised copy of the raw exports; the fingerprints that "
    "name it and the final audit's hashes are recomputed. Every record, label, vote, note and "
    "decision is unchanged.\n"
)
COUNTED = (  # what the summary counts
    "draft names and e-mails",
    "local PDF paths",
    "paper passages removed",
    "fingerprints",
    "final-audit hashes",
)

# The JSON byte formats (`files.json_text` settings) that the two exports' members are written in: a changed member is written back in the one it had.
JSON_FORMATS: dict[str, dict[str, Any]] = {
    "Label Studio": dict(
        indent=None, separators=(",", ":"), escape_slashes=True, end=""
    ),
    "the curation app": dict(indent=2, end=""),
    # a file the app wrote on its own, outside an export: with a final newline
    "the curation app's own file": dict(indent=2),
}

# What must not be left in a copy.
EMAIL = files.EMAIL
LOCAL_PATH = re.compile(
    r"(?<![\w.:/])(?:"
    + "|".join(map(re.escape, files.LOCAL_FOLDERS))
    + r"|[A-Za-z]:\\)"
)
PERSON_KEYS = {
    "user",
    "username",
    "created_username",
    "email",
    "first_name",
    "last_name",
}
PSEUDONYMOUS = re.compile(r"account-\d+(?:, \d+)?")

# The members of the curator's export that the rules change.
PROVENANCE = "curation-provenance.json"
PROPOSALS = "curation-suggestions.json"
FINAL_AUDIT = "final-touches.json"
PDF_SOURCES = "pdf-sources.json"
RAW_COPY = "annotations-original.zip"
README = "README.txt"


# The zips


def unpacked(data: bytes) -> tuple[list[zipfile.ZipInfo], dict[str, bytes]]:
    with zipfile.ZipFile(BytesIO(data)) as archive:
        infos = archive.infolist()
        return infos, {info.filename: archive.read(info) for info in infos}


def packed(infos: list[zipfile.ZipInfo], members: dict[str, bytes]) -> bytes:
    """The members again, with the names, order, times, compression and attributes of INFOS."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for info in infos:
            member = zipfile.ZipInfo(info.filename, info.date_time)
            member.compress_type = info.compress_type
            member.external_attr = info.external_attr
            member.create_system = info.create_system
            archive.writestr(member, members[info.filename])
    return buffer.getvalue()


def rewritten(name: str, raw: bytes, value: Any) -> bytes:
    """VALUE as the bytes of member NAME: RAW if nothing changed, else in RAW's own format."""
    original = json.loads(raw)
    if value == original:
        return raw
    for byte_format in JSON_FORMATS.values():
        if files.json_text(original, **byte_format).encode() == raw:
            return files.json_text(value, **byte_format).encode()
    raise SystemExit(
        f"{name} is not written as {' or '.join(JSON_FORMATS)} writes JSON, so a copy would "
        "change more than the replaced values."
    )


# The rules


def pseudonym(account: int) -> str:
    return PSEUDONYM.format(account)


def draft_account(draft: dict) -> int:
    """The Label Studio account of a draft: `created_username` is '<name> <e-mail>, <account>'."""
    match = re.fullmatch(r".*, (\d+)", str(draft.get("created_username", "")), re.S)
    if not match:
        raise SystemExit(
            f"Draft {draft.get('id')} names no account number in created_username."
        )
    return int(match[1])


def pseudonymised_annotations(data: bytes, counts: Counter) -> bytes:
    """The raw Label Studio exports, with each draft's name and e-mail replaced (rule 1)."""
    infos, members = unpacked(data)
    for name, raw in members.items():
        if not name.endswith(".json"):
            continue
        tasks = json.loads(raw)
        for task in tasks:
            for draft in task.get("drafts") or []:
                account = draft_account(draft)
                draft["user"] = pseudonym(account)
                draft["created_username"] = f"{pseudonym(account)}, {account}"
                counts["draft names and e-mails"] += 1
        members[name] = rewritten(name, raw, tasks)
    return packed(infos, members)


def remove_passages(proposals: dict, counts: Counter) -> None:
    """Rule 3 on the curator's suggestions (curation-suggestions.json): each proposal's sources keep their block number and lose the paper passage."""
    for proposal in proposals.get("records", {}).values():
        for source in proposal.get("source", []):
            if source.pop("text", None) is not None:
                counts["paper passages removed"] += 1


def neutral_paths(pdfs: list[dict], counts: Counter) -> None:
    """Rule 2 on PDF entries: each local path becomes papers/<file name>."""
    for pdf in pdfs:
        if "path" in pdf:
            pdf["path"] = f"{PDF_FOLDER}/{PurePosixPath(pdf['path']).name}"
            counts["local PDF paths"] += 1


def pseudonymised_export(
    data: bytes, raw_exports: bytes, raw_copy: bytes, counts: Counter
) -> bytes:
    """The curator's export with rules 2 to 4, holding RAW_COPY instead of RAW_EXPORTS."""
    infos, members = unpacked(data)
    docs = {
        name: json.loads(members[name])
        for name in (PROVENANCE, PROPOSALS, FINAL_AUDIT, PDF_SOURCES)
        if name in members
    }
    provenance = docs[PROVENANCE]
    named = provenance["fingerprint"]["archive_sha256"]
    if named != files.sha256(raw_exports):
        raise SystemExit(
            f"The curator's export was made from other raw exports: its fingerprint names "
            f"{named}, not {files.sha256(raw_exports)}."
        )
    if RAW_COPY in members:
        if members[RAW_COPY] != raw_exports:
            raise SystemExit(
                f"The export's {RAW_COPY} is not the raw exports it names."
            )
        members[RAW_COPY] = raw_copy
    for doc in docs.values():  # rule 4: the fingerprints name the copy
        if "fingerprint" in doc:
            doc["fingerprint"]["archive_sha256"] = files.sha256(raw_copy)
            counts["fingerprints"] += 1

    audit = docs.get(FINAL_AUDIT, {})
    touches = audit.get("records", {})
    sealed = {key: groundtruth.object_digest(t) for key, t in touches.items()}
    pdfs = [
        *docs.get(PDF_SOURCES, {}).values(),
        *(t["pdf"] for t in touches.values() if "pdf" in t),
    ]
    neutral_paths(pdfs, counts)  # rule 2
    remove_passages(docs.get(PROPOSALS, {}), counts)  # rule 3

    # Rule 4: the final audit's seals. Each touched decision (and its history) names the digest of its touch, and the audit and the provenance name the audit's digest.
    for key, old in sealed.items():
        new = groundtruth.object_digest(touches[key])
        for entry in [provenance["decisions"][key], *provenance["history"][key]]:
            if new != old and entry.get("final_review_sha256") == old:
                entry["final_review_sha256"] = new
                counts["final-audit hashes"] += 1
    if "sha256" in audit:
        old = audit["sha256"]
        new = groundtruth.object_digest(
            {k: v for k, v in audit.items() if k != "sha256"}
        )
        if new != old:
            audit["sha256"] = new
            counts["final-audit hashes"] += 1
            if provenance.get("final_audit_sha256") == old:
                provenance["final_audit_sha256"] = new
                counts["final-audit hashes"] += 1

    for name, doc in docs.items():
        members[name] = rewritten(name, members[name], doc)
    if README in members:
        members[README] += README_NOTE.encode()
    return packed(infos, members)


# The check before writing


def traces(zip_name: str, members: dict[str, bytes]) -> list[str]:
    """What still names a person or a local folder in the members of ZIP_NAME (a zip inside is checked on its own)."""
    found: list[str] = []

    def check(where: str, text: str) -> None:
        for kind, pattern in (
            ("an e-mail address", EMAIL),
            ("a local path", LOCAL_PATH),
        ):
            if pattern.search(text):
                found.append(f"{kind} in {where}")

    def walk(where: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                check(f"{where}.{key}", key)
                if key in PERSON_KEYS and not (
                    item is None
                    or isinstance(item, int)
                    or PSEUDONYMOUS.fullmatch(str(item))
                ):
                    found.append(f"a person's {key} in {where}.{key}")
                walk(f"{where}.{key}", item)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(f"{where}[{i}]", item)
        elif isinstance(value, str):
            check(where, value)

    for name, content in members.items():
        if name.endswith((".zip", "/")):
            continue
        if name.endswith(".json"):
            walk(f"{zip_name}: {name} at $", json.loads(content))
        else:
            check(f"{zip_name}: {name}", content.decode("utf-8", errors="replace"))
    return list(dict.fromkeys(found))


# The command


def pseudonymise(out: str, export: str | None, annotations: str | None) -> int:
    """Write publishable copies of the curator's export and the raw exports into OUT."""
    target = Path(out)
    export_path, _ = files.located(export, groundtruth.EXPORT)
    annotations_path, _ = files.located(annotations, groundtruth.ANNOTATIONS)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is not empty; choose a new folder for --out.")
    counts = Counter(dict.fromkeys(COUNTED, 0))
    raw_exports = annotations_path.read_bytes()
    raw_copy = pseudonymised_annotations(raw_exports, counts)
    export_copy = pseudonymised_export(
        export_path.read_bytes(), raw_exports, raw_copy, counts
    )
    left = traces(PurePosixPath(ANNOTATIONS_COPY).name, unpacked(raw_copy)[1]) + traces(
        EXPORT_COPY, unpacked(export_copy)[1]
    )
    if left:
        shown = "; ".join(left[:6]) + (
            f"; and {len(left) - 6} more" if len(left) > 6 else ""
        )
        raise SystemExit(
            f"Not written: the copies would still name a person or a local folder ({shown}). "
            "Add a rule for it (docs/step-3-extraction/groundtruth.md)."
        )

    outputs = {
        EXPORT_COPY: target / EXPORT_COPY,
        ANNOTATIONS_COPY: target / ANNOTATIONS_COPY,
    }
    for path, content in zip(outputs.values(), (export_copy, raw_copy), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    receipt = files.receipt(
        "groundtruth pseudonymise",
        settings={
            "pseudonym": PSEUDONYM,
            "pdf_folder": PDF_FOLDER,
            "replaced": dict(counts),
        },
        inputs={"export": export_path, "annotations": annotations_path},
        outputs=outputs,
    )
    files.write_json(target / files.RECEIPT, receipt)
    export_sha = files.sha256(export_copy)
    print(
        json.dumps(
            {
                "replaced": dict(counts),
                "export": str(outputs[EXPORT_COPY]),
                "export_sha256": export_sha,
                "annotations": str(outputs[ANNOTATIONS_COPY]),
                "annotations_sha256": files.sha256(raw_copy),
                "pinned": export_sha == groundtruth.PSEUDONYMISED_EXPORT_SHA256,
            },
            indent=2,
        )
    )
    return 0
