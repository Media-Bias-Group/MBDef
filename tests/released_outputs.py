"""What the original implementation wrote on the miniature inputs of some tests, read from a stored bundle.

`tests/fixtures/released-outputs.zip` holds the outputs of the original scoring and selection code on small made-up inputs. Its README.md says what made each file and which test reads it. The tests compare the package with these files.
"""

import json
import zipfile
from pathlib import Path

BUNDLE = Path(__file__).with_name("fixtures") / "released-outputs.zip"


def read(name: str, **folders: str) -> bytes:
    """The bytes of file NAME of the bundle. A placeholder `@KEY@` in it stands for a temporary folder of the test that made the file; each keyword argument puts that test's folder in."""
    with zipfile.ZipFile(BUNDLE) as bundle:
        data = bundle.read(name)
    for key, folder in folders.items():
        data = data.replace(f"@{key}@".encode(), folder.encode())
    return data


def read_json(name: str, **folders: str):
    return json.loads(read(name, **folders))
