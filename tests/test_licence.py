"""The licence files agree: the code is Apache 2.0, the data is CC BY 4.0, and the README says what else is under which licence."""

import tomllib
from pathlib import Path

from mbdef import dataset

REPO = Path(__file__).resolve().parents[1]


def test_license_is_the_apache_2_0_text():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")

    assert text.lstrip().startswith("Apache License\n")
    assert "Version 2.0, January 2004" in text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text
    assert 'distributed under the License is distributed on an "AS IS" BASIS' in text


def test_notice_names_the_group_and_2026():
    text = (REPO / "NOTICE").read_text(encoding="utf-8")

    assert "Copyright 2026 Media Bias Group" in text


def test_the_package_metadata_names_the_same_licence():
    project = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))["project"]

    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]


def test_the_readme_has_a_licence_section_for_code_data_and_quotations():
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    section = readme.split("\n## Licence\n")[1].split("\n## ")[0]
    assert "](LICENSE)" in section
    assert "Apache License 2.0" in section
    assert "CC BY 4.0" in section
    assert "authors" in section


def test_the_dataset_cards_stay_on_cc_by_4_0():
    assert dataset.LICENCE == "cc-by-4.0"
