"""`mbdef repo check-links` on throwaway git repositories."""

import subprocess
import sys


def check_links(repo, *flags):
    command = [sys.executable, "-m", "mbdef", "repo", "check-links", *flags]
    return subprocess.run(command, cwd=repo, capture_output=True, text=True)


def test_a_template_placeholder_is_not_a_link_to_check(git_repo):
    """A page template's `{{name}}` is filled in when the page is built (for example the dataset card's links)."""
    repo = git_repo({"card_template.md": "See [the other dataset]({{gold_url}}).\n"})

    done = check_links(repo)

    assert done.returncode == 0, done.stdout
    assert done.stdout.startswith("OK: all 0 internal links in 1 file")


def test_links_that_resolve_pass(git_repo):
    repo = git_repo(
        {
            "README.md": "# Top\n\nSee [the step](docs/step.md) and the [data folder](data/).\n",
            "docs/step.md": "# Step\n## A\nBack to [README](../README.md#top), [x](step.md?plain=1#a).\n",
            "data/table.tsv": "a\tb\n",
        }
    )

    done = check_links(repo)

    assert done.returncode == 0, done.stdout + done.stderr
    assert (
        done.stdout
        == "OK: all 4 internal links in 2 files of the index point to files, folders and headings in the tree.\n"
    )


def test_a_link_to_a_missing_file_is_reported_with_its_line(git_repo):
    repo = git_repo(
        {
            "README.md": "# Title\n\nSee [the step](docs/gone.md).\n",
            "docs/step.md": "Back to [README](../README.md).\n",
        }
    )

    done = check_links(repo)

    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        "README.md:3: docs/gone.md: no such file or folder in the tree",
        "Broken: 1 of 2 internal links in 2 files of the index.",
    ]


def test_code_is_not_a_link(git_repo):
    repo = git_repo(
        {
            "README.md": (
                "Write `[text](docs/example.md)` for a link.\n"
                "\n"
                "```markdown\n"
                "[also an example](docs/other.md)\n"
                "```\n"
                "\n"
                "~~~\n"
                "[and this](docs/third.md)\n"
                "~~~\n"
            ),
        }
    )

    done = check_links(repo)

    assert done.returncode == 0, done.stdout
    assert (
        done.stdout
        == "OK: all 0 internal links in 1 file of the index point to files, folders and headings in the tree.\n"
    )


def test_other_ways_to_write_a_link(git_repo):
    repo = git_repo(
        {
            "README.md": (
                "[ref]: docs/a%20b.md\n"
                'An <a href="docs/gone.html">anchor</a> and ![fig](<docs/my figure.png>).\n'
                'A [titled](docs/a%20b.md "the title") link and a [paren](docs/f(1).md) one.\n'
            ),
            "docs/a b.md": "x\n",
            "docs/my figure.png": b"\x89PNG",
            "docs/f(1).md": "x\n",
        }
    )

    done = check_links(repo)

    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        "README.md:2: docs/gone.html: no such file or folder in the tree",
        "Broken: 1 of 5 internal links in 3 files of the index.",
    ]


def test_an_anchor_must_be_a_heading_of_the_page_it_points_at(git_repo):
    repo = git_repo(
        {
            "README.md": (
                "# Title\n"
                "[same page](#title) [other page](docs/a.md#the-cluster-job-evaluate-score)\n"
                "[spaces and symbols](docs/a.md#p--28-papers) [second heading](docs/a.md#run-it-1)\n"
                "[a line of code](docs/code.py#L3) [html anchor](docs/a.md#custom)\n"
                "[no such heading](docs/a.md#nope) [not here](#missing)\n"
                "[example in code](docs/a.md#in-a-fence)\n"
            ),
            "docs/a.md": (
                "# A page\n"
                "## The cluster job: `evaluate score`\n"
                "### P ≥ 28 papers\n"
                "## Run it\n"
                "## Run it\n"
                '<a id="custom"></a>\n'
                "```bash\n"
                "# in a fence\n"
                "```\n"
            ),
            "docs/code.py": "x = 1\n",
        }
    )

    done = check_links(repo)

    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        "README.md:5: docs/a.md#nope: no heading with that anchor in docs/a.md",
        "README.md:5: #missing: no heading with that anchor in README.md",
        "README.md:6: docs/a.md#in-a-fence: no heading with that anchor in docs/a.md",
        "Broken: 3 of 9 internal links in 2 files of the index.",
    ]


def test_web_addresses_are_not_checked(git_repo):
    repo = git_repo(
        {
            "README.md": (
                "[repo](https://github.com/someone/repo/blob/main/docs/gone.md)\n"
                "[web](https://example.org/docs/gone.md) [mail](mailto:someone@example.org)\n"
            ),
        }
    )

    done = check_links(repo)

    assert done.returncode == 0, done.stdout
    assert (
        done.stdout
        == "OK: all 0 internal links in 1 file of the index point to files, folders and headings in the tree.\n"
    )


def test_latex_links(git_repo):
    repo = git_repo(
        {
            "docs/draft.tex": (
                "A \\href{gone.md}{study}, \\href{step.md}{a step} and "
                "\\href{https://example.org}{a site}.\n"
            ),
            "docs/step.md": "x\n",
        }
    )

    done = check_links(repo)

    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        "docs/draft.tex:1: gone.md: no such file or folder in the tree",
        "Broken: 1 of 2 internal links in 2 files of the index.",
    ]


def test_hash_gated_files_are_not_read_but_can_be_linked_to(git_repo):
    repo = git_repo(
        {
            "README.md": "[frozen](data/frozen/note.md)\n",
            "data/frozen/note.md": "[gone](missing.md)\n",
        }
    )

    done = check_links(repo)

    assert done.returncode == 0, done.stdout
    skipped, summary = done.stdout.splitlines()
    assert skipped.startswith("Skipped 1 hash-gated file (data/frozen/**")
    assert (
        summary
        == "OK: all 1 internal link in 1 file of the index point to files, folders and headings in the tree."
    )


def test_rev_reads_that_commit_not_the_index(git_repo):
    repo = git_repo({"README.md": "[step](docs/step.md)\n", "docs/step.md": "x\n"})
    (repo / "README.md").write_text("[step](docs/gone.md)\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)

    in_the_index = check_links(repo)
    in_the_commit = check_links(repo, "--rev", "HEAD")

    assert in_the_index.returncode == 1
    assert in_the_commit.returncode == 0
    assert "of commit HEAD" in in_the_commit.stdout


def test_absolute_paths_and_root_paths(git_repo):
    repo = git_repo(
        {
            "README.md": (
                "[laptop](/home/someone/Documents/notes.md)\n"
                "[from the root](/docs/step.md)\n"
                "[up too far](../outside.md)\n"
            ),
            "docs/step.md": "x\n",
        }
    )

    done = check_links(repo)

    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        "README.md:1: /home/someone/Documents/notes.md: no such file or folder in the tree",
        "README.md:3: ../outside.md: no such file or folder in the tree",
        "Broken: 2 of 3 internal links in 2 files of the index.",
    ]
