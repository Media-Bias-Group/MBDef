"""The `mbdef` command line: `mbdef <group> <action> [flags]`."""

import argparse
import json
import sys
from pathlib import Path

import mbdef
from mbdef import (
    cluster,
    doc_links,
    files,
    groundtruth,
    models,
    pseudonymise,
)
from mbdef.closure import bundle as closure_bundle_step
from mbdef.closure import (
    corpus_check,
    extend_gt,
    manual_citations,
    pdf_identity,
)
from mbdef.closure import inputs as closure_inputs_step
from mbdef.closure import run as closure_run
from mbdef.collect import crossref as collect_crossref
from mbdef.collect import download as collect_download
from mbdef.collect import grobid as collect_grobid
from mbdef.collect import ledger as collect_ledger
from mbdef.collect import open_access, openalex, semantic_scholar
from mbdef.evaluate import (
    agreement,
    grid,
    heldout,
    metric,
    paired,
    selection,
)
from mbdef.evaluate import score as scoring
from mbdef.extract import derive as derive_step
from mbdef.extract import filter as filter_step
from mbdef.extract import plan as plan_step
from mbdef.extract import relevance
from mbdef.extract import run as extract
from mbdef.extract import train as extract_train
from mbdef.extract import train_filter as filter_train
from mbdef.extract import verify as verify_step
from mbdef.relevance import classify as relevance_classify
from mbdef.relevance import decide as relevance_decide


def main(argv: list[str] | None = None) -> int:
    warning = mbdef.untested_python(sys.version_info[:3])
    if warning:
        print(warning, file=sys.stderr)
    parser = argparse.ArgumentParser(
        prog="mbdef",
        description="Run the steps of the pipeline behind the MBDef datasets, and the link check of the docs.",
    )
    groups = parser.add_subparsers(dest="group", required=True, metavar="<group>")

    repo_parser = groups.add_parser(
        "repo",
        help="repository checks: the links of the docs (CONTRIBUTING.md)",
    )
    repo_actions = repo_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    check_links = repo_actions.add_parser(
        "check-links",
        help="check that every internal link of the docs points at a file, folder or heading in the tree",
        description=(
            "Read the markdown and LaTeX files of a commit's tree (or the index) without a "
            "checkout and list every relative link or path from the repository root whose "
            "target is not in the tree, or whose #anchor is not a heading of the page it points "
            "at. Exit 1 if there is any. Web addresses are not checked, and neither are the "
            "hash-gated "
            f"files ({', '.join(doc_links.HASH_GATED)}), whose text a stored hash pins."
        ),
    )
    check_links.add_argument(
        "--rev",
        help="check this commit's tree instead of the index (the files the next commit would hold)",
    )
    check_links.set_defaults(run=lambda args: doc_links.check_links(args.rev))

    data_parser = groups.add_parser(
        "data",
        help="the data kept outside git, under DATA_ROOT (docs/setup/data-root.md)",
    )
    data_actions = data_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    fetch = data_actions.add_parser(
        "fetch",
        help="fetch data-root files from Hugging Face and the public deposit and check their sha256",
        description=(
            f"Put every file that {files.DATA_ROOT_MANIFEST} lists under PREFIX into DATA_ROOT: "
            "files already there are checked, missing files are downloaded from Hugging Face "
            "(`hf:` sources, the datasets MBDef and MBDef-Gold) or the public deposit (`https:` "
            "sources) and kept only if their sha256 matches; `derive:` sources print the command "
            "that makes the file. "
            "Nothing is ever overwritten (docs/setup/data-fetch.md)."
        ),
    )
    fetch.add_argument(
        "prefix", nargs="*", help="only paths starting with this, e.g. MBDef-Gold/"
    )
    fetch.set_defaults(run=lambda args: files.fetch(args.prefix))

    def positive(kind):  # argparse types for the collect flags
        def parse(text: str):
            if (value := kind(text)) <= 0:
                raise argparse.ArgumentTypeError(f"must be positive, not {text}")
            return value

        return parse

    def not_negative(kind):
        def parse(text: str):
            if (value := kind(text)) < 0:
                raise argparse.ArgumentTypeError(f"cannot be negative, not {text}")
            return value

        return parse

    relevance_parser = groups.add_parser(
        "relevance",
        help="the relevance step: which papers are about media bias, from title and abstract (docs/step-1-2-papers/relevance.md)",
    )
    relevance_actions = relevance_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    papers_help = "the papers file: corpus_id, paperid, title, abstract, doi (default: $DATA_ROOT/relevance/papers.csv)"
    classify_cmd = relevance_actions.add_parser(
        "classify",
        help="ask one classifier about every paper, or replay its recorded answers",
        description=(
            "Write answers.csv into --out: one answer and its label for each paper with a title or abstract, from "
            f"the classifier's model on an OpenAI-compatible server (DSPY_API_BASE and DSPY_API_KEY in .env; "
            f"max_tokens {relevance_classify.MAX_TOKENS}, no temperature). Papers without an answer are listed in "
            f"{relevance_classify.NOT_ANSWERED}. With --replay, the answers come from a recording and nothing is "
            "sent; a recorded answer that answers no request built from the papers stops the run."
        ),
    )
    classify_cmd.add_argument(
        "--classifier",
        required=True,
        choices=sorted(relevance_classify.CLASSIFIERS),
        help="qwen and gemma classify all papers; glm arbitrates their disagreements",
    )
    classify_cmd.add_argument("--papers", help=papers_help)
    classify_cmd.add_argument("--out", required=True, help="the output folder")
    classify_cmd.add_argument(
        "--replay",
        metavar="FOLDER",
        help="answer from the recording in FOLDER (answers and metadata.json), send nothing",
    )
    classify_cmd.add_argument(
        "--examples",
        help="the arbiter's labelled examples (default: $DATA_ROOT/relevance/examples.csv)",
    )
    classify_cmd.add_argument(
        "--model", help="the model's name on the server (default: the classifier's)"
    )
    classify_cmd.add_argument(
        "--max-tokens",
        type=positive(int),
        default=relevance_classify.MAX_TOKENS,
        help="the answer's token limit (default: %(default)s)",
    )
    classify_cmd.add_argument(
        "--temperature",
        type=float,
        help="send this temperature (default: none, the model's own setting)",
    )
    classify_cmd.add_argument(
        "--api-base", help="the server (default: DSPY_API_BASE from .env)"
    )
    classify_cmd.add_argument(
        "--workers",
        type=positive(int),
        help="requests in parallel (default: the classifier's, 32 or 8)",
    )
    classify_cmd.set_defaults(
        run=lambda args: relevance_classify.classify_command(
            args.classifier,
            args.papers,
            args.out,
            args.replay,
            args.examples,
            args.model,
            args.max_tokens,
            args.temperature,
            args.api_base,
            args.workers,
        )
    )
    compare_cmd = relevance_actions.add_parser(
        "compare",
        help="the papers Qwen and Gemma both marked relevant, and those they disagree on",
        description=(
            f"Write {relevance_decide.AGREED} (both answered relevant) and {relevance_decide.DISAGREEMENTS} (two "
            "different labels; the arbiter's papers file) into --out."
        ),
    )
    compare_cmd.add_argument("--papers", help=papers_help)
    compare_cmd.add_argument(
        "--qwen",
        required=True,
        help="the output folder of `classify --classifier qwen`",
    )
    compare_cmd.add_argument(
        "--gemma",
        required=True,
        help="the output folder of `classify --classifier gemma`",
    )
    compare_cmd.add_argument("--out", required=True, help="the output folder")
    compare_cmd.set_defaults(
        run=lambda args: relevance_decide.compare_command(
            args.papers, args.qwen, args.gemma, args.out
        )
    )
    export_cmd = relevance_actions.add_parser(
        "export",
        help="the relevant papers: the agreed ones and the disagreements the arbiter marked relevant",
        description=f"Write {relevance_decide.RELEVANT} into --out.",
    )
    export_cmd.add_argument(
        "--compare", required=True, help="the output folder of `relevance compare`"
    )
    export_cmd.add_argument(
        "--glm", required=True, help="the output folder of `classify --classifier glm`"
    )
    export_cmd.add_argument("--out", required=True, help="the output folder")
    export_cmd.set_defaults(
        run=lambda args: relevance_decide.export_command(
            args.compare, args.glm, args.out
        )
    )

    collect_parser = groups.add_parser(
        "collect",
        help="the collect step: find, download and convert PDFs (docs/step-1-2-papers/collect.md)",
    )
    collect_actions = collect_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    folder_help = f"the step's folder with its ledgers (default: $DATA_ROOT/{collect_ledger.FOLDER})"
    queue_help = (
        f"the collection queue (default: {collect_ledger.COLLECTION} in the folder)"
    )
    open_access_cmd = collect_actions.add_parser(
        "open-access",
        help="find open-access PDFs of the queued papers with Unpaywall",
        description=(
            "For each `missing` row of the collection queue that OpenAlex left without a PDF "
            "(run `collect openalex` first), ask Unpaywall for an open-access PDF of the DOI "
            "OpenAlex names, or of the queue's or a found DOI if OpenAlex has not answered. "
            f"Appends to {open_access.STATUS}, rewrites {open_access.CANDIDATES} and appends a "
            "receipt. Needs UNPAYWALL_EMAIL in .env. Rerun to resume."
        ),
    )
    open_access_cmd.add_argument("--folder", help=folder_help)
    open_access_cmd.add_argument("--collection", help=queue_help)
    open_access_cmd.add_argument(
        "--api-url",
        default=open_access.UNPAYWALL_URL,
        help="the Unpaywall API (default: %(default)s)",
    )
    open_access_cmd.add_argument(
        "--sleep-seconds",
        type=not_negative(float),
        default=open_access.SLEEP_SECONDS,
        help="pause after each request (default: %(default)s)",
    )
    open_access_cmd.add_argument(
        "--timeout-seconds",
        type=positive(float),
        default=open_access.TIMEOUT_SECONDS,
        help="request timeout (default: %(default)s)",
    )
    open_access_cmd.add_argument(
        "--limit", type=positive(int), help="look up at most this many papers"
    )
    open_access_cmd.add_argument(
        "--reference-urls",
        action="store_true",
        help="keep the reference's own web address as a source: a fallback when Unpaywall has no "
        "PDF, and the only one of a row without a DOI (the v1.1 closure did not)",
    )
    open_access_cmd.set_defaults(
        run=lambda args: open_access.command(
            args.folder,
            args.collection,
            args.api_url,
            args.sleep_seconds,
            args.timeout_seconds,
            args.limit,
            args.reference_urls,
        )
    )

    openalex_cmd = collect_actions.add_parser(
        "openalex",
        help="find PDFs of the queued papers with OpenAlex (by DOI or by title)",
        description=(
            "For each `missing` row of the collection queue, look its DOI up in OpenAlex (a DOI "
            "OpenAlex lacks is left to Unpaywall), or search its title and accept only a result "
            "with the same title, an author surname and the year (or, under the default title "
            "rule, a year one off when no result has the row's year). A row whose Semantic "
            "Scholar match has a PDF waits until that download failed for good. Appends to "
            f"{openalex.STATUS}, rewrites {openalex.CANDIDATES} and appends a receipt. Uses "
            "OPENALEX_API_KEY from .env if it is set. Rerun to resume."
        ),
    )
    openalex_cmd.add_argument("--folder", help=folder_help)
    openalex_cmd.add_argument("--collection", help=queue_help)
    openalex_cmd.add_argument(
        "--api-url",
        default=openalex.OPENALEX_URL,
        help="the OpenAlex API (default: %(default)s)",
    )
    openalex_cmd.add_argument(
        "--sleep-seconds",
        type=not_negative(float),
        default=openalex.SLEEP_SECONDS,
        help="pause after each request; retries wait twice as long each time (default: %(default)s)",
    )
    openalex_cmd.add_argument(
        "--timeout-seconds",
        type=positive(float),
        default=openalex.TIMEOUT_SECONDS,
        help="request timeout (default: %(default)s)",
    )
    openalex_cmd.add_argument(
        "--max-retries",
        type=not_negative(int),
        default=openalex.MAX_RETRIES,
        help="retries after an HTTP 429 or 5xx answer (default: %(default)s)",
    )
    openalex_cmd.add_argument(
        "--limit", type=positive(int), help="look up at most this many papers"
    )
    openalex_cmd.add_argument(
        "--title-rule",
        choices=openalex.TITLE_RULES,
        default=openalex.TITLE_RULE,
        help="the title search for a row without a DOI: 2026-09-28 is a full-text search, "
        "2026-09-29 OpenAlex's title filter, the one the v1.1 closure used (the names are "
        "stored in the ledgers; docs/step-1-2-papers/collect.md; default: %(default)s)",
    )
    openalex_cmd.set_defaults(
        run=lambda args: openalex.command(
            args.folder,
            args.collection,
            args.api_url,
            args.sleep_seconds,
            args.timeout_seconds,
            args.max_retries,
            args.limit,
            args.title_rule,
        )
    )

    crossref_cmd = collect_actions.add_parser(
        "crossref",
        help="find the DOIs of queued papers that have none with Crossref",
        description=(
            "For each `missing` row of the collection queue without a DOI, search Crossref by "
            "its title and authors (the first --rows records) and take a DOI only if the records "
            "with the same title, a year the row names and an author surname carry exactly one DOI. "
            f"Appends to {collect_crossref.STATUS} "
            "and appends a receipt; `openalex`, `open-access` and `semantic-scholar` read the "
            "DOIs found. Uses CROSSREF_MAILTO (else UNPAYWALL_EMAIL) from .env, if set, for "
            "Crossref's polite pool. Rerun to resume."
        ),
    )
    crossref_cmd.add_argument("--folder", help=folder_help)
    crossref_cmd.add_argument("--collection", help=queue_help)
    crossref_cmd.add_argument(
        "--api-url",
        default=collect_crossref.CROSSREF_URL,
        help="the Crossref API (default: %(default)s)",
    )
    crossref_cmd.add_argument(
        "--sleep-seconds",
        type=not_negative(float),
        default=collect_crossref.SLEEP_SECONDS,
        help="pause before each request, longer if Crossref asks (default: %(default)s)",
    )
    crossref_cmd.add_argument(
        "--timeout-seconds",
        type=positive(float),
        default=collect_crossref.TIMEOUT_SECONDS,
        help="request timeout (default: %(default)s)",
    )
    crossref_cmd.add_argument(
        "--max-attempts",
        type=positive(int),
        default=collect_crossref.MAX_ATTEMPTS,
        help="attempts per paper after network errors and HTTP 429 or 5xx (default: %(default)s)",
    )
    crossref_cmd.add_argument(
        "--backoff-seconds",
        type=not_negative(float),
        default=collect_crossref.BACKOFF_SECONDS,
        help="wait after a failed attempt, doubling each time (default: %(default)s)",
    )
    crossref_cmd.add_argument(
        "--limit", type=positive(int), help="search at most this many papers"
    )
    crossref_cmd.add_argument(
        "--rows",
        type=positive(int),
        default=collect_crossref.ROWS,
        help="records per search, the most relevant first (default: %(default)s)",
    )
    crossref_cmd.add_argument(
        "--min-title-chars",
        type=positive(int),
        default=collect_crossref.MIN_TITLE_CHARS,
        help="a row whose compared title is shorter is too little evidence for a DOI "
        "(default: %(default)s)",
    )
    crossref_cmd.set_defaults(
        run=lambda args: collect_crossref.command(
            args.folder,
            args.collection,
            args.api_url,
            args.sleep_seconds,
            args.timeout_seconds,
            args.max_attempts,
            args.backoff_seconds,
            args.limit,
            args.rows,
            args.min_title_chars,
        )
    )

    s2_cmd = collect_actions.add_parser(
        "semantic-scholar",
        help="match the papers Crossref found no DOI for by title with Semantic Scholar",
        description=(
            "For each queued paper without a DOI that `collect crossref` settled as `no_match`, "
            "take Semantic Scholar's best title match if its title, year and an author surname "
            "agree; keep the DOI it names only if Crossref's record of that DOI passes the same "
            f"rule. Appends to {semantic_scholar.STATUS}, rewrites {semantic_scholar.CANDIDATES} "
            "(the matches' open PDFs and arXiv copies) and appends a receipt. Uses "
            "SEMANTIC_SCHOLAR_API_KEY from .env if it is set. Rerun to resume."
        ),
    )
    s2_cmd.add_argument("--folder", help=folder_help)
    s2_cmd.add_argument("--collection", help=queue_help)
    s2_cmd.add_argument(
        "--api-url",
        default=semantic_scholar.SEMANTIC_SCHOLAR_URL,
        help="the Semantic Scholar Graph API (default: %(default)s)",
    )
    s2_cmd.add_argument(
        "--crossref-url",
        default=collect_crossref.CROSSREF_URL,
        help="the Crossref API that checks a DOI Semantic Scholar names (default: %(default)s)",
    )
    s2_cmd.add_argument(
        "--sleep-seconds",
        type=not_negative(float),
        default=semantic_scholar.SLEEP_SECONDS,
        help="pause before each title match, doubled after an HTTP 429 (default: %(default)s)",
    )
    s2_cmd.add_argument(
        "--check-sleep-seconds",
        type=not_negative(float),
        default=semantic_scholar.CHECK_SLEEP_SECONDS,
        help="pause before each Crossref check (default: %(default)s)",
    )
    s2_cmd.add_argument(
        "--timeout-seconds",
        type=positive(float),
        default=semantic_scholar.TIMEOUT_SECONDS,
        help="request timeout (default: %(default)s)",
    )
    s2_cmd.add_argument(
        "--max-attempts",
        type=positive(int),
        default=semantic_scholar.MAX_ATTEMPTS,
        help="attempts per paper after network errors and HTTP 429 or 5xx (default: %(default)s)",
    )
    s2_cmd.add_argument(
        "--backoff-seconds",
        type=not_negative(float),
        default=semantic_scholar.BACKOFF_SECONDS,
        help="wait after a failed attempt, this much longer each time (default: %(default)s)",
    )
    s2_cmd.add_argument(
        "--limit", type=positive(int), help="match at most this many papers"
    )
    s2_cmd.set_defaults(
        run=lambda args: semantic_scholar.command(
            args.folder,
            args.collection,
            args.api_url,
            args.crossref_url,
            args.sleep_seconds,
            args.check_sleep_seconds,
            args.timeout_seconds,
            args.max_attempts,
            args.backoff_seconds,
            args.limit,
        )
    )

    s2_doi_cmd = collect_actions.add_parser(
        "semantic-scholar-doi",
        help="ask Semantic Scholar for open copies of the DOIs whose PDF is still missing",
        description=(
            "For each queued paper that `collect openalex` settled and that has no downloaded PDF, "
            f"ask Semantic Scholar about its DOI ({semantic_scholar.BATCH_SIZE} at a time) for "
            "the open copies it knows: its open-access PDF, the arXiv copy and Europe PMC's PDF "
            "of a PubMed Central paper. Appends to "
            f"{semantic_scholar.DOI_STATUS}, rewrites {semantic_scholar.DOI_CANDIDATES} and "
            "appends a receipt. Uses SEMANTIC_SCHOLAR_API_KEY from .env if it is set. Rerun to "
            "ask about DOIs that are new or whose batch failed."
        ),
    )
    s2_doi_cmd.add_argument("--folder", help=folder_help)
    s2_doi_cmd.add_argument("--collection", help=queue_help)
    s2_doi_cmd.add_argument(
        "--api-url",
        default=semantic_scholar.SEMANTIC_SCHOLAR_URL,
        help="the Semantic Scholar Graph API (default: %(default)s)",
    )
    s2_doi_cmd.add_argument(
        "--sleep-seconds",
        type=not_negative(float),
        default=semantic_scholar.BATCH_SLEEP_SECONDS,
        help="pause after each batch (default: %(default)s)",
    )
    s2_doi_cmd.add_argument(
        "--timeout-seconds",
        type=positive(float),
        default=semantic_scholar.BATCH_TIMEOUT_SECONDS,
        help="request timeout (default: %(default)s)",
    )
    s2_doi_cmd.add_argument(
        "--backoff-seconds",
        type=not_negative(float),
        default=semantic_scholar.BACKOFF_SECONDS,
        help="wait after a failed batch, this much longer each time (default: %(default)s)",
    )
    s2_doi_cmd.set_defaults(
        run=lambda args: semantic_scholar.doi_command(
            args.folder,
            args.collection,
            args.api_url,
            args.sleep_seconds,
            args.timeout_seconds,
            args.backoff_seconds,
        )
    )

    download_cmd = collect_actions.add_parser(
        "download",
        help="download the PDFs a resolver found into zip batches",
        description=(
            "Try each candidate's PDF addresses in order, with a browser's TLS fingerprint, until "
            "one answers with a PDF under the size cap, and store it in "
            f"{collect_download.ARCHIVES}/pdf_batch_<n>.zip. If none does, try Europe PMC's "
            "rendering of a PubMed Central paper and the PDF a landing page declares. Europe PMC "
            "is asked at most once every 3 seconds. Appends to "
            f"{collect_download.STATUS}, rewrites {collect_download.FAILED} and appends a receipt. "
            "Rerun to resume."
        ),
    )
    download_cmd.add_argument("--folder", help=folder_help)
    download_cmd.add_argument(
        "--candidates",
        help=(
            f"the candidates to download (default: {open_access.CANDIDATES} in the folder; "
            f"{openalex.CANDIDATES} for the OpenAlex ones)"
        ),
    )
    download_cmd.add_argument(
        "--concurrency",
        type=positive(int),
        default=collect_download.CONCURRENCY,
        help="parallel downloads (default: %(default)s)",
    )
    download_cmd.add_argument(
        "--timeout-seconds",
        type=positive(float),
        default=collect_download.TIMEOUT_SECONDS,
        help="time per address, including redirects and the body (default: %(default)s)",
    )
    download_cmd.add_argument(
        "--max-pdf-mb",
        type=positive(float),
        default=collect_download.MAX_PDF_MB,
        help="largest PDF kept, in MiB (default: %(default)s)",
    )
    download_cmd.add_argument(
        "--pdf-batch-size",
        type=positive(int),
        default=collect_download.PDF_BATCH_SIZE,
        help="PDFs per zip (default: %(default)s)",
    )
    download_cmd.add_argument(
        "--limit", type=positive(int), help="download at most this many papers"
    )
    download_cmd.add_argument(
        "--europe-pmc-url",
        default=collect_download.EUROPE_PMC_URL,
        help="Europe PMC, whose PDF rendering of a PubMed Central paper is tried after a failed "
        "download (default: %(default)s)",
    )
    download_cmd.set_defaults(
        run=lambda args: collect_download.command(
            args.folder,
            args.candidates,
            args.concurrency,
            args.timeout_seconds,
            args.max_pdf_mb,
            args.pdf_batch_size,
            args.limit,
            args.europe_pmc_url,
        )
    )

    grobid_cmd = collect_actions.add_parser(
        "grobid",
        help="convert the downloaded PDFs to TEI with a running GROBID service",
        description=(
            "Send each downloaded PDF to GROBID and keep the TEI only if its title page shows the "
            "requested work (the same DOI, or the same title, an author and the year), with a "
            f".closure.json receipt beside it, in {collect_grobid.TEI_FOLDER}/. Appends to "
            f"{collect_grobid.STATUS}, rewrites {collect_grobid.FAILED} and appends a receipt. "
            "Exits 1 if a PDF was not converted. Rerun to resume."
        ),
    )
    grobid_cmd.add_argument("--folder", help=folder_help)
    grobid_cmd.add_argument(
        "--downloads",
        help=f"the download ledger (default: {collect_download.STATUS} in the folder)",
    )
    grobid_cmd.add_argument(
        "--identity-verdicts",
        help=(
            "accepted.json of the closure step's PDF identity check: PDFs the rule could not "
            "verify but a checked verdict accepts, bound to their exact bytes"
        ),
    )
    grobid_cmd.add_argument(
        "--grobid-url",
        default=collect_grobid.GROBID_URL,
        help="the running GROBID service (default: %(default)s)",
    )
    grobid_cmd.add_argument(
        "--concurrency",
        type=positive(int),
        default=collect_grobid.CONCURRENCY,
        help="parallel GROBID requests (default: %(default)s)",
    )
    grobid_cmd.add_argument(
        "--timeout-seconds",
        type=positive(float),
        default=collect_grobid.TIMEOUT_SECONDS,
        help="time per PDF (default: %(default)s)",
    )
    grobid_cmd.add_argument(
        "--consolidate-header",
        action="store_true",
        default=collect_grobid.CONSOLIDATE_HEADER,
        help="let GROBID complete the header from online sources (default: off)",
    )
    grobid_cmd.add_argument(
        "--limit", type=positive(int), help="convert at most this many PDFs"
    )
    grobid_cmd.set_defaults(
        run=lambda args: collect_grobid.command(
            args.folder,
            args.downloads,
            args.identity_verdicts,
            args.grobid_url,
            args.concurrency,
            args.timeout_seconds,
            args.consolidate_header,
            args.limit,
        )
    )

    closure_parser = groups.add_parser(
        "closure",
        help="the closure step: the works the seed papers cite, and the release bundle (docs/step-4-closure/closure.md)",
    )
    closure_actions = closure_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    closure_run_cmd = closure_actions.add_parser(
        "run",
        help="resolve the seed papers' citations, check them against the corpus, queue what is missing",
        description=(
            "Resolve every citation string of the seed papers' extracted definitions to a reference-list "
            "entry, decide which cited works are already in the corpus roots, and write the rest as the "
            "collection queue `before-queue.csv` (the input of `collect`). With --collect-folder, check "
            "again with the TEI files `collect grobid` kept (`after*`). Reruns resume in the same --out; "
            "changed inputs need a new folder."
        ),
    )
    closure_run_cmd.add_argument(
        "--dataset",
        required=True,
        help="the seed papers' extracted definitions (JSON: papers with definition_records)",
    )
    closure_run_cmd.add_argument("--out", required=True, help="the output folder")
    closure_run_cmd.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="PATH",
        help="a corpus root: a folder of TEI files or a zip of them (repeat for several)",
    )
    closure_run_cmd.add_argument(
        "--tei-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="a folder with the seed papers' TEI files, `<paper id>.tei.xml` (repeat for several)",
    )
    closure_run_cmd.add_argument(
        "--source-zip",
        help="a zip with the seed papers' TEI files as `<paper id>.tei.xml`",
    )
    closure_run_cmd.add_argument(
        "--collect-folder",
        help="the collect step's folder: its TEI files are one more corpus root for the `after` check",
    )
    closure_run_cmd.add_argument(
        "--workers",
        type=positive(int),
        default=closure_run.WORKERS,
        help="corpus files scanned at the same time (default: %(default)s)",
    )
    closure_run_cmd.set_defaults(
        run=lambda args: closure_run.command(
            args.dataset,
            args.out,
            args.corpus,
            args.tei_dir,
            args.source_zip,
            args.collect_folder,
            args.workers,
        )
    )

    closure_resolve = closure_actions.add_parser(
        "resolve",
        help="resolve the citation strings of the ground truth's extracted items",
        description=(
            "Resolve every citation string of the ground truth's items to a reference-list entry of "
            "its paper's TEI. Writes citations.json, report.json and review.csv into --out and merges "
            "the unresolved ones into the manual-link ledger (YAML), keeping completed entries."
        ),
    )
    closure_resolve.add_argument(
        "--ground-truth", required=True, help="the ground truth (JSON)"
    )
    closure_resolve.add_argument("--out", required=True, help="the output folder")
    closure_resolve.add_argument(
        "--manual-links",
        required=True,
        help="the manual-link ledger (YAML), read and rewritten",
    )
    closure_resolve.add_argument(
        "--source-zip", help="a zip with the papers' TEI files as `<paper id>.tei.xml`"
    )
    closure_resolve.add_argument(
        "--grobid-archive", help="a folder of `paper_<paper id>.grobid.tei.xml`"
    )
    closure_resolve.add_argument(
        "--grobid-files", help="a folder of `doi_<doi>.grobid.tei.xml`"
    )
    closure_resolve.add_argument(
        "--tei-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="a folder of TEI files named by paper id (repeat for several)",
    )
    closure_resolve.add_argument(
        "--refresh", action="store_true", help="ignore the cached TEI files"
    )
    closure_resolve.add_argument(
        "--min-linked",
        type=not_negative(int),
        default=0,
        help="stop if fewer markers than this are linked (default: %(default)s: report only)",
    )
    closure_resolve.set_defaults(
        run=lambda args: extend_gt.command(
            args.ground_truth,
            args.out,
            args.manual_links,
            args.source_zip,
            args.grobid_archive,
            args.grobid_files,
            args.tei_dir,
            args.refresh,
            args.min_linked,
        )
    )

    closure_manual = closure_actions.add_parser(
        "manual",
        help="apply the completed manual links of the ledger",
        description=(
            "Validate the completed rows of the manual-link ledger and apply them to the files of "
            "`closure resolve` (rewritten in place, all or none). Needs no network."
        ),
    )
    closure_manual.add_argument(
        "--manual-links", required=True, help="the ledger (YAML)"
    )
    closure_manual.add_argument(
        "--dataset", required=True, help="citations.json of `closure resolve`"
    )
    closure_manual.add_argument(
        "--report", required=True, help="report.json of `closure resolve`"
    )
    closure_manual.add_argument(
        "--review", required=True, help="review.csv of `closure resolve`"
    )
    closure_manual.set_defaults(
        run=lambda args: manual_citations.command(
            args.manual_links, args.dataset, args.report, args.review
        )
    )

    closure_check = closure_actions.add_parser(
        "check",
        help="check every cited work of a resolved dataset against the corpus roots",
        description=(
            "Decide for each cited work of the dataset (citations.json of `closure resolve`) whether "
            "a paper with that identity is in the corpus roots. Writes the dataset with each work's "
            f"match, {corpus_check.COVERAGE} and {corpus_check.QUEUE} (the `collect` queue) into --out."
        ),
    )
    closure_check.add_argument("--dataset", required=True, help="citations.json")
    closure_check.add_argument("--out", required=True, help="the output folder")
    closure_check.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="PATH",
        help="a corpus root: a folder of TEI files or a zip of them (repeat for several)",
    )
    closure_check.add_argument(
        "--workers",
        type=positive(int),
        default=16,
        help="corpus files scanned at the same time (default: %(default)s)",
    )
    closure_check.add_argument(
        "--fuzzy-suggestions",
        action="store_true",
        help="add near-title suggestions for review (slow; never a match)",
    )
    closure_check.add_argument(
        "--allow-empty-root",
        action="store_true",
        help="treat a corpus root without TEI files as complete instead of `review`",
    )
    closure_check.set_defaults(
        run=lambda args: corpus_check.command(
            args.dataset,
            args.corpus,
            args.out,
            args.workers,
            args.fuzzy_suggestions,
            args.allow_empty_root,
        )
    )

    closure_identity = closure_actions.add_parser(
        "identity",
        help="the PDF identity check: a model confirms PDFs the rule could not verify",
        description=(
            "Four actions, in order: `evidence` (the PDFs `collect grobid` refused as unverified, plus "
            "controls), `judge` (a model answers each item; needs a server, or --replay), `decide` "
            "(program checks turn answers into accepted.json, the input of `collect grobid "
            "--identity-verdicts`) and `merge` (the accepted lists of two rounds in one)."
        ),
    )
    identity_actions = closure_identity.add_subparsers(
        dest="step", required=True, metavar="<step>"
    )
    identity_evidence = identity_actions.add_parser(
        "evidence",
        help="the cited reference and the PDF's first pages, for each unverified PDF and for controls",
        description=(
            f"Write {pdf_identity.EVIDENCE} into --out from the collect step's folder: a candidate for "
            "each PDF the identity rule refused, a positive control for each of a sample of verified "
            "PDFs and negative controls pairing a reference with another paper's PDF. Needs `pdftotext` "
            "(poppler). With --earlier, only candidates whose PDF was not judged before are kept and no "
            f"control is repeated; the unfiltered evidence is kept as {pdf_identity.EVIDENCE_FULL}."
        ),
    )
    identity_evidence.add_argument(
        "--collect-folder",
        help=f"the collect step's folder (default: $DATA_ROOT/{collect_ledger.FOLDER})",
    )
    identity_evidence.add_argument("--out", required=True, help="the output folder")
    identity_evidence.add_argument(
        "--controls",
        type=positive(int),
        default=pdf_identity.CONTROLS,
        help="positive controls to draw (default: %(default)s)",
    )
    identity_evidence.add_argument(
        "--seed",
        type=int,
        default=pdf_identity.SEED,
        help="the draw's seed (default: %(default)s)",
    )
    identity_evidence.add_argument(
        "--earlier",
        action="append",
        default=[],
        metavar="EVIDENCE",
        help="the evidence.json of an earlier round (repeat for several)",
    )
    identity_evidence.set_defaults(
        run=lambda args: pdf_identity.evidence_command(
            args.collect_folder, args.out, args.controls, args.seed, args.earlier
        )
    )

    identity_judge = identity_actions.add_parser(
        "judge",
        help="ask the model whether each PDF is the cited work",
        description=(
            f"Write {pdf_identity.VERDICTS} into --out: the model's JSON answer to every item of the "
            f"evidence, one request each (temperature 0, {pdf_identity.MAX_TOKENS} tokens, thinking off, "
            f"the JSON schema), with every request recorded in {pdf_identity.REQUESTS}/. With --replay, "
            "answers come from recorded requests and nothing is sent; a request that was not recorded "
            "stops the run."
        ),
    )
    identity_judge.add_argument("--evidence", required=True, help="evidence.json")
    identity_judge.add_argument("--out", required=True, help="the output folder")
    identity_judge.add_argument(
        "--base",
        default=pdf_identity.BASE_URL,
        help="the OpenAI-compatible server (default: %(default)s)",
    )
    identity_judge.add_argument(
        "--model",
        default=pdf_identity.MODEL,
        help="the model's name on the server (default: %(default)s)",
    )
    identity_judge.add_argument(
        "--revision",
        default=pdf_identity.REVISION,
        help="the model revision, recorded in the verdicts (default: %(default)s)",
    )
    identity_judge.add_argument(
        "--workers",
        type=positive(int),
        default=pdf_identity.WORKERS,
        help="requests in parallel (default: %(default)s)",
    )
    identity_judge.add_argument(
        "--replay",
        action="append",
        default=[],
        metavar="FOLDER",
        help="answer from the request receipts in FOLDER, send nothing (repeat for several)",
    )
    identity_judge.set_defaults(
        run=lambda args: pdf_identity.judge_command(
            args.evidence,
            args.out,
            args.base,
            args.model,
            args.revision,
            args.workers,
            args.replay,
        )
    )

    identity_decide = identity_actions.add_parser(
        "decide",
        help="apply the program checks to the verdicts",
        description=(
            f"Write {pdf_identity.ACCEPTED} (the candidates a verdict and the program checks accept, "
            f"with the controls' error rates) and {pdf_identity.DECISIONS} into --out. No model call."
        ),
    )
    identity_decide.add_argument("--evidence", required=True, help="evidence.json")
    identity_decide.add_argument("--verdicts", required=True, help="verdicts.json")
    identity_decide.add_argument("--out", required=True, help="the output folder")
    identity_decide.set_defaults(
        run=lambda args: pdf_identity.decide_command(
            args.evidence, args.verdicts, args.out
        )
    )

    identity_merge = identity_actions.add_parser(
        "merge",
        help="join the accepted lists of two rounds",
        description=(
            "Write one accepted.json from the earlier rounds' (already merged) and a new round's; "
            "a paper accepted in both takes the later verdict."
        ),
    )
    identity_merge.add_argument(
        "--earlier", required=True, help="accepted.json of the earlier round(s)"
    )
    identity_merge.add_argument(
        "--new", required=True, help="accepted.json of the new round"
    )
    identity_merge.add_argument(
        "--round", type=positive(int), required=True, help="the new round's number"
    )
    identity_merge.add_argument("--out", required=True, help="the merged accepted.json")
    identity_merge.set_defaults(
        run=lambda args: pdf_identity.merge_command(
            args.earlier, args.new, args.round, args.out
        )
    )

    closure_inputs = closure_actions.add_parser(
        "inputs",
        help="rebuild the four lists `closure bundle` reads (all-candidates, papers-info, uncited, existing-exclude)",
        description=(
            "Write all-candidates.json (every collected paper), papers-info.json (what was known of each "
            "when the duplicate check ran), uncited.json (collected papers no seed reference cites) and "
            "existing-exclude.json (existing-corpus works whose text another such work has) into --out, "
            "with a receipt, from the closure run's own files. The defaults are the files of the v1.1 "
            "closure under $DATA_ROOT; docs/step-4-closure/closure.md says how the v1.1 lists are rebuilt byte for byte."
        ),
    )
    closure_inputs.add_argument("--out", required=True, help="the output folder")
    for flag, default, what in (
        (
            "ledgers",
            closure_bundle_step.LEDGERS,
            "the closure run's ledgers (before.json, provider.csv, ...)",
        ),
        (
            "earlier-ledgers",
            closure_inputs_step.EARLIER_LEDGERS,
            "the ledgers as they were when the extracted batches ran; papers-info's citing counts",
        ),
        (
            "closure-set",
            closure_bundle_step.CLOSURE_SET,
            "the closure papers' extraction: rows-all.json and batches.json",
        ),
        (
            "closure-tei",
            closure_bundle_step.CLOSURE_TEI,
            "the collected closure TEI files with their receipts, as the duplicate check saw them",
        ),
        ("seeds", closure_bundle_step.SEEDS, "the seed run's derived/"),
        (
            "seed-inputs",
            closure_bundle_step.SEED_INPUTS,
            "the seed run's seed-tei/ and the existing corpus's TEI zip and sha256 list",
        ),
    ):
        closure_inputs.add_argument(
            f"--{flag}", help=f"{what} (default: $DATA_ROOT/{default})"
        )
    closure_inputs.add_argument(
        "--extracted-batches",
        type=int,
        default=closure_inputs_step.EXTRACTED_BATCHES,
        help="how many of the closure set's extraction batches had run when the duplicate check ran "
        "(default: %(default)s, as in v1.1)",
    )
    closure_inputs.add_argument(
        "--tei-prefix",
        help="the folder written in front of each TEI path of all-candidates.json (default: --closure-tei)",
    )
    closure_inputs.set_defaults(
        run=lambda args: closure_inputs_step.command(
            args.out,
            args.ledgers,
            args.earlier_ledgers,
            args.closure_set,
            args.closure_tei,
            args.seeds,
            args.seed_inputs,
            args.extracted_batches,
            args.tei_prefix,
        )
    )

    closure_bundle = closure_actions.add_parser(
        "bundle",
        help="rebuild the closure's release folder from the run's saved files (docs/step-4-closure/closure.md)",
        description=(
            "Leave out the papers whose text is another paper's, add the works found only in the "
            "existing corpus, link every seed definition to its cited work and closure paper, take the "
            "PDF identity check's decisions, and build the release folder with its README and "
            "MANIFEST.sha256, then verify every manifest line. Writes audit/, existing/, links/, "
            "identity/, release/ and a receipt into --out. The defaults are the files of the v1.1 "
            "closure under $DATA_ROOT (`mbdef data fetch frozen/`); the other flags swap "
            "one folder each (docs/step-4-closure/closure.md lists what each holds)."
        ),
    )
    closure_bundle.add_argument("--out", required=True, help="the output folder")
    closure_bundle.add_argument(
        "--version",
        default=closure_bundle_step.VERSION,
        help="the release's name, in its README (default: %(default)s)",
    )
    for flag, default, what in (
        (
            "ledgers",
            closure_bundle_step.LEDGERS,
            "the closure run's ledgers (before.json, provider.csv, ...)",
        ),
        (
            "closure-set",
            closure_bundle_step.CLOSURE_SET,
            "the closure papers' extraction: rows-all.json, derived/, compare/, ...",
        ),
        (
            "closure-tei",
            closure_bundle_step.CLOSURE_TEI,
            "the collected closure TEI files with their receipts",
        ),
        (
            "left-out-tei",
            closure_bundle_step.LEFT_OUT_TEI,
            "TEI files moved out of the collected folder after the duplicate check",
        ),
        (
            "audit-inputs",
            closure_bundle_step.AUDIT_INPUTS,
            "all-candidates.json, papers-info.json, uncited.json, existing-exclude.json",
        ),
        ("seeds", closure_bundle_step.SEEDS, "the seed run's derived/ and compare/"),
        (
            "seed-inputs",
            closure_bundle_step.SEED_INPUTS,
            "the seed run's runner-rows.json, seed-tei/ and the existing corpus's TEI zip",
        ),
        (
            "identity",
            closure_bundle_step.IDENTITY,
            "the PDF identity check's rounds: evidence.json and verdicts.json, round N in rN/",
        ),
    ):
        closure_bundle.add_argument(
            f"--{flag}", help=f"{what} (default: $DATA_ROOT/{default})"
        )
    closure_bundle.add_argument(
        "--loss-audit",
        help=f"the closure loss audit, copied into the release (default: {closure_bundle_step.LOSS_AUDIT})",
    )
    closure_bundle.add_argument(
        "--audit-reference",
        help="a folder with the released audit results: the rerun must reproduce them",
    )
    closure_bundle.add_argument(
        "--identity-reference",
        help="a folder with the released identity decisions: the rerun must reproduce them",
    )
    closure_bundle.set_defaults(
        run=lambda args: closure_bundle_step.command(
            args.out,
            args.version,
            args.ledgers,
            args.closure_set,
            args.closure_tei,
            args.left_out_tei,
            args.audit_inputs,
            args.seeds,
            args.seed_inputs,
            args.identity,
            args.loss_audit,
            args.audit_reference,
            args.identity_reference,
        )
    )

    gt_parser = groups.add_parser(
        "groundtruth",
        help="the ground-truth step: build MBDef-Gold (docs/step-3-extraction/groundtruth.md)",
    )
    gt_actions = gt_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    gt_build = gt_actions.add_parser(
        "build",
        help="build MBDef-Gold, the ground truth, from the curator's export",
        description=(
            "Check the curator's export against the raw annotations and V3, then write the "
            "relevant and non-relevant datasets, their statistics, the release report, a hash "
            "manifest, the release zip and a receipt into OUT. Nothing is written elsewhere."
        ),
    )
    gt_build.add_argument("--out", required=True, help="a new or empty folder")
    gt_build.add_argument(
        "--export",
        help=f"the curator's export zip (default: $DATA_ROOT/{groundtruth.EXPORT})",
    )
    gt_build.add_argument(
        "--annotations",
        help=f"the raw Label Studio exports (default: $DATA_ROOT/{groundtruth.ANNOTATIONS})",
    )
    gt_build.add_argument(
        "--wiki",
        help=f"a text snapshot of the annotation's method notes, put into the release zip if it exists (default: $DATA_ROOT/{groundtruth.WIKI})",
    )
    gt_build.add_argument(
        "--data",
        default="data",
        help="the repository's data folder with the earlier ground truths (default: data)",
    )
    gt_build.add_argument(
        "--export-sha256",
        help="the export's expected sha256 (default: the curator's final export or its "
        "pseudonymised copy)",
    )
    gt_build.set_defaults(
        run=lambda args: groundtruth.build(
            args.out,
            args.export,
            args.annotations,
            args.wiki,
            args.data,
            args.export_sha256,
        )
    )
    gt_pseudonymise = gt_actions.add_parser(
        "pseudonymise",
        help="write publishable copies of the two annotation exports",
        description=(
            "Write copies of the curator's export and the raw Label Studio exports that name no "
            "annotator and hold no local path and no paper text, and keep everything `groundtruth "
            "build` and `evaluate agreement` read, into OUT (docs/step-3-extraction/groundtruth.md). Nothing is "
            "written if a copy would still hold an e-mail address, a local path or a name field. "
            "It reads the two original exports, which are not public: the public deposit holds the "
            "copies it wrote."
        ),
    )
    gt_pseudonymise.add_argument("--out", required=True, help="a new or empty folder")
    gt_pseudonymise.add_argument(
        "--export",
        help=f"the curator's export zip (default: $DATA_ROOT/{groundtruth.EXPORT})",
    )
    gt_pseudonymise.add_argument(
        "--annotations",
        help=f"the raw Label Studio exports (default: $DATA_ROOT/{groundtruth.ANNOTATIONS})",
    )
    gt_pseudonymise.set_defaults(
        run=lambda args: pseudonymise.pseudonymise(
            args.out, args.export, args.annotations
        )
    )

    extract_parser = groups.add_parser(
        "extract",
        help="the extraction step: definition records from papers (docs/step-3-extraction/extract.md)",
    )
    extract_actions = extract_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    # Training: `extract train` and `extract check-programs` (docs/step-3-extraction/extract.md, Training).
    train_cmd = extract_actions.add_parser(
        "train",
        help="train an extractor program with Bootstrap, as the released programs were",
        description=(
            "Train a prompt program for one window size the way the released programs were "
            "(DSPy BootstrapFewShot, one demo, Gemma 4 12B with thinking): the TRAIN papers in "
            "the seed-9 order, each read window by window and scored with the full-record "
            "score until one scores above 0. Writes the program, its manifest, every call and "
            "a receipt into OUT. With --replay, every model call is answered from recorded "
            "responses, and a call that was not recorded stops the run."
        ),
    )
    train_cmd.add_argument("--out", required=True, help="a new or empty folder")
    train_cmd.add_argument(
        "--window",
        required=True,
        choices=tuple(extract_train.WINDOWS),
        help="the window size: "
        + ", ".join(
            f"{k} = {s:,}/{o:,} characters"
            for k, (s, o) in extract_train.WINDOWS.items()
        ),
    )
    train_cmd.add_argument(
        "--paper",
        action="append",
        help="train on this paper of the dataset only (repeatable; default: all 80 TRAIN "
        "papers, with the 21 DEV papers held out)",
    )
    train_cmd.add_argument(
        "--dataset",
        default=str(extract_train.DATASET),
        help=f"the papers and their references (default: {extract_train.DATASET})",
    )
    train_cmd.add_argument(
        "--split",
        default=str(extract_train.SPLIT),
        help=f"TRAIN, DEV and the refused TEST ids (default: {extract_train.SPLIT})",
    )
    train_cmd.add_argument(
        "--teis",
        default=str(extract.TEIS),
        help=f"a folder or zip of <paper id>.tei.xml files (default: {extract.TEIS})",
    )
    train_cmd.add_argument(
        "--model", default=extract.MODEL, help=f"default: {extract.MODEL}"
    )
    train_cmd.add_argument(
        "--revision",
        default=extract.REVISION,
        help=f"the model revision the server runs, recorded (default: {extract.REVISION})",
    )
    train_cmd.add_argument(
        "--max-tokens",
        type=int,
        default=extract.MAX_TOKENS,
        help=f"output tokens per call, thinking included (default: {extract.MAX_TOKENS})",
    )
    train_cmd.add_argument(
        "--seed", type=int, default=extract.SEED, help=f"default: {extract.SEED}"
    )
    train_cmd.add_argument(
        "--threshold",
        type=float,
        default=extract_train.THRESHOLD,
        help=f"the full-record score's match cutoff (default: {extract_train.THRESHOLD})",
    )
    train_cmd.add_argument(
        "--replay",
        action="append",
        help="a folder of recorded requests to answer every call from (repeatable)",
    )
    train_cmd.add_argument(
        "--workers",
        type=int,
        default=extract_train.WORKERS,
        help="windows of one paper sent to the server at the same time "
        f"(default: {extract_train.WORKERS})",
    )
    train_cmd.add_argument(
        "--replay-scores",
        help="answer every BGE similarity and entailment of the metric from this file, "
        "which --record-scores wrote (no scoring models)",
    )
    train_cmd.add_argument(
        "--record-scores",
        help="also write every similarity and entailment the models computed to this new file",
    )
    train_cmd.add_argument(
        "--nli-batch-size",
        type=int,
        help="pairs per entailment batch (default: that of the released scoring runs)",
    )
    thinking_flag(train_cmd)
    train_cmd.set_defaults(
        run=lambda args: extract_train.train(
            Path(args.out),
            args.window,
            args.paper,
            Path(args.dataset),
            Path(args.split),
            Path(args.teis),
            args.model,
            args.revision,
            args.max_tokens,
            args.seed,
            args.threshold,
            [Path(folder) for folder in args.replay or []],
            args.workers,
            args.replay_scores,
            args.record_scores,
            args.nli_batch_size,
            args.thinking,
        )
    )
    check_programs = extract_actions.add_parser(
        "check-programs",
        help="check saved programs: they load, keep the signature, and quote their windows",
        description=(
            "Load each program and check that it keeps the student's signature, has at most "
            "one demo, and that every demo is, word for word, a window of a TRAIN paper of the "
            "dataset with that paper's title. Reports how many of each demo's record strings "
            "occur in its window. Writes nothing; exit code 1 if a check fails."
        ),
    )
    check_programs.add_argument(
        "--programs",
        action="append",
        help="a program folder, or a folder of them (repeatable; default: "
        f"{extract.PROGRAMS})",
    )
    check_programs.add_argument(
        "--dataset",
        default=str(extract_train.DATASET),
        help=f"default: {extract_train.DATASET}",
    )
    check_programs.add_argument(
        "--split",
        default=str(extract_train.SPLIT),
        help=f"default: {extract_train.SPLIT}",
    )
    check_programs.add_argument(
        "--teis", default=str(extract.TEIS), help=f"default: {extract.TEIS}"
    )
    check_programs.set_defaults(
        run=lambda args: extract_train.check_programs(
            [Path(p) for p in args.programs or [extract.PROGRAMS]],
            Path(args.dataset),
            Path(args.split),
            Path(args.teis),
        )
    )
    # Training inputs: `extract train-dataset` and `extract train-filter` (docs/step-3-extraction/extract.md).
    train_dataset_cmd = extract_actions.add_parser(
        "train-dataset",
        help="build the prompt-training dataset and its split from the ground truth and the TEIs",
        description=(
            "Write the dataset that `extract train` reads: the ground truth's TRAIN and DEV "
            "papers in its order, each with its title, its TEI's name and sha256, its text's "
            "sha256 and length, its number of windows per size, and its reference records "
            "(train-dataset.json); and the TRAIN, DEV and refused TEST ids (split.json). TEST "
            "papers are read for their ids only. Writes both and a receipt into OUT."
        ),
    )
    train_dataset_cmd.add_argument("--out", required=True, help="a new or empty folder")
    gold = f"$DATA_ROOT/{files.MBDEF_GOLD}"  # MBDef-Gold in the data root (`data fetch MBDef-Gold/`)
    train_dataset_cmd.add_argument(
        "--ground-truth",
        help="the TRAIN and DEV papers and their reference records "
        f"(default: {gold}/{extract_train.GROUND_TRUTH}, MBDef-Gold)",
    )
    train_dataset_cmd.add_argument(
        "--non-relevant",
        help=f"read for its TEST ids (default: {gold}/{extract_train.NON_RELEVANT})",
    )
    train_dataset_cmd.add_argument(
        "--metadata",
        default=str(extract_train.METADATA),
        help="the papers' titles, and more TEST ids (a paper missing there takes its TEI's "
        f"title; default: {extract_train.METADATA})",
    )
    train_dataset_cmd.add_argument(
        "--teis",
        default=str(extract.TEIS),
        help=f"a folder or zip of <paper id>.tei.xml files (default: {extract.TEIS})",
    )
    train_dataset_cmd.add_argument(
        "--plan",
        default=str(extract_train.PLAN),
        help=f"the training plan, recorded by sha256 (default: {extract_train.PLAN})",
    )
    train_dataset_cmd.add_argument(
        "--amendment",
        default=str(extract_train.AMENDMENT),
        help=f"the plan's amendment, recorded by sha256 (default: {extract_train.AMENDMENT})",
    )
    train_dataset_cmd.set_defaults(
        run=lambda args: extract_train.train_dataset(
            Path(args.out),
            args.ground_truth,
            args.non_relevant,
            Path(args.metadata),
            Path(args.teis),
            Path(args.plan),
            Path(args.amendment),
        )
    )

    def encoder_flags(action) -> None:
        """--bge, --replay and --filters: where `extract filter` and `extract train-filter` get the BGE vectors and the frozen filter zips."""
        action.add_argument(
            "--bge",
            help="the folder of BAAI/bge-base-en-v1.5 at its pinned revision "
            "(default: the Hugging Face cache)",
        )
        action.add_argument(
            "--replay",
            action="append",
            help="a recorded encoder/ folder to answer every encoding from (repeatable)",
        )
        action.add_argument(
            "--filters",
            default=str(relevance.ZIPS),
            help=f"the folder of the frozen filter zips (default: {relevance.ZIPS})",
        )

    train_filter_cmd = extract_actions.add_parser(
        "train-filter",
        help="fit a relevance filter on the ground truth's TRAIN records, as the frozen ones were",
        description=(
            "Fit the classifier of one relevance filter the way the frozen filters were fitted: "
            "the records of the ground truth's TRAIN papers (relevant and not relevant), their "
            "BGE vectors, --folds paper-grouped folds for out-of-fold probabilities, the cutoff "
            "that keeps --target-recall of the relevant records out of fold, and a logistic "
            "regression on all records. Writes the classifier, the training summary, the folds and "
            "every record's out-of-fold probability into OUT, with a receipt that compares them "
            "with the frozen filter (in the --filters folder). BGE runs on SCIDEF_NLI_DEVICE "
            "(.env); with --stored-vectors the vectors come from the frozen filter's zip, with "
            "--replay from a recording."
        ),
    )
    train_filter_cmd.add_argument("--out", required=True, help="a new or empty folder")
    train_filter_cmd.add_argument(
        "--filter",
        default=filter_step.FILTER,
        choices=filter_step.NAMES,
        help=f"which filter's classifier input and frozen zip (default: {filter_step.FILTER})",
    )
    train_filter_cmd.add_argument(
        "--ground-truth",
        help=f"the relevant records, label 1 (default: {gold}/{extract_train.GROUND_TRUTH}, MBDef-Gold)",
    )
    train_filter_cmd.add_argument(
        "--non-relevant",
        help=f"the records that are not relevant, label 0 (default: {gold}/{extract_train.NON_RELEVANT})",
    )
    train_filter_cmd.add_argument(
        "--stored-vectors",
        action="store_true",
        help="take the BGE vectors from the frozen filter's zip (no model, no GPU)",
    )
    encoder_flags(train_filter_cmd)
    train_filter_cmd.add_argument(
        "--folds",
        type=int,
        default=filter_train.FOLDS,
        help="paper-grouped folds for the out-of-fold probabilities (default: %(default)s)",
    )
    train_filter_cmd.add_argument(
        "--seed",
        type=int,
        default=filter_train.SEED,
        help="the folds' shuffle and the classifier's seed (default: %(default)s)",
    )
    train_filter_cmd.add_argument(
        "--target-recall",
        type=float,
        default=filter_train.TARGET_RECALL,
        help="the share of relevant records the cutoff keeps out of fold (default: %(default)s)",
    )
    train_filter_cmd.add_argument(
        "--regularisation",
        type=lambda text: int(float(text)) if float(text).is_integer() else float(text),
        default=filter_train.REGULARISATION,
        help="the logistic regression's C, the inverse of its regularisation strength "
        "(default: %(default)s)",
    )
    train_filter_cmd.set_defaults(
        run=lambda args: filter_train.train_filter(
            Path(args.out),
            args.filter,
            args.ground_truth,
            args.non_relevant,
            Path(args.filters),
            args.stored_vectors,
            Path(args.bge) if args.bge else None,
            [Path(folder) for folder in args.replay or []],
            args.folds,
            args.seed,
            args.target_recall,
            args.regularisation,
        )
    )

    extract_run = extract_actions.add_parser(
        "run",
        help="extract definition records with the saved programs, window by window",
        description=(
            "Read each paper's TEI, cut it into windows, ask the model for the definition records "
            "of every window, keep only strings copied from the window, and merge each paper's "
            "windows into one union per part. Writes everything into OUT, with a receipt, and "
            "every paper's records, the parts pooled, to OUT/predictions.jsonl. "
            "With --replay, every model call is answered from recorded responses, and a call "
            "that was not recorded stops the run."
        ),
    )
    extract_run.add_argument("--out", required=True, help="a new or empty folder")
    papers_from = extract_run.add_mutually_exclusive_group()
    papers_from.add_argument(
        "--teis",
        default=str(extract.TEIS),
        help=f"a folder or zip of <paper id>.tei.xml files (default: {extract.TEIS})",
    )
    papers_from.add_argument(
        "--pdfs",
        help="a folder of <paper id>.pdf files, converted first by a running GROBID into "
        "OUT/tei/",
    )
    extract_run.add_argument(
        "--grobid-url",
        default=collect_grobid.GROBID_URL,
        help=f"the GROBID of --pdfs (default: {collect_grobid.GROBID_URL})",
    )
    extract_run.add_argument(
        "--paper",
        action="append",
        help="run only this paper id (repeatable; default: every TEI in --teis)",
    )
    extract_run.add_argument(
        "--titles",
        help="a JSON file {paper id: title}; these titles replace the TEIs' own",
    )
    extract_run.add_argument(
        "--shard",
        help="a shard of `extract plan` (shards/NAME.json): its papers, their titles and "
        "their TEIs' sha256, instead of --paper and --titles",
    )
    extract_run.add_argument(
        "--programs",
        default=str(extract.PROGRAMS),
        help=f"the folder of program folders (default: {extract.PROGRAMS})",
    )
    extract_run.add_argument(
        "--part",
        action="append",
        help=(
            "programs to run together, joined by +, e.g. w16k+w32k (repeatable; default: "
            f"{' and '.join(extract.PARTS)}, the released extractor)"
        ),
    )
    extract_run.add_argument(
        "--format",
        default=extract.FORMAT,
        choices=("chat", "json"),
        help=f"the output format the model is asked for (default: {extract.FORMAT})",
    )
    extract_run.add_argument(
        "--model", default=extract.MODEL, help=f"default: {extract.MODEL}"
    )
    extract_run.add_argument(
        "--revision",
        default=extract.REVISION,
        help="the model revision the server runs, recorded in the receipt "
        f"(default: {extract.REVISION})",
    )
    extract_run.add_argument(
        "--max-tokens",
        type=int,
        default=extract.MAX_TOKENS,
        help=f"output tokens per call, thinking included (default: {extract.MAX_TOKENS})",
    )
    extract_run.add_argument(
        "--seed", type=int, default=extract.SEED, help=f"default: {extract.SEED}"
    )
    extract_run.add_argument(
        "--replay",
        action="append",
        help="a folder of recorded requests to answer every call from (repeatable)",
    )
    extract_run.add_argument(
        "--workers",
        type=int,
        default=extract.WORKERS,
        help=f"windows sent to the server at the same time (default: {extract.WORKERS})",
    )
    thinking_flag(extract_run)
    extract_run.set_defaults(
        run=lambda args: extract.run(
            Path(args.out),
            Path(args.teis),
            args.paper,
            Path(args.titles) if args.titles else None,
            Path(args.programs),
            args.part or extract.PARTS,
            args.format,
            args.model,
            args.revision,
            args.max_tokens,
            args.seed,
            [Path(folder) for folder in args.replay or []],
            args.workers,
            Path(args.shard) if args.shard else None,
            Path(args.pdfs) if args.pdfs else None,
            args.grobid_url,
            args.thinking,
        )
    )
    extract_verify = extract_actions.add_parser(
        "verify",
        help="ask a second model whether each extracted definition is one (the candidate verifier)",
        description=(
            "Read the unions and windows of one extraction run (a folder with papers/, such as "
            "one part of `extract run`), ask the model for a verdict on every candidate "
            "definition in batches per window, and keep the records whose definitions a verdict "
            "accepts. Writes everything into OUT, with a receipt. With --replay, every model call "
            "is answered from recorded responses, and a call that was not recorded stops the run."
        ),
    )
    extract_verify.add_argument(
        "--extraction",
        required=True,
        help="the extraction run: a folder with papers/ (one part of `extract run`)",
    )
    extract_verify.add_argument("--out", required=True, help="a new or empty folder")
    extract_verify.add_argument(
        "--model", default=verify_step.MODEL, help=f"default: {verify_step.MODEL}"
    )
    extract_verify.add_argument(
        "--revision",
        default=verify_step.REVISION,
        help="the model revision: recorded in the receipt, and the tokenizer's revision "
        f"(default: {verify_step.REVISION})",
    )
    extract_verify.add_argument(
        "--tokenizer",
        help="a folder with the model's tokenizer, which counts each batch's prompt "
        "(default: the model at --revision in the Hugging Face cache)",
    )
    extract_verify.add_argument(
        "--max-tokens",
        type=int,
        default=verify_step.MAX_TOKENS,
        help=f"output tokens per call (default: {verify_step.MAX_TOKENS})",
    )
    extract_verify.add_argument(
        "--seed",
        type=int,
        default=verify_step.SEED,
        help=f"default: {verify_step.SEED}",
    )
    extract_verify.add_argument(
        "--replay",
        action="append",
        help="a folder of recorded requests to answer every call from (repeatable)",
    )
    extract_verify.add_argument(
        "--workers",
        type=int,
        default=verify_step.WORKERS,
        help=f"papers verified at the same time (default: {verify_step.WORKERS})",
    )
    extract_verify.set_defaults(
        run=lambda args: verify_step.verify(
            Path(args.extraction),
            Path(args.out),
            args.model,
            args.revision,
            Path(args.tokenizer) if args.tokenizer else None,
            args.max_tokens,
            args.seed,
            [Path(folder) for folder in args.replay or []],
            args.workers,
        )
    )
    extract_filter = extract_actions.add_parser(
        "filter",
        help="keep the records the frozen relevance classifier calls relevant to media bias",
        description=(
            "Embed every record of every paper in RUN with BGE, give it the frozen classifier's "
            "probability of being relevant, and keep the records at or above the cutoff. RUN is "
            "a folder with papers/ (a verifier run, or an extraction run with --records union). "
            "Writes everything into OUT, with a receipt and a recording of the encoder's vectors. "
            "BGE runs on SCIDEF_NLI_DEVICE (.env). With --replay, the vectors come from a "
            "recording, and a text that was not recorded stops the run."
        ),
    )
    extract_filter.add_argument(
        "--run",
        dest="records_from",
        required=True,
        help="a folder with papers/ (from extract verify or extract run)",
    )
    extract_filter.add_argument("--out", required=True, help="a new or empty folder")
    extract_filter.add_argument(
        "--filter",
        default=filter_step.FILTER,
        choices=filter_step.NAMES,
        help=f"which frozen filter (default: {filter_step.FILTER}, the released one)",
    )
    extract_filter.add_argument(
        "--records",
        default=filter_step.RECORDS[0],
        choices=filter_step.RECORDS,
        help="filter each paper's verified.json or its union's records "
        f"(default: {filter_step.RECORDS[0]})",
    )
    extract_filter.add_argument(
        "--cutoff",
        type=float,
        help="keep records with at least this probability (default: the filter's frozen cutoff)",
    )
    encoder_flags(extract_filter)
    extract_filter.set_defaults(
        run=lambda args: filter_step.filter_papers(
            Path(args.records_from),
            Path(args.out),
            args.filter,
            args.records,
            args.cutoff,
            Path(args.bge) if args.bge else None,
            [Path(folder) for folder in args.replay or []],
            Path(args.filters),
        )
    )
    extract_derive = extract_actions.add_parser(
        "derive",
        help="pool a run's saved parts into its variants (P>=20 ... P>=28) and compare them",
        description=(
            "Read the saved stage records of every part a variant uses (no model call): from the "
            "per-paper folders that the run's plan names (a run planned by `extract plan`), or "
            "from the run's part tarballs (the recorded runs); pool each variant's parts per "
            "paper with the merge rule, "
            "and write per variant one file per paper and predictions.json, plus the union of "
            "all variants per paper (derived/); then compare the variants without gold "
            "(compare/). Writes everything into OUT, with a receipt. The default run is the "
            "released seed run under DATA_ROOT."
        ),
    )
    extract_derive.add_argument("--out", required=True, help="a new or empty folder")
    extract_derive.add_argument(
        "--set",
        dest="set_dir",
        help="the run's folder with picks.json, plans/grid.json, shards/, and runs/ (from "
        f"`extract plan`) or parts/ (default: DATA_ROOT/{derive_step.SEEDS})",
    )
    extract_derive.add_argument(
        "--picks",
        help="the variants to derive (default: picks.json in the run's folder)",
    )
    extract_derive.set_defaults(
        run=lambda args: derive_step.run(
            Path(args.out),
            Path(args.set_dir) if args.set_dir else None,
            Path(args.picks) if args.picks else None,
        )
    )
    # The corpus run's plan: shards and the jobs of each stage (docs/step-3-extraction/extract.md).
    extract_plan = extract_actions.add_parser(
        "plan",
        help="cut a corpus into shards and list the jobs that make its variants",
        description=(
            "Count each paper's windows, cut the papers into shards of about the same work, "
            "and list, for every shard, the extract run, verify and filter commands that make "
            "each stage the variants (--picks) need. Writes the run's folder (OUT): "
            "shards/, plans/grid.json, plans/jobs.tsv, picks.json, paper-metadata.json and a "
            "receipt. Run the jobs, then `extract derive --set OUT`. No model is called."
        ),
    )
    extract_plan.add_argument("--out", required=True, help="a new or empty folder")
    extract_plan.add_argument(
        "--teis",
        required=True,
        help="a folder or zip of <paper id>.tei.xml files",
    )
    extract_plan.add_argument(
        "--rows",
        help="a JSON list of rows with paper_id and frozen_title (default: one row per TEI, "
        "with its own title)",
    )
    extract_plan.add_argument(
        "--paper",
        action="append",
        help="plan only this paper id (repeatable; default: every row or TEI)",
    )
    extract_plan.add_argument(
        "--picks",
        default=str(plan_step.PICKS),
        help=f"the variants to make (default: {plan_step.PICKS})",
    )
    extract_plan.add_argument(
        "--shard-papers",
        type=int,
        default=plan_step.SHARD_PAPERS,
        help=f"papers per shard, on average (default: {plan_step.SHARD_PAPERS})",
    )
    extract_plan.add_argument(
        "--prefix",
        default=plan_step.PREFIX,
        help=f"the shards' names: PREFIX-000, ... (default: {plan_step.PREFIX})",
    )
    extract_plan.add_argument(
        "--metadata",
        help="a JSON file {paper id: {doi or tei_header_doi}} for the variants' DOIs "
        "(default: the first DOI each TEI names)",
    )
    extract_plan.set_defaults(
        run=lambda args: plan_step.plan(
            Path(args.out),
            Path(args.teis),
            Path(args.rows) if args.rows else None,
            args.paper,
            Path(args.picks),
            args.shard_papers,
            args.prefix,
            Path(args.metadata) if args.metadata else None,
        )
    )

    evaluate_parser = groups.add_parser(
        "evaluate",
        help="the evaluate step: the agreement numbers (Table 3) and the scoring of extractions "
        "(docs/step-3-extraction/evaluate.md)",
    )
    evaluate_actions = evaluate_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    ev_agreement = evaluate_actions.add_parser(
        "agreement",
        help="the annotators' agreement and the curation numbers of the ground truth (Table 3)",
        description=(
            "Count how the first two annotators of each record answered Q1 (is it a definition?) "
            "and Q2 (is the concept relevant?), with Krippendorff's alpha and the numbers around "
            "it, and what the curator did. Writes agreement.json, curation.json, two figures "
            "and a receipt into OUT. Nothing is written elsewhere."
        ),
    )
    ev_agreement.add_argument("--out", required=True, help="a new or empty folder")
    ev_agreement.add_argument(
        "--annotations",
        help=f"the raw Label Studio exports (default: $DATA_ROOT/{agreement.ANNOTATIONS})",
    )
    ev_agreement.add_argument(
        "--export",
        help=f"the curator's export zip (default: $DATA_ROOT/{agreement.EXPORT})",
    )
    ev_agreement.add_argument(
        "--data",
        default="data",
        help="the repository's data folder with V3 and the labelling form (default: data)",
    )
    ev_agreement.set_defaults(
        run=lambda args: agreement.agreement_command(
            args.out, args.annotations, args.export, args.data
        )
    )

    def pair_score_flags(action) -> None:
        """The flags of every scoring action: where the pair scores come from."""
        action.add_argument(
            "--replay-scores",
            help="answer every BGE similarity and entailment from this file, which "
            "--record-scores wrote; a pair that is not in it stops the scoring (no models)",
        )
        action.add_argument(
            "--record-scores",
            help="also write every similarity and entailment the models computed to this new file",
        )
        action.add_argument(
            "--nli-batch-size",
            type=int,
            default=metric.NLI_BATCH_SIZE,
            help=f"pairs per entailment batch (default: {metric.NLI_BATCH_SIZE}, as the released "
            "scoring runs; another size moves the probabilities by about 1e-6)",
        )

    ev_score = evaluate_actions.add_parser(
        "score",
        help="score predicted definition records against reference definitions (DefScore: the "
        "paper's pair score of word overlap, embedding similarity and entailment)",
        description=(
            "Score each --predictions file against one split of MBDef-Gold (or each scoring of "
            "the --scorings file against its reference), paper by paper: each pair of "
            "definitions gets its DefScore (the mean "
            "of word overlap, BGE embedding similarity and entailment both ways, calibrated to "
            "0-1), pairs at or above the cutoff are matched one to one, and the full-record "
            "score compares the matched records field by field. Writes scores.json and a receipt into OUT. Fails, and writes "
            "nothing, if any scoring fails. The models run on SCIDEF_NLI_DEVICE (.env)."
        ),
    )
    ev_score.add_argument(
        "--predictions",
        action="append",
        help='a JSONL file, one record per line: {"paper_id", "term", "definitions": [...]} '
        "and optionally the other lists, as extract run, verify and filter write it "
        "(repeatable: each file is one scoring)",
    )
    ev_score.add_argument(
        "--split",
        default=scoring.GOLD_SPLIT,
        choices=tuple(scoring.GOLD_SPLITS),
        help=f"the split of MBDef-Gold that --predictions is scored on (default: {scoring.GOLD_SPLIT})",
    )
    ev_score.add_argument(
        "--paper",
        action="append",
        help="score on this paper of the split only (repeatable; default: every paper of the "
        "split, so a paper without predictions counts as found nothing)",
    )
    ev_score.add_argument(
        "--gold",
        help="MBDef-Gold's record file (default: $DATA_ROOT/MBDef-Gold/validated_gt.json, "
        "from `data fetch MBDef-Gold/`)",
    )
    ev_score.add_argument(
        "--scorings",
        help='instead of --predictions, a JSON list of {"id", "reference", "papers"}: a '
        "reference file and a folder of predicted paper files, relative to the list's folder",
    )
    ev_score.add_argument("--out", required=True, help="a new or empty folder")
    ev_score.add_argument(
        "--threshold",
        type=float,
        default=metric.MATCH_THRESHOLD,
        help=f"the DefScore cutoff of a match (default: {metric.MATCH_THRESHOLD})",
    )
    pair_score_flags(ev_score)
    ev_score.set_defaults(
        run=lambda args: scoring.score_command(
            args.scorings,
            args.out,
            args.replay_scores,
            args.record_scores,
            args.threshold,
            args.nli_batch_size,
            args.predictions,
            args.gold,
            args.split,
            args.paper,
        )
    )

    ev_heldout = evaluate_actions.add_parser(
        "heldout",
        help="rescore the rows of the held-out TEST evaluation and apply the frozen pick rule",
        description=(
            "Rescore every saved row of the held-out TEST evaluation (12 rows, 29 papers, 365 "
            "reference definitions) with the general scorer, then apply the frozen plan's "
            "eligibility and pick rules. Writes results.json, results.md (the released "
            "format) and a receipt into OUT. Fails, and writes nothing, if a row cannot be "
            "scored. The models run on SCIDEF_NLI_DEVICE (.env)."
        ),
    )
    ev_heldout.add_argument("--out", required=True, help="a new or empty folder")
    ev_heldout.add_argument(
        "--data",
        default="data",
        help="the repository's data folder with the frozen plan and reference (default: data)",
    )
    ev_heldout.add_argument(
        "--plan",
        help=f"the frozen plan (default: <data>/{heldout.PLAN}; its sha256 is checked)",
    )
    ev_heldout.add_argument(
        "--reference", help=f"the TEST reference (default: <data>/{heldout.REFERENCE})"
    )
    ev_heldout.add_argument(
        "--ready", help=f"its readiness record (default: <data>/{heldout.READY})"
    )
    ev_heldout.add_argument(
        "--sources",
        help=f"the cohort's sources (default: $DATA_ROOT/{heldout.SOURCES})",
    )
    ev_heldout.add_argument(
        "--rows",
        action="append",
        help=f"a folder of row folders (repeatable; default: $DATA_ROOT/{heldout.ROWS})",
    )
    pair_score_flags(ev_heldout)
    ev_heldout.set_defaults(
        run=lambda args: heldout.heldout_command(
            args.out,
            args.data,
            args.plan,
            args.reference,
            args.ready,
            args.sources,
            args.rows,
            args.replay_scores,
            args.record_scores,
            args.nli_batch_size,
        )
    )

    ev_paired = evaluate_actions.add_parser(
        "paired",
        help="the paired comparison: our P≥28 extractor against Opus, paper by paper, on DEV and TEST",
        description=(
            "Compare the definitions our P≥28 extractor and Opus find, paper by paper, from "
            "saved per-paper counts: pooled differences with paired-bootstrap intervals, a "
            "paired permutation test and a sign test (the paper's comparison, its seeds "
            "pinned). Writes output.txt, paired.json and a receipt into OUT."
        ),
    )
    ev_paired.add_argument("--out", required=True, help="a new or empty folder")
    ev_paired.add_argument(
        "--selection",
        help=f"the selection's results.json (default: $DATA_ROOT/{paired.SELECTION})",
    )
    ev_paired.add_argument(
        "--test-results",
        help="the TEST rescoring's results.json, with the Opus row "
        f"(default: $DATA_ROOT/{paired.TEST_RESULTS})",
    )
    ev_paired.add_argument(
        "--opus-dev",
        help=f"Opus's DEV evaluation.json (default: $DATA_ROOT/{paired.OPUS_DEV})",
    )
    ev_paired.set_defaults(
        run=lambda args: paired.paired_command(
            args.out, args.selection, args.test_results, args.opus_dev
        )
    )

    ev_grid = evaluate_actions.add_parser(
        "grid",
        help="score every setup of the extractor grid (2,268 scorings)",
        description=(
            "Score every setup of the extractor grid on DEV and TEST with the general scorer: "
            "each pooled cohort's records against the reference, plus each setup's "
            "eligibility and LLM calls. All setups of one model on one split run in one "
            "process (a definition pair they share is scored once). Writes scores.json and a "
            "receipt into OUT; `evaluate select` picks from it. Fails, and writes nothing, if "
            "a setup cannot be scored. The models run on SCIDEF_NLI_DEVICE (.env)."
        ),
    )
    ev_grid.add_argument("--out", required=True, help="a new or empty folder")
    ev_grid.add_argument(
        "--grid",
        help=f"the grid's cohorts (default: $DATA_ROOT/{grid.GRID})",
    )
    ev_grid.add_argument(
        "--trained",
        help=f"the trained-prompt cohorts (default: $DATA_ROOT/{grid.TRAINED})",
    )
    ev_grid.add_argument(
        "--data",
        default="data",
        help="the repository's data folder with the DEV and TEST references (default: data)",
    )
    ev_grid.add_argument(
        "--test-results",
        help="the TEST evaluation's results.json, for Opus's TEST calls and the replicate "
        f"note (default: $DATA_ROOT/{grid.TEST_RESULTS})",
    )
    ev_grid.add_argument(
        "--family",
        action="append",
        choices=[f"{s}/{m}" for s in grid.SPLITS for m in grid.MODELS],
        help="score only this split and model (repeatable; default: all four)",
    )
    ev_grid.add_argument(
        "--workers",
        type=int,
        default=4,
        help="processes, one per split and model at a time (default: 4)",
    )
    ev_grid.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPUs to spread the processes over (default: 1)",
    )
    ev_grid.add_argument(
        "--replay-scores",
        help="answer every BGE similarity and entailment from this folder, which "
        "--record-scores wrote (one file per split and model); no models",
    )
    ev_grid.add_argument(
        "--record-scores",
        help="also write every similarity and entailment the models computed into this new "
        "folder, one file per split and model",
    )
    ev_grid.add_argument(
        "--nli-batch-size",
        type=int,
        default=metric.NLI_BATCH_SIZE,
        help=f"pairs per entailment batch (default: {metric.NLI_BATCH_SIZE}, as the released "
        "scoring runs)",
    )
    ev_grid.add_argument(
        "--write-scorings",
        action="store_true",
        help="score nothing: write every distinct record set as a folder of paper files, and "
        "one scorings list per split and model, into OUT, for `evaluate score` (for example "
        "on a cluster, whose recorded pair scores --replay-scores then reads)",
    )
    ev_grid.set_defaults(
        run=lambda args: grid.grid_command(
            args.out,
            args.grid,
            args.trained,
            args.data,
            args.test_results,
            args.family,
            args.workers,
            args.gpus,
            args.replay_scores,
            args.record_scores,
            args.nli_batch_size,
            args.write_scorings,
        )
    )

    ev_select = evaluate_actions.add_parser(
        "select",
        help="pick the extractors from the scored grid by the frozen rule (plan amendment 8)",
        description=(
            "Apply the audit and the frozen pick rules to the scores `evaluate grid` wrote: "
            "for each precision floor (20, 24, 28%), the TEST setup with the most of the 365 "
            "definitions found among those with at least 292 and that precision; within 2 "
            "matches the higher precision, then fewer LLM calls, wins. Writes results.json, "
            "results.md and a receipt into OUT."
        ),
    )
    ev_select.add_argument(
        "--scores", required=True, help="scores.json of `evaluate grid`"
    )
    ev_select.add_argument(
        "--out", required=True, help="a folder without results in it"
    )
    ev_select.add_argument(
        "--data",
        default="data",
        help="the repository's data folder with the frozen plan (default: data)",
    )
    ev_select.add_argument(
        "--audit",
        help=f"the cohorts' audit (default: <data>/{selection.AUDIT})",
    )
    ev_select.add_argument(
        "--extra-audit",
        action="append",
        help="a further audit file whose verdicts fill in cohorts the audit lacks (repeatable)",
    )
    ev_select.add_argument(
        "--replaces",
        help="results this selection corrects: first prove that their own audit gives their "
        "picks on these scores, then record the correction (needs --reason)",
    )
    ev_select.add_argument("--reason", help="why the results are replaced")
    ev_select.add_argument(
        "--match-floor",
        type=int,
        default=selection.FINAL_MATCH_FLOOR,
        help="the fewest TEST matches (of 365) a pick needs (default: %(default)s, plan "
        "amendment 8)",
    )
    ev_select.add_argument(
        "--precision-floors",
        type=int,
        nargs="+",
        default=list(selection.FINAL_PRECISION_FLOORS.values()),
        metavar="PERCENT",
        help="one pick per precision floor, in percent (default: %(default)s)",
    )
    ev_select.add_argument(
        "--tie-matches",
        type=int,
        default=selection.FINAL_TIE_MATCHES,
        help="setups within this many matches of the most are tied (default: %(default)s)",
    )
    ev_select.set_defaults(
        run=lambda args: selection.select_command(
            args.scores,
            args.out,
            args.data,
            args.audit,
            args.extra_audit,
            args.replaces,
            args.reason,
            args.match_floor,
            args.precision_floors,
            args.tie_matches,
        )
    )

    cluster_parser = groups.add_parser(
        "cluster",
        help="run a step on a Slurm GPU cluster: render the job, models, checks (docs/setup/cluster.md)",
    )
    cluster_actions = cluster_parser.add_subparsers(
        dest="action", required=True, metavar="<action>"
    )
    render = cluster_actions.add_parser(
        "render",
        help="fill the launcher template in for one site and one step; submits nothing",
        description=(
            f"Fill {cluster.TEMPLATE} in with cluster/SITE.toml and the flags below, and print "
            "the job script (or write it to --out). It never submits anything: submit the file "
            "on the login node with sbatch. The QOS is always the normal one."
        ),
    )
    render.add_argument(
        "--site",
        required=True,
        help="a site file in cluster/, e.g. example (cluster/example.toml)",
    )
    render.add_argument(
        "--step",
        required=True,
        help="the command line the job runs in the checkout, e.g. 'mbdef evaluate score ...'",
    )
    render.add_argument(
        "--time",
        required=True,
        help="the walltime, HH:MM:SS; ask for what the step needs",
    )
    render.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPUs on one node (default: 1); 0 asks for the site's CPU partition",
    )
    render.add_argument(
        "--model",
        action="append",
        default=[],
        help=f"a model the step reads, checked (and on a site that stages models copied to node-local disk) before "
        f"it runs: {', '.join(models.PINNED)} or REPO@REVISION (repeatable)",
    )
    render.add_argument(
        "--serve",
        choices=sorted(cluster.SERVE),
        help="serve this pinned model with vLLM on every GPU of the job, check the server, "
        "and point the step at it (DSPY_API_BASE)",
    )
    render.add_argument(
        "--think",
        action="store_true",
        help="serve it with its reasoning parser, as the thinking runs did",
    )
    render.add_argument(
        "--job-name", default="taxonomy", help="the Slurm job name (default: taxonomy)"
    )
    render.add_argument(
        "--partition", help="another partition of the site, e.g. a test partition"
    )
    render.add_argument(
        "--qos",
        help=f"the QOS: only {cluster.NORMAL_QOS} is accepted (the default); render never "
        "writes a QOS that pre-empts other users' jobs or jumps ahead of them",
    )
    render.add_argument(
        "--out", help="write the script here (must not exist) instead of printing it"
    )
    render.set_defaults(
        run=lambda args: cluster.render(
            args.site,
            args.step,
            args.gpus,
            args.time,
            args.job_name,
            args.partition,
            args.qos,
            args.model,
            args.serve,
            args.think,
            args.out,
        )
    )
    download = cluster_actions.add_parser(
        "download",
        help="download the pinned models into the Hugging Face cache and check every file",
        description=(
            "Download each model at its pinned revision (on a login node: compute nodes have "
            "no internet), or with --offline only find it in the cache. Then check every file "
            "against the Hub object id that names it, and write a receipt with each file's "
            "size and sha256. Exits 1 if a model is missing or differs."
        ),
    )
    download.add_argument(
        "--model",
        action="append",
        default=[],
        help=f"{', '.join(models.PINNED)} or REPO@REVISION (repeatable; default: all four)",
    )
    download.add_argument(
        "--cache",
        help="the hub cache folder (default: HF_HUB_CACHE or the Hub's default)",
    )
    download.add_argument(
        "--offline",
        action="store_true",
        help="do not download; only find and check the models in the cache",
    )
    download.add_argument("--out", required=True, help="where to write the receipt")
    download.set_defaults(
        run=lambda args: cluster.download(
            args.model, args.cache, args.offline, args.out
        )
    )
    vllm = cluster_actions.add_parser(
        "vllm-check",
        help="check that a running vLLM server serves the expected model and answers",
        description=(
            "Check a running vLLM server: /v1/models lists MODEL alone with the expected "
            "max_model_len, /tokenize counts the tokens of a short chat, and /metrics lists its "
            "engines. Writes what it found (with the KV-cache lines of --log) to --out and "
            "exits 1 if a check fails."
        ),
    )
    vllm.add_argument(
        "--base", required=True, help="the server, e.g. http://127.0.0.1:8000"
    )
    vllm.add_argument(
        "--model", required=True, help="the model id the server must serve, alone"
    )
    vllm.add_argument(
        "--max-model-len",
        type=int,
        default=cluster.MAX_MODEL_LEN,
        help=f"the context the server must have (default: {cluster.MAX_MODEL_LEN}, as released)",
    )
    vllm.add_argument(
        "--think", action="store_true", help="tokenize with thinking on, as served"
    )
    vllm.add_argument("--log", help="the server's log, for its KV-cache lines")
    vllm.add_argument("--out", required=True, help="where to write the health JSON")
    vllm.set_defaults(
        run=lambda args: cluster.vllm_check(
            args.base, args.model, args.max_model_len, args.think, args.log, args.out
        )
    )
    runtime = cluster_actions.add_parser(
        "check-runtime",
        help="check that Python and the pinned packages are the versions in uv.lock",
        description=(
            "Compare the running Python and the installed versions of the released runs' "
            f"packages ({', '.join(cluster.RUNTIME_PACKAGES)}) with uv.lock. Writes what it "
            "found to --out and exits 1 on any difference."
        ),
    )
    runtime.add_argument(
        "--lock", help=f"the lock file (default: {cluster.LOCK} in this folder)"
    )
    runtime.add_argument(
        "--package",
        action="append",
        default=[],
        help="check this package instead (repeatable)",
    )
    runtime.add_argument("--out", required=True, help="where to write what was found")
    runtime.set_defaults(
        run=lambda args: cluster.check_runtime(args.lock, args.package, args.out)
    )

    explorer_parser(groups)
    embed_parser(groups)
    postprocess_parser(groups)
    dataset_parser(groups)

    files.load_env()
    args = parser.parse_args(argv)
    return args.run(args)


def thinking_flag(action) -> None:
    """`--thinking` of `extract run` and `extract train`: the thinking switch each request sends."""
    action.add_argument(
        "--thinking",
        default="on",
        choices=tuple(extract.THINKING_SWITCH),
        help="on (default, the released runs) or off sends vLLM's chat_template_kwargs "
        "enable_thinking; unset sends no switch, for an API that refuses it",
    )


def explorer_parser(groups) -> None:
    """The `explorer` group (docs/dataset/explorer.md): the MBDef Explorer, a static web page."""
    from mbdef import explorer

    parser = groups.add_parser(
        "explorer",
        help="the MBDef Explorer, a static web page in explorer/ (docs/dataset/explorer.md)",
    )
    actions = parser.add_subparsers(dest="action", required=True, metavar="<action>")
    build = actions.add_parser(
        "build",
        help=f"write {explorer.PAGE} and {explorer.DATA}/ from {explorer.SRC}/ and the MBDef dataset",
        description=(
            f"Run node --check on every script in {explorer.SRC}/ and inline them, with the CSS and the tabs' "
            f"sections, into {explorer.PAGE}. Check the page data that comes with the MBDef dataset (its folder "
            f"{explorer.BUNDLE}/) against its {explorer.SUMS} and against the dataset's tables, and write it to "
            f"{explorer.DATA}/ with a new {explorer.SUMS}. Run it from the repository root; it needs node. "
            "Nothing is written when a check fails."
        ),
    )
    build.add_argument(
        "--dataset",
        type=Path,
        help=f"the MBDef dataset's folder (default: $DATA_ROOT/{explorer.MBDEF}, where `data fetch` puts it)",
    )
    build.set_defaults(run=lambda args: explorer.build(args.dataset))
    data = actions.add_parser(
        "data",
        help="write the page data for MBDef's released mentions only, with the map and topics made on them",
        description=(
            f"Read the MBDef dataset in --dataset (its tables and its page data in {explorer.BUNDLE}/) and write page "
            "data that holds only the term mentions of its instances table. The map (PCA, then UMAP), the 36 topics "
            "(k-means), the nearest mentions and each concept's spread are computed again on those mentions, from the "
            "embedding vectors of their definitions in --vectors. The seeds are fixed: a second run writes the same "
            "files. Nothing is written when a check fails."
        ),
    )
    data.add_argument(
        "--dataset",
        type=Path,
        help=f"the MBDef dataset's folder (default: $DATA_ROOT/{explorer.MBDEF}, where `data fetch` puts it)",
    )
    data.add_argument(
        "--vectors",
        type=Path,
        default=Path("embeddings/qwen3-emb-8b"),
        help="the Qwen3-Embedding-8B cache of the definitions, below DATA_ROOT unless absolute (default: embeddings/qwen3-emb-8b)",
    )
    data.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the folder to write the page data to (replaced); give it to `dataset build --explorer-data`",
    )
    data.set_defaults(run=run_explorer_data)


def run_explorer_data(args) -> int:
    from mbdef import explorer, explorer_data
    from mbdef.postprocess.embeddings import in_data_root

    dataset = explorer.dataset_folder() if args.dataset is None else args.dataset
    return explorer_data.released(dataset, in_data_root(args.vectors), args.out)


def embed_parser(groups) -> None:
    """The `embed` group (docs/step-5-post-processing/embed.md): the embedding cache that Step 5 reads."""
    from mbdef import embed
    from mbdef.postprocess import corpus
    from mbdef.postprocess import groundtruth as gold

    parser = groups.add_parser(
        "embed",
        help="the embedding cache that Step 5 reads (docs/step-5-post-processing/embed.md)",
    )
    actions = parser.add_subparsers(dest="action", required=True, metavar="<action>")
    build = actions.add_parser(
        "build",
        help="embed every text the cache lacks: corpus definitions and terms, ground truth",
        description=(
            "Fill the cache OUT/<model>/ in two passes, the seed papers first, then the seed "
            "and the closure papers. Each pass takes '<term>: <definition>' of every "
            "definition in the corpus store (in the store's row order), of every ground-truth "
            "definition, then every term (sorted) and the comparison texts, and embeds the "
            "texts the cache does not have yet, on a GPU. A cache already in OUT keeps its "
            "rows and must have the same settings (manifest.json); a complete cache is left "
            "unchanged. With --replay, every vector comes from a recorded cache instead (a "
            "check on a CPU)."
        ),
    )
    build.add_argument(
        "--out", required=True, help="the folder of caches; the cache is OUT/<model>/"
    )
    build.add_argument(
        "--store",
        help=f"the corpus store (default: $DATA_ROOT/{corpus.STORE})",
    )
    build.add_argument(
        "--ground-truth",
        action="append",
        help="a ground-truth file, in order (repeatable; default: "
        + " and ".join(f"{gold.FOLDER}/{name}" for name, _ in gold.FILES)
        + ", MBDef-Gold)",
    )
    build.add_argument(
        "--comparison",
        help=f"the comparison texts, a JSON list (default: $DATA_ROOT/{embed.COMPARISON})",
    )
    build.add_argument(
        "--model",
        default=embed.MODEL,
        choices=sorted(embed.MODELS),
        help="default: %(default)s",
    )
    build.add_argument(
        "--replay",
        help="a recorded cache folder (keys.txt, vectors.npy) to take every vector from; "
        "a text it lacks stops the build",
    )
    build.add_argument(
        "--sample",
        type=int,
        help=f"embed only this many texts, drawn with seed {embed.SAMPLE_SEED} (a quick check)",
    )
    build.add_argument(
        "--gpu-share",
        type=float,
        default=embed.GPU_SHARE,
        help="the share of GPU memory vLLM may take (default: %(default)s)",
    )
    build.set_defaults(
        run=lambda args: embed.build(
            Path(args.out),
            files.located(args.store, corpus.STORE)[0],
            files.located(args.comparison, embed.COMPARISON)[0],
            args.ground_truth,
            args.model,
            args.replay,
            args.sample,
            args.gpu_share,
        )
    )


def postprocess_parser(groups) -> None:
    """The `postprocess` group (Step 5, docs/step-5-post-processing/postprocess.md): one action per command, every setting a flag."""
    from mbdef import postprocess
    from mbdef.postprocess import (
        compare,
        decide,
        ensemble,
        extractor_gate,
        frozen,
        label_groups,
        label_match,
        term_merging,
        testrun,
    )
    from mbdef.postprocess.paths.clustering import PINNED

    frozen_15 = postprocess.REPO / postprocess.FROZEN / "20261015"
    post = groups.add_parser(
        "postprocess",
        help="Step 5: which corpus Extracted Concepts are about media bias (docs/step-5-post-processing/postprocess.md)",
        description=(
            "Step 5's commands. Each reads Step 5's frozen files from "
            f"{postprocess.FROZEN}/<date>/ (those over 10 MB from the same path under "
            "$DATA_ROOT/frozen/) and writes only to the folder --out names, never into a frozen "
            "folder."
        ),
    )
    actions = post.add_subparsers(dest="action", required=True, metavar="<action>")

    def out_flag(p, what):
        p.add_argument(
            "--out",
            type=Path,
            required=True,
            help=f"the folder to write {what} to (not inside a frozen folder)",
        )

    def rule_flag(p, what):
        p.add_argument(
            "--rule",
            choices=label_match.RULE_VERSIONS,
            default=label_match.RULE_VERSION,
            help=f"the version of the Label Match rule {what} (default: %(default)s; C-20261015 is the "
            "earlier version, whose Label Groups the frozen paths were evaluated on)",
        )

    def cluster_flags(p):
        p.add_argument(
            "--clusters",
            type=Path,
            default=PINNED,
            help="path 1's pinned cluster map (default: the saved map in the frozen folder 20261015/)",
        )
        p.add_argument(
            "--refit-clusters",
            action="store_true",
            help="fit a new k-means instead of reading the map (not reproducible with 4 or more "
            "CPU threads; docs/step-5-post-processing/postprocess.md)",
        )

    def pinned(args):
        return None if args.refit_clusters else args.clusters

    def inputs_flag(p, what):
        p.add_argument(
            "--inputs",
            type=Path,
            default=frozen_15,
            help=f"the frozen folder of {what} (default: the folder 20261015/)",
        )

    from mbdef.postprocess import corpus, store

    bs = actions.add_parser(
        "build-store",
        help="the corpus store that Step 5 reads, from the extractions of the corpus release v1.1 (Step 4's release folder)",
        description=(
            "Read each corpus's variants (<release>/{seeds,closure}/derived/: DONE.json and "
            "<variant>/predictions.json), check them against DONE.json, and write the tables "
            "papers, mentions, occurrences and occurrence_sources to --out/library.sqlite3, with "
            "receipt.json. No paper text: the paragraph contexts are not built."
        ),
    )
    bs.add_argument(
        "--release",
        type=Path,
        help=f"the release folder (default: $DATA_ROOT/{store.RELEASE})",
    )
    out_flag(bs, "library.sqlite3 and receipt.json")
    bs.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a library.sqlite3 already in --out",
    )
    bs.add_argument(
        "--compare",
        type=Path,
        metavar="STORE",
        help="then compare the four tables with this store's, row for row (exit 1 if one differs); "
        f"the stored one is $DATA_ROOT/{corpus.STORE}",
    )

    def run_build_store(args):
        return store.build_command(
            args.release,
            postprocess.refuse_frozen(args.out),
            args.overwrite,
            args.compare,
        )

    bs.set_defaults(run=run_build_store)

    lg = actions.add_parser(
        "label-groups",
        help="Step 5.1: the Label Groups of the corpus under the Label Match rule",
        description=(
            "Group the corpus's Extracted Concepts whose labels match under the Label Match rule and write "
            "terms.jsonl, groups.jsonl, labels.jsonl (not under the earlier rule version), summary.json and run.json."
        ),
    )
    lg.add_argument(
        "--store", type=Path, help="the corpus store (default: the one in DATA_ROOT)"
    )
    lg.add_argument(
        "--links",
        type=Path,
        help="the definition links of the corpus release v1.1 (Step 4's release folder; default: the ones in DATA_ROOT)",
    )
    out_flag(lg, "the Label Groups")
    rule_flag(lg, "to group by")
    lg.add_argument(
        "--overwrite",
        action="store_true",
        help="replace Label Group files already in --out",
    )

    def run_label_groups(args):
        out = postprocess.refuse_frozen(args.out)
        recorded = ["--out", str(args.out), "--rule", args.rule]
        recorded += ["--store", str(args.store)] if args.store else []
        recorded += ["--links", str(args.links)] if args.links else []
        return label_groups.label_groups_command(
            args.store, args.links, out, args.rule, args.overwrite, recorded
        )

    lg.set_defaults(run=run_label_groups)

    ev = actions.add_parser(
        "evaluate",
        help="Step 5.3: cross-validate relevance paths on train + dev",
        description=(
            "Score each path out of fold on the 1,700 train + dev records (5 folds by paper) and "
            "write scores/<path>.csv, metrics.json and curves.json. Reads no TEST record."
        ),
    )
    ev.add_argument(
        "--paths",
        default="keyword",
        help="comma-separated path names (keyword, clustering, neighbours, classifier, "
        "representative, zero-shot; '@group' for the Label Group setting; default: %(default)s)",
    )
    ev.add_argument(
        "--extra",
        nargs="*",
        type=Path,
        default=[],
        help="score tables made elsewhere, compared beside the paths",
    )
    inputs_flag(ev, "the Representative Definitions and the extractor filter's scores")
    out_flag(ev, "the comparison")
    ev.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="bootstrap draws for the intervals (default: %(default)s)",
    )
    rule_flag(ev, "of the Label Group setting")
    cluster_flags(ev)

    def run_evaluate(args):
        compare.evaluate_paths(
            [p for p in args.paths.split(",") if p],
            inputs=args.inputs,
            out=postprocess.refuse_frozen(args.out),
            extra=args.extra,
            draws=args.draws,
            pinned=pinned(args),
            rule=args.rule,
        )
        return 0

    ev.set_defaults(run=run_evaluate)

    de = actions.add_parser(
        "decide",
        help="keep or drop every corpus Extracted Concept with one path at its 95%%-recall cut",
        description=(
            "Fit the path on all train + dev records, score every corpus Extracted Concept and keep "
            "those at or above the cut that keeps --recall of the relevant train + dev records "
            "(set on the path's saved out-of-fold scores in --inputs, made when missing). Writes "
            "<path>.csv and <path>.json."
        ),
    )
    de.add_argument("--path", required=True, help="the path, e.g. classifier")
    de.add_argument(
        "--recall",
        type=float,
        default=decide.RECALL,
        help="the share of relevant train + dev records the cut keeps (default: %(default)s)",
    )
    de.add_argument(
        "--fast",
        action="store_true",
        help="entailment paths: the fast scorer, with exact rescoring near the cut",
    )
    de.add_argument(
        "--band",
        type=float,
        default=decide.BAND,
        help="with --fast: rescore pairs this close to the cut (default: %(default)s)",
    )
    inputs_flag(de, "the saved out-of-fold scores and the Representative Definitions")
    out_flag(de, "the decisions")
    rule_flag(de, "of the Label Groups (the @group setting and the counts)")
    cluster_flags(de)

    def run_decide(args):
        return decide.decide_path(
            args.path,
            inputs=args.inputs,
            out=postprocess.refuse_frozen(args.out),
            recall=args.recall,
            fast=args.fast,
            band=args.band,
            pinned=pinned(args),
            rule=args.rule,
        )

    de.set_defaults(run=run_decide)

    en = actions.add_parser(
        "ensemble",
        help="Step 5.4: keep or drop every corpus Extracted Concept with the ensemble",
        description=(
            "Score every corpus Extracted Concept with the ensemble (classifier, basal, judge: "
            "each member's score ranked against its own train + dev scores, weighted with the "
            "frozen weights of pick.json) and keep those at or above the cut that keeps --recall of "
            "the relevant train + dev records. Writes ensemble.csv and ensemble.json. Each input "
            "defaults to the frozen file of the released run (docs/step-5-post-processing/postprocess.md, Inputs)."
        ),
    )

    def member_files(text):
        member, _, path = text.partition("=")
        if not path:
            raise argparse.ArgumentTypeError(f"{text!r} is not MEMBER=FILE")
        return member, Path(path)

    en.add_argument(
        "--pick", type=Path, help="the frozen members and weights (pick.json)"
    )
    for flag, what in (
        ("--weight-table", "a member's train + dev scores the weights were fitted on"),
        (
            "--reference",
            "a member's train + dev scores its corpus scores are ranked against",
        ),
        ("--corpus-scores", "a member's scores of the corpus"),
    ):
        en.add_argument(
            flag,
            type=member_files,
            action="append",
            default=[],
            metavar="MEMBER=FILE",
            help=f"{what} (repeatable; members not named keep the frozen file)",
        )
    en.add_argument(
        "--terms",
        type=Path,
        help="every corpus Extracted Concept with its Label Group (the rows and their order)",
    )
    en.add_argument(
        "--classifier-decisions",
        type=Path,
        help="the classifier's own corpus decisions (for the comparison counts)",
    )
    en.add_argument(
        "--recall",
        type=float,
        default=ensemble.RECALL,
        help="the share of relevant train + dev records the cut keeps (default: %(default)s)",
    )
    en.add_argument(
        "--sanity",
        type=float,
        nargs=2,
        default=list(ensemble.SANITY),
        metavar=("LOW", "HIGH"),
        help="write nothing unless the share of corpus terms kept lies in [LOW, HIGH] "
        "(default: %(default)s)",
    )
    out_flag(en, "the decisions")

    def run_ensemble(args):
        out = postprocess.refuse_frozen(args.out)
        frozen_file = (
            postprocess.frozen_file
        )  # each default is resolved only when its flag is not given
        inputs = {
            "pick": args.pick or frozen_file(ensemble.PICK),
            "terms": args.terms or frozen_file(ensemble.TERMS),
            "classifier_decisions": args.classifier_decisions
            or frozen_file(ensemble.CLASSIFIER_DECISIONS),
        }
        for key, given, defaults in (
            ("weight_tables", args.weight_table, ensemble.WEIGHT_TABLES),
            ("references", args.reference, ensemble.REFERENCES),
            ("corpus_scores", args.corpus_scores, ensemble.CORPUS_SCORES),
        ):
            named = dict(given)
            inputs[key] = {
                m: named[m] if m in named else frozen_file(defaults[m])
                for m in ensemble.MEMBERS
            }
        return ensemble.ensemble_command(
            out, **inputs, recall=args.recall, sanity=tuple(args.sanity)
        )

    en.set_defaults(run=run_ensemble)

    mt = actions.add_parser(
        "merge-terms",
        help="merge the kept Extracted Concepts by Label Group, setting aside the split labels",
        description=(
            "Merge the kept Extracted Concepts by Label Group and write concepts.jsonl, "
            "split-labels.csv, majority-disagreements.csv and summary.json."
        ),
    )
    mt.add_argument(
        "--decisions",
        type=Path,
        help="per-term decisions with mention_id and keep (default: the released ensemble's)",
    )
    mt.add_argument(
        "--label-groups",
        type=Path,
        help="a folder written by `postprocess label-groups` (default: the released Label "
        "Groups, made with the adopted Label Match rule)",
    )
    out_flag(mt, "the merged rows")

    def run_merge_terms(args):
        out = postprocess.refuse_frozen(args.out)
        return term_merging.merge_command(
            args.decisions or postprocess.frozen_file(term_merging.DECISIONS),
            args.label_groups
            or postprocess.REPO / postprocess.FROZEN / term_merging.LABEL_GROUPS,
            out,
        )

    mt.set_defaults(run=run_merge_terms)

    fr = actions.add_parser(
        "freeze",
        help="record frozen paths and their evidence before the TEST split is read",
        description=(
            "Write the paths' settings and the train + dev evidence for them into "
            "--out/frozen-paths.json; `postprocess test --read --out` the same folder reads only "
            "paths recorded there."
        ),
    )
    fr.add_argument(
        "--paths",
        default=None,
        help="comma-separated paths with their frozen settings (default: all of them)",
    )
    fr.add_argument(
        "--outside",
        metavar="NAME",
        help="record an outside method (for example another group's judge) instead of --paths",
    )
    fr.add_argument(
        "--settings",
        default="{}",
        help="with --outside: its fixed settings as JSON (e.g. arm and policy)",
    )
    fr.add_argument("--evidence", default="", help="with --outside: why these settings")
    out_flag(fr, "frozen-paths.json")

    def run_freeze(args):
        out = postprocess.refuse_frozen(args.out)
        if args.outside:
            return frozen.freeze_outside(
                out, args.outside, json.loads(args.settings), args.evidence
            )
        names = args.paths.split(",") if args.paths else list(frozen.FROZEN_SETTINGS)
        return frozen.freeze_paths(out, [n for n in names if n])

    fr.set_defaults(run=run_freeze)

    te = actions.add_parser(
        "test",
        help="compare test-eval.json by hash (default), or read the TEST split once more (asks first)",
        description=(
            "Without --read: check that the saved TEST result still has its sha256; "
            "nothing of the TEST split is read. With --read: the frozen paths on the TEST split "
            "once more, into --out; it warns, needs --reason, asks at the prompt and logs the read."
        ),
    )
    te.add_argument(
        "--read",
        action="store_true",
        help="read the TEST split again (warns, needs --reason, asks at the prompt, is logged)",
    )
    te.add_argument(
        "--paths",
        default=None,
        help="with --read: comma-separated (default: every frozen path)",
    )
    te.add_argument("--reason", help="with --read: why; the read log keeps it")
    te.add_argument(
        "--from-saved",
        action="store_true",
        help="with --read: summarise the saved TEST scores of the same frozen paths instead of "
        "refitting",
    )
    inputs_flag(te, "the frozen record, the saved scores and the read log")
    te.add_argument(
        "--out",
        type=Path,
        help="with --read: the new folder for test-eval.json, the scores and the read log",
    )

    def run_test(args):
        if not args.read:
            return testrun.compare_test()
        if args.out is None:
            raise SystemExit("A read of the TEST split needs --out, a new folder.")
        return testrun.read_test(
            postprocess.refuse_frozen(args.out),
            args.inputs,
            names=[p for p in args.paths.split(",") if p] if args.paths else None,
            reason=args.reason,
            from_saved=args.from_saved,
        )

    te.set_defaults(run=run_test)

    eg = actions.add_parser(
        "extractor-gate",
        help="the extractor's relevance filter replayed on the ground truth",
        description=(
            "Score the ground truth's train + dev records with the extraction run's own relevance "
            "filter (extract/relevance.py, `title`) and write gate-scores.csv and gate.json. "
            "--read-test scores the TEST records too, which is a read of the TEST split."
        ),
    )
    eg.add_argument(
        "--device",
        default="auto",
        help="where BGE runs: cpu, cuda or auto (default: %(default)s)",
    )
    eg.add_argument(
        "--read-test",
        action="store_true",
        help="also score the TEST records (warns, needs --reason, asks at the prompt, is logged)",
    )
    eg.add_argument("--reason", help="with --read-test: why; the read log keeps it")
    out_flag(eg, "the filter's scores")

    def run_extractor_gate(args):
        return extractor_gate.gate_command(
            postprocess.refuse_frozen(args.out),
            device=args.device,
            read_test=args.read_test,
            reason=args.reason,
        )

    eg.set_defaults(run=run_extractor_gate)

    basal_parser(actions)
    judge_parser(actions)


def basal_parser(actions) -> None:
    """`postprocess basal <step>`: the ensemble member basal's items, its client and its score tables (Step 5)."""
    from mbdef import postprocess
    from mbdef.postprocess import basal

    parser = actions.add_parser(
        "basal",
        help="the decision model basal: its items, the requests to its server, its score tables",
        description=(
            "Three steps, in order: `items` (the corpus's Extracted Concepts and the ground truth's "
            "train + dev records, as basal is shown them), `ask` (send each item to basal's server; "
            "needs the server, which needs a large GPU) and `table` (the score table from the answers)."
        ),
    )
    steps = parser.add_subparsers(dest="step", required=True, metavar="<step>")

    items = steps.add_parser(
        "items",
        help="the items basal is asked about: each term with its first definitions",
        description=(
            f"Write --out/{basal.ITEMS}: one line {{id, split, state}} per Extracted Concept of the "
            f"corpus view, and --out/{basal.GROUNDTRUTH_ITEMS}: one per train + dev record of the "
            "ground truth (the reference the corpus scores are ranked against; no TEST record). "
            "The state is the term and its first --max-definitions definitions, each with its "
            "whitespace collapsed, U+FFFD removed and cut to --max-chars minus 1 characters + '…' "
            "when longer."
        ),
    )
    items.add_argument(
        "--store", type=Path, help="the corpus store (default: the one in DATA_ROOT)"
    )
    items.add_argument(
        "--links",
        type=Path,
        help="the definition links of the corpus release v1.1 (Step 4's release folder; default: the ones in DATA_ROOT)",
    )
    items.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the folder to write the items to (not inside a frozen folder)",
    )
    items.add_argument(
        "--max-definitions",
        type=int,
        default=basal.MAX_DEFINITIONS,
        help="the definitions of a term that its state shows, the first ones (default: %(default)s)",
    )
    items.add_argument(
        "--max-chars",
        type=int,
        default=basal.MAX_CHARS,
        help="a longer definition is cut to one character less + '…' (default: %(default)s)",
    )
    items.set_defaults(
        run=lambda args: basal.items_command(
            args.store,
            args.links,
            postprocess.refuse_frozen(args.out),
            args.max_definitions,
            args.max_chars,
        )
    )

    ask = steps.add_parser(
        "ask",
        help="send each item to basal's server and record its answers",
        description=(
            f"POST each item of --items to the server's {basal.ENDPOINT} and append one line per "
            "item to --out as the answers come back: {id, p: {core, satellite, irrelevant}, "
            "choice, model, repeat}, or {id, error} for an item whose requests failed "
            f"{basal.ATTEMPTS} times or whose answer is malformed (its probabilities must be the "
            "three options, summing to 1). Items that already have an answer in --out are "
            "skipped, so running the command again asks only the rest; it exits 1 while an item "
            "has no answer. Needs basal's server (docs/step-5-post-processing/postprocess.md)."
        ),
    )
    ask.add_argument(
        "--items", type=Path, required=True, help="an items file of `items`"
    )
    ask.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the answers file (JSONL, appended to; not inside a frozen folder)",
    )
    ask.add_argument(
        "--server",
        default=basal.SERVER,
        help="the address of basal's server (default: %(default)s)",
    )
    ask.add_argument(
        "--concurrency",
        type=int,
        default=basal.CONCURRENCY,
        help="requests in flight (default: %(default)s)",
    )
    ask.add_argument(
        "--order-seed",
        type=int,
        default=basal.ORDER_SEED,
        help="the items are sent in the order random.Random(seed).shuffle gives (default: "
        "%(default)s, the recorded runs')",
    )
    ask.add_argument(
        "--repeat",
        type=int,
        default=basal.REPEAT,
        help="the run's number, written into every answer (default: %(default)s)",
    )
    ask.add_argument(
        "--limit",
        type=int,
        default=0,
        help="ask only the first N items of the file (a quick trial; default: all)",
    )
    ask.set_defaults(
        run=lambda args: basal.ask_command(
            args.items,
            postprocess.refuse_frozen(args.out),
            server=args.server,
            concurrency=args.concurrency,
            order_seed=args.order_seed,
            repeat=args.repeat,
            limit=args.limit,
        )
    )

    table = steps.add_parser(
        "table",
        help="basal's score table: P(core) + P(satellite) of each item",
        description=(
            "Write the score table (method, unit, item_id, split, score) of the items of --items, "
            "in their order, from the first valid answer of each item in --answers; error lines "
            "are skipped. Refuses while an item has no answer."
        ),
    )
    table.add_argument(
        "--items", type=Path, required=True, help="the items that were asked"
    )
    table.add_argument(
        "--answers", type=Path, required=True, help="the answers file of `ask`"
    )
    table.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the score table to write (CSV; not inside a frozen folder)",
    )
    label = table.add_mutually_exclusive_group()
    label.add_argument(
        "--method",
        default=basal.METHOD,
        help="the method column (default: %(default)s)",
    )
    label.add_argument(
        "--method-of",
        type=Path,
        metavar="TABLE",
        help="take the method column from this score table (to rebuild a recorded table)",
    )
    table.set_defaults(
        run=lambda args: basal.table_command(
            args.items,
            args.answers,
            postprocess.refuse_frozen(args.out),
            basal.method_of(args.method_of) if args.method_of else args.method,
        )
    )


def judge_parser(actions) -> None:
    """`postprocess judge <step>`: the packages, requests and score tables of the ensemble member GT-RAG judge (a judge shown retrieved ground-truth definitions; Step 5)."""
    from mbdef import postprocess
    from mbdef.postprocess import judge

    parser = actions.add_parser(
        "judge",
        help="the GT-RAG judge (shown the most similar labelled ground-truth definitions): its request packages, the requests to its server, its score tables",
        description=(
            f"Three steps, in order: `package` (each statement with its {judge.K} nearest train records, for "
            "the corpus and for the ground truth's train + dev records), `ask` (send the requests to the "
            "model's server; needs the server, which needs a large GPU) and `table` (the score table "
            "from the run folders)."
        ),
    )
    steps = parser.add_subparsers(dest="step", required=True, metavar="<step>")

    package = steps.add_parser(
        "package",
        help=f"the request packages: each statement with its {judge.K} nearest train records",
        description=(
            f"Write --out/{judge.PACKAGE} and --out/{judge.REQUESTS} (every definition of the corpus "
            f"view's terms, in {judge.SHARDS} shards), and --out/{judge.GROUNDTRUTH_PACKAGE} and "
            f"--out/{judge.GROUNDTRUTH_REQUESTS} (the ground truth's train + dev statements; no TEST "
            "record). Each statement gets its --neighbours nearest train records by the exact inner "
            "product of the cached embeddings of 'term: definition', never from its own paper. Needs the "
            "embedding cache (CPU only)."
        ),
    )
    package.add_argument(
        "--store", type=Path, help="the corpus store (default: the one in DATA_ROOT)"
    )
    package.add_argument(
        "--links",
        type=Path,
        help="the definition links of the corpus release v1.1 (Step 4's release folder; default: the ones in DATA_ROOT)",
    )
    package.add_argument(
        "--cache",
        type=Path,
        help="the embedding cache folder (default: Qwen3-Embedding-8B's in DATA_ROOT)",
    )
    package.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the folder to write the packages to (not inside a frozen folder)",
    )
    package.add_argument(
        "--ground-truth-folder",
        type=Path,
        help=f"the folder holding the two ground-truth files (default: {judge.groundtruth.FOLDER}, MBDef-Gold)",
    )
    package.add_argument(
        "--neighbours",
        type=int,
        default=judge.K,
        help="train records shown with each statement (default: %(default)s)",
    )
    package.add_argument(
        "--batch",
        type=int,
        default=judge.BATCH,
        help="statements of a paper per request, for the request lists; give `ask` the same "
        "number (default: %(default)s)",
    )
    package.set_defaults(
        run=lambda args: judge.package_command(
            postprocess.refuse_frozen(args.out),
            args.store,
            args.links,
            args.cache,
            args.ground_truth_folder,
            args.neighbours,
            args.batch,
        )
    )

    ask = steps.add_parser(
        "ask",
        help="send a package's requests to the judge's server and record the answers",
        description=(
            "Send each request (--batch statements of a paper, shown as c1, c2, ..., with their "
            "neighbours) to the server's /v1/chat/completions (--model, thinking off, the answer's JSON "
            f"schema, {judge.TOP_LOGPROBS} log-probabilities per token), --concurrency at a "
            f"time; write one line per request to --out/{judge.ANSWERS} as the answers come back and, "
            f"once every request has one, --out/{judge.DECISIONS}. A malformed answer is an error line, "
            "asked again by the next run; the command exits 1 while a request has no answer."
        ),
    )
    ask.add_argument(
        "--package", type=Path, required=True, help="a package of `package`"
    )
    ask.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the run folder (resumed if it exists; not inside a frozen folder)",
    )
    ask.add_argument(
        "--split", help="only the papers of this shard or split (e.g. shard0, train)"
    )
    ask.add_argument(
        "--server",
        default=judge.SERVER,
        help="the server's address (default: %(default)s)",
    )
    ask.add_argument(
        "--concurrency",
        type=int,
        default=judge.CONCURRENCY,
        help="requests in flight (default: %(default)s)",
    )
    ask.add_argument(
        "--model",
        default=judge.MODEL,
        help="the model the server serves (default: %(default)s)",
    )
    for flag, kind in (
        ("--temperature", float),
        ("--max-tokens", int),
        ("--seed", int),
    ):
        ask.add_argument(
            flag,
            type=kind,
            default=judge.SETTINGS[flag[2:].replace("-", "_")],
            help="a setting of every request (default: %(default)s)",
        )
    ask.add_argument(
        "--batch",
        type=int,
        default=judge.BATCH,
        help="statements of a paper per request, as `package --batch` (default: %(default)s)",
    )
    ask.set_defaults(
        run=lambda args: judge.ask_command(
            args.package,
            postprocess.refuse_frozen(args.out),
            args.split,
            args.server,
            args.concurrency,
            args.model,
            {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            },
            args.batch,
        )
    )

    table = steps.add_parser(
        "table",
        help="the judge's score table: each term's highest P(yes) + 0.5 P(uncertain)",
        description=(
            "Write the score table (method, unit, item_id, split, score) of the run folders of a package: "
            "for the corpus, one row per term (every statement judged exactly once); for the ground truth, "
            "one row per train or dev record. A statement scores P(yes) + 0.5 P(uncertain) of its "
            "relevance answer, a term or record its highest statement."
        ),
    )
    table.add_argument(
        "--package", type=Path, required=True, help="the package the runs answered"
    )
    table.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="the run folders of `ask`"
    )
    table.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the score table to write (CSV; not inside a frozen folder)",
    )
    table.add_argument(
        "--method",
        help=f"the method column (default: {judge.CORPUS_METHOD!r} or {judge.METHOD!r})",
    )
    table.set_defaults(
        run=lambda args: judge.table_command(
            args.package, args.runs, postprocess.refuse_frozen(args.out), args.method
        )
    )


def dataset_parser(groups) -> None:
    """The `dataset` group (docs/dataset/dataset.md): MBDef's tables, card and JSON copy, and MBDef-Gold's split files and card."""
    from mbdef import dataset, postprocess
    from mbdef.dataset import build as dataset_build
    from mbdef.dataset import export as dataset_export
    from mbdef.dataset import gold as dataset_gold
    from mbdef.postprocess.groundtruth import FILES as gt_files
    from mbdef.postprocess.groundtruth import FOLDER as gt_folder
    from mbdef.postprocess.groundtruth import VOTES as gt_votes

    frozen = f"{postprocess.FROZEN}/"

    def repo_path(path: Path) -> str:
        return path.relative_to(postprocess.REPO).as_posix()

    def banner_flags(parser, default: Path) -> None:
        parser.add_argument(
            "--banner",
            type=Path,
            default=default,
            help=f"the image at the top of the card, copied into the dataset folder as {dataset.BANNER_PATH} "
            f"(default: {repo_path(default)})",
        )
        parser.add_argument(
            "--no-banner", action="store_true", help="a card without the image"
        )

    group = groups.add_parser(
        "dataset",
        help="MBDef and MBDef-Gold, the released datasets: parquet tables, dataset cards, JSON copy (docs/dataset/dataset.md)",
        description=(
            f"Build MBDef, the released dataset, from Step 5's frozen files ({frozen}<date>/), the corpus "
            "store and the corpus release v1.1 in DATA_ROOT; and MBDef-Gold's split files and card from the "
            "ground truth. Write only to the folder --out names, never into a frozen folder."
        ),
    )
    actions = group.add_subparsers(dest="action", required=True, metavar="<action>")
    listing = (
        "a sha256sum listing (paths taxonomy-dataset/... and taxonomy-dataset-json/...); fail unless "
        "every file written has the hash it records, as data/frozen/dataset/release-1.2.1.sha256 records version "
        "1.2.1 (files it records that are not written, such as 1.2.1's ground-truth tables, are named)"
    )

    build = actions.add_parser(
        "build",
        help=f"write --out/{dataset.DATASET}/ (card, parquet tables, provenance) and the validation report",
        description=(
            f"Write --out/{dataset.DATASET}/: README.md (the dataset card), data/*.parquet "
            "(terms, concepts, instances, definitions, papers) "
            "and provenance.json; and --out/validation.{json,md}. A build that fails a check is moved "
            f"to --out/{dataset.DATASET}-FAILED/ and the command exits 1. Every input flag defaults "
            "to the file of the released run (docs/dataset/dataset.md, Inputs)."
        ),
    )
    build.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the folder to write to (not inside a frozen folder)",
    )
    for flag, what in (
        (
            "--decisions",
            f"the keep or drop of every corpus term, a csv with its json beside it (default: {frozen}{dataset.DECISIONS}.csv)",
        ),
        (
            "--label-groups",
            f"the Label Groups over all corpus terms (default: {frozen}{dataset.LABEL_GROUPS})",
        ),
        (
            "--term-merging",
            f"the merging of the kept terms (default: {frozen}{dataset.TERM_MERGING})",
        ),
        ("--pick", f"the ensemble's frozen weights (default: {frozen}{dataset.PICK})"),
        ("--store", f"the corpus store (default: $DATA_ROOT/{dataset.STORE})"),
        (
            "--corpus",
            f"the corpus release v1.1's folder (Step 4), its paper lists and links (default: $DATA_ROOT/{dataset.CORPUS})",
        ),
        (
            "--recorded",
            f"the recorded OpenAlex answers, likely-wrong merges and card numbers (default: {repo_path(dataset.RECORDED)})",
        ),
        (
            "--ground-truth",
            f"MBDef-Gold's relevant records, for the card's count of labelled train and dev records (default: {gt_folder}/{gt_files[0][0]})",
        ),
        (
            "--non-relevant",
            f"MBDef-Gold's non-relevant records (default: {gt_folder}/{gt_files[1][0]})",
        ),
    ):
        build.add_argument(flag, type=Path, help=what)
    build.add_argument(
        "--licence",
        default=dataset.LICENCE,
        help="the data licence's id for the card (default: %(default)s)",
    )
    build.add_argument(
        "--repo-id",
        default=dataset.REPO_ID,
        help="the dataset's repository on a dataset host, for the card's loading example (default: %(default)s)",
    )
    build.add_argument(
        "--gold-repo-id",
        default=files.MBDEF_GOLD_REPO,
        help="MBDef-Gold's repository, which the card links (default: %(default)s)",
    )
    build.add_argument("--same-data-as", type=Path, metavar="LISTING", help=listing)
    banner_flags(build, dataset.BANNER)
    build.add_argument(
        "--explorer-data",
        type=Path,
        metavar="FOLDER",
        help="the MBDef Explorer's page data (a folder with its SHA256SUMS, as `explorer build` writes it to "
        "explorer/data/): checked against the tables and copied into the dataset as explorer/ (docs/dataset/explorer.md)",
    )

    def run_build(args):
        out = postprocess.refuse_frozen(args.out)
        out.mkdir(parents=True, exist_ok=True)
        inputs = dataset.inputs(
            decisions=args.decisions,
            label_groups=args.label_groups,
            term_merging=args.term_merging,
            pick=args.pick,
            store=args.store,
            corpus=args.corpus,
            recorded=args.recorded,
            ground_truth=args.ground_truth,
            non_relevant=args.non_relevant,
        )
        return dataset_build.build(
            inputs,
            out,
            version=dataset.VERSION,
            licence=args.licence,
            repo_id=args.repo_id,
            gold_repo_id=args.gold_repo_id,
            same_data_as=args.same_data_as,
            banner=None if args.no_banner else args.banner,
            explorer=args.explorer_data,
        )

    build.set_defaults(run=run_build)

    gold = actions.add_parser(
        "build-gold",
        help=f"write --out/{dataset_gold.GOLD}/: MBDef-Gold's split files, card and record files",
        description=(
            f"Write --out/{dataset_gold.GOLD}/: README.md (MBDef-Gold's dataset card), "
            "data/{train,validation,test}.parquet (one row per record, with its label, votes, fields and paper), "
            "the two record files unchanged and provenance.json; and --out/gold-validation.{json,md}. A build that "
            f"fails a check is moved to --out/{dataset_gold.GOLD}-FAILED/ and the command exits 1. MBDef-Gold's "
            "other files (statistics, guide, report) come from `groundtruth build` (docs/step-3-extraction/groundtruth.md)."
        ),
    )
    gold.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the folder to write to (not inside a frozen folder)",
    )
    for flag, what in (
        (
            "--ground-truth",
            f"the relevant records (default: {gt_folder}/{gt_files[0][0]})",
        ),
        (
            "--non-relevant",
            f"the non-relevant records (default: {gt_folder}/{gt_files[1][0]})",
        ),
        ("--votes", f"each record's votes (default: {repo_path(gt_votes)})"),
        (
            "--recorded",
            f"the folder of the card's recorded Step 5 numbers (default: {repo_path(dataset.RECORDED)})",
        ),
    ):
        gold.add_argument(flag, type=Path, help=what)
    gold.add_argument(
        "--licence",
        default=dataset.LICENCE,
        help="the labels' licence id for the card (default: %(default)s)",
    )
    gold.add_argument(
        "--repo-id",
        default=files.MBDEF_GOLD_REPO,
        help="MBDef-Gold's repository, for the card's loading example (default: %(default)s)",
    )
    gold.add_argument(
        "--mbdef-repo-id",
        default=dataset.REPO_ID,
        help="MBDef's repository, which the card links (default: %(default)s)",
    )
    banner_flags(gold, dataset.GOLD_BANNER)
    gold.add_argument(
        "--samples-image",
        type=Path,
        default=dataset.GOLD_SAMPLES,
        help=f"the picture of sample records on the card, copied into the dataset folder as {dataset.SAMPLES_PATH} "
        f"(default: {repo_path(dataset.GOLD_SAMPLES)})",
    )
    gold.add_argument(
        "--no-samples-image",
        action="store_true",
        help="a card without the picture of sample records",
    )

    def run_gold(args):
        out = postprocess.refuse_frozen(args.out)
        out.mkdir(parents=True, exist_ok=True)
        from mbdef.postprocess import groundtruth as gt

        relevant, non_relevant = (
            (args.ground_truth, args.non_relevant)
            if args.ground_truth and args.non_relevant
            else (p for p, _ in gt.paths())
        )
        return dataset_gold.build_gold(
            args.ground_truth or relevant,
            args.non_relevant or non_relevant,
            args.votes or gt_votes,
            args.recorded or dataset.RECORDED,
            out,
            licence=args.licence,
            repo_id=args.repo_id,
            mbdef_repo_id=args.mbdef_repo_id,
            banner=None if args.no_banner else args.banner,
            samples_image=None if args.no_samples_image else args.samples_image,
        )

    gold.set_defaults(run=run_gold)

    export = actions.add_parser(
        "export",
        help=f"write --out/{dataset.JSON_COPY}/ from the tables a build wrote to --out/{dataset.DATASET}/",
        description=(
            f"Write the built tables as JSON into --out/{dataset.JSON_COPY}/: terms.json, papers.json, "
            "tables/*.json, SCHEMA.md, DATASET-CARD.md and SHA256SUMS. Each file is "
            "read back and must equal its parquet table."
        ),
    )
    export.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the folder a `dataset build --out` wrote to",
    )
    export.add_argument("--same-data-as", type=Path, metavar="LISTING", help=listing)
    export.set_defaults(
        run=lambda args: dataset_export.export(
            postprocess.refuse_frozen(args.out), args.same_data_as
        )
    )
