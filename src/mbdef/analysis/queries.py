"""Lookups in MBDef's tables: a merged term by its name, its definitions with their papers, and the subtype hierarchy that the papers report."""

from collections import defaultdict

import polars as pl


def find_concept(data: dict[str, pl.DataFrame], label: str) -> str:
    """The concept_id of the merged term one of whose spellings (`concepts.labels`) is LABEL, ignoring case."""
    wanted = label.casefold()
    matches = (
        data["concepts"]
        .filter(
            pl.col("labels")
            .list.eval(pl.element().str.to_lowercase())
            .list.contains(wanted)
        )
        .get_column("concept_id")
        .to_list()
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one merged term spelled {label!r}, found {len(matches)}"
        )
    return matches[0]


def definitions_of(data: dict[str, pl.DataFrame], label: str) -> pl.DataFrame:
    """Every definition of the merged term LABEL, with its paper's title, year, DOI and corpus."""
    papers = data["papers"].select("paper_id", "title", "year", "doi", "corpus")
    return (
        data["definitions"]
        .filter(pl.col("concept_id") == find_concept(data, label))
        .join(papers, on="paper_id", how="left")
        .sort("paper_id", "def_id")
    )


def hierarchy(
    data: dict[str, pl.DataFrame],
) -> tuple[dict[tuple[str, str], list[dict]], list[dict]]:
    """Subtype-to-supertype edges between merged terms, each with the papers' claims behind it, and the claims that link a term to itself.

    An edge comes from a `subtypes` or `supertypes` item of the terms table whose `links_to` names a merged term; every source of the item is one claim.
    """
    corpus = dict(data["papers"].select("paper_id", "corpus").iter_rows())
    evidence: dict[tuple[str, str], list[dict]] = defaultdict(list)
    self_relations = []
    for field in ("subtypes", "supertypes"):
        items = (
            data["terms"]
            .select("concept_id", "term", field)
            .explode(field)
            .unnest(field)
            .filter(pl.col("links_to").struct.field("concept_id").is_not_null())
            .explode("sources")
        )
        for row in items.iter_rows(named=True):
            cid, other = row["concept_id"], row["links_to"]["concept_id"]
            child, parent = (other, cid) if field == "subtypes" else (cid, other)
            source = row["sources"]
            claim = {
                **source,
                "field": field,
                "term": row["term"],
                "related_term": row["text"],
                "corpus": corpus[source["paper_id"]],
            }
            if child == parent:
                self_relations.append(claim)
            else:
                evidence[child, parent].append(claim)
    return evidence, self_relations
