# Glossary: MBDef, the Media Bias Definitions Dataset

The words of the project: the datasets of definitions quoted from the scientific literature on media bias, how they are scored, and the taxonomy of media bias concepts built from them, which keeps for every concept and relation the evidence and the people's decisions behind it.

## Datasets and scores

**MBDef**:
The Media Bias Definitions Dataset: the Definitions found in the Corpus Closure, as Extracted Concepts with their papers, whether each is Relevant, and the Label Groups they merge into.
_Avoid_: taxonomy dataset, the dataset (when MBDef-Gold could be meant)

**MBDef-Gold**:
The ground truth: the definition records of 130 papers that annotators judged and a curator checked and edited, against which extracted Definitions are scored.
_Avoid_: BiasDef130, validated GT, the gold set

**DefScore**:
How closely two Definitions say the same thing, from 0 to 1: the calibrated mean of their word overlap, embedding similarity and Entailment in both directions; a predicted Definition matches a reference Definition at 0.5 or more.
_Avoid_: M3, definition similarity (unqualified)

**MBDef Explorer**:
The web page that shows MBDef: its Extracted Concepts on a map, with their Definitions and papers.
_Avoid_: taxonomy explorer, dashboard

## Corpus

**Seed Corpus**:
The papers judged relevant to media bias that the work starts from.
_Avoid_: seeds set, base corpus

**Corpus Closure**:
The Seed Corpus plus every work a seed definition is attributed to (one step deep), as found by the chosen extractor.
_Avoid_: closure set, expanded corpus

**Definition**:
A quoted passage in which a paper states what a term means; a passage that only describes or uses the term is not one.
_Avoid_: description, quote (when meaning a definition)

**Accepted Definition**:
A Definition the screening judged both Relevant and a real definition.
_Avoid_: usable definition, clean definition

**Relevant**:
Said of an Extracted Concept that is a kind of media bias (core) or a concept needed to study media bias (satellite); judged per term in its paper, not per Definition.
_Avoid_: in scope, on topic

## Concepts

**Extracted Concept**:
A term as one paper defines it: one paper, one label, and that paper's Definitions of it.
_Avoid_: mention, concept (unqualified)

**Media Bias Concept**:
A group of Extracted Concepts judged to mean the same thing (a study of one counts as a study of the other); the unit the taxonomy is made of, named by an opaque identifier that never changes meaning.
_Avoid_: MBC (in prose), cluster, group, concept (unqualified)

**Preferred Label**:
The name a Media Bias Concept is shown under: the form used by the most distinct papers, unless a Curator sets another.
_Avoid_: name, title, main label

**Label Attestation**:
One written form of a Media Bias Concept's name together with the works it appears in.
_Avoid_: alias, variant

**Label Match**:
Two labels that are equal once case, punctuation, spacing, plural forms, British or American spelling, word order and the stop words between content words are ignored, and a bracketed acronym or alternative name is dropped; any other bracketed qualifier still counts.
_Avoid_: exact match, same label

**Label Group**:
All Extracted Concepts whose labels Label-Match; the grouping made before any meaning is compared, so one Label Group can hold different meanings and is never itself a Media Bias Concept.
_Avoid_: label cluster, concept group, normalised term

**Candidate Pair**:
Two Extracted Concepts that are compared because at least one signal (label, meaning, a reported synonym, a shared cited work) links them.
_Avoid_: pair (unqualified), match

**Case**:
The category of a Candidate Pair by Label Match and the direction of entailment between its Definitions (Cases 1–6); it explains a grouping decision but does not by itself make it, except Case 3 (same label, no entailment), which keeps the pair apart until a Curator groups it.
_Avoid_: rule, bucket

**Entailment**:
One Definition entails another when everything the second says is implied by the first.
_Avoid_: similarity, equivalence

**Representative Definition**:
The Definition chosen to stand for a set of Definitions (a Media Bias Concept or a Label Group): the one that entails the largest share of the set's other Definitions.
_Avoid_: canonical definition, main definition

**Representative Media Bias Definition**:
The Representative Definition of the `media bias` Label Group; the anchor that other Definitions are compared with when relevance is judged by entailment.
_Avoid_: wiki definition, reference definition

**Coverage**:
The share of the other Definitions in its set (a Media Bias Concept or a Label Group) that one Definition entails.

**Deprecated Concept**:
A Media Bias Concept that no longer exists after a merge or split; it points to the concept that replaced it and its identifier is never reused.
_Avoid_: deleted concept, retired id

## Attribution

**Attribution Graph**:
The graph of works in which an edge runs from a paper to the work it attributes one of its Definitions to, labelled with the Media Bias Concept being defined; it is deliberately not a full bibliography.
_Avoid_: citation graph, temporal trend graph, concept-labeled citation graph

**Bibliographic Entity**:
A node of the Attribution Graph: a corpus paper with full text, or a cited work without it.
_Avoid_: source, node, reference (when meaning the work)

**Definition Citation**:
One edge of the Attribution Graph: one Definition attributed to one work.
_Avoid_: citation (unqualified), link

**Concept Subgraph**:
The part of the Attribution Graph whose edges carry one Media Bias Concept.
_Avoid_: subgraph (unqualified)

**Originator**:
A paper that defines a Media Bias Concept without attributing it, directly or through other papers, to another corpus paper that defines it; a concept can have several.
_Avoid_: root, source, inventor

**Lineage**:
The papers a given paper's Definition of a Media Bias Concept descends from in that concept's Concept Subgraph.
_Avoid_: ancestry, chain

**Lineage Support**:
The number of independent Lineages among the papers backing a claim: papers connected in the Concept Subgraph count once, a paper with no resolved attribution counts on its own.
_Avoid_: paper count, citation count, discounted weight

## Relations

**Narrower**:
The relation between two Media Bias Concepts where one is a kind of the other, typically by adding a restriction (political, visual, online); the taxonomy allows a concept several broader concepts but no cycles.
_Avoid_: subtype, is_subconcept_of, child

**Related**:
The relation between two Media Bias Concepts where one names a cause, effect, measure or part of the other, and neither is a kind of the other.
_Avoid_: is_related_to, associated

**Editorial Concept**:
A top-level concept a Curator adds to gather Media Bias Concepts, marked as not found in the literature.
_Avoid_: category, umbrella concept

**Primary Parent**:
The one broader Media Bias Concept a concept is drawn under in the hierarchy view, normally the one with the most Lineage Support; its other broader concepts are listed beside it.
_Avoid_: parent (unqualified), main category

**Reported Term**:
A synonym, broader, narrower or related term that a paper names for one of its Extracted Concepts.
_Avoid_: extracted synonym (for the whole set), keyword

**Relation Assertion**:
One piece of evidence that two Media Bias Concepts stand in a Narrower or Related relation.
_Avoid_: edge, link

**Published Relation**:
A relation supported by at least two independent criteria (definition entailment, model verification, human review, occurrence in the literature); only published relations appear in the taxonomy.
_Avoid_: fixed relation, accepted relation

**Proposed Relation**:
A relation supported by exactly one criterion; visible and flagged for expert review, but not part of the taxonomy.
_Avoid_: candidate relation, pending relation

**Rejected Relation**:
A relation a model verified as wrong with no other criterion supporting it.

## Review

**Evidence Record**:
Anything kept to justify a grouping or a relation: a grouping judgement, a Relation Assertion, a Comment, or a Curator Decision.
_Avoid_: log entry, annotation

**Pipeline Version**:
One numbered, reproducible run of the whole pipeline, fixed by its code, inputs, models, settings and the Curator Decisions it took in; everything shown on the platform belongs to exactly one.
_Avoid_: run, snapshot, release

**Curator**:
One of the few co-authors whose decisions bind the pipeline.
_Avoid_: admin, owner

**Expert**:
A member of the Media Bias Group whose judgements are evidence that a Curator weighs.
_Avoid_: reviewer (unqualified), annotator

**Commenter**:
Anyone else given access, who can leave Comments that are not counted as evidence.

**Curator Decision**:
A binding ruling by a Curator (for example "these two are the same concept") that every later Pipeline Version must respect; it is held between Extracted Concepts so it survives concepts changing, and contradicting rulings go back to a Curator.
_Avoid_: label, vote, override

**Review Queue**:
The ordered list of pairs, concepts and relations a person may review, most consequential first; anything left unreviewed is decided by the evidence already there.
_Avoid_: manual review (as a required step), backlog

**Comment**:
A remark attached to a concept, relation or Definition that binds nothing.
_Avoid_: note, feedback
