"""The relevance step (paper Steps 1 and 2, docs/step-1-2-papers/relevance.md): which papers of the starting corpus are about media bias.

Two language models classify each paper from its title and abstract (`classify`); the papers both mark relevant are kept, and the papers on which they disagree go to a third model (`compare`), whose relevant ones are kept too (`export`). Every classification can be replayed from recorded answers, with no model and no key.
"""
