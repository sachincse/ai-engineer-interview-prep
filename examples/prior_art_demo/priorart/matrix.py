"""The element × document coverage matrix — the product insight of the system.

Reading the matrix:

  * **A full column** (one document disclosing every element) is an
    anticipation candidate under Art. 54.  It is the only shape that can kill
    novelty on its own.
  * **No full column** means "no anticipation found *by this system, at this
    recall level*".  It does **not** mean "novel".  That distinction is the
    whole reason the system outputs evidence rather than a verdict.
  * **A row that is empty across every column** is the **point of novelty** —
    the element currently carrying the invention.  It is the single most
    actionable output in the pipeline, and it feeds the design-around module.
  * **A small set of columns that jointly cover every row** is an
    inventive-step *hypothesis* under Art. 56, not a finding.  Whether the
    skilled person would have combined those documents is a legal judgement
    about motivation, which no similarity score expresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Corpus, DateVerdict, classify_date
from .elements import Cell, Element, DISCLOSED, PARTIAL, NOT_FOUND, assess


@dataclass
class Column:
    doc_id: str
    verdict: DateVerdict
    cells: dict[str, Cell] = field(default_factory=dict)

    def disclosed_ids(self, include_partial: bool = False) -> set[str]:
        ok = {DISCLOSED} | ({PARTIAL} if include_partial else set())
        return {eid for eid, c in self.cells.items() if c.status in ok}


@dataclass
class Matrix:
    elements: list[Element]
    columns: list[Column]

    # -------------------------------------------------------------- queries

    def anticipations(self) -> list[Column]:
        """Columns where a single document discloses every element AND is
        usable for novelty (Art. 54(2) or 54(3))."""
        need = {e.id for e in self.elements}
        return [
            c for c in self.columns
            if c.verdict.usable_for_novelty and c.disclosed_ids() >= need
        ]

    def point_of_novelty(self) -> list[Element]:
        """Elements disclosed by no document anywhere in the candidate pool."""
        out = []
        for e in self.elements:
            if not any(
                c.cells[e.id].status == DISCLOSED
                for c in self.columns
                if c.verdict.usable_for_novelty
            ):
                out.append(e)
        return out

    def minimal_cover(self) -> list[str]:
        """Greedy set cover over documents usable for *inventive step*.

        Greedy is used deliberately: exact minimum set cover is NP-hard, greedy
        is within a ln(n) factor, and the output is a hypothesis for a human
        rather than an optimum worth paying for.
        """
        remaining = {e.id for e in self.elements}
        usable = [c for c in self.columns if c.verdict.usable_for_inventive_step]
        chosen: list[str] = []
        while remaining:
            best, gain = None, 0
            for col in usable:
                if col.doc_id in chosen:
                    continue
                g = len(col.disclosed_ids() & remaining)
                if g > gain:
                    best, gain = col, g
            if best is None:
                break
            chosen.append(best.doc_id)
            remaining -= best.disclosed_ids()
        return chosen

    def uncovered_after(self, chosen: list[str]) -> set[str]:
        covered: set[str] = set()
        for col in self.columns:
            if col.doc_id in chosen:
                covered |= col.disclosed_ids()
        return {e.id for e in self.elements} - covered


def build(
    corpus: Corpus, elements: list[Element], candidate_ids: list[str]
) -> Matrix:
    """Assess every candidate document against every claim element.

    Note the order of operations: the **date verdict is computed first and is
    never delegated to a model**.  A document filed after the priority date can
    match the disclosure word-for-word and is still not prior art; a system that
    ranks on similarity alone will happily present it as a perfect anticipation.
    """
    columns: list[Column] = []
    for doc_id in candidate_ids:
        doc = corpus.by_id(doc_id)
        verdict = classify_date(doc, corpus.disclosure.priority_date)
        col = Column(doc_id=doc_id, verdict=verdict)
        for element in elements:
            cell = assess(element, doc.full_text)
            cell.doc_id = doc_id
            col.cells[element.id] = cell
        columns.append(col)
    return Matrix(elements=elements, columns=columns)
