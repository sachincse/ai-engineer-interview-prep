"""The "what can we tweak?" module.

Two hard constraints make this legitimate rather than a hallucination machine:

1. **Constrained by the landscape.** A proposed change must land in a region the
   retrieved prior art does not occupy — computed from the documents, not
   imagined by a model.
2. **Constrained by support in the specification.** You cannot later claim what
   your own application as filed does not support (EPC Art. 123(2): an amendment
   must be *directly and unambiguously derivable* from the application as filed).
   A clever narrowing that appears nowhere in your own text is not a rescue, it
   is an added-matter objection.

Consequence, and it inverts where the value sits: at prosecution time the set of
available tweaks is already frozen.  The high-value moment is **before drafting**
— run the landscape first, then make sure the fallback ladder (sub-ranges,
preferred embodiments, individual substituents) is actually written into the
specification, so those positions exist later.

Everything below produces **ranked hypotheses with evidence**, for counsel to
accept or reject.  Nothing here decides patentability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Corpus
from .elements import Element, NumericSpec, _ranges
from . import text as T


@dataclass
class Occupied:
    doc_id: str
    low: float
    high: float
    span: str


@dataclass
class WhiteSpace:
    element: Element
    occupied: list[Occupied] = field(default_factory=list)
    merged: list[tuple[float, float]] = field(default_factory=list)
    gaps: list[tuple[float, float]] = field(default_factory=list)
    claimed_sits_in_gap: bool = False
    overlaps_occupied: bool = False
    fully_inside_occupied: bool = False

    @property
    def quantity(self) -> str:
        return self.element.numeric.quantity if self.element.numeric else ""

    def sources(self, lo: float, hi: float) -> str:
        ids = sorted({o.doc_id for o in self.occupied if o.low <= hi and o.high >= lo})
        return ", ".join(ids)


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        plo, phi = merged[-1]
        if lo <= phi:
            merged[-1] = (plo, max(phi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _gaps(merged: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [
        (merged[i][1], merged[i + 1][0])
        for i in range(len(merged) - 1)
        if merged[i + 1][0] > merged[i][1]
    ]


def map_white_space(
    corpus: Corpus, element: Element, candidate_ids: list[str]
) -> WhiteSpace | None:
    """Project the retrieved landscape onto one numeric axis and find the holes."""
    spec: NumericSpec | None = element.numeric
    if spec is None:
        return None

    ws = WhiteSpace(element=element)
    for doc_id in candidate_ids:
        doc = corpus.by_id(doc_id)
        for raw in T.sentences(doc.full_text):
            norm = " " + T.normalise(raw) + " "
            if not all(c in set(T.concepts(raw)) for c in element.required):
                continue
            for lo, hi in _ranges(norm, spec):
                ws.occupied.append(Occupied(doc_id, lo, hi, raw))

    ws.merged = _merge([(o.low, o.high) for o in ws.occupied])
    ws.gaps = _gaps(ws.merged)
    claimed = (spec.low, spec.high)
    ws.overlaps_occupied = any(
        max(claimed[0], lo) < min(claimed[1], hi) or (lo <= claimed[0] <= hi)
        for lo, hi in ws.merged
    )
    ws.fully_inside_occupied = any(
        lo <= claimed[0] and claimed[1] <= hi for lo, hi in ws.merged
    )
    ws.claimed_sits_in_gap = (not ws.overlaps_occupied) or any(
        g[0] <= claimed[0] and claimed[1] <= g[1] for g in ws.gaps
    )
    return ws


@dataclass
class Hypothesis:
    element_id: str
    headline: str
    evidence: str
    legal_flag: str


def propose(
    corpus: Corpus, elements: list[Element], candidate_ids: list[str],
    point_of_novelty_ids: set[str],
) -> list[Hypothesis]:
    """Rank tweak hypotheses. Elements already carrying the novelty come first,
    because they are what the filing currently rests on."""
    out: list[Hypothesis] = []
    for e in elements:
        ws = map_white_space(corpus, e, candidate_ids)
        if ws is None or not ws.occupied:
            continue
        spec = e.numeric
        assert spec is not None
        occ = ", ".join(f"{lo}–{hi} [{ws.sources(lo, hi)}]" for lo, hi in ws.merged)
        gaps = ", ".join(f"{g[0]}–{g[1]}" for g in ws.gaps) or "none between the occupied bands"
        claimed = f"{spec.low}–{spec.high} {spec.unit_label}".strip()

        if ws.fully_inside_occupied:
            headline = (f"the claimed {ws.quantity} ({claimed}) lies wholly INSIDE a "
                        f"range already disclosed — this element does not differentiate "
                        f"on its own")
            flag = ("A narrower range carved out of a broader disclosed range is the "
                    "selection-invention question (EPO Guidelines G-VI, 8): is it narrow, "
                    "and sufficiently far removed from the disclosed examples and "
                    "end-points? That is a legal call, not a computed one.")
        elif not ws.overlaps_occupied:
            headline = (f"the claimed {ws.quantity} ({claimed}) is UNOCCUPIED by any "
                        f"document in this pool — the strongest differentiator found")
            flag = ("Unoccupied in the retrieved pool is not the same as unoccupied in "
                    "the art: this is bounded by the recall of the search, and by the "
                    "18-month publication blackout.")
        else:
            headline = (f"the claimed {ws.quantity} ({claimed}) partially overlaps "
                        f"occupied territory")
            flag = ("Partial overlap does not differentiate on its own, and any "
                    "narrowing to escape it must still be directly and unambiguously "
                    "derivable from the application as filed (Art. 123(2)).")
        out.append(Hypothesis(
            element_id=e.id,
            headline=headline,
            evidence=f"occupied: {occ}  |  unoccupied gaps: {gaps}",
            legal_flag=flag,
        ))
    out.sort(key=lambda h: (h.element_id not in point_of_novelty_ids, h.element_id))
    return out
