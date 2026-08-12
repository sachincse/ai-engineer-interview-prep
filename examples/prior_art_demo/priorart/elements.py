"""Claim-element decomposition and per-element evidence extraction.

This module is where the legal rule becomes an objective function.

    Novelty (EPC Art. 54) requires that **one single document** disclose
    **every** element of the claim, directly and unambiguously.

So the right primitive is not "how similar are these two patents".  It is a
**coverage question, per element, per document**.  A document scoring 0.9 on
whole-document similarity but missing element 4 is worth nothing; a document
scoring 0.3 that happens to disclose all five is decisive.

In production the decomposition is proposed by an LLM and then *edited by the
attorney* before anything is searched — it is the highest-leverage human
checkpoint in the whole system, because every downstream result is conditioned
on it.  Here the elements are written out by hand so the demo is reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from . import text as T

Status = Literal["disclosed", "partial", "not_found"]

DISCLOSED: Status = "disclosed"
PARTIAL: Status = "partial"
NOT_FOUND: Status = "not_found"

# A numeric range written as "X to Y", optionally followed by a unit.
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(?P<tail>[a-z0-9 ]{0,16})")
# A single value, e.g. "at 175 degrees c" — treated as a degenerate range.
_POINT_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?P<tail>degrees c|wt)")


@dataclass(frozen=True)
class NumericSpec:
    """A quantity the element constrains, e.g. 'cerium loading, 0.4–0.6 wt%'.

    ``owner_terms`` / ``rival_terms`` solve an attribution problem that looks
    trivial and is not.  In the sentence

        "Example 2: nickel (5 wt%) with cerium (0.2 wt%) on gamma-alumina"

    a naive "find a wt% number in a sentence that mentions cerium" rule happily
    reports a cerium loading of 5 wt%.  The fix is nearest-cue-wins: a number
    belongs to the quantity whose name sits closest to it.  Get this wrong and
    the white-space map in ``designaround.py`` inherits phantom occupied
    territory, which is worse than having no map at all.
    """
    quantity: str
    low: float
    high: float
    unit_label: str
    unit_cue: str                          # substring that must follow the number
    sentence_cue: str = ""                 # substring required in the sentence
    owner_terms: tuple[str, ...] = ()      # this quantity's own names
    rival_terms: tuple[str, ...] = ()      # names that would steal the number


@dataclass(frozen=True)
class Element:
    id: str
    label: str
    required: tuple[str, ...] = ()          # concepts that must ALL be present
    broader: tuple[str, ...] = ()           # concepts that make it *arguable*
    numeric: NumericSpec | None = None


@dataclass
class Cell:
    """One (element, document) result: a status plus the evidence for it."""
    element_id: str
    doc_id: str
    status: Status
    span: str = ""
    note: str = ""
    found_range: tuple[float, float] | None = None

    @property
    def symbol(self) -> str:
        return (_ASCII_SYMBOLS if USE_ASCII else _SYMBOLS)[self.status]


#: Cell glyphs. Some Windows consoles cannot encode the block glyphs, so the
#: demo falls back to ASCII rather than crashing on a UnicodeEncodeError.
_SYMBOLS = {DISCLOSED: "●", PARTIAL: "◐", NOT_FOUND: "○"}
_ASCII_SYMBOLS = {DISCLOSED: "[X]", PARTIAL: "[~]", NOT_FOUND: "[ ]"}
USE_ASCII = False


def set_ascii(enabled: bool) -> None:
    global USE_ASCII
    USE_ASCII = enabled


def _nearest(text: str, terms: tuple[str, ...], pos: int) -> int | None:
    """Distance from ``pos`` to the closest occurrence of any term."""
    best: int | None = None
    for term in terms:
        start = 0
        while True:
            i = text.find(term, start)
            if i < 0:
                break
            d = abs(i - pos)
            if best is None or d < best:
                best = d
            start = i + 1
    return best


def _owned(norm_sentence: str, spec: NumericSpec, pos: int) -> bool:
    """Nearest-cue-wins: does this number belong to *this* quantity?"""
    if not spec.owner_terms:
        return True
    mine = _nearest(norm_sentence, spec.owner_terms, pos)
    if mine is None:
        return False
    theirs = _nearest(norm_sentence, spec.rival_terms, pos)
    return theirs is None or mine <= theirs


def _ranges(norm_sentence: str, spec: NumericSpec) -> list[tuple[float, float]]:
    """Every range in the sentence that belongs to this quantity."""
    out: list[tuple[float, float]] = []
    if spec.sentence_cue and spec.sentence_cue not in norm_sentence:
        return out
    for m in _RANGE_RE.finditer(norm_sentence):
        tail = m.group("tail")
        if spec.unit_cue:
            if spec.unit_cue not in tail:
                continue
        else:
            # A unitless quantity (a ratio): reject ranges that carry a unit.
            if "degrees" in tail or "wt" in tail:
                continue
        if not _owned(norm_sentence, spec, m.start()):
            continue
        out.append((float(m.group(1)), float(m.group(2))))
    if not out and spec.unit_cue:
        for m in _POINT_RE.finditer(norm_sentence):
            if spec.unit_cue in m.group("tail") and _owned(norm_sentence, spec, m.start()):
                v = float(m.group(1))
                out.append((v, v))
    return out


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def assess(element: Element, doc_text: str) -> Cell:
    """Decide whether one document discloses one element, and quote the span.

    The rule that keeps this honest: **no span, no cell.**  A status is only
    ever returned together with the exact sentence it came from, so a reviewer
    can disagree with the machine by reading one line.
    """
    raw_sents = T.sentences(doc_text)
    best = Cell(element_id=element.id, doc_id="", status=NOT_FOUND)

    for raw in raw_sents:
        norm = " " + T.normalise(raw) + " "
        present = set(T.concepts(raw))

        has_required = all(c in present for c in element.required)
        has_broader = bool(element.broader) and any(c in present for c in element.broader)
        if not has_required and not has_broader:
            continue

        if element.numeric is None:
            if has_required:
                return Cell(element.id, "", DISCLOSED, raw)
            if has_broader and best.status == NOT_FOUND:
                best = Cell(
                    element.id, "", PARTIAL, raw,
                    note=f"only the broader concept ({', '.join(element.broader)}) is disclosed",
                )
            continue

        if not has_required:
            continue

        spec = element.numeric
        target = (spec.low, spec.high)
        for rng in _ranges(norm, spec):
            if rng[0] <= spec.low and rng[1] >= spec.high:
                return Cell(
                    element.id, "", DISCLOSED, raw,
                    note=(f"the disclosed range {rng[0]}–{rng[1]} {spec.unit_label} "
                          f"encompasses the claimed {spec.low}–{spec.high} — this is the "
                          f"selection-invention question, and it is a legal call"),
                    found_range=rng,
                )
            if _overlap(target, rng) > 0:
                if best.status != DISCLOSED:
                    best = Cell(
                        element.id, "", PARTIAL, raw,
                        note=(f"ranges overlap but do not contain: document "
                              f"{rng[0]}–{rng[1]} vs claimed {spec.low}–{spec.high} {spec.unit_label}"),
                        found_range=rng,
                    )
            elif best.status == NOT_FOUND:
                best = Cell(
                    element.id, "", NOT_FOUND, raw,
                    note=(f"quantity present but outside the claimed range: "
                          f"{rng[0]}–{rng[1]} vs {spec.low}–{spec.high} {spec.unit_label}"),
                    found_range=rng,
                )
    return best


#: The decomposition of the demo disclosure.  Five elements, two of them purely
#: conceptual and three of them numeric ranges — which is what a formulation or
#: process claim in chemistry actually looks like.
DEMO_ELEMENTS: list[Element] = [
    Element("E1", "palladium catalyst", required=("pd",), broader=("noble_metal",)),
    Element("E2", "gamma-alumina support", required=("gamma_alumina",),
            broader=("alumina", "oxide_support")),
    Element("E3", "reaction temperature 180–220 °C", required=("temperature",),
            numeric=NumericSpec("temperature", 180, 220, "°C", "degrees c")),
    Element("E4", "cerium promoter 0.4–0.6 wt%", required=("ce",),
            numeric=NumericSpec(
                "cerium loading", 0.4, 0.6, "wt%", "wt",
                owner_terms=("cerium", "ce "),
                rival_terms=("nickel", "palladium", "lanthanum", "platinum",
                             "praseodymium", "noble metal"))),
    Element("E5", "H2:substrate molar ratio 2.2–2.4", required=("h2_ratio",),
            numeric=NumericSpec("H2:substrate ratio", 2.2, 2.4, "", "",
                                sentence_cue="molar ratio")),
]
