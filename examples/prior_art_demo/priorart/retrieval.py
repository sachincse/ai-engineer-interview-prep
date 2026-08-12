"""Retrieval channels and rank fusion.

Three channels, deliberately built on *different* evidence:

  1. ``BM25Channel``      — surface words only.  Exact, unforgiving, and blind
                            to synonyms it has never seen.
  2. ``ConceptChannel``   — words plus canonical concept ids, so
                            "Al2O3 carrier of the gamma phase" and
                            "gamma-aluminium oxide" land in the same place.
                            Stands in for a dense/embedding retriever.
  3. ``StructureChannel`` — no text at all.  Tanimoto over structural
                            fingerprints, because a patent can disclose a
                            molecule it never names in words.

Channel diversity is not a nice-to-have.  The recall estimate in ``recall.py``
is only honest if the channels can fail *independently* — if all three shared
one embedding they would miss the same document together and then confidently
report that nothing was missed.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .corpus import Corpus, Document
from . import text as T


@dataclass
class Hit:
    doc_id: str
    score: float
    rank: int


class _VectorChannel:
    """Shared BM25 machinery; subclasses only choose the tokenizer."""

    name = "channel"
    tokenizer: Callable[[str], list[str]] = staticmethod(T.tokenize)
    k1 = 1.5
    b = 0.75

    def __init__(self, docs: Sequence[Document]) -> None:
        self.docs = list(docs)
        self.tf: dict[str, Counter[str]] = {}
        self.len: dict[str, int] = {}
        df: Counter[str] = Counter()
        for d in self.docs:
            toks = type(self).tokenizer(d.full_text)
            self.tf[d.id] = Counter(toks)
            self.len[d.id] = len(toks)
            df.update(set(toks))
        self.df = df
        self.N = len(self.docs)
        self.avgdl = (sum(self.len.values()) / self.N) if self.N else 0.0

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, doc: Document) -> float:
        tf = self.tf[doc.id]
        dl = self.len[doc.id]
        total = 0.0
        for term in type(self).tokenizer(query):
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            total += self.idf(term) * (f * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scored = [(d.id, self.score(query, d)) for d in self.docs]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [Hit(doc_id, sc, i + 1) for i, (doc_id, sc) in enumerate(scored[:k])]


class BM25Channel(_VectorChannel):
    """Pure lexical. The floor you must not fall below — and, on patents,
    a floor that is surprisingly hard to beat out-of-domain."""

    name = "bm25"
    tokenizer = staticmethod(T.tokenize)


class ConceptChannel(_VectorChannel):
    """Lexical + normalised concepts. Stands in for a dense retriever: it
    closes the synonym hole that BM25 structurally cannot close."""

    name = "concept"
    tokenizer = staticmethod(T.concept_tokens)


class StructureChannel:
    """Chemistry, with no text involved.

    Tanimoto on binary fingerprints::

        T = c / (a + b - c)

    and the bound every practitioner should be able to state on demand::

        T <= min(a, b) / max(a, b)

    That bound means any *fixed* similarity threshold is silently also a
    molecular-size filter — a fragment can never be 0.85-similar to a molecule
    twice its size, because the ceiling is 0.5.
    """

    name = "structure"

    def __init__(self, docs: Sequence[Document]) -> None:
        self.docs = [d for d in docs if d.fingerprint]

    @staticmethod
    def tanimoto(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        c = len(a & b)
        return c / (len(a) + len(b) - c)

    @staticmethod
    def size_bound(a: set[str], b: set[str]) -> float:
        """The maximum Tanimoto these two fingerprints could possibly reach."""
        if not a or not b:
            return 0.0
        return min(len(a), len(b)) / max(len(a), len(b))

    def search(self, query_fp: set[str], k: int = 10) -> list[Hit]:
        scored = [
            (d.id, self.tanimoto(query_fp, d.fingerprint))
            for d in self.docs
        ]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [Hit(doc_id, sc, i + 1) for i, (doc_id, sc) in enumerate(scored[:k])]


# ------------------------------------------------------------------- fusion


def reciprocal_rank_fusion(
    channel_hits: dict[str, list[Hit]], k: int = 60
) -> list[tuple[str, float, dict[str, int]]]:
    """RRF: ``score(d) = sum over channels of 1 / (k + rank_c(d))``.

    Why RRF rather than a weighted blend of the raw scores: a BM25 score of 18.3
    and a Tanimoto of 0.7 are not on comparable scales and never will be.  RRF
    only looks at *ranks*, so it needs no calibration and does not have to be
    re-tuned every time the corpus changes.  Learn weights later, from feedback,
    once you actually have labels.
    """
    fused: dict[str, float] = {}
    provenance: dict[str, dict[str, int]] = {}
    for channel, hits in channel_hits.items():
        for h in hits:
            fused[h.doc_id] = fused.get(h.doc_id, 0.0) + 1.0 / (k + h.rank)
            provenance.setdefault(h.doc_id, {})[channel] = h.rank
    ordered = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return [(doc_id, score, provenance[doc_id]) for doc_id, score in ordered]


def run_all_channels(corpus: Corpus, k: int = 10) -> dict[str, list[Hit]]:
    """Query every channel with the disclosure and return per-channel hits."""
    docs = corpus.documents
    disclosure = corpus.disclosure
    return {
        BM25Channel.name: BM25Channel(docs).search(disclosure.text, k=k),
        ConceptChannel.name: ConceptChannel(docs).search(disclosure.text, k=k),
        StructureChannel.name: StructureChannel(docs).search(disclosure.fingerprint, k=k),
    }
