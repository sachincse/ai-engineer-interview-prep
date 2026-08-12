"""Loading, family de-duplication, and the date engine.

The date engine is the part people skip, and it is the part that decides whether
the answer is legally meaningful at all.  It is deliberately *deterministic*: no
model is involved, because no embedding represents "was filed before my priority
date but published after it".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import text as T

# ---------------------------------------------------------------- data model


def _d(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


@dataclass
class Document:
    id: str
    family_id: str
    kind: str
    title: str
    filing_date: date | None
    priority_date: date | None
    publication_date: date | None
    cpc: list[str]
    sections: dict[str, str]
    compound: dict | None

    @property
    def full_text(self) -> str:
        return " ".join(
            f"{self.title}. " + v for v in self.sections.values() if v
        ) or self.title

    @property
    def fingerprint(self) -> set[str]:
        """Toy structural fingerprint: the set of substructure feature tokens.

        A real system uses Morgan/ECFP bit vectors from RDKit.  The *shape* of
        the maths is identical — Tanimoto over two sets of set bits — which is
        all this demo needs to make the size-bound point honestly.
        """
        if not self.compound:
            return set()
        return set(self.compound.get("features", []))


@dataclass
class Disclosure:
    id: str
    title: str
    priority_date: date
    cpc: list[str]
    text: str
    compound: dict | None

    @property
    def fingerprint(self) -> set[str]:
        if not self.compound:
            return set()
        return set(self.compound.get("features", []))


# ------------------------------------------------------------- date verdicts

#: Why a document is (or is not) available as prior art against a disclosure.
PRIOR_ART_54_2 = "54(2)"      # published before the priority date: full prior art
PRIOR_ART_54_3 = "54(3)"      # filed before, published on/after: novelty only
NOT_PRIOR_ART = "not-prior-art"


@dataclass
class DateVerdict:
    status: str
    reason: str

    @property
    def usable_for_novelty(self) -> bool:
        return self.status in (PRIOR_ART_54_2, PRIOR_ART_54_3)

    @property
    def usable_for_inventive_step(self) -> bool:
        # EPC Art. 56 expressly excludes Art. 54(3) documents.
        return self.status == PRIOR_ART_54_2


def classify_date(doc: Document, priority: date) -> DateVerdict:
    """Apply the EPC date rules to one document.

    * published before the priority date            -> Art. 54(2), full prior art
    * filed before but published on/after           -> Art. 54(3), novelty only
    * filed on/after the priority date              -> not prior art at all
    """
    pub = doc.publication_date
    filed = doc.filing_date or doc.priority_date

    if pub is not None and pub < priority:
        return DateVerdict(
            PRIOR_ART_54_2,
            f"published {pub.isoformat()}, before the priority date {priority.isoformat()}",
        )
    if filed is not None and filed < priority:
        return DateVerdict(
            PRIOR_ART_54_3,
            f"filed {filed.isoformat()} (before priority) but published "
            f"{pub.isoformat() if pub else 'later'} — novelty only, never inventive step",
        )
    return DateVerdict(
        NOT_PRIOR_ART,
        f"filed {filed.isoformat() if filed else '?'}, on or after the priority date "
        f"{priority.isoformat()} — not prior art",
    )


# ------------------------------------------------------------------- loading


@dataclass
class Corpus:
    disclosure: Disclosure
    documents: list[Document] = field(default_factory=list)

    def by_id(self, doc_id: str) -> Document:
        for d in self.documents:
            if d.id == doc_id:
                return d
        raise KeyError(doc_id)

    def dedup_to_family(self, doc_ids: list[str]) -> list[str]:
        """Collapse to one representative per DOCDB-simple family, keeping the
        best-ranked member.  Showing a reviewer six members of one family as six
        separate hits is the cheapest way to destroy their trust in the tool."""
        seen: set[str] = set()
        out: list[str] = []
        for did in doc_ids:
            fam = self.by_id(did).family_id
            if fam in seen:
                continue
            seen.add(fam)
            out.append(did)
        return out


def load(path: str | Path) -> Corpus:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    d = raw["disclosure"]
    disclosure = Disclosure(
        id=d["id"],
        title=d["title"],
        priority_date=date.fromisoformat(d["priority_date"]),
        cpc=d["cpc"],
        text=d["text"],
        compound=d.get("compound"),
    )
    docs = [
        Document(
            id=x["id"],
            family_id=x["family_id"],
            kind=x["kind"],
            title=x["title"],
            filing_date=_d(x.get("filing_date")),
            priority_date=_d(x.get("priority_date")),
            publication_date=_d(x.get("publication_date")),
            cpc=x.get("cpc", []),
            sections=x.get("sections", {}),
            compound=x.get("compound"),
        )
        for x in raw["documents"]
    ]
    return Corpus(disclosure=disclosure, documents=docs)


__all__ = [
    "Corpus", "Document", "Disclosure", "DateVerdict", "load", "classify_date",
    "PRIOR_ART_54_2", "PRIOR_ART_54_3", "NOT_PRIOR_ART", "T",
]
