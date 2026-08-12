"""Tokenisation and concept normalisation.

The single most important idea in this file: in chemistry, the *same thing* is
written many different ways.  A lexical index that has never seen the word
"alumina" cannot match a document that says "Al2O3 carrier of the gamma phase",
no matter how good its scoring function is.

So we keep two views of every document:

  * the **surface** tokens  -> what BM25 searches
  * the **concept** tokens  -> surface tokens plus canonical concept ids,
                               which is what the semantic channel searches

In production the concept layer is a chemical NER + name-to-structure pipeline
(OPSIN, a synonym dictionary, an embedding model).  Here it is a small hand
written table, because the point of the demo is the *architecture*, not the
extractor.
"""

from __future__ import annotations

import re
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

# Surface phrase -> canonical concept id.  Longest phrases are matched first so
# that "gamma-aluminium oxide" wins over a bare "aluminium".
SYNONYMS: dict[str, str] = {
    # support
    "gamma alumina": "gamma_alumina",
    "gamma aluminium oxide": "gamma_alumina",
    "gamma aluminum oxide": "gamma_alumina",
    "aluminium oxide of the gamma phase": "gamma_alumina",
    "al2o3 carrier of the gamma phase": "gamma_alumina",
    "al2o3": "alumina",
    "alumina": "alumina",
    "aluminium oxide": "alumina",
    "refractory oxide": "oxide_support",
    "activated carbon": "carbon_support",
    # metals
    "palladium": "pd",
    "pd": "pd",
    "nickel": "ni",
    "platinum": "pt",
    "noble metal": "noble_metal",
    # promoters
    "cerium": "ce",
    "ce": "ce",
    "lanthanum": "la",
    "praseodymium": "pr",
    "rare earth": "rare_earth",
    "promoter": "promoter",
    # process
    "hydrogenation": "hydrogenation",
    "hydrogenating": "hydrogenation",
    "unsaturated aldehyde": "unsaturated_aldehyde",
    "hydrogen to substrate molar ratio": "h2_ratio",
    "temperature": "temperature",
    "degrees c": "temperature",
}

# Concepts that imply other concepts.  "gamma_alumina" is a kind of "alumina",
# which is a kind of "oxide_support".  Cheap stand-in for an ontology.
HYPERNYMS: dict[str, tuple[str, ...]] = {
    "gamma_alumina": ("alumina", "oxide_support"),
    "alumina": ("oxide_support",),
    "carbon_support": ("support",),
    "oxide_support": ("support",),
    "pd": ("noble_metal",),
    "pt": ("noble_metal",),
    "ce": ("rare_earth", "promoter"),
    "la": ("rare_earth", "promoter"),
    "pr": ("rare_earth", "promoter"),
}

_PHRASES = sorted(SYNONYMS, key=len, reverse=True)


def normalise(text: str) -> str:
    """Lowercase and flatten punctuation so phrase matching is predictable.

    Decimal points are preserved ("0.5 wt%") but sentence-ending periods are
    not.  This matters more than it looks: leave the full stop attached and
    "supported on gamma-alumina." never matches the phrase "gamma alumina",
    so the concept layer silently loses the document.  A whole retrieval
    channel can be defeated by punctuation.
    """
    flat = re.sub(r"[^a-z0-9.]+", " ", text.lower())
    flat = re.sub(r"\.(?!\d)|(?<!\d)\.", " ", flat)   # keep 0.5, drop "alumina."
    return re.sub(r"\s+", " ", flat).strip()


def tokenize(text: str) -> list[str]:
    """Surface tokens. This is what a lexical (BM25) index sees."""
    return _TOKEN_RE.findall(normalise(text))


def concepts(text: str) -> list[str]:
    """Canonical concept ids found in the text, expanded by hypernym."""
    flat = " " + normalise(text) + " "
    found: list[str] = []
    for phrase in _PHRASES:
        if " " + phrase + " " in flat:
            found.append(SYNONYMS[phrase])
    expanded: list[str] = []
    for c in found:
        expanded.append(c)
        expanded.extend(HYPERNYMS.get(c, ()))
    # stable de-duplication
    seen: set[str] = set()
    out: list[str] = []
    for c in expanded:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def concept_tokens(text: str) -> list[str]:
    """Surface tokens PLUS concept ids — what the semantic channel indexes."""
    return tokenize(text) + ["@" + c for c in concepts(text)]


def sentences(text: str) -> list[str]:
    """Split into sentences. Evidence spans are quoted at sentence granularity
    so a reviewer can read the disclosure in context."""
    parts = re.split(r"(?<=[.;])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
