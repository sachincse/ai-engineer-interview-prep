"""Rendering the evidence report.

The deliverable is **evidence, not a verdict**.  Every assertion the system
makes resolves to a quoted span in a real document with a real date, and the
things it did not search are stated out loud rather than left implied.
"""

from __future__ import annotations

from .corpus import Corpus, PRIOR_ART_54_2, PRIOR_ART_54_3, NOT_PRIOR_ART
from .elements import DISCLOSED, PARTIAL
from .matrix import Matrix

RULE = "=" * 78
THIN = "-" * 78

USE_ASCII = False

#: Last-resort transliteration so the report survives a legacy code page.
_ASCII_MAP = str.maketrans({
    "–": "-", "—": "-", "°": " deg ", "×": "x", "≥": ">=", "≤": "<=",
    "→": "->", "’": "'", "“": '"', "”": '"', "●": "[X]", "◐": "[~]", "○": "[ ]",
})


def set_ascii(enabled: bool) -> None:
    global USE_ASCII
    USE_ASCII = enabled


def _out(s: str) -> str:
    if not USE_ASCII:
        return s
    return s.translate(_ASCII_MAP).encode("ascii", "replace").decode("ascii")


def _wrap(s: str, width: int = 74, indent: str = "      ") -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return ("\n" + indent).join(lines)


def render_matrix(matrix: Matrix, corpus: Corpus) -> str:
    cols = matrix.columns
    label_w = 34
    cell_w = 13

    out: list[str] = []
    header = " " * label_w + "".join(c.doc_id[:cell_w - 1].ljust(cell_w) for c in cols)
    out.append(header)
    out.append(" " * label_w + "".join(
        (c.verdict.status if c.verdict.status != NOT_PRIOR_ART else "NOT ART").ljust(cell_w)
        for c in cols))
    out.append(THIN)
    for e in matrix.elements:
        row = f"  {e.id}  {e.label[:label_w - 6]}".ljust(label_w)
        for c in cols:
            cell = c.cells[e.id]
            mark = cell.symbol if c.verdict.usable_for_novelty else f"({cell.symbol})"
            row += mark.ljust(cell_w)
        out.append(row)
    out.append(THIN)
    tally = "  elements disclosed (● only)".ljust(label_w)
    for c in cols:
        tally += str(len(c.disclosed_ids())).ljust(cell_w)
    out.append(tally)
    tally2 = "  elements disclosed (● or ◐)".ljust(label_w)
    for c in cols:
        tally2 += str(len(c.disclosed_ids(include_partial=True))).ljust(cell_w)
    out.append(tally2)
    out.append("")
    out.append("  ● disclosed    ◐ arguably disclosed    ○ not found")
    out.append("  (brackets) = the document is NOT prior art on the dates, so its")
    out.append("               content is shown but must not be relied on.")
    return "\n".join(out)


def render(corpus: Corpus, matrix: Matrix, channel_hits, fused, chao, hypotheses) -> str:
    d = corpus.disclosure
    o: list[str] = []
    a = o.append

    a(RULE)
    a("  PRIOR-ART SCREENING REPORT — DECISION SUPPORT, NOT A LEGAL OPINION")
    a(RULE)
    a(f"  Disclosure    : {d.id} — {d.title}")
    a(f"  Priority date : {d.priority_date.isoformat()}")
    a(f"  CPC scope     : {', '.join(d.cpc)}")
    a(f"  Corpus        : {len(corpus.documents)} documents (toy corpus, fictional)")
    a("")

    a("  1. RETRIEVAL — each channel searched independently")
    a(THIN)
    for name, hits in channel_hits.items():
        listed = ", ".join(f"{h.doc_id}({h.score:.2f})" for h in hits[:5]) or "no hits"
        a(f"  {name:<10} {listed}")
    a("")
    a("  Note how the channels disagree. That disagreement is the point: it is")
    a("  what makes the recall estimate in section 5 meaningful.")
    a("")

    a("  2. FUSION — reciprocal rank fusion, then de-duplicate to family level")
    a(THIN)
    for doc_id, score, prov in fused[:8]:
        where = ", ".join(f"{k}#{v}" for k, v in sorted(prov.items()))
        a(f"  {doc_id:<22} rrf={score:.4f}   found by: {where}")
    a("")

    a("  3. DATE ENGINE — deterministic, never delegated to a model")
    a(THIN)
    for col in matrix.columns:
        tag = {PRIOR_ART_54_2: "Art.54(2) full prior art",
               PRIOR_ART_54_3: "Art.54(3) NOVELTY ONLY",
               NOT_PRIOR_ART: "NOT PRIOR ART"}[col.verdict.status]
        a(f"  {col.doc_id:<22} {tag}")
        a(f"      {_wrap(col.verdict.reason)}")
    a("")

    a("  4. ELEMENT x DOCUMENT COVERAGE MATRIX")
    a(THIN)
    a(render_matrix(matrix, corpus))
    a("")

    ants = matrix.anticipations()
    if ants:
        a("  >>> ANTICIPATION CANDIDATE(S): " + ", ".join(c.doc_id for c in ants))
        for c in ants:
            a(f"      {c.doc_id} discloses every element and is usable for novelty.")
            a(f"      {_wrap(c.verdict.reason)}")
    else:
        a("  >>> No single prior-art document discloses every element at this")
        a("      recall level. That is NOT a finding of novelty — it is the")
        a("      absence of an anticipation in what was searched.")
    a("")

    pon = matrix.point_of_novelty()
    if pon:
        a("  >>> POINT OF NOVELTY: " + ", ".join(f"{e.id} ({e.label})" for e in pon))
        a("      No document in the pool discloses this. It is what the filing")
        a("      currently rests on, and it is the input to the design-around step.")
    a("")

    cover = matrix.minimal_cover()
    if cover:
        a(f"  >>> INVENTIVE-STEP HYPOTHESIS (Art. 56): {' + '.join(cover)}")
        left = matrix.uncovered_after(cover)
        a(f"      Jointly cover all elements except: {', '.join(sorted(left)) or 'none'}")
        a("      This is a HYPOTHESIS, not a finding. Whether the skilled person")
        a("      WOULD have combined these documents is a legal judgement about")
        a("      motivation that no similarity score expresses.")
    a("")

    a("  5. EVIDENCE — every non-empty cell, with its quoted span")
    a(THIN)
    for col in matrix.columns:
        shown = [c for c in col.cells.values() if c.status in (DISCLOSED, PARTIAL)]
        if not shown:
            continue
        a(f"  {col.doc_id}  [{col.verdict.status}]")
        for cell in shown:
            a(f"    {cell.symbol} {cell.element_id}: \"{_wrap(cell.span, indent='         ')}\"")
            if cell.note:
                a(f"         -> {_wrap(cell.note, indent='            ')}")
        a("")

    a("  6. HOW MUCH DID WE MISS? (capture-recapture across channels)")
    a(THIN)
    a(f"  {chao.describe()}")
    a(f"  singletons f1={chao.f1}  doubletons f2={chao.f2}")
    a("  Assumption that would invalidate this: channel independence. If every")
    a("  channel shared one embedding they would miss the same document")
    a("  together, f1 would collapse, and this would report a false all-clear.")
    a("")

    a("  7. DESIGN-AROUND HYPOTHESES — for counsel, ranked, never a decision")
    a(THIN)
    for h in hypotheses:
        a(f"  {h.element_id}: {_wrap(h.headline)}")
        a(f"      {_wrap(h.evidence)}")
        a(f"      LEGAL: {_wrap(h.legal_flag, indent='             ')}")
        a("")

    a("  8. WHAT WAS NOT SEARCHED — state this every time")
    a(THIN)
    a("  * Applications published within ~18 months of the priority date are")
    a("    structurally invisible to every search system, including this one.")
    a("  * Non-patent literature coverage here is one document; in reality NPL")
    a("    appears in roughly one in five EPO search reports.")
    a("  * No Markush / generic-structure matching: a claim can cover a compound")
    a("    it never names, and only curated generic-structure search finds that.")
    a("  * This is a toy corpus of fictional documents. Nothing here is a real")
    a("    patent, and no conclusion about any real invention follows from it.")
    a("")
    a(RULE)
    a("  Novelty and inventive step are legal determinations. This report ranks")
    a("  evidence for a qualified attorney; it does not decide anything.")
    a(RULE)
    return _out("\n".join(o))
