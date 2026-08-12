#!/usr/bin/env python3
"""End-to-end prior-art novelty screening demo.

    python run_demo.py

No dependencies beyond the Python standard library (3.10+).  The corpus is
twelve fictional documents invented for teaching; nothing here is a real patent.

The pipeline, in the order it runs:

    disclosure
        -> decompose into claim elements          (elements.py)
        -> search 3 independent channels          (retrieval.py)
        -> fuse the rankings with RRF             (retrieval.py)
        -> de-duplicate to patent family          (corpus.py)
        -> apply the EPC date rules               (corpus.py)
        -> build the element x document matrix    (matrix.py)
        -> estimate the recall we cannot observe  (recall.py)
        -> propose design-around hypotheses       (designaround.py)
        -> render evidence for a human            (report.py)

Read the report from the bottom up if you are short of time: section 8 says what
was *not* searched, and that is the part that decides how much the rest is worth.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from priorart import corpus as C          # noqa: E402
from priorart import designaround, recall, report  # noqa: E402
from priorart import matrix as M          # noqa: E402
from priorart import retrieval as R       # noqa: E402
from priorart.elements import DEMO_ELEMENTS  # noqa: E402

HERE = Path(__file__).parent


def _setup_console(argv: list[str]) -> None:
    """Make the report printable on any terminal.

    Prefer UTF-8 so the matrix glyphs render; if the console genuinely cannot
    encode them (some Windows code pages), drop to ASCII markers instead of
    dying with a UnicodeEncodeError halfway through the report.
    """
    from priorart import elements as E
    from priorart import report as REP

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    if "--ascii" in argv:
        E.set_ascii(True)
        REP.set_ascii(True)
        return
    try:
        "●◐○–".encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        E.set_ascii(True)
        REP.set_ascii(True)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    _setup_console(argv)
    corpus = C.load(HERE / "corpus" / "patents.json")

    # 1. Retrieve from every channel independently.
    channel_hits = R.run_all_channels(corpus, k=5)

    # 2. Fuse, then collapse to one representative per DOCDB-simple family.
    fused = R.reciprocal_rank_fusion(channel_hits)
    candidates = corpus.dedup_to_family([doc_id for doc_id, _, _ in fused])[:6]

    # 3. Build the coverage matrix (the date engine runs inside build()).
    matrix = M.build(corpus, DEMO_ELEMENTS, candidates)

    # 4. Estimate what the channels between them did not find.
    chao = recall.chao1(recall.capture_counts_from_channels(channel_hits))

    # 5. Design-around hypotheses, ranked with the point of novelty first.
    pon_ids = {e.id for e in matrix.point_of_novelty()}
    hypotheses = designaround.propose(corpus, DEMO_ELEMENTS, candidates, pon_ids)

    print(report.render(corpus, matrix, channel_hits, fused, chao, hypotheses))

    if "--explain-tanimoto" in argv:
        _explain_tanimoto(corpus)
    return 0


def _explain_tanimoto(corpus: C.Corpus) -> None:
    """Show the size bound that makes any fixed Tanimoto threshold a size filter."""
    print()
    print("  APPENDIX — Tanimoto and its size bound")
    print("  " + "-" * 74)
    q = corpus.disclosure.fingerprint
    print(f"  query fingerprint: {sorted(q)}  (|a| = {len(q)})")
    for doc in corpus.documents:
        fp = doc.fingerprint
        if not fp:
            continue
        t = R.StructureChannel.tanimoto(q, fp)
        bound = R.StructureChannel.size_bound(q, fp)
        print(f"  {doc.id:<22} T={t:.3f}   ceiling min(a,b)/max(a,b)={bound:.3f}"
              f"   |b|={len(fp)}")
    print()
    print("  A fixed threshold such as 0.85 cannot be reached at all by a")
    print("  fingerprint whose size ratio is below 0.85 — so the threshold is")
    print("  silently filtering by molecule size, not only by similarity.")


if __name__ == "__main__":
    raise SystemExit(main())
