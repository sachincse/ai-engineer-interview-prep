# Prior-art novelty screening — a runnable demo

A small, complete, dependency-free implementation of the pipeline described in
[Chapter 50](../../interview/50_prior_art_novelty_system_design.md), so you can
*run* the idea instead of only reading it.

```bash
cd examples/prior_art_demo
python run_demo.py                    # the full evidence report
python run_demo.py --explain-tanimoto # appendix: the Tanimoto size bound
python run_demo.py --ascii            # if your console cannot render ● ◐ ○

python -m unittest discover -s tests -v   # 25 tests, no dependencies
```

Requires Python 3.10+. **No third-party packages.** In production you would swap
the toy pieces for real ones — RDKit for fingerprints, an encoder for the dense
channel, OPSIN for name-to-structure, EPO OPS/PATSTAT for the corpus — but the
control flow, the date logic and the maths are the real thing.

> The corpus is **twelve fictional documents** written for teaching. Nothing in
> it is a real patent, and no conclusion about any real invention follows from it.

## What the demo actually demonstrates

The point is not that it retrieves well on twelve documents. It is that the
**shape** of the system is right, and that shape is driven by four facts about
the domain that most demos ignore:

| Fact about patents | What it forces in the code |
|---|---|
| Novelty needs **one** document disclosing **every** feature (EPC Art. 54) | The primitive is a coverage **matrix**, not a similarity score — `matrix.py` |
| A document filed before but published after your priority date counts for novelty but **never** for inventive step (Art. 54(3)) | A deterministic **date engine** that runs before any scoring — `corpus.py` |
| The same molecule is written four different ways | A **concept layer** on top of the lexical one, plus a structure channel that reads no text at all — `text.py`, `retrieval.py` |
| You cannot see what you missed | **Capture–recapture** across channels that fail independently — `recall.py` |

## The three things worth looking at in the output

**1. A perfect textual match that is not prior art.** `EP3500007A1` recites the
invention almost word for word and tops the fused ranking. It was filed *after*
the priority date, so it is not prior art at all. The report shows it in brackets
and excludes it from every conclusion. A system that ranks on similarity alone
reports it as a flawless anticipation — this is the single most expensive bug
available in this domain, and no embedding can prevent it.

**2. The structure channel finding what the text channels miss.** At `k=5` the
lexical and concept channels never surface `EP3000006A1` or `NPL-JCAT-2019-0112`;
the fingerprint channel ranks them 1st and 3rd. That disagreement is not noise —
it is exactly what makes the recall estimate in section 6 meaningful. If every
channel shared one embedding they would miss the same document together, the
singleton count would collapse, and the estimator would confidently report a
false all-clear.

**3. The point of novelty falling out of the matrix.** No prior-art document
discloses element E5. That single empty row is what the filing currently rests
on, and it is the input to the design-around step — which then reports that E5
is unoccupied in the pool while E3 and E4 sit *inside* ranges already disclosed.

## Files

| File | Does |
|---|---|
| `corpus/patents.json` | Twelve fictional documents + the invention disclosure |
| `priorart/text.py` | Tokenising, concept normalisation, hypernyms |
| `priorart/corpus.py` | Loading, family de-duplication, **the EPC date engine** |
| `priorart/retrieval.py` | BM25, concept and structure channels; Tanimoto; RRF |
| `priorart/elements.py` | Claim-element decomposition, span extraction, range logic |
| `priorart/matrix.py` | The element × document matrix, anticipation, set cover |
| `priorart/recall.py` | Chao1 and Lincoln–Petersen/Chapman |
| `priorart/designaround.py` | White-space mapping over numeric ranges |
| `priorart/report.py` | Evidence rendering |
| `tests/test_priorart.py` | 25 tests, weighted toward the quiet failure modes |

## Deliberate simplifications

Stated plainly, because pretending otherwise is how demos mislead:

- The **dense channel is a stand-in.** It is BM25 over concept-expanded tokens,
  not a neural encoder. It demonstrates *why* you need a second channel that
  closes the synonym hole; it does not demonstrate embedding quality.
- **Fingerprints are feature sets**, not Morgan/ECFP bit vectors. The Tanimoto
  maths and the `T ≤ min(a,b)/max(a,b)` bound are identical; the chemistry is not.
- **No Markush matching.** A claim can cover a compound it never names, and only
  curated generic-structure search finds that. It is the hardest part of real
  chemical prior-art search and it is absent here.
- **Element decomposition is hand-written.** In production an LLM proposes it and
  an attorney edits it before anything is searched — the highest-leverage human
  checkpoint in the system, because everything downstream is conditioned on it.
- **The corpus is twelve documents.** Every retrieval number is therefore a
  demonstration of mechanics, not a benchmark result.

## The line the system never crosses

It reports *evidence*, never a verdict. It does not output "novel", "obvious",
"patentable" or "clear to operate". Those are legal determinations made by
qualified attorneys and examiners. Everything here exists to make one of those
people faster and better-informed, and to be auditable afterwards.
