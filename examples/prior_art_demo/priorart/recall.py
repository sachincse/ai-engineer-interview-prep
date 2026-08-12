"""Estimating the recall you cannot observe.

The hardest honest question in prior-art search is not "what did we find?" but
"what did we miss?".  You cannot measure that directly — the missed documents
are, by definition, the ones you did not see.

Capture–recapture answers it the way ecologists count fish: run several
*independent* searches and look at the overlap.  If two searches of similar size
barely overlap, there is a lot you have not seen; if they agree almost
perfectly, you are probably near the ceiling.

The assumption that destroys it — and the reason to raise it before anyone else
does — is **independence**.  If every "channel" is a dense retriever over the
same embedding, they all miss the same synonym-hidden document together, the
singleton count collapses, and the estimator confidently reports that nothing
was missed.  Positive dependence biases the estimate *downward*.  Channel
diversity is therefore a statistical requirement, not an engineering
preference.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class Chao1:
    found: int
    f1: int
    f2: int
    estimated_total: float
    estimated_missing: float
    ci_low: float
    ci_high: float

    def describe(self) -> str:
        return (
            f"found {self.found}; estimated total {self.estimated_total:.1f} "
            f"(95% CI {self.ci_low:.1f}–{self.ci_high:.1f}); "
            f"estimated missed {self.estimated_missing:.1f}"
        )


def chao1(capture_counts: dict[str, int]) -> Chao1:
    """Chao1 lower-bound estimator of the true number of relevant documents.

    ``capture_counts`` maps document id -> how many independent channels found
    it.  With ``f1`` singletons (found by exactly one channel) and ``f2``
    doubletons::

        f0_hat = f1^2 / (2*f2)                if f2 > 0
        f0_hat = f1*(f1-1) / (2*(f2+1))       if f2 == 0     (bias-corrected)
        N_hat  = n + f0_hat

    The confidence interval is **log-normal, not symmetric**: a symmetric
    interval can put the lower bound below the number of documents you have
    already physically seen, which is nonsense.
    """
    counts = Counter(capture_counts.values())
    n = len(capture_counts)
    f1, f2 = counts.get(1, 0), counts.get(2, 0)

    if f2 > 0:
        f0 = (f1 * f1) / (2 * f2)
    else:
        f0 = (f1 * (f1 - 1)) / (2 * (f2 + 1))

    n_hat = n + f0
    if f0 <= 0:
        return Chao1(n, f1, f2, float(n), 0.0, float(n), float(n))

    if f2 > 0:
        r = f1 / f2
        var = f2 * (0.25 * r ** 4 + r ** 3 + 0.5 * r ** 2)
    else:
        var = f0 ** 2  # crude fallback; the f2 == 0 regime is inherently noisy

    denom = (n_hat - n) ** 2
    q = math.exp(1.96 * math.sqrt(math.log(1 + var / denom))) if denom > 0 else 1.0
    return Chao1(
        found=n, f1=f1, f2=f2,
        estimated_total=n_hat,
        estimated_missing=f0,
        ci_low=n + (n_hat - n) / q,
        ci_high=n + (n_hat - n) * q,
    )


def lincoln_petersen(n1: int, n2: int, m: int) -> tuple[float, float]:
    """Two-searcher estimate with the Chapman correction.

        N_hat = (n1+1)(n2+1)/(m+1) - 1

    Chapman's version is near-unbiased for m >~ 7 and, unlike the naive
    ``n1*n2/m``, is defined when the two searches share nothing.
    """
    n_hat = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = ((n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m)) / (((m + 1) ** 2) * (m + 2))
    return n_hat, math.sqrt(max(var, 0.0))


def capture_counts_from_channels(channel_hits: dict[str, list]) -> dict[str, int]:
    """How many distinct channels surfaced each document."""
    counts: Counter[str] = Counter()
    for hits in channel_hits.values():
        for h in {x.doc_id for x in hits}:
            counts[h] += 1
    return dict(counts)
