"""Tests for the prior-art demo.

    python -m unittest discover -s tests -v        (no dependencies)
    pytest tests -q                                (if you have pytest)

These are deliberately weighted toward the places where a prior-art system goes
quietly wrong: punctuation defeating the concept layer, a number being attributed
to the wrong chemical, and a document that matches perfectly but is not prior art
because of its dates.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from priorart import corpus as C          # noqa: E402
from priorart import designaround, recall  # noqa: E402
from priorart import matrix as M          # noqa: E402
from priorart import retrieval as R       # noqa: E402
from priorart import text as T            # noqa: E402
from priorart.elements import (           # noqa: E402
    DEMO_ELEMENTS, DISCLOSED, NOT_FOUND, PARTIAL, assess,
)

CORPUS_PATH = ROOT / "corpus" / "patents.json"


class TestText(unittest.TestCase):
    def test_decimal_points_survive_but_sentence_periods_do_not(self):
        self.assertIn("0.5", T.normalise("Cerium at 0.5 wt%."))
        # The regression that silently broke a whole retrieval channel:
        self.assertIn(" gamma alumina ", " " + T.normalise("supported on gamma-alumina.") + " ")

    def test_all_four_spellings_of_the_support_resolve_to_one_concept(self):
        for surface in [
            "gamma-alumina",
            "gamma-aluminium oxide",
            "aluminium oxide of the gamma phase",
            "Al2O3 carrier of the gamma phase",
        ]:
            with self.subTest(surface=surface):
                self.assertIn("gamma_alumina", T.concepts(surface))

    def test_hypernyms_expand(self):
        cs = T.concepts("palladium on gamma-alumina")
        self.assertIn("noble_metal", cs)   # pd -> noble_metal
        self.assertIn("oxide_support", cs)  # gamma_alumina -> alumina -> oxide_support


class TestDateEngine(unittest.TestCase):
    def setUp(self):
        self.corpus = C.load(CORPUS_PATH)
        self.priority = self.corpus.disclosure.priority_date

    def test_published_before_priority_is_full_prior_art(self):
        v = C.classify_date(self.corpus.by_id("JP2015000005A"), self.priority)
        self.assertEqual(v.status, C.PRIOR_ART_54_2)
        self.assertTrue(v.usable_for_novelty)
        self.assertTrue(v.usable_for_inventive_step)

    def test_filed_before_published_after_is_novelty_only(self):
        v = C.classify_date(self.corpus.by_id("EP3000006A1"), self.priority)
        self.assertEqual(v.status, C.PRIOR_ART_54_3)
        self.assertTrue(v.usable_for_novelty)
        # EPC Art. 56 expressly excludes Art. 54(3) documents.
        self.assertFalse(v.usable_for_inventive_step)

    def test_filed_after_priority_is_not_prior_art_at_all(self):
        v = C.classify_date(self.corpus.by_id("EP3500007A1"), self.priority)
        self.assertEqual(v.status, C.NOT_PRIOR_ART)
        self.assertFalse(v.usable_for_novelty)

    def test_family_dedup_keeps_the_first_ranked_member_only(self):
        kept = self.corpus.dedup_to_family(["EP1000001A1", "US9000002B2", "JP2015000005A"])
        self.assertEqual(kept, ["EP1000001A1", "JP2015000005A"])


class TestElements(unittest.TestCase):
    def setUp(self):
        self.corpus = C.load(CORPUS_PATH)
        self.by_id = {e.id: e for e in DEMO_ELEMENTS}

    def test_encompassing_range_counts_as_disclosed(self):
        doc = self.corpus.by_id("EP3000006A1")     # cerium 0.35-0.75 wt%
        cell = assess(self.by_id["E4"], doc.full_text)
        self.assertEqual(cell.status, DISCLOSED)
        self.assertEqual(cell.found_range, (0.35, 0.75))
        self.assertIn("selection-invention", cell.note)

    def test_overlapping_but_not_containing_range_is_only_partial(self):
        doc = self.corpus.by_id("US8500003B1")     # 190-260 C vs claimed 180-220
        cell = assess(self.by_id["E3"], doc.full_text)
        self.assertEqual(cell.status, PARTIAL)

    def test_disjoint_range_is_not_found(self):
        doc = self.corpus.by_id("WO2019000004A1")  # cerium 0.1-0.3 vs claimed 0.4-0.6
        cell = assess(self.by_id["E4"], doc.full_text)
        self.assertEqual(cell.status, NOT_FOUND)

    def test_wrong_metal_does_not_satisfy_the_element(self):
        doc = self.corpus.by_id("WO2019000004A1")  # nickel, not palladium
        self.assertEqual(assess(self.by_id["E1"], doc.full_text).status, NOT_FOUND)

    def test_number_is_attributed_to_the_nearest_chemical_not_any_chemical(self):
        """'nickel (5 wt%) with cerium (0.2 wt%)' must not yield a 5 wt% cerium
        loading. Nearest-cue-wins is what stops phantom occupied territory from
        leaking into the white-space map."""
        doc = self.corpus.by_id("WO2019000004A1")
        ws = designaround.map_white_space(self.corpus, self.by_id["E4"], [doc.id])
        assert ws is not None
        values = {(o.low, o.high) for o in ws.occupied}
        self.assertIn((0.1, 0.3), values)
        self.assertNotIn((5.0, 5.0), values)


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.corpus = C.load(CORPUS_PATH)

    def test_tanimoto_and_its_size_bound(self):
        a = {"Pd", "Ce", "Al2O3"}
        b = {"Pd", "Ce", "Al2O3"}
        self.assertAlmostEqual(R.StructureChannel.tanimoto(a, b), 1.0)
        small, big = {"Pd"}, {"Pd", "Ce", "Al2O3", "gamma_phase"}
        t = R.StructureChannel.tanimoto(small, big)
        bound = R.StructureChannel.size_bound(small, big)
        self.assertLessEqual(t, bound)
        # A 1-feature fragment can never be 0.85-similar to a 4-feature molecule.
        self.assertLess(bound, 0.85)

    def test_structure_channel_finds_what_the_text_channels_miss(self):
        """The whole argument for channel diversity, in one assertion."""
        hits = R.run_all_channels(self.corpus, k=5)
        text_ids = {h.doc_id for h in hits["bm25"]} | {h.doc_id for h in hits["concept"]}
        struct_ids = {h.doc_id for h in hits["structure"]}
        self.assertTrue(struct_ids - text_ids, "structure channel added nothing")

    def test_rrf_rewards_documents_found_by_several_channels(self):
        hits = {
            "a": [R.Hit("D1", 9.0, 1), R.Hit("D2", 8.0, 2)],
            "b": [R.Hit("D2", 9.0, 1), R.Hit("D3", 8.0, 2)],
        }
        fused = R.reciprocal_rank_fusion(hits)
        self.assertEqual(fused[0][0], "D2")           # found by both channels
        self.assertEqual(set(fused[0][2]), {"a", "b"})


class TestRecall(unittest.TestCase):
    def test_chao1_reproduces_the_published_worked_example(self):
        """van Dijk et al.: f1=40, f2=33, n=92 -> N_hat = 92 + 1600/66 ~= 116.24"""
        counts = {f"one-{i}": 1 for i in range(40)}
        counts.update({f"two-{i}": 2 for i in range(33)})
        counts.update({f"three-{i}": 3 for i in range(19)})
        est = recall.chao1(counts)
        self.assertEqual(est.found, 92)
        self.assertEqual((est.f1, est.f2), (40, 33))
        self.assertAlmostEqual(est.estimated_total, 116.24, places=2)
        self.assertLess(est.ci_low, est.estimated_total)
        self.assertGreater(est.ci_high, est.estimated_total)
        # The interval must be asymmetric, and never below what we already saw.
        self.assertGreaterEqual(est.ci_low, est.found)

    def test_chapman_is_defined_when_the_two_searches_share_nothing(self):
        n_hat, sd = recall.lincoln_petersen(10, 10, 0)
        self.assertGreater(n_hat, 100)     # no overlap => huge unseen population
        self.assertGreater(sd, 0)


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.corpus = C.load(CORPUS_PATH)
        hits = R.run_all_channels(self.corpus, k=5)
        fused = R.reciprocal_rank_fusion(hits)
        self.candidates = self.corpus.dedup_to_family([d for d, _, _ in fused])[:6]
        self.matrix = M.build(self.corpus, DEMO_ELEMENTS, self.candidates)

    def test_no_anticipation_is_found(self):
        self.assertEqual(self.matrix.anticipations(), [])

    def test_the_post_priority_document_is_excluded_despite_matching_perfectly(self):
        """EP3500007A1 discloses the invention almost word for word. A system
        ranking on similarity alone would call it a perfect anticipation. It was
        filed after the priority date, so it is not prior art at all."""
        col = next(c for c in self.matrix.columns if c.doc_id == "EP3500007A1")
        self.assertFalse(col.verdict.usable_for_novelty)
        self.assertGreaterEqual(len(col.disclosed_ids()), 3)
        self.assertNotIn(col, self.matrix.anticipations())

    def test_point_of_novelty_is_the_hydrogen_ratio(self):
        self.assertEqual([e.id for e in self.matrix.point_of_novelty()], ["E5"])

    def test_54_3_document_is_excluded_from_the_inventive_step_cover(self):
        cover = self.matrix.minimal_cover()
        self.assertNotIn("EP3000006A1", cover)


class TestDesignAround(unittest.TestCase):
    def setUp(self):
        self.corpus = C.load(CORPUS_PATH)
        hits = R.run_all_channels(self.corpus, k=5)
        fused = R.reciprocal_rank_fusion(hits)
        self.candidates = self.corpus.dedup_to_family([d for d, _, _ in fused])[:6]
        self.by_id = {e.id: e for e in DEMO_ELEMENTS}

    def test_unoccupied_range_is_flagged_as_the_strongest_differentiator(self):
        ws = designaround.map_white_space(self.corpus, self.by_id["E5"], self.candidates)
        assert ws is not None
        self.assertFalse(ws.overlaps_occupied)
        self.assertTrue(ws.claimed_sits_in_gap)

    def test_range_inside_a_broader_disclosure_is_flagged_as_not_differentiating(self):
        ws = designaround.map_white_space(self.corpus, self.by_id["E3"], self.candidates)
        assert ws is not None
        self.assertTrue(ws.fully_inside_occupied)

    def test_hypotheses_rank_the_point_of_novelty_first(self):
        matrix = M.build(self.corpus, DEMO_ELEMENTS, self.candidates)
        pon = {e.id for e in matrix.point_of_novelty()}
        hyps = designaround.propose(self.corpus, DEMO_ELEMENTS, self.candidates, pon)
        self.assertTrue(hyps)
        self.assertIn(hyps[0].element_id, pon)


class TestEndToEnd(unittest.TestCase):
    def test_demo_runs_clean_and_reports_the_right_headlines(self):
        sys.path.insert(0, str(ROOT))
        import io
        import contextlib
        import run_demo

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_demo.main([])
        out = buf.getvalue()

        self.assertEqual(rc, 0)
        self.assertIn("No single prior-art document discloses every element", out)
        self.assertIn("POINT OF NOVELTY", out)
        self.assertIn("NOT PRIOR ART", out)
        self.assertIn("Art.54(3) NOVELTY ONLY", out)
        self.assertIn("WHAT WAS NOT SEARCHED", out)
        # The report must never assert the legal conclusion.
        self.assertNotIn("is novel", out.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
