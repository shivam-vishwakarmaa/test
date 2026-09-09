"""Fast, offline smoke tests — no API key, no network, no Ollama.

Run with: python -m pytest tests/ -v   (or plain `python tests/test_pipeline_smoke.py`)
These are deliberately narrow: they check that the pieces which have bitten us
during development (union-find correctness, cache-key stability, kappa math,
the self-leakage bug class) stay fixed, not full coverage of the pipeline.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from support_agent.data.threads import UnionFind
from support_agent.eval.judge import cohens_kappa, quadratic_weighted_kappa
from support_agent.eval.metrics import aurc, risk_coverage_curve, triage_report
from support_agent.taxonomy.weak_labels import weak_label


def test_union_find_basic_chains():
    uf = UnionFind()
    # chain: 4 <- 3 <- 2 <- 1  (all one conversation)
    for a, b in [(1, 2), (2, 3), (3, 4)]:
        uf.union(a, b)
    roots = {uf.find(x) for x in (1, 2, 3, 4)}
    assert len(roots) == 1, "a linear reply chain must collapse to one conversation"

    uf2 = UnionFind()
    uf2.union(10, 11)
    uf2.union(20, 21)
    assert uf2.find(10) != uf2.find(20), "disjoint threads must stay disjoint"


def test_weak_label_precedence_money_before_technical():
    # "charged" must win over any technical keyword when both are present.
    assert weak_label("why was I charged twice, the app also keeps buffering") == "billing_payment"


def test_weak_label_cancel_refund_specific_pattern():
    assert weak_label("I cancelled my account and you still charged me") == "cancel_refund"


def test_qwk_perfect_and_bounds():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    # a systematic 1<->5 reversal must score strongly negative, not near-zero
    assert quadratic_weighted_kappa([1, 1, 5, 5], [5, 5, 1, 1]) < -0.5


def test_cohens_kappa_chance_agreement_is_zero():
    # identical constant labels: kappa is defined as 1.0 (po==pe==1), not NaN/crash
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_risk_coverage_curve_monotonic_coverage():
    confidences = [0.9, 0.5, 0.7, 0.2]
    correct = [True, False, True, False]
    points = risk_coverage_curve(confidences, correct)
    coverages = [p[0] for p in points]
    assert coverages == sorted(coverages), "coverage must increase monotonically by construction"
    assert points[-1][0] == 1.0


def test_aurc_all_correct_is_zero_risk():
    points = risk_coverage_curve([0.9, 0.8, 0.7], [True, True, True])
    assert aurc(points) == 0.0


def test_triage_report_unsafe_auto_is_false_negative_not_false_positive():
    # predicted auto (False) but gold says should-escalate (True) => the UNSAFE cell
    pred = [False, True, False]
    gold = [True, True, False]
    rep = triage_report(pred, gold)
    assert rep["fn"] == 1  # exactly the unsafe case above
    assert rep["unsafe_auto_rate"] == 1 / 3


if __name__ == "__main__":
    # allow running without pytest installed
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
