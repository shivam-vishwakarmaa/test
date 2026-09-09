"""Automated metrics: classification, triage/selective-prediction, and reply
quality proxies that don't need an LLM judge.

These run today, with no API key, against the golden set -- they are the
"automated metrics" half of the eval harness the brief asks for. The LLM-judge
rubric (the other half) lives in eval/judge.py.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


def classification_report(preds: list[str], golds: list[str]) -> dict:
    classes = sorted(set(golds) | set(preds))
    n = len(golds)
    acc = sum(p == g for p, g in zip(preds, golds)) / n

    per_class = {}
    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(1 for g in golds if g == c)}
        f1s.append(f1)

    return {
        "accuracy": acc,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "n": n,
        "per_class": per_class,
    }


def confusion_matrix(preds: list[str], golds: list[str]) -> dict[str, Counter]:
    cm: dict[str, Counter] = {}
    for p, g in zip(preds, golds):
        cm.setdefault(g, Counter())[p] += 1
    return cm


@dataclass
class TriageOutcome:
    """One row's triage classification against the 2x2 cost structure.

    The costly cell is auto_send_wrong: the agent sent a reply on its own
    authority for a case a human labeller said should have been escalated.
    escalate_wrong (over-escalation) wastes agent time but is not unsafe.
    """
    true_positive: int   # predicted escalate, gold escalate  (correctly caught)
    false_positive: int  # predicted escalate, gold auto      (over-escalation, wasted human time)
    false_negative: int  # predicted auto,     gold escalate  (UNSAFE: auto-sent something that needed a human)
    true_negative: int   # predicted auto,     gold auto      (correctly automated)


def triage_report(pred_escalate: list[bool], gold_escalate: list[bool]) -> dict:
    tp = sum(1 for p, g in zip(pred_escalate, gold_escalate) if p and g)
    fp = sum(1 for p, g in zip(pred_escalate, gold_escalate) if p and not g)
    fn = sum(1 for p, g in zip(pred_escalate, gold_escalate) if not p and g)
    tn = sum(1 for p, g in zip(pred_escalate, gold_escalate) if not p and not g)
    n = len(gold_escalate)

    auto_send_rate = (fn + tn) / n  # fraction the system would send with no human in the loop
    unsafe_auto_rate = fn / n       # fraction of ALL traffic that is an unsafe auto-send
    catch_rate = tp / (tp + fn) if (tp + fn) else float("nan")  # recall on "should escalate"
    precision = tp / (tp + fp) if (tp + fp) else float("nan")   # of what we escalate, how much needed it

    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "auto_send_rate": auto_send_rate,
        "unsafe_auto_rate": unsafe_auto_rate,
        "escalate_recall": catch_rate,
        "escalate_precision": precision,
    }


def risk_coverage_curve(confidences: list[float], correct: list[bool]) -> list[tuple[float, float, float]]:
    """Sort by descending confidence; at each threshold, report
    (coverage, selective_risk, threshold). Used to compute AURC.

    coverage = fraction of examples the system would auto-answer at this
    threshold; selective_risk = error rate among ONLY those auto-answered
    examples (lower is better; this is what "how bad are the ones we let
    through" measures, distinct from overall accuracy).
    """
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    n = len(confidences)
    points = []
    wrong_so_far = 0
    for rank, i in enumerate(order, start=1):
        if not correct[i]:
            wrong_so_far += 1
        coverage = rank / n
        risk = wrong_so_far / rank
        points.append((coverage, risk, confidences[i]))
    return points


def aurc(points: list[tuple[float, float, float]]) -> float:
    """Trapezoidal area under the risk-coverage curve, sorted by coverage."""
    pts = sorted(points, key=lambda p: p[0])
    area = 0.0
    prev_cov, prev_risk = 0.0, pts[0][1] if pts else 0.0
    for cov, risk, _ in pts:
        area += (cov - prev_cov) * (risk + prev_risk) / 2
        prev_cov, prev_risk = cov, risk
    return area


# --------------------------------------------------------------------------- reply proxies
_ABSOLUTE_PROMISE_RE = re.compile(
    r"\b(?:guarantee[ds]?|promise[ds]?|will definitely|100%|will (?:be )?fix(?:ed)? by|"
    r"refund(?:ed)? (?:immediately|right away)|will never happen again)\b", re.IGNORECASE)

_PROFANITY_RE = re.compile(r"\b(?:fuck|shit|damn|hell|bitch|asshole)\w*\b", re.IGNORECASE)


def reply_automated_checks(draft: str, precedents: list[str]) -> dict:
    """Cheap, deterministic proxies computed without any LLM call.

    - lexical_grounding: max n-gram (4-gram) overlap ratio between the draft
      and any retrieved precedent, as a floor-level "did this come from
      somewhere real" signal. It cannot detect paraphrased grounding (that is
      exactly what the LLM judge's 'groundedness' criterion is for), but a
      score of 0 here is a hard, uncontestable red flag.
    - has_unsupported_promise: regex flag for absolute commitments the agent
      is not authorized to make ("guaranteed", "100%", specific fix ETAs).
    - has_profanity: safety floor -- should always be False.
    - word_count.
    """
    draft_words = draft.lower().split()

    def ngrams(words, n=4):
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    draft_grams = ngrams(draft_words)
    best_overlap = 0.0
    if draft_grams:
        for prec in precedents:
            prec_grams = ngrams(prec.lower().split())
            if not prec_grams:
                continue
            overlap = len(draft_grams & prec_grams) / len(draft_grams)
            best_overlap = max(best_overlap, overlap)

    return {
        "word_count": len(draft_words),
        "lexical_grounding": best_overlap,
        "has_unsupported_promise": bool(_ABSOLUTE_PROMISE_RE.search(draft)),
        "has_profanity": bool(_PROFANITY_RE.search(draft)),
    }
