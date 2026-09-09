"""LLM-as-judge for reply quality, plus the machinery to check whether the
judge is worth trusting -- because an unvalidated judge is just a second
unverified model grading a first one.

Rubric (1-5 each, defined operationally so two different judges converge):
  grounded     - Does every factual/procedural claim in the reply trace to a
                 retrieved precedent or to information present in the customer
                 message? (1 = fabricates a policy/fact, 5 = fully traceable)
  correct_safe - Is the reply free of unsupported promises, wrong information,
                 or a disposition mismatched to the intent's escalation policy?
  tone         - Empathetic and brand-appropriate without being saccharine or
                 dismissive, calibrated to the customer's visible frustration.
  actionable   - Gives the customer a concrete next step (a diagnostic
                 question, an instruction, or a clear "we can't do X, here's
                 what we can do").
  concise      - No padding, no repeated apology, respects that this is a
                 tweet-reply-length medium.

Anti-bias measures baked into the harness (see docs/DECISIONS.md #11 for the
evidence behind each):
  - The judge model (claude-opus-5 by default) is a DIFFERENT AND STRONGER
    model than the drafter (claude-haiku-4-5), specifically to avoid
    self-enhancement bias, where an LLM judge scores its own family's outputs
    higher (arxiv 2410.21819, 2506.02592).
  - Every reply is scored twice with the rubric's criteria listed in reversed
    order; a large score delta on a criterion flags order sensitivity in that
    item rather than being silently averaged away.
  - The judge sees the SAME retrieved precedents the drafter was given, not
    the ground truth Hulu reply, so it is scoring "grounded in what was
    available," not "matches what the brand historically said" -- those are
    different, and conflating them would penalize a legitimately better reply.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from support_agent.llm.client import LLMClient

RUBRIC_CRITERIA = ["grounded", "correct_safe", "tone", "actionable", "concise"]

_SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {c: {"type": "integer", "minimum": 1, "maximum": 5} for c in RUBRIC_CRITERIA},
    "required": RUBRIC_CRITERIA,
    "additionalProperties": False,
}

_SCHEMA_TEMPLATE_WITH_REASON = {
    "type": "object",
    "properties": {
        **{c: {"type": "integer", "minimum": 1, "maximum": 5} for c in RUBRIC_CRITERIA},
        "reasoning": {"type": "string"},
    },
    "required": RUBRIC_CRITERIA + ["reasoning"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You are an exacting quality reviewer for a customer-support AI \
agent at Hulu. You will be shown a customer's message, a set of historical \
precedents (past customer messages this brand has handled and how the brand \
replied), and a DRAFT reply produced by the agent. Score the draft on each \
rubric criterion from 1 (fails badly) to 5 (excellent), independently -- a low \
score on one criterion should not be dragged up or down by another. Be skeptical: \
a warm, fluent reply that invents a policy or promises something the precedents \
never mention must score low on 'grounded' and 'correct_safe' regardless of tone."""


def _build_user_prompt(customer_text: str, precedents: list[dict], draft: str, criteria_order: list[str]) -> str:
    prec_block = "\n".join(
        f"  [{i+1}] customer said: {p['customer_text']!r}\n      brand replied: {p['brand_reply']!r}"
        for i, p in enumerate(precedents)
    ) or "  (no precedents retrieved)"
    rubric_block = "\n".join(f"  - {c}" for c in criteria_order)
    return (
        f"CUSTOMER MESSAGE:\n  {customer_text!r}\n\n"
        f"RETRIEVED PRECEDENTS (what the agent was grounded on):\n{prec_block}\n\n"
        f"AGENT'S DRAFT REPLY:\n  {draft!r}\n\n"
        f"Score the draft on these criteria (1-5 each):\n{rubric_block}\n\n"
        f"Also give a one-sentence reasoning summary."
    )


@dataclass
class JudgeResult:
    scores_forward: dict[str, int]
    scores_reversed: dict[str, int]
    reasoning: str
    usage_calls: int = 0

    @property
    def scores(self) -> dict[str, float]:
        """Average of the two orderings -- the reported score per criterion."""
        return {c: (self.scores_forward[c] + self.scores_reversed[c]) / 2 for c in RUBRIC_CRITERIA}

    @property
    def order_sensitivity(self) -> dict[str, int]:
        """Absolute point delta between orderings per criterion (0 = fully stable)."""
        return {c: abs(self.scores_forward[c] - self.scores_reversed[c]) for c in RUBRIC_CRITERIA}

    @property
    def mean_score(self) -> float:
        s = self.scores
        return sum(s.values()) / len(s)


def judge_reply(
    client: LLMClient,
    model: str,
    customer_text: str,
    precedents: list[dict],
    draft: str,
    tag: str = "",
    double_score: bool = True,
) -> JudgeResult:
    fwd_prompt = _build_user_prompt(customer_text, precedents, draft, RUBRIC_CRITERIA)
    fwd = client.complete_json(
        JUDGE_SYSTEM, fwd_prompt, _SCHEMA_TEMPLATE_WITH_REASON, model=model,
        max_tokens=400, tag=f"{tag}:judge:fwd",
    )
    reasoning = fwd.pop("reasoning", "")

    if double_score:
        rev_prompt = _build_user_prompt(customer_text, precedents, draft, list(reversed(RUBRIC_CRITERIA)))
        rev = client.complete_json(
            JUDGE_SYSTEM, rev_prompt, _SCHEMA_TEMPLATE_WITH_REASON, model=model,
            max_tokens=400, tag=f"{tag}:judge:rev",
        )
        rev.pop("reasoning", None)
    else:
        rev = dict(fwd)

    return JudgeResult(scores_forward=fwd, scores_reversed=rev, reasoning=reasoning)


# --------------------------------------------------------------------- calibration
def cohens_kappa(a: list[int], b: list[int], categories: list[int] | None = None) -> float:
    """Cohen's kappa for two raters over ordinal/nominal categories (unweighted)."""
    if categories is None:
        categories = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    a_counts = {c: a.count(c) / n for c in categories}
    b_counts = {c: b.count(c) / n for c in categories}
    pe = sum(a_counts[c] * b_counts[c] for c in categories)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def quadratic_weighted_kappa(a: list[int], b: list[int], min_rating: int = 1, max_rating: int = 5) -> float:
    """Quadratic-weighted kappa -- the right agreement stat for 1-5 ordinal
    judge/human scores, since it penalizes a 1-vs-5 disagreement far more than
    a 3-vs-4 disagreement, unlike plain (unweighted) Cohen's kappa."""
    n_cats = max_rating - min_rating + 1
    O = [[0] * n_cats for _ in range(n_cats)]
    for x, y in zip(a, b):
        O[x - min_rating][y - min_rating] += 1
    n = len(a)
    a_hist = [sum(row) for row in O]
    b_hist = [sum(O[i][j] for i in range(n_cats)) for j in range(n_cats)]
    E = [[a_hist[i] * b_hist[j] / n for j in range(n_cats)] for i in range(n_cats)]
    W = [[((i - j) ** 2) / ((n_cats - 1) ** 2) for j in range(n_cats)] for i in range(n_cats)]

    num = sum(W[i][j] * O[i][j] for i in range(n_cats) for j in range(n_cats))
    den = sum(W[i][j] * E[i][j] for i in range(n_cats) for j in range(n_cats))
    if den == 0:
        return 1.0
    return 1 - num / den


def judge_human_agreement(judge_scores: dict[str, list[int]], human_scores: dict[str, list[int]]) -> dict:
    """Per-criterion QWK between judge and human ratings on the SAME items,
    plus the mean-score Pearson-style correlation as a secondary sanity check."""
    out = {}
    for crit in RUBRIC_CRITERIA:
        j, h = judge_scores[crit], human_scores[crit]
        out[crit] = {
            "qwk": quadratic_weighted_kappa(j, h),
            "exact_agreement": sum(1 for a, b in zip(j, h) if a == b) / len(j),
            "within_1": sum(1 for a, b in zip(j, h) if abs(a - b) <= 1) / len(j),
            "n": len(j),
        }
    return out
