"""The trivial baseline -- what a competent engineer builds in twenty minutes
with no ML and no LLM. Every headline number in the report is judged against
this floor: an agent that can't beat it is not worth the retrieval/LLM
machinery.

Trivial policy, applied uniformly regardless of message content:
  - classify: always predict the single most frequent intent in the training
    distribution (`playback_error` for hulu_support).
  - draft: always send the brand's own single most common reply-opening verbatim
    (the real, most-frequent canned deflection this brand actually sends --
    NOT one we invented, so the baseline is as strong as "do what the brand
    already mostly does" and not a strawman).
  - triage: fixed policy, either 'always escalate' (maximally safe, zero
    automation) or 'always auto-send' (maximally automated, zero safety) --
    we report BOTH fixed points because they bound the risk-coverage curve
    any real triage policy must beat.
"""
from __future__ import annotations

import json
from collections import Counter


class TrivialClassifier:
    def __init__(self, majority_intent: str) -> None:
        self.majority_intent = majority_intent

    @classmethod
    def fit(cls, intents: list[str]) -> "TrivialClassifier":
        majority = Counter(intents).most_common(1)[0][0]
        return cls(majority)

    def predict(self, texts: list[str]) -> list[str]:
        return [self.majority_intent] * len(texts)


class TrivialReplier:
    def __init__(self, canned_reply: str) -> None:
        self.canned_reply = canned_reply

    @classmethod
    def fit_from_cases(cls, brand_replies: list[str]) -> "TrivialReplier":
        # Group near-duplicate openings (first 6 words) to find the brand's
        # single most repeated canned opening -- a fair "what the brand already
        # does most of the time" reference reply.
        def key(r: str) -> str:
            return " ".join(r.split()[:6]).lower()

        counts = Counter(key(r) for r in brand_replies if r.strip())
        top_key, _ = counts.most_common(1)[0]
        for r in brand_replies:
            if key(r) == top_key:
                return cls(r)
        return cls("We're here to help! Please reach out with more details.")

    def draft(self, customer_text: str) -> str:
        return self.canned_reply


def fixed_triage(policy: str, n: int) -> list[bool]:
    """policy in {'always_escalate', 'always_auto'} -> list of `escalate` bools."""
    if policy == "always_escalate":
        return [True] * n
    if policy == "always_auto":
        return [False] * n
    raise ValueError(policy)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/interim/hulu_support_cases.jsonl")
    ap.add_argument("--weak-labels", default="data/interim/hulu_weak_labels.jsonl")
    args = ap.parse_args()

    cases = [json.loads(line) for line in open(args.cases, encoding="utf-8")]
    weak = {json.loads(line)["case_id"]: json.loads(line)["weak_label"]
            for line in open(args.weak_labels, encoding="utf-8")}
    intents = [weak[c["case_id"]] for c in cases if c["case_id"] in weak]

    clf = TrivialClassifier.fit(intents)
    replier = TrivialReplier.fit_from_cases([c["brand_reply"] for c in cases])
    print(f"trivial majority intent: {clf.majority_intent}")
    print(f"trivial canned reply:    {replier.canned_reply!r}")
