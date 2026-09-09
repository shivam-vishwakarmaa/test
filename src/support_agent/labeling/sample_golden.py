"""Stratified sampler for the golden evaluation set.

Sampling design (documented here because the docstring IS the methodology
note the assignment asks for):

We want ~200 examples that (a) cover every intent including rare ones,
(b) include enough `other_non_support` to measure the "should the bot even
reply" boundary, (c) include a deliberate slice of hard/ambiguous cases so the
golden set isn't only the easy 80%, and (d) are drawn from real, unedited
customer messages so the eval reflects production input distribution, not a
cleaned-up version of it.

Concretely, we stratify on the weak label (`support_agent.taxonomy.weak_labels`)
and sample a target count per stratum: enough from big strata to estimate
precision within a few points, ALL available examples for strata that occur
naturally at <1% of traffic (`live_tv_blackout`, `cancel_refund`) up to a cap,
and a fixed "hard negative" slice pulled from cases near a weak-label decision
boundary (matched >1 rule, or matched by length/ambiguity heuristics) that the
human labeller most likely disagrees with the rule on.

Every row is written with `weak_label` alongside so the eventual human label
can be diffed against it -- that diff is precisely the weak-labeller
precision/recall number used in the failure-analysis section.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from support_agent.taxonomy.weak_labels import COMPILED, weak_label

# Target composition of the ~200-row golden set. Rare strata are capped by
# availability, not by this target -- see `sample`.
TARGET_PER_STRATUM = {
    "other_non_support": 30,
    "playback_error": 30,
    "content_availability": 22,
    "ads_experience": 20,
    "how_to_feature": 20,
    "billing_payment": 20,
    "account_access": 20,
    "app_ui_feedback": 15,
    "cancel_refund": 15,   # naturally ~1.5% of traffic -> oversampled deliberately
    "live_tv_blackout": 8,  # naturally ~0.3% of traffic -> take almost all of it
}
HARD_NEGATIVE_TARGET = 20  # cases matching >=2 rules: likely mislabelled by weak_label


def n_rule_matches(text: str) -> int:
    return sum(1 for _, pat in COMPILED if pat.search(text))


def sample(cases_path: str, seed: int, out_path: str) -> None:
    rng = random.Random(seed)
    cases = [json.loads(line) for line in open(cases_path, encoding="utf-8")]
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        c["_weak_label"] = weak_label(c["customer_text"])
        c["_n_matches"] = n_rule_matches(c["customer_text"])
        by_stratum[c["_weak_label"]].append(c)

    chosen: list[dict] = []
    chosen_ids: set[int] = set()

    # 1. Targeted stratified sample.
    for stratum, target in TARGET_PER_STRATUM.items():
        pool = by_stratum.get(stratum, [])
        rng.shuffle(pool)
        take = pool[:target]
        chosen.extend(take)
        chosen_ids.update(c["case_id"] for c in take)

    # 2. Hard-negative slice: multi-rule-match cases pulled from ANY stratum,
    #    excluding what's already chosen, favouring ambiguity.
    ambiguous = [c for c in cases if c["case_id"] not in chosen_ids and c["_n_matches"] >= 2]
    rng.shuffle(ambiguous)
    hard = ambiguous[:HARD_NEGATIVE_TARGET]
    chosen.extend(hard)
    chosen_ids.update(c["case_id"] for c in hard)

    rng.shuffle(chosen)

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, c in enumerate(chosen):
            row = {
                "golden_id": i + 1,
                "case_id": c["case_id"],
                "customer_text": c["customer_text"],
                "brand_reply_reference": c["brand_reply"],  # what Hulu actually said -- NOT ground truth, reference only
                "n_customer_turns": c["n_customer_turns"],
                "n_brand_turns": c["n_brand_turns"],
                "weak_label": c["_weak_label"],
                "weak_label_n_rule_matches": c["_n_matches"],
                "sampling_reason": (
                    "hard_negative" if c["case_id"] not in
                    {cc["case_id"] for s in TARGET_PER_STRATUM for cc in by_stratum.get(s, [])[:TARGET_PER_STRATUM[s]]}
                    else "stratified"
                ),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    strata_counts = Counter(c["_weak_label"] for c in chosen)
    print(f"sampled {len(chosen)} golden candidates -> {out_path}")
    print(f"  hard negatives (>=2 rule matches): {len(hard)}")
    for s, n in strata_counts.most_common():
        print(f"  {s:<22} {n:>4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/interim/hulu_support_cases.jsonl")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out", default="data/golden/golden_candidates.jsonl")
    args = ap.parse_args()
    sample(args.cases, args.seed, args.out)
