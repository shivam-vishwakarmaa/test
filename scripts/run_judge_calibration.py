#!/usr/bin/env python
"""Score the 45-item judge calibration set with the LLM judge, then compute
judge-vs-human agreement (quadratic-weighted kappa per rubric criterion).

This is the "evidence of how well your judge agrees with a human" deliverable.
Needs ANTHROPIC_API_KEY (or a populated artifacts/llm_cache/). Cost: a few cents
(45 items x 2 orderings x 1 judge call).

Usage: python scripts/run_judge_calibration.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml

from support_agent.config import load_config
from support_agent.eval.judge import RUBRIC_CRITERIA, judge_human_agreement, judge_reply
from support_agent.llm.client import CacheMiss, client_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--calibration", default="data/golden/judge_calibration.json")
    ap.add_argument("--out", default="artifacts/judge_calibration_results.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    client = client_from_config(cfg)
    judge_model = cfg["judge"]["model"]

    data = json.load(open(args.calibration, encoding="utf-8"))
    items = data["items"]

    judge_scores_by_crit = {c: [] for c in RUBRIC_CRITERIA}
    human_scores_by_crit = {c: [] for c in RUBRIC_CRITERIA}
    per_item = []
    n_miss = 0

    for it in items:
        try:
            jr = judge_reply(client, judge_model, it["customer_text"], [], it["draft"],
                              tag=f"cal:{it['id']}", double_score=True)
        except CacheMiss as e:
            n_miss += 1
            if n_miss == 1:
                print(f"CACHE MISS: {e}")
            continue
        scores = jr.scores
        for c in RUBRIC_CRITERIA:
            judge_scores_by_crit[c].append(round(scores[c]))  # QWK needs integer categories
            human_scores_by_crit[c].append(it["human_scores"][c])
        per_item.append({"id": it["id"], "source": it["source"], "judge_scores": scores,
                          "human_scores": it["human_scores"], "reasoning": jr.reasoning})

    if n_miss:
        print(f"\n{n_miss}/{len(items)} calibration items have no cached judge response.")
        print("Run with ANTHROPIC_API_KEY set to populate the cache, then re-run.")
        if n_miss == len(items):
            return

    agreement = judge_human_agreement(judge_scores_by_crit, human_scores_by_crit)
    print(f"\njudge-vs-human agreement over {len(per_item)} calibration items:")
    print(f"{'criterion':<14}{'QWK':>8}{'exact':>9}{'within_1':>10}")
    for c in RUBRIC_CRITERIA:
        a = agreement[c]
        print(f"{c:<14}{a['qwk']:>8.3f}{a['exact_agreement']:>9.1%}{a['within_1']:>10.1%}")

    overall_qwk = sum(agreement[c]["qwk"] for c in RUBRIC_CRITERIA) / len(RUBRIC_CRITERIA)
    print(f"\nmean QWK across criteria: {overall_qwk:.3f}")
    print("(Landis & Koch heuristic: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, "
          "0.61-0.80 substantial, >0.80 almost perfect)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"agreement": agreement, "mean_qwk": overall_qwk, "per_item": per_item}, fh,
                   ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")
    print(f"usage: {client.usage.summary()}")


if __name__ == "__main__":
    main()
