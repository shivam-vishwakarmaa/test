"""Merge candidate rows + hand-label batches into the frozen golden set.

Labelling protocol (see docs/LABELING_PROTOCOL.md for the full writeup):
each of the 220 sampled candidates was read against config/taxonomy.yaml by a
single annotator and assigned an intent, zero or more flags, and an escalation
decision, with a free-text note on any judgment call worth recording (taxonomy
edge cases, disagreement with the weak/rule-based label, genuine ambiguity).

This script:
  1. Joins the label batches onto the sampled candidates.
  2. Applies ONE deterministic escalation override on top of the hand label:
     if 'angry' or 'churn_risk' is flagged, escalate is forced True regardless
     of intent, because reputational risk should route to a human even for an
     otherwise auto-answerable intent. This rule is applied in code (not by
     hand per-row) so it is consistent and auditable -- see docs/DECISIONS.md.
  3. Computes and prints weak-label vs gold-label agreement (accuracy,
     per-class precision/recall) -- this is the number used in the report's
     "what a rule-based system gets wrong" failure analysis.
  4. Writes the final data/golden/golden_v1.jsonl, frozen (never edited after
     this point -- a v2 would be a new file, so every reported number stays
     reproducible against the exact set that produced it).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

FORCE_ESCALATE_FLAGS = {"angry", "churn_risk"}


def load_batches(paths: list[str]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            merged.update(json.load(fh))
    return merged


def build(candidates_path: str, batch_paths: list[str], out_path: str) -> None:
    candidates = [json.loads(l) for l in open(candidates_path, encoding="utf-8")]
    labels = load_batches(batch_paths)

    missing = [c["golden_id"] for c in candidates if str(c["golden_id"]) not in labels]
    if missing:
        raise SystemExit(f"missing hand labels for golden_id(s): {missing}")

    rows = []
    for c in candidates:
        lab = labels[str(c["golden_id"])]
        flags = lab["flags"]
        escalate = bool(lab["escalate"]) or any(f in FORCE_ESCALATE_FLAGS for f in flags)
        rows.append({
            **c,
            "gold_intent": lab["intent"],
            "gold_flags": flags,
            "gold_escalate": escalate,
            "gold_escalate_hand": lab["escalate"],
            "annotator_note": lab["note"],
        })

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- weak-label diagnostic -------------------------------------------------
    n = len(rows)
    correct = sum(1 for r in rows if r["weak_label"] == r["gold_intent"])
    print(f"wrote {n} golden rows -> {out_path}")
    print(f"\nweak-label (rule-based) vs gold-label agreement: {correct}/{n} = {correct/n:.1%}")

    per_class_tp: Counter = Counter()
    per_class_fp: Counter = Counter()
    per_class_fn: Counter = Counter()
    for r in rows:
        w, g = r["weak_label"], r["gold_intent"]
        if w == g:
            per_class_tp[g] += 1
        else:
            per_class_fp[w] += 1
            per_class_fn[g] += 1

    classes = sorted(set(r["gold_intent"] for r in rows) | set(r["weak_label"] for r in rows))
    print(f"\n{'intent':<22}{'gold_n':>7}{'precision':>11}{'recall':>9}")
    for cls in classes:
        gold_n = sum(1 for r in rows if r["gold_intent"] == cls)
        tp, fp, fn = per_class_tp[cls], per_class_fp[cls], per_class_fn[cls]
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"{cls:<22}{gold_n:>7}{prec:>11.1%}{rec:>9.1%}")

    esc_n = sum(r["gold_escalate"] for r in rows)
    print(f"\ngold escalate=True: {esc_n}/{n} = {esc_n/n:.1%}")

    intent_counts = Counter(r["gold_intent"] for r in rows)
    print(f"\ngold intent distribution:")
    for cls, cnt in intent_counts.most_common():
        print(f"  {cls:<22}{cnt:>4}  {cnt/n:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/golden/golden_candidates.jsonl")
    ap.add_argument("--batches", nargs="+", default=[
        "data/golden/labels_batch1.json",
        "data/golden/labels_batch2.json",
        "data/golden/labels_batch3.json",
        "data/golden/labels_batch4.json",
    ])
    ap.add_argument("--out", default="data/golden/golden_v1.jsonl")
    args = ap.parse_args()
    build(args.candidates, args.batches, args.out)
