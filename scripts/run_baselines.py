#!/usr/bin/env python
"""Run the trivial and simple baselines against the golden set and print/save
every headline number the report cites for them. No API key needed -- this
is the part of the pipeline that runs today, deterministically, in under a
minute.

Usage: python scripts/run_baselines.py [--config config/config.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml

from support_agent.baselines.simple import (
    NearestNeighborReplier, SimpleClassifier, simple_triage, taxonomy_handling,
)
from support_agent.baselines.trivial import TrivialClassifier, TrivialReplier, fixed_triage
from support_agent.eval.metrics import (
    aurc, classification_report, reply_automated_checks, risk_coverage_curve, triage_report,
)
from support_agent.retrieval.index import PrecedentIndex


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--taxonomy", default="config/taxonomy.yaml")
    ap.add_argument("--out", default="artifacts/baseline_results.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    taxonomy = yaml.safe_load(open(args.taxonomy, encoding="utf-8"))
    handling_by_intent = taxonomy_handling(taxonomy)
    never_auto = set(cfg["triage"]["never_auto_intents"])

    cases_path = os.path.join(cfg["paths"]["interim_dir"], f"{cfg['brand']}_cases.jsonl")
    brand_short = cfg["brand"].replace("_support", "")
    weak_path = os.path.join(cfg["paths"]["interim_dir"], f"{brand_short}_weak_labels.jsonl")
    golden = load_jsonl(cfg["paths"]["golden"])
    cases = load_jsonl(cases_path)
    weak = {r["case_id"]: r["weak_label"] for r in load_jsonl(weak_path)}

    g_texts = [g["customer_text"] for g in golden]
    g_intents = [g["gold_intent"] for g in golden]
    g_escalate = [g["gold_escalate"] for g in golden]
    g_case_ids = [g["case_id"] for g in golden]
    n = len(golden)

    results = {"n_golden": n}

    # ---------------------------------------------------------------- TRIVIAL
    print("=" * 70)
    print("TRIVIAL BASELINE")
    print("=" * 70)
    train_intents = [weak[c["case_id"]] for c in cases if c["case_id"] in weak]
    trivial_clf = TrivialClassifier.fit(train_intents)
    trivial_replier = TrivialReplier.fit_from_cases([c["brand_reply"] for c in cases])
    trivial_preds = trivial_clf.predict(g_texts)
    trivial_cls_report = classification_report(trivial_preds, g_intents)
    print(f"majority intent: {trivial_clf.majority_intent}")
    print(f"classification accuracy: {trivial_cls_report['accuracy']:.1%}  macro-F1: {trivial_cls_report['macro_f1']:.1%}")

    trivial_drafts = [trivial_replier.draft(t) for t in g_texts]
    trivial_reply_checks = [reply_automated_checks(d, []) for d in trivial_drafts]
    mean_grounding = sum(c["lexical_grounding"] for c in trivial_reply_checks) / n
    print(f"canned reply: {trivial_replier.canned_reply!r}")
    print(f"(automated) mean lexical_grounding vs nothing retrieved: {mean_grounding:.3f} (expected 0 -- no retrieval used)")

    for policy in ["always_escalate", "always_auto"]:
        tri = fixed_triage(policy, n)
        rep = triage_report(tri, g_escalate)
        print(f"triage policy={policy}: auto_send_rate={rep['auto_send_rate']:.1%}  "
              f"unsafe_auto_rate={rep['unsafe_auto_rate']:.1%}  escalate_recall={rep['escalate_recall']:.1%}")
        results[f"trivial_triage_{policy}"] = rep

    results["trivial_classification"] = trivial_cls_report
    results["trivial_majority_intent"] = trivial_clf.majority_intent
    results["trivial_canned_reply"] = trivial_replier.canned_reply
    results["trivial_mean_lexical_grounding"] = mean_grounding

    # ----------------------------------------------------------------- SIMPLE
    print("\n" + "=" * 70)
    print("SIMPLE BASELINE (TF-IDF+LR classifier, nearest-neighbour reply, rule triage)")
    print("=" * 70)
    train_texts = [c["customer_text"] for c in cases if c["case_id"] in weak]
    simple_clf = SimpleClassifier.fit(train_texts, train_intents, seed=cfg["seed"])
    simple_preds, simple_confs = simple_clf.predict_proba_top(g_texts)
    simple_cls_report = classification_report(simple_preds, g_intents)
    print(f"trained on {len(train_texts)} weakly-labelled cases")
    print(f"classification accuracy: {simple_cls_report['accuracy']:.1%}  macro-F1: {simple_cls_report['macro_f1']:.1%}")
    print(f"mean predicted-class confidence: {sum(simple_confs)/n:.3f}")

    idx = PrecedentIndex.load("artifacts/precedent_index")
    nn = NearestNeighborReplier(idx, cfg["embedding"]["backend"], cfg["embedding"]["model"],
                                 os.path.join(cfg["paths"]["interim_dir"], "emb_cache"))
    nn_out = nn.draft_batch(g_texts, exclude_case_ids=g_case_ids)
    nn_drafts = [r[0] for r in nn_out]
    nn_sims = [r[1] for r in nn_out]
    nn_source_ids = [r[2] for r in nn_out]
    # exclude self-match leakage: if the retrieved case IS this golden case, similarity==1.0 exactly
    self_leak = sum(1 for sid, cid in zip(nn_source_ids, g_case_ids) if sid == cid)
    print(f"nearest-neighbour reply: mean top-1 similarity={sum(nn_sims)/n:.3f}  "
          f"self-match leakage (retrieved its own case)={self_leak}/{n}")

    nn_checks = [reply_automated_checks(d, [d]) for d in nn_drafts]  # trivially grounds in itself; see note below
    print("NOTE: for a nearest-neighbour reply, 'lexical_grounding' against its own source is definitionally "
          "1.0 and not informative -- the real quality question is whether that source reply fits THIS query, "
          "which is what mean top-1 similarity and the judge/human read address instead.")

    # selective prediction: if we used the classifier's own confidence to decide
    # coverage (independent of the taxonomy-policy triage rule above), how good
    # a signal is that confidence for "is this classification correct"?
    correct = [p == g for p, g in zip(simple_preds, g_intents)]
    rc_points = risk_coverage_curve(simple_confs, correct)
    rc_aurc = aurc(rc_points)
    print(f"selective-prediction AURC on classifier confidence: {rc_aurc:.4f} (lower is better; "
          f"a random confidence signal gives ~{1 - sum(correct)/n:.4f})")
    results["simple_classifier_aurc"] = rc_aurc
    results["simple_classifier_risk_coverage_curve"] = rc_points

    simple_tri = simple_triage(simple_preds, handling_by_intent)
    simple_tri_report = triage_report(simple_tri, g_escalate)
    print(f"triage (rule-on-predicted-intent): auto_send_rate={simple_tri_report['auto_send_rate']:.1%}  "
          f"unsafe_auto_rate={simple_tri_report['unsafe_auto_rate']:.1%}  "
          f"escalate_recall={simple_tri_report['escalate_recall']:.1%}  "
          f"escalate_precision={simple_tri_report['escalate_precision']:.1%}")

    results["simple_classification"] = simple_cls_report
    results["simple_mean_confidence"] = sum(simple_confs) / n
    results["simple_nn_mean_top1_similarity"] = sum(nn_sims) / n
    results["simple_nn_self_leakage"] = self_leak
    results["simple_triage"] = simple_tri_report

    # dump per-example predictions for downstream judge/report use
    per_example = []
    for i, g in enumerate(golden):
        per_example.append({
            "golden_id": g["golden_id"], "case_id": g["case_id"],
            "customer_text": g["customer_text"], "gold_intent": g["gold_intent"],
            "gold_escalate": g["gold_escalate"],
            "trivial_pred_intent": trivial_preds[i], "trivial_draft": trivial_drafts[i],
            "simple_pred_intent": simple_preds[i], "simple_confidence": simple_confs[i],
            "simple_escalate": simple_tri[i],
            "nn_draft": nn_drafts[i], "nn_similarity": nn_sims[i], "nn_source_case_id": nn_source_ids[i],
        })
    results["per_example"] = per_example

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote full results -> {args.out}")


if __name__ == "__main__":
    main()
