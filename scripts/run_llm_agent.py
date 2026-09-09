#!/usr/bin/env python
"""Run the full LLM agent (classify -> retrieve -> draft -> triage) over the
golden set, then run the LLM judge over every draft, then compute judge/human
agreement on the calibration set.

Default backend is 'gemini' (config/config.yaml), needs GEMINI_API_KEY in
.env or the environment. On this submission's free-tier key the run costs
$0 (see Decision #18 in docs/DECISIONS.md); on a paid Anthropic key at the
default models (Haiku 4.5 agent, Opus 5 judge) budget ~$2-5 for the
220-example golden set + 45-item calibration set. Every response is cached
to artifacts/llm_cache/ as it's produced, so this is safe to interrupt and
re-run -- it will only pay for what's missing.

Usage:
  python scripts/run_llm_agent.py                     # backend from config.yaml (gemini)
  LLM_BACKEND=anthropic python scripts/run_llm_agent.py
  LLM_BACKEND=cached python scripts/run_llm_agent.py   # replay committed cache, no network/cost
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml
from dotenv import load_dotenv

from support_agent.agent.pipeline import run_case
from support_agent.baselines.simple import taxonomy_handling
from support_agent.config import load_config
from support_agent.eval.judge import RUBRIC_CRITERIA, judge_reply
from support_agent.eval.metrics import classification_report, triage_report
from support_agent.llm.client import CacheMiss, QuotaExhausted, client_from_config
from support_agent.retrieval.embed import embed_texts
from support_agent.retrieval.index import PrecedentIndex


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main():
    load_dotenv()  # picks up GEMINI_API_KEY / ANTHROPIC_API_KEY from .env if present
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--taxonomy", default="config/taxonomy.yaml")
    ap.add_argument("--out", default="artifacts/llm_agent_results.json")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N golden rows (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    taxonomy = yaml.safe_load(open(args.taxonomy, encoding="utf-8"))
    handling_by_intent = taxonomy_handling(taxonomy)
    never_auto = set(cfg["triage"]["never_auto_intents"])

    client = client_from_config(cfg)
    agent_model = cfg["agent"]["model"]
    judge_model = cfg["judge"]["model"]
    k = cfg["retrieval"]["k"]

    golden = load_jsonl(cfg["paths"]["golden"])
    if args.limit:
        golden = golden[: args.limit]
    idx = PrecedentIndex.load("artifacts/precedent_index")

    texts = [g["customer_text"] for g in golden]
    qvecs = embed_texts(texts, backend=cfg["embedding"]["backend"], model=cfg["embedding"]["model"],
                         cache_dir=os.path.join(cfg["paths"]["interim_dir"], "emb_cache"))

    print(f"running agent ({agent_model}) over {len(golden)} golden examples...")
    outputs = []
    n_cache_miss = 0
    n_error = 0
    quota_hit: str | None = None
    for i, (g, v) in enumerate(zip(golden, qvecs)):
        if quota_hit:
            # Stop calling the model entirely once a daily quota is confirmed
            # exhausted -- every remaining item would fail the identical way
            # after burning max_retries x backoff apiece for nothing.
            outputs.append(None)
            continue
        try:
            out = run_case(
                client, agent_model, taxonomy, idx, g["customer_text"], v, k,
                handling_by_intent, never_auto,
                cfg["triage"]["min_confidence"], cfg["triage"]["min_grounding"],
                case_id=g["case_id"], exclude_case_id=g["case_id"], tag=f"golden{g['golden_id']}",
            )
            outputs.append(out)
        except CacheMiss as e:
            n_cache_miss += 1
            if n_cache_miss == 1:
                print(f"\nCACHE MISS on example {i}: {e}\n")
            outputs.append(None)
        except QuotaExhausted as e:
            quota_hit = str(e)
            print(f"\nSTOPPING at example {i}/{len(golden)}: {e}\n")
            outputs.append(None)
        except RuntimeError as e:
            # A single item's API call failed for a reason that ISN'T a
            # confirmed daily-quota wall (e.g. one transient network error
            # that outlasted all retries) -- log it and keep going rather
            # than losing every already-completed example in the batch.
            n_error += 1
            print(f"  [error on example {i}, skipping]: {e}")
            outputs.append(None)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(golden)}  ({client.usage.summary()})")

    if n_cache_miss:
        print(f"\n{n_cache_miss}/{len(golden)} examples had no cached LLM response.")
        print("Run with GEMINI_API_KEY set (LLM_BACKEND=gemini) to populate the cache, then re-run.")
        if n_cache_miss == len(golden):
            print("(0 cached responses found at all -- this is expected on a fresh clone; "
                  "see README for the one command to populate the cache.)")
            return
    if n_error:
        print(f"\n{n_error}/{len(golden)} examples failed with a non-quota error and were skipped.")
    if quota_hit:
        print(f"{sum(1 for o in outputs if o is None) - n_cache_miss - n_error}/{len(golden)} "
              "examples were never attempted (quota exhausted before reaching them).")

    valid = [(g, o) for g, o in zip(golden, outputs) if o is not None]
    print(f"\n{len(valid)}/{len(golden)} examples completed. {client.usage.summary()}")

    # ------------------------------------------------------------- classification
    preds = [o.classify.intent for _, o in valid]
    golds = [g["gold_intent"] for g, _ in valid]
    cls_report = classification_report(preds, golds)
    print(f"\nagent classification accuracy: {cls_report['accuracy']:.1%}  macro-F1: {cls_report['macro_f1']:.1%}")

    # ------------------------------------------------------------------- triage
    pred_escalate = [o.triage.escalate for _, o in valid]
    gold_escalate = [g["gold_escalate"] for g, _ in valid]
    tri_report = triage_report(pred_escalate, gold_escalate)
    print(f"agent triage: auto_send_rate={tri_report['auto_send_rate']:.1%}  "
          f"unsafe_auto_rate={tri_report['unsafe_auto_rate']:.1%}  "
          f"escalate_recall={tri_report['escalate_recall']:.1%}  "
          f"escalate_precision={tri_report['escalate_precision']:.1%}")

    results = {
        "n_golden": len(golden), "n_completed": len(valid),
        "classification": cls_report, "triage": tri_report,
        "usage": {"calls": client.usage.calls, "cache_hits": client.usage.cache_hits,
                  "cost_usd": client.usage.cost_usd},
        "per_example": [],
    }
    for g, o in valid:
        results["per_example"].append({
            "golden_id": g["golden_id"], "case_id": g["case_id"],
            "customer_text": g["customer_text"],
            "gold_intent": g["gold_intent"], "pred_intent": o.classify.intent,
            "pred_confidence": o.classify.confidence, "pred_flags": o.classify.flags,
            "gold_escalate": g["gold_escalate"], "pred_escalate": o.triage.escalate,
            "triage_reasons": o.triage.reasons,
            "draft": o.draft.reply, "cannot_ground": o.draft.cannot_ground,
            "n_precedents_used": len(o.draft.used_precedent_indices),
        })

    # ----------------------------------------------------------------------- judge
    if not args.skip_judge:
        print(f"\nrunning judge ({judge_model}) over {len(valid)} drafts...")
        judge_scores = []  # parallel to results["per_example"]; None where judging didn't happen
        judge_quota_hit = False
        for i, (g, o) in enumerate(valid):
            if judge_quota_hit:
                judge_scores.append(None)
                continue
            prec_dicts = [{"customer_text": p.customer_text, "brand_reply": p.brand_reply} for p in o.draft.precedents]
            try:
                jr = judge_reply(client, judge_model, g["customer_text"], prec_dicts, o.draft.reply,
                                  tag=f"golden{g['golden_id']}", double_score=cfg["judge"]["double_score_for_consistency"])
                judge_scores.append(jr)
            except QuotaExhausted as e:
                print(f"\nSTOPPING judge at example {i}/{len(valid)}: {e}\n")
                judge_quota_hit = True
                judge_scores.append(None)
            except RuntimeError as e:
                print(f"  [judge error on example {i}, skipping]: {e}")
                judge_scores.append(None)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(valid)}  ({client.usage.summary()})")

        scored = [jr for jr in judge_scores if jr is not None]
        for row, jr in zip(results["per_example"], judge_scores):
            if jr is None:
                continue
            row["judge_scores"] = jr.scores
            row["judge_order_sensitivity"] = jr.order_sensitivity
            row["judge_reasoning"] = jr.reasoning

        if scored:
            mean_scores = {c: sum(jr.scores[c] for jr in scored) / len(scored) for c in RUBRIC_CRITERIA}
            mean_sensitivity = {c: sum(jr.order_sensitivity[c] for jr in scored) / len(scored) for c in RUBRIC_CRITERIA}
            print(f"\nmean judge scores (n={len(scored)}/{len(valid)}):", {k: round(v, 2) for k, v in mean_scores.items()})
            print("mean order-sensitivity (0=stable):", {k: round(v, 2) for k, v in mean_sensitivity.items()})
            results["judge_mean_scores"] = mean_scores
            results["judge_order_sensitivity"] = mean_sensitivity
            results["judge_n_scored"] = len(scored)
        else:
            print("\nno drafts were judged (quota exhausted immediately).")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")
    print(f"final usage: {client.usage.summary()}")


if __name__ == "__main__":
    main()
