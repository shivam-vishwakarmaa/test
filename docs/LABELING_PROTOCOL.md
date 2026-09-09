# Labeling protocol — golden evaluation set (`data/golden/golden_v1.jsonl`)

## Who labeled this, and why that matters

Every gold label in this repository — intent, flags, and escalation decision
for all 220 golden examples, plus the 45-item judge-calibration scores — was
assigned by a **single annotator**: the author of this repository, working
directly from `config/taxonomy.yaml`. There was no second labeler and no
inter-annotator agreement study. That is a real limitation, not a formality,
and it is treated as one throughout: the "what's misleading about my headline
number" section of `docs/REPORT.md` names it explicitly, and the recomputed
agreement numbers should be read as **test-retest / single-rater** evidence,
not the inter-rater kappa the field normally means by "human agreement." A
second independent labeler on even 50 of these 220 examples would be the
single highest-value addition to this project's evidence base — see "what I'd
do with one more week."

## Sampling

Source: `hulu_support_cases.jsonl` — 14,703 conversations surviving the
cleaning funnel in `support_agent/data/corpus.py` (98.9% retention from
14,868 raw hulu_support threads; see `docs/DECISIONS.md` #3 for why hulu_support
was chosen over larger handles like AmazonHelp or AppleSupport).

`support_agent/labeling/sample_golden.py` draws 220 candidates in two passes:

1. **Stratified by weak label** (`support_agent/taxonomy/weak_labels.py`, a
   hand-written regex classifier over the taxonomy). Target counts per stratum
   range from 30 (common intents, `other_non_support`, `playback_error`) down
   to 8 (`live_tv_blackout`, which is only ~0.3% of raw traffic) — rare
   intents are deliberately oversampled relative to their true frequency so
   the golden set can say something statistically meaningful about them at
   all. **This means the golden set's intent distribution is NOT the true
   traffic distribution** — using it to estimate "X% of tickets are billing
   issues" would be wrong; it exists to measure per-intent quality, not
   traffic share. `docs/REPORT.md` states this plainly in the misleading-number
   section.
2. **20 hard negatives**: cases matching ≥2 weak-label rules simultaneously,
   pulled from whatever stratum they land in. These are the examples most
   likely to expose where the taxonomy's edge rules actually matter (see
   `config/taxonomy.yaml`'s `edge_rules` per intent) and most likely to be
   labeled wrong by a naive rule-based system — which is exactly what
   happened: 20/20 hard negatives disagree with their weak label at a much
   higher rate than the stratified sample (see the per-class precision/recall
   table `support_agent/labeling/build_golden.py` prints).

## What was actually labeled, per example

- **`gold_intent`**: exactly one label from `config/taxonomy.yaml`'s 10
  intents, chosen by reading the message against each intent's `summary`,
  `includes` examples, and `edge_rules`. Where two intents plausibly applied
  (e.g. a cancellation *threat* used as leverage inside a UI complaint), the
  taxonomy's edge rule was followed and the underlying issue was labeled, not
  the threat — this single rule accounts for a large share of the
  golden/weak-label disagreements and is called out per-example in
  `annotator_note` wherever it applied.
- **`gold_flags`**: zero or more of `angry`, `churn_risk`, `repeat_contact`,
  `outage_signal`, `needs_account_data`, per the definitions in
  `config/taxonomy.yaml`.
- **`gold_escalate`**: whether a human should see this case before any reply
  goes out. Computed in two layers, both auditable: (1) the annotator's direct
  judgment (`gold_escalate_hand`), and (2) a fixed, code-level override in
  `build_golden.py` that forces `escalate=True` whenever `angry` or
  `churn_risk` is flagged, regardless of intent or the annotator's hand
  decision — applied uniformly in code specifically so this policy is
  consistent across all 220 rows rather than subject to per-row inconsistency.
- **`annotator_note`**: free text, used for exactly three things — flagging a
  disagreement with the weak label (and why), flagging a genuine taxonomy gap
  (e.g. caption-translation-quality complaints, which fit neither
  `playback_error` nor `how_to_feature` cleanly — see golden_id 98), and
  flagging a real data-quality artifact found while labeling (golden_id 106's
  and 165's real historical brand replies are themselves bad — one appears to
  answer an unrelated conversation, the other ignores an answer sitting in
  the brand's own retrievable precedents — both are used as intentional
  low-anchors in the judge calibration set instead of being discarded).

## Reproducing the merge

```
python -m support_agent.labeling.sample_golden      # -> golden_candidates.jsonl (needs a fresh sample; not required, output already committed)
python -m support_agent.labeling.build_golden        # merges labels_batch{1..4}.json onto the candidates -> golden_v1.jsonl
```

`build_golden.py` also prints the weak-label-vs-gold agreement table used in
the failure analysis — rerun it any time to regenerate that diagnostic from
the frozen label files.

## Judge calibration set (`data/golden/judge_calibration.json`)

45 items: 30 real historical Hulu replies (paired with precedents actually
retrieved by our own index) plus 15 synthetic "bad" drafts written by the same
annotator at the low end of the scale (fabricated promises, ignored context,
factually wrong claims, one hard safety violation). The synthetic half exists
because real historical replies cluster in the 3–5 range; a kappa computed
over a range-restricted sample is close to meaningless, and forcing the scale
to actually be used is the standard fix. See `docs/REPORT.md` section 4 for
the resulting judge/human agreement numbers (populated once
`scripts/run_judge_calibration.py` is run with an API key).
