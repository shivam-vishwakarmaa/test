# Hulu AI Support Agent — take-home submission

An AI support agent for **`hulu_support`** (chosen from the Customer Support on Twitter
dataset by a computed metric, not by volume — see Decision #1 in [`docs/DECISIONS.md`](docs/DECISIONS.md)) that
classifies incoming customer messages into a 10-intent taxonomy, drafts a reply grounded in
retrieved historical precedent, and decides auto-handle vs. escalate with a stated reason.

**Read this first:** [`docs/REPORT.md`](docs/REPORT.md) has the actual results, two baselines,
failure analysis, and the mandatory "what's misleading about my headline number" section.
This README is reproduction instructions and a map of the repo.

## Reproduce the headline numbers — under 1 minute, no API key, no download

Every non-LLM number in the report (both baselines, retrieval quality, selective-prediction
AURC, the weak-label-vs-gold diagnostic) is reproducible from artifacts already committed to
this repo — no need to touch the 493MB raw dataset or run any model.

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .    # puts src/support_agent on the path so `python -m support_agent....` works

python -m support_agent.retrieval.index \
  --cases data/interim/hulu_support_cases.jsonl --out-prefix artifacts/precedent_index
  # rebuilds the retrieval index from the COMMITTED embedding cache: a pure cache hit,
  # a few seconds, no network, no Ollama needed (this is what `make index` does)

python scripts/run_baselines.py
```

This prints (and writes to `artifacts/baseline_results.json`) the trivial baseline, the simple
(TF-IDF+LR + nearest-neighbor retrieval) baseline, and every metric in report section 2.1–2.2.
On Linux/Mac with `make` installed, `make index && make baselines` does the same thing.

## Reproduce the LLM-agent numbers — needs `ANTHROPIC_API_KEY`, ~10–15 min, ~$2–5

```bash
export ANTHROPIC_API_KEY=sk-...
export LLM_BACKEND=anthropic   # config.yaml defaults to 'cached' (never spends money unless asked) -- this opts in explicitly
python scripts/run_llm_agent.py            # classify + retrieve + draft + triage over all 220 golden examples, then judges every draft
python scripts/run_judge_calibration.py    # scores the 45-item human-calibration set, prints judge-vs-human quadratic-weighted kappa
```

Every model response is cached to `artifacts/llm_cache/` **as it's produced** (content-addressed,
keyed by the exact request — see Decision #14 in [`docs/DECISIONS.md`](docs/DECISIONS.md)), so:

- Both scripts are safe to interrupt and re-run; you only pay for what's missing.
- Once `artifacts/llm_cache/` is committed (after one real run), anyone can replay the exact
  recorded outputs with **zero cost and zero network calls**: `LLM_BACKEND=cached python
  scripts/run_llm_agent.py` (or `make llm-agent-cached`). This is what actually makes the LLM
  numbers reproducible by a grader who doesn't want to spend their own API budget.
- Run with no key and an empty cache and you get a clear, actionable error naming exactly what's
  missing — not a crash and not a silently wrong number.

## Rebuild everything from the raw dataset (optional — not needed for the numbers above)

```bash
mkdir -p data/raw && curl -sL -o data/raw/twcs.csv \
  https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv
  # 493MB, a verified mirror of the exact Kaggle thoughtvector/customer-support-on-twitter
  # twcs.csv (byte-identical header/rows checked during development) -- no Kaggle account or
  # API key needed. This is also `make download`.

python -m support_agent.data.build_brand_dataset --brands hulu_support --out-dir data/interim
  # reconstructs 14,868 hulu_support conversation threads from the flat 3M-row tweet table
  # via union-find over the reply graph (support_agent/data/threads.py)

python -m support_agent.data.corpus --brand hulu_support --in-dir data/interim
  # cleans + filters -> 14,703 cases (98.9% retention; funnel printed on run)

python -m support_agent.taxonomy.weak_labels \
  --cases data/interim/hulu_support_cases.jsonl --out data/interim/hulu_weak_labels.jsonl

python -m support_agent.retrieval.index \
  --cases data/interim/hulu_support_cases.jsonl --backend ollama --out-prefix artifacts/precedent_index
  # embeds all 14,703 customer openings via a local Ollama nomic-embed-text model (no API key,
  # fully offline once pulled: `ollama pull nomic-embed-text`). Takes ~15-20 minutes on CPU the
  # FIRST time; every subsequent run is a per-text cache hit (Decision #9) and takes seconds.
  # No Ollama? Pass --backend tfidf for a fully offline, deterministic (if lower-quality) fallback.
```

The golden set itself (`data/golden/golden_v1.jsonl`) is frozen and committed — see
[`docs/LABELING_PROTOCOL.md`](docs/LABELING_PROTOCOL.md) for exactly how it was sampled and
labeled, and who labeled it (a disclosure that matters — read it).

## What's in this repo

```
config/
  config.yaml       # every knob: brand, thresholds, model choices, retrieval k
  taxonomy.yaml     # the 10-intent taxonomy: definitions, edge rules, handling policy per intent
src/support_agent/
  data/             # thread reconstruction (union-find), brand learnability survey, corpus cleaning
  taxonomy/         # weak (rule-based) labeler used for stratification + distant supervision
  retrieval/        # embeddings (Ollama + offline TF-IDF fallback, per-text cached), precedent index
  baselines/        # trivial (majority-class + canned reply) and simple (TF-IDF+LR + kNN reply)
  agent/            # the real pipeline: classify -> retrieve -> draft -> triage
  llm/              # Anthropic client with a content-addressed, replayable on-disk cache
  eval/             # automated metrics (classification, selective-prediction/AURC, reply proxies)
                    # + LLM-judge rubric with anti-bias measures + human-agreement (QWK) computation
  labeling/         # golden-set sampler + the merge/build script for the hand labels
data/
  golden/           # golden_v1.jsonl (220 examples), judge_calibration.json (45 items), labels
  interim/          # cleaned corpus, weak labels, embedding cache (committed; raw CSV is not)
scripts/            # run_baselines.py, run_llm_agent.py, run_judge_calibration.py
docs/
  REPORT.md              # the actual report: framing, results, failure analysis, misleading-number section, next steps
  DECISIONS.md            # 15 non-obvious decisions and why
  LABELING_PROTOCOL.md    # golden-set sampling + labeling method, and who labeled it
```

## A note on how this was built

This repository, including the taxonomy, the golden-set labels, and the report, was produced
with AI assistance (Claude) working from the assignment brief, under the repo owner's direction.
The golden-set labels in particular were assigned by a single annotator following the written
codebook in `config/taxonomy.yaml` — see the disclosure at the top of
[`docs/LABELING_PROTOCOL.md`](docs/LABELING_PROTOCOL.md) before treating them as independently
verified ground truth, and consider spot-checking a sample against your own reading before
relying on them further.
