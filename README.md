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

## Reproduce the LLM-agent numbers — no key needed by default (a real cache is already committed)

```bash
LLM_BACKEND=cached python scripts/run_llm_agent.py          # replays the real, cached Gemini responses
LLM_BACKEND=cached python scripts/run_judge_calibration.py  # ditto, for the judge-calibration set
```

Every model response is cached to `artifacts/llm_cache/` **as it's produced** (content-addressed,
keyed by the exact request — see Decision #14 in [`docs/DECISIONS.md`](docs/DECISIONS.md)), and
that cache is committed. `LLM_BACKEND=cached` (the `config.yaml` default) replays it byte-for-byte
with zero cost and zero network calls — this is what actually makes the LLM numbers reproducible by
a grader who doesn't want to spend their own API budget. Run with no key and an empty cache and you
get a clear, actionable error naming exactly what's missing, not a crash or a silently wrong number.

**Read this before assuming the LLM numbers are as complete as the baseline numbers: they aren't,
on purpose and for a stated reason.** The committed cache holds a real but small sample — 5/220
golden examples and 5/45 judge-calibration items — because this submission's Gemini API key has a
hard free-tier cap of **20 requests/day per model** (confirmed from the live API's error body, not
assumed; see Decision #19 in `DECISIONS.md`), not the "10–15 minutes" this section used to promise
before that limit was discovered by actually running it. Section 2.3 and section 4 of
[`docs/REPORT.md`](docs/REPORT.md) report this exact number honestly and explain what can and can't
be concluded from a sample that size. To extend it:

```bash
export GEMINI_API_KEY=...          # get a free key at https://aistudio.google.com/apikey
                                    # -- or copy .env.example to .env and put it there instead
make llm-agent                     # continues from cache; ~10 more examples/day on this tier
make judge-calibration              # same, for the calibration set
# ANTHROPIC_API_KEY set instead? `make llm-agent-anthropic` / `-judge-calibration-anthropic`
# restores the originally-designed Haiku 4.5 (drafter) / Opus 5 (judge) pairing -- see Decision #18.
```

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

## Architecture

Two things worth stating up front, because they shape every box in the diagram below: **this is a
batch/offline pipeline, not a deployed service** (a deliberate scope choice: the assignment asks for
a runnable pipeline and a report, not a live API — building an unrequested FastAPI+Docker+k8s stack
around it would be scope creep dressed up as rigor); and **classification, retrieval, drafting, and triage are
four separately-callable stages, not one prompt** (Decision #6) specifically so each stage's errors
are independently measurable — the whole point of the eval harness on the right-hand side.

```mermaid
flowchart TD
    raw[("twcs.csv — 3M tweets\n(Kaggle Customer Support on Twitter)")]
    raw --> threads["thread reconstruction\nunion-find over in_response_to_tweet_id\ndata/threads.py"]
    threads --> corpus["clean + filter\ndata/corpus.py"]
    corpus --> cases[("hulu_support_cases.jsonl\n14,703 cases")]

    cases --> weak["weak / rule-based labeler\ntaxonomy/weak_labels.py"]
    cases --> embedstep["embed customer openings\nOllama nomic-embed-text (cached)\nretrieval/embed.py"]
    embedstep --> index[("precedent_index\n14,697 retrievable precedents\n(deflection-only replies excluded)")]
    cases --> sampler["stratified golden sampling\nlabeling/sample_golden.py"]
    sampler --> golden[("golden_v1.jsonl\n220 hand-labeled examples")]
    weak --> simple["simple baseline\nTF-IDF+LR + kNN reply\nbaselines/simple.py"]

    golden --> classify
    subgraph pipeline["agent pipeline — support_agent/agent/pipeline.py"]
        direction TB
        classify["1 classify (LLM)\nintent + confidence + flags"]
        retrieve["2 retrieve\ncosine top-k over precedent_index"]
        draft["3 draft (LLM)\nreply grounded ONLY in retrieved precedents"]
        triage["4 triage — DETERMINISTIC RULE\n(not an LLM call — Decision #6)"]
        classify --> retrieve --> draft --> triage
    end
    index -.k=6.-> retrieve
    triage -->|auto_answer / policy_answer,\nhigh confidence, grounded,\nnot angry / churn| autosend(["auto-send"])
    triage -->|everything else,\nincl. angry/churn override\n(Decision #11)| human(["escalate to human\n+ drafted reply attached"])

    llmclient["LLMClient — gemini / anthropic / ollama / cached\ncontent-addressed on-disk cache, retries,\nQuotaExhausted vs. per-item-error handling\nllm/client.py"]
    classify -.-> llmclient
    draft -.-> llmclient

    subgraph evalharness["evaluation harness"]
        direction TB
        judge["LLM judge (different model,\ndouble-scored, order-reversed)\neval/judge.py"]
        metrics["classification / triage /\nselective-prediction (AURC)\neval/metrics.py"]
        calib[("judge_calibration.json\n45 items, human-scored")]
        qwk["judge-vs-human\nquadratic-weighted kappa"]
        judge --> qwk
        calib --> qwk
    end
    draft -.-> judge
    classify --> metrics
    triage --> metrics
    judge -.-> llmclient

    metrics --> report[["docs/REPORT.md\nresults · failure analysis ·\nmisleading-number section"]]
    qwk --> report
```

**Why triage is a plain function and not a fifth LLM call:** the thing that decides whether a human
ever sees a case has to be auditable as a rule with named reasons ("intent is on the
never-auto list", "confidence 0.61 < threshold 0.75"), not another model's opinion you'd then have
to trust recursively. `agent/pipeline.py::triage` is ~25 lines and every reason it can emit is
enumerable by reading it — that auditability is a requirement here, not a nicety, given the brief
asks for a stated reason on every escalation decision.

**If this went from a batch pipeline to a live service** (deliberately not built — see the top of
this section): the LLM-facing pieces (`agent/pipeline.py`, `llm/client.py`) already have no
dependency on argparse, stdin, or the filesystem beyond the retrieval index and taxonomy config, so
they'd wrap into a queue consumer or a thin FastAPI handler without restructuring — `run_case()` is
already the exact unit of work a ticket-arrival event would call. What such a deployment would still
need that this repo intentionally doesn't build: an idempotency key per ticket (so a retried queue
message doesn't double-draft), a human-review UI for the escalate path (`triage.reasons` is already
structured for exactly this), production observability on the four-stage latency/error breakdown the
architecture above already separates, and a feedback loop from human edits back into the retrieval
corpus (today's precedent pool is static, built once from historical data).

## What's in this repo

```
.github/workflows/ci.yml  # lint + unit tests + the free/offline path, on every push (no API key used)
Dockerfile, .dockerignore # containerized free/offline path -- `docker build . && docker run <image>`
.env.example              # copy to .env; GEMINI_API_KEY / ANTHROPIC_API_KEY as needed (.env is git-ignored)
config/
  config.yaml       # every knob: brand, thresholds, model choices, retrieval k
  taxonomy.yaml     # the 10-intent taxonomy: definitions, edge rules, handling policy per intent
src/support_agent/
  data/             # thread reconstruction (union-find), brand learnability survey, corpus cleaning
  taxonomy/         # weak (rule-based) labeler used for stratification + distant supervision
  retrieval/        # embeddings (Ollama + offline TF-IDF fallback, per-text cached), precedent index
  baselines/        # trivial (majority-class + canned reply) and simple (TF-IDF+LR + kNN reply)
  agent/            # the real pipeline: classify -> retrieve -> draft -> triage
  llm/              # multi-backend client (gemini/anthropic/ollama/cached) with a content-addressed,
                    # replayable on-disk cache and quota-vs-transient-error-aware retries
  eval/             # automated metrics (classification, selective-prediction/AURC, reply proxies)
                    # + LLM-judge rubric with anti-bias measures + human-agreement (QWK) computation
  labeling/         # golden-set sampler + the merge/build script for the hand labels
data/
  golden/           # golden_v1.jsonl (220 examples), judge_calibration.json (45 items), labels
  interim/          # cleaned corpus, weak labels, embedding cache (committed; raw CSV is not)
scripts/            # run_baselines.py, run_llm_agent.py, run_judge_calibration.py
tests/              # offline unit tests (union-find, kappa math, cache-key stability, ...)
docs/
  REPORT.md              # the actual report: framing, results, failure analysis, misleading-number section, next steps
  DECISIONS.md            # 20 non-obvious decisions and why
  LABELING_PROTOCOL.md    # golden-set sampling + labeling method, and who labeled it
```

## Development: lint, tests, CI, Docker

```bash
make lint   # ruff check src/ scripts/ tests/
make test   # pytest tests/ -v  (8 offline unit tests, no key/network needed, ~3s)
```

Both run in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) on every push/PR to `main`,
alongside a rebuild of the retrieval index and both baselines from the committed artifacts (i.e. CI
re-verifies the "reproduces in under a minute, no API key" claim on every commit, not just at
submission time) and a `LLM_BACKEND=cached` replay of the real LLM-agent/judge sample. CI
deliberately makes zero LLM API calls — no secret is available to a fork's PR run, and the
free/offline path is exactly what should stay honest on every commit.

```bash
docker build -t hulu-support-agent .
docker run --rm hulu-support-agent              # runs scripts/run_baselines.py by default
docker run --rm hulu-support-agent python -m pytest tests/ -v
docker run --rm --env-file .env -e LLM_BACKEND=gemini hulu-support-agent python scripts/run_llm_agent.py
```

## A note on how this was built

This repository, including the taxonomy, the golden-set labels, and the report, was produced
with AI assistance (Claude) working from the assignment brief, under the repo owner's direction.
The golden-set labels in particular were assigned by a single annotator following the written
codebook in `config/taxonomy.yaml` — see the disclosure at the top of
[`docs/LABELING_PROTOCOL.md`](docs/LABELING_PROTOCOL.md) before treating them as independently
verified ground truth, and consider spot-checking a sample against your own reading before
relying on them further.
