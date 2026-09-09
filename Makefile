.PHONY: setup download data index baselines llm-agent judge-calibration all-free clean smoke

PY := python

setup:
	$(PY) -m pip install -r requirements.txt

# Only needed if you want to rebuild data/interim/*.jsonl from scratch. The
# fast reproduction path below does NOT need this -- hulu_support_cases.jsonl
# and the embedding cache are already committed.
download:
	mkdir -p data/raw
	curl -sL -o data/raw/twcs.csv \
	  https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv
	@echo "downloaded $$(du -h data/raw/twcs.csv | cut -f1) to data/raw/twcs.csv"

# Full rebuild from the raw CSV: union-find thread reconstruction, brand
# extraction, cleaning, weak labels, and full-corpus embeddings. Takes ~15-20
# minutes the FIRST time (dominated by embedding 14.7k texts through a local
# Ollama model with no GPU) and is not needed to reproduce the headline
# numbers -- see `baselines` below, which uses the committed artifacts instead.
data: download
	$(PY) -m support_agent.data.build_brand_dataset --brands hulu_support --out-dir data/interim
	$(PY) -m support_agent.data.corpus --brand hulu_support --in-dir data/interim
	$(PY) -m support_agent.taxonomy.weak_labels --cases data/interim/hulu_support_cases.jsonl --out data/interim/hulu_weak_labels.jsonl

# Rebuilds the retrieval index from the COMMITTED embedding cache -- a pure
# cache-hit, a few seconds, no network and no Ollama required.
index:
	$(PY) -m support_agent.retrieval.index --cases data/interim/hulu_support_cases.jsonl --out-prefix artifacts/precedent_index

# --- reproduce the headline numbers: < 1 minute, no API key, no download ---
baselines: index
	$(PY) scripts/run_baselines.py

# --- optional: the LLM agent + judge (needs ANTHROPIC_API_KEY; ~$2-5, ~10-15 min) ---
# config/config.yaml defaults llm.backend to 'cached' (safe: never spends money unless asked),
# so these targets set LLM_BACKEND=anthropic explicitly rather than relying on ANTHROPIC_API_KEY
# alone to switch modes.
llm-agent:
	LLM_BACKEND=anthropic $(PY) scripts/run_llm_agent.py

judge-calibration:
	LLM_BACKEND=anthropic $(PY) scripts/run_judge_calibration.py

# One command replaying already-committed results: no key, no cost, no wait.
llm-agent-cached:
	LLM_BACKEND=cached $(PY) scripts/run_llm_agent.py

judge-calibration-cached:
	LLM_BACKEND=cached $(PY) scripts/run_judge_calibration.py

smoke:
	LLM_BACKEND=cached $(PY) scripts/run_llm_agent.py --limit 2

clean:
	rm -rf artifacts/precedent_index.*
